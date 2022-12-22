#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 10:54:25 2020

@author: yumin
"""
import os
import time
import json
import torch
import models
#import utils
import numpy as np
from skimage.transform import resize
start_time = time.time()

#%% parameter settings
#is_debug = True # False # 
variable = 'tmax' # 'tmin' # 'ppt' #
readme = 'single_location' # 'multiple_locations'
loc_lat, loc_lon = 30,20 # 120,240 # latitude and longitude of location we want to get saliency map
resolution = 0.125 # 0.25 # 0.5 # 
scale = int(1/resolution) # 8 # 2 # downscaling factor
model_name = 'YNet'
model_saved_date = '2021-01-11_17.42.26.526113' # '2021-01-11_17.33.42.045321' # '2020-12-25_21.18.55.261503'
model_path = '../results/Climate/PRISM_GCM/{}/{}/scale{}/{}/'.format(model_name,variable,scale,model_saved_date)

datapath = '../data/Climate/PRISM_GCMdata/{}/{}by{}/'.format(variable,resolution,resolution)
#train_datapath = datapath+'train/'
#valid_datapath = datapath+'val/'
test_datapath = datapath+'test/'
save_root_path = '../results/Climate/PRISM_GCM/{}/{}/scale{}/'.format(model_name,variable,scale)
filenames = [f for f in os.listdir(test_datapath) if f.endswith('.npz')]
filenames = sorted(filenames)

#### model parameter setting
if variable=='ppt':
    input_channels = 35 #
elif variable=='tmax' or variable=='tmin':
    input_channels = 33 # 35 #
output_channels = 1
hidden_channels = 64 # number of feature maps for hidden layers
num_layers = 15 # number of Conv/Deconv layer pairs
use_climatology = True
model_name = 'YNet_epoch_best.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hyparas = {}
hyparas['input_channels'] = input_channels
hyparas['output_channels'] = output_channels
hyparas['hidden_channels'] = hidden_channels
hyparas['num_layers'] = num_layers
hyparas['scale'] = scale
hyparas['use_climatology'] = use_climatology

paras = {}
#paras['is_debug'] = is_debug
paras['readme'] = readme
paras['loc_lat_lon'] = 'lat_'+str(loc_lat)+'_lon_'+str(loc_lon)
paras['resolution'] = resolution
paras['variable'] = variable
paras['model_path'] = model_path
paras['test_datapath'] = test_datapath
paras['save_root_path'] = save_root_path
paras['model_name'] = model_name
paras['device'] = str(device)
savepath = save_root_path+model_saved_date+'_Saliency/lat_{}_lon_{}/'.format(loc_lat,loc_lon)
paras['savepath'] = savepath
if savepath and not os.path.exists(savepath):
    os.makedirs(savepath)
#%% load trained model
checkpoint_pathname = model_path+model_name
checkpoint = torch.load(checkpoint_pathname, map_location=lambda storage, loc: storage)
model = models.YNet(input_channels=input_channels,output_channels=output_channels,
                    hidden_channels=hidden_channels,num_layers=num_layers,
                    scale=scale,use_climatology=use_climatology)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

y_preds = []
saliency_gcms = []
saliency_gcms_large = []
saliency_aux = []
#%% load test data and forward and backward pass to get saliency map
for filename in filenames:#[0:10]:
    data = np.load(test_datapath+filename)
    X = data['gcms'] # tmax tmin:[-1,1], ppt: [0.0,1.0], [Ngcm,Nlat,Nlon]
    y = data['prism'] # [1,Nlat,Nlon], tmax/tmin:[-1.0,1.0], ppt:[0.0,1.0]
    
    input1 = torch.from_numpy(X[np.newaxis,...]).float() #coarse resolution GCM, [Ngcm,Nlat,Nlon]-->[1,Ngcm,Nlat,Nlon]
    X2 = resize(np.transpose(X,axes=(1,2,0)),y.shape[1:],order=1,preserve_range=True) # [Nlat,Nlon,Ngcm]
    X2 = np.transpose(X2,axes=(2,0,1))# [Ngcm,Nlat,Nlon]
    input2 = torch.from_numpy(X2[np.newaxis,...]).float() # interpolated coarse resolution GCM, [Ngcm,Nlat,Nlon]
    inputs = [input1,input2]
    if use_climatology:
        Xaux = np.concatenate((data['climatology'],data['elevation']),axis=0)  # [2,Nlat,Nlon]
        input3 = torch.from_numpy(Xaux[np.newaxis,...]).float() # auxiliary data, [1,2,Nlat,Nlon] --> [1,2,Nlat,Nlon]
        inputs += [input3]
    inputs = [e.to(device) for e in inputs]  
    inputs = [e.requires_grad_() for e in inputs]  
    y_pred = model(*inputs) # [1,1,Nlat,Nlon]
    y_pred_single_locations = y_pred[0,0,loc_lat,loc_lon]
    
    y_pred_single_locations.backward()
    
    gradient_maps = [e.grad.data.abs() for e in inputs]
    #print('saliency_map.size={}{}{}'.format(gradient_maps[0].size(),gradient_maps[1].size(),gradient_maps[2].size()))
    y_preds.append(y_pred)
    saliency_gcms.append(gradient_maps[0])
    saliency_gcms_large.append(gradient_maps[1])
    saliency_aux.append(gradient_maps[2])

y_preds = torch.cat(y_preds,dim=0)
saliency_gcms = torch.cat(saliency_gcms,dim=0)
saliency_gcms_large = torch.cat(saliency_gcms_large,dim=0)
saliency_aux = torch.cat(saliency_aux,dim=0)

print('y_preds.size()={}'.format(y_preds.size()))
print('saliency_gcms.size()={}'.format(saliency_gcms.size()))
print('saliency_gcms_large.size()={}'.format(saliency_gcms_large.size()))
print('saliency_aux.size()={}'.format(saliency_aux.size()))

y_preds = y_preds.cpu().detach().numpy()
saliency_gcms = saliency_gcms.cpu().detach().numpy()
saliency_gcms_large = saliency_gcms_large.cpu().detach().numpy()
saliency_aux = saliency_aux.cpu().detach().numpy()
savename = 'lat_{}_lon_{}_y_preds_saliency_gcms_saliency_gcms_large_saliency_aux'.format(loc_lat,loc_lon) 
np.savez(savepath+savename+'.npz',y_preds=y_preds,saliency_gcms=saliency_gcms,
         saliency_gcms_large=saliency_gcms_large,saliency_aux=saliency_aux)

configs = {**hyparas,**paras}
with open(savepath+'configs.txt', 'w') as file:
     file.write(json.dumps(configs,indent=0)) # use `json.loads` to do the reverse

#import matplotlib
#matplotlib.use('Qt5Agg')
#import matplotlib.pyplot as plt
#
#fig = plt.figure()
#plt.imshow(50*y_preds[0,0,:,:])
#plt.colorbar()
#
#fig = plt.figure()
#plt.imshow(np.abs(50*y_preds[0,0,:,:]))
#plt.colorbar()
