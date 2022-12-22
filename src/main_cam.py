#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 23:27:29 2021

@author: wang.zife
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 12:57:29 2021

@author: wang.zife
"""
#%%
import os
import json
import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
import models
import datasets

#%%
model_name = 'CAM'
model_saved_date = '2021-02-02_00.25.33.460016' # '2021-02-01_22.44.06.929407'
model_path = '../results/CAM/lag_0/{}/'.format(model_saved_date)
save_root_path = model_path+'{}_Saliency/'.format(model_saved_date)
checkpoint_pathname = model_path+'CAM_epoch_best.pth'

#### model parameters
input_channels = 1 # 3 # 32 # 4 # 12 # 24
output_channels = 1
hidden_channels = [64,128,256,512] # [8,16,32,64,128,256] #[8,8,16,16,32,32,64,46,128,128,256,256]
fc_channels = [128]
drop_rate = 0 # 0.5
activation = 'ReLU' # 'LeakyReLU' # 'RReLU' # 

hyparas = {}
hyparas['input_channels'] = input_channels # 6 # 8 # 8*lag # 7 #
hyparas['output_channels'] = output_channels # 1
hyparas['hidden_channels'] = hidden_channels # len(hidden_channels)
hyparas['fc_channels'] = fc_channels
hyparas['drop_rate'] = drop_rate
hyparas['activation'] = activation
#### other parameters
window = 3 # moving window
data_name = '/scratch/wang.zife/YuminLiu/DATA/GCM/GCMdata/tas/processeddata/32gcms_tas_monthly_1by1_195001-200512_World.npy' # 
#data_name = '../data/Climate/SST/sst_cobe_hadley_noaa_uod_195001-200512_1by1_world.npy' # 'UoD_NOAA_ExtendedReconstructed_RiverFlow' # 'CPC' #
variable = 'nino3' # 'amazon' # 'ppt'
verbose = True # False
seed = 123
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = True # true if input size not vary else false
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('using device {}'.format(device))
nGPU = torch.cuda.device_count() # number of GPU used, 0,1,..
paras = {}
paras['window'] = window
paras['model_name'] = model_name
paras['data_name'] = data_name
paras['verbose'] = verbose
paras['save_root_path'] = save_root_path
paras['device'] = str(device)
paras['nGPU'] = nGPU
paras['variable'] = variable

#%%
test_dataset = datasets.myDataset_CNN(fold='test',window=window)
test_loader = DataLoader(dataset=test_dataset,batch_size=1,shuffle=False)
#%%
model = models.CAM(**hyparas)
checkpoint = torch.load(checkpoint_pathname, map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['model_state_dict'])

model = model.to(device)
model.eval()

fc_weights = model.state_dict()['fc_layers.0.weight'].detach().cpu().numpy().squeeze()[:,np.newaxis,np.newaxis]

if save_root_path and not os.path.exists(save_root_path):
    os.makedirs(save_root_path)
saliency_maps = []
inputs,y = next(iter(test_loader))
cam_maps = []
for inputs,y in test_loader:
    #inputs # [batch,channel,height,width]
    inputs = inputs.to(device)
    inputs.requires_grad_()
    y_pred,cnn = model(inputs) # [1,1]
    
    cam_map = cnn.detach().cpu().numpy().squeeze()*fc_weights
    cam_map = np.sum(cam_map,axis=0)
    cam_maps.append(cam_map)
    
    
    y_pred = y_pred.squeeze() # [1,1] --> []
    y_pred.backward()
    gradient_map = inputs.grad.data.abs()
    saliency_maps.append(gradient_map)

saliency_maps = torch.cat(saliency_maps,dim=0)
print('saliency_maps.size()={}'.format(saliency_maps.size()))
saliency_maps = saliency_maps.cpu().detach().numpy()

cam_maps = np.stack(cam_maps,axis=0)
from skimage.transform import resize
cam_maps = np.transpose(resize(np.transpose(cam_maps,axes=(1,2,0)),saliency_maps.shape[2:],order=1,preserve_range=True),axes=(2,0,1))

if save_root_path:
    savename = '{}_{}_saliency_maps'.format(model_name,variable) 
    np.save(save_root_path+savename+'.npy',saliency_maps)
    np.save(save_root_path+savename.replace('saliency','cam')+'.npy',cam_maps)

    configs = {**hyparas,**paras}
    with open(save_root_path+'configs.txt', 'w') as file:
         file.write(json.dumps(configs,indent=0)) # use `json.loads` to do the reverse

from mpl_toolkits.basemap import Basemap
#import matplotlib
#matplotlib.use('Agg')
##matplotlib.use('Qt5Agg')
#import matplotlib.pyplot as plt

#%%
#def get_saliency(datapathname,varname,fillnan=None,threshold=0.05):
def get_saliency(saliency_maps,fillnan=None,threshold=0.05):
    #data = np.load(datapathname)
    #saliency_maps = data[varname]
    #saliency_maps = abs(saliency_maps[:,gcm,:,:])
    #saliency_maps = np.amax(abs(saliency_maps),axis=1)
    saliency_maps = abs(saliency_maps)
    #saliency_maps = np.sum(abs(saliency_maps),axis=1)
    #saliency_maps = np.std(abs(saliency_maps),axis=1)
    max_per_map = np.amax(saliency_maps,axis=(1,2))
    if fillnan!=None:
        for mon in range(len(saliency_maps)):
            saliency_maps[mon,:,:][saliency_maps[mon,:,:]<threshold*max_per_map[mon]] = fillnan
    return saliency_maps

def get_months(method):
    if method=='Months':
        months = [[i] for i in range(36)]
    elif method=='NaturalMonths':
        months = [[i,i+12,i+24] for i in range(12)]
    elif method=='Seasons':
        months = [[-1,0,1,11,12,13,23,24,25],[2,3,4,14,15,16,26,27,28],[5,6,7,17,18,19,29,30,31],[8,9,10,20,21,22,32,33,34]]
    elif method=='Years':
        months = [list(range(i,i+12)) for i in range(0,36,12)]
    return months

def get_title_savename(method,i,varname,variable,mapname):
    var2title = {'nino3':'Nino3','amazon':'Amazon River Flow','ppt':'Precipitation','tmax':'Max Temperature','tmin':'Min Temperature'}
    ind2month = {1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',
                 7:'July',8:'August',9:'September',10:'October',11:'November',12:'December'}
    ind2season = {1:'Spring(MAM)',2:'Summer(JJA)',3:'Autumn(SON)',4:'Winter(DJF)'}  
    ind2year = {1:2003,2:2004,3:2005}
    if method=='Months':
        YEAR, MONTH = str(2003+i//12), ind2month[i%12+1] 
        savename = '{}_{}_month_{}'.format(varname,variable,i)
        title = '{} Map for {} in {} {}'.format(mapname,var2title[variable],MONTH,YEAR)
        #title = 'Saliency Map std for {} in {} {}'.format(var2title[variable],MONTH,YEAR)
    elif method=='NaturalMonths':
        savename = '{}_{}_month_{}'.format(varname,variable,ind2month[i+1])
        title = '{} Map for {} in {}'.format(mapname,var2title[variable],ind2month[i+1])
        #title = 'Saliency Map std for {} in {}'.format(var2title[variable],ind2month[i+1])
    elif method=='Seasons':
        savename = '{}_{}_season_{}'.format(varname,variable,ind2season[i+1])
        title = '{} Map for {} in {}'.format(mapname,var2title[variable],ind2season[i+1])
        #title = 'Saliency Map std for {} in {}'.format(var2title[variable],ind2season[i+1])
    elif method=='Years':
        savename = '{}_{}_year_{}'.format(varname,variable,ind2year[i+1])
        title = '{} Map for {} in {}'.format(mapname,var2title[variable],ind2year[i+1])
        #title = 'Saliency Map std for {} in {}'.format(var2title[variable],ind2year[i+1])
    return title,savename


#%%
def plot_results(method='Years'):
    #import os
    #import numpy as np
    import plots
    
    #method = 'Years' # 'Seasons' # 'NaturalMonths' # 'Months' # 
    ## gcm region 1
    #lonlat = [50.5,-42.5,349.5,37.5] # [-124.5,24.5,-66.5,49.5] # map area, [left,bottom,right,top]
    #parallels = np.arange(-40.0,40.0,10.0)
    #meridians = np.arange(60.0,350.0,30.0) # label lons, 60E to -9.5W, or 60E to 350E
    ## sst region 1
    lonlat = [50.5,-42.5,349.5,37.5] # [-124.5,24.5,-66.5,49.5] # map area, [left,bottom,right,top]
    parallels = np.arange(-40.0,40.0,10.0)
    meridians = np.arange(60.0,350.0,30.0) # label lons, 60E to -9.5W, or 60E to 350E
    ## sst region 2
    #lonlat = [50.5,-20.5,259.5,20.5] # # map area, [left,bottom,right,top]
    #parallels = np.arange(-20.0,21.0,10.0)
    #meridians = np.arange(60.0,300.0,30.0) # label lons, 60E to -9.5W, or 60E to 350E
    ## sst region 3
    #lonlat = [150.5,-39.5,259.5,29.5] # # map area, [left,bottom,right,top]
    #parallels = np.arange(-30.0,30.0,10.0)
    #meridians = np.arange(150.0,300.0,30.0) # label lons, 60E to -9.5W, or 60E to 350E
    
    mapname = 'CAM' # 'Saliency'
    varname = '{}_gcms'.format(mapname.lower()) # 'saliency_gcms' # 'saliency_sst' # 'saliency_aux' # 
    variable = 'nino3' # 'amazon' # 
    
    #gcm_names = np.load('/scratch/wang.zife/YuminLiu/DATA/GCM/GCMdata/tas/processeddata/gcm_names.npy')
    #saliency_maps = np.load('../results/myCAM/lag_0/2021-01-18_12.40.16.939557_gcm_amazon/2021-01-18_12.40.16.939557_Saliency')
    #model_saved_date = '2021-01-31_15.58.06.538504_debug' #'2021-01-18_12.40.16.939557' # '2021-01-19_16.59.14.648607_debug' 
    #model_path = '../results/CAM/lag_0/{}_gcm_amazon/'.format(model_saved_date)
    #model_path = '../results/CAM/lag_0/{}_sst_amazon/'.format(model_saved_date)
    #save_root_path = model_path+'{}_Saliency/'.format(model_saved_date)
    
    #savepath_prefix = save_root_path+'SaliencyMaps/singleGCM/{}/'.format(method) # None # 
    watercolor = 'white' # '#46bcec'
    cmap = 'YlOrRd' # 'rainbow' # 'Accent' #'YlGn' #'hsv' #'seismic' # 
    alpha = 0.7
    projection = 'merc' # 'cyl' # 
    resolution = 'i' # 'h'
    area_thresh = 10000
    clim = None
    pos_lons, pos_lats = [], [] # None, None # to plot specific locations
    verbose = False # True
    savename = 'CAM_{}_{}_maps'.format(variable,mapname.lower())
    saliency_maps = np.load(save_root_path+savename+'.npy') # [Nmonth,Ngcm,Nlat,Nlon]    
    
#    for gcm in range(len(gcm_names)):
#        print('processing {}: {}-th GCM'.format(method,gcm+1))
#        gcm_name = str(gcm+1)+'_'+gcm_names[gcm].split('_')[4]
#        savepath = savepath_prefix+gcm_name+'/'
#        saliencymaps = plots.get_saliency(abs(saliency_maps[:,gcm,:,:]),fillnan=0.0,threshold=0.15)
        
    #%% 
    savepath = save_root_path+'{}Maps/{}/'.format(mapname,method) # None # 
    saliencymaps = get_saliency(saliency_maps,fillnan=0.0,threshold=0.15) # [Nmonth,Nlat,Nlon]
    if savepath and not os.path.exists(savepath):
        os.makedirs(savepath)
    months = get_months(method)  
    for i,month in enumerate(months):
        #if i>0: break
        title,savename = get_title_savename(method,i,varname,variable,mapname)
        img = np.mean(saliencymaps[month,:,:],axis=0)
        img[img==0] = np.nan
        plots.plot_map(img,title=title,savepath=savepath,savename=savename,cmap=cmap,alpha=alpha,
                     lonlat=lonlat,projection=projection,resolution=resolution,area_thresh=area_thresh,
                     parallels=parallels,meridians=meridians,pos_lons=pos_lons, pos_lats=pos_lats,clim=clim,
                     watercolor=watercolor,verbose=verbose)

for method in ['Years']: # ['Years','Seasons','NaturalMonths','Months']: # ['Seasons','NaturalMonths','Months']: # 
    plot_results(method)