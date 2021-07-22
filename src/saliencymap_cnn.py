#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 12:57:29 2021

@author: wang.zife
"""
#%%
import os
#import json
import torch
import numpy as np
#from torch.utils.data.dataloader import DataLoader
import models
#import datasets
import plots

#%%
def plot_saliency_maps(method='Years',predictand='',predictor='',save_root_path=None):
    fillnan = None # 0.0
    threshold = 0.05
    # if not method:
    #     method = 'Years' # 'Seasons' # 'NaturalMonths' # 'Months' # 
    # if not predictand:
    #     predictand = 'nino34' # 'congo' # 'amazon' # 'nino3' # 
    # if not predictor:
    #     #predictor = 'saliency_gcm_{}_masked'.format(ngcm) #'saliency_gcms_masked_land' # 'saliency_gcms_masked_ocean' # 'saliency_sst_masked_ocean' # 'saliency_sst_masked_land' # 'saliency_sst_masked' # 'saliency_gcms_masked' # 'saliency_sst_hadley_masked_land' # 'saliency_sst_cobe_land' # 'saliency_sst_hadley_masked' # 'saliency_aux' # 
    #     predictor = 'gcm_masked'
    # if not save_root_path:
    #     model_saved_date = '' # '2021-03-23_14.25.56.880963_debug'
    #     save_root_path = '../results/myCNN/lag_0/{}/{}/Saliency/'.format(predictand,model_saved_date)
    
    ## gcm region 1
    lonlat = [50.5,-41.5,349.5,37.5] # [50.5,-42.5,349.5,37.5] # [-124.5,24.5,-66.5,49.5] # map area, [left,bottom,right,top]
    
    #lonlat = [50.5,-20.5,349.5,20.5] # region 4
    
    parallels = np.arange(-40.0,40.0,10.0)
    meridians = np.arange(60.0,350.0,30.0) # label lons, 60E to -9.5W, or 60E to 350E
    ## sst region 1
#    lonlat = [50.5,-41.5,349.5,37.5] # [50.5,-42.5,349.5,37.5] # [-124.5,24.5,-66.5,49.5] # map area, [left,bottom,right,top]
#    parallels = np.arange(-40.0,40.0,10.0)
#    meridians = np.arange(60.0,350.0,30.0) # label lons, 60E to -9.5W, or 60E to 350E
    ## region 1 mask
    masks = np.load('../data/Climate/Reanalysis/masks_cobe_hadley_noaa_uod_1by_world.npy')
    mask = masks[0,52:132,50:350] # cobe region 1
    #mask = masks[0,52:132,50:350] # cobe region 1
    #mask = masks[0,69:111,50:350] # cobe region 4
    ## sst region 2
    # lonlat = [50.5,-20.5,259.5,20.5] # # map area, [left,bottom,right,top]
    # parallels = np.arange(-20.0,21.0,10.0)
    # meridians = np.arange(60.0,300.0,30.0) # label lons, 60E to -9.5W, or 60E to 350E
    ## sst region 3
#    lonlat = [150.5,-39.5,259.5,29.5] # # map area, [left,bottom,right,top]
#    parallels = np.arange(-30.0,30.0,10.0)
#    meridians = np.arange(150.0,300.0,30.0) # label lons, 60E to -9.5W, or 60E to 350E 
    
    watercolor = 'white' # '#46bcec'
    cmap = 'YlOrRd' # 'bwr' # 'rainbow' # 'Accent' #'YlGn' #'hsv' #'seismic' # 
    alpha = 1.0 # 0.7
    projection = 'merc' # 'cyl' # 
    resolution = 'i' # 'h'
    area_thresh = 10000
    clim = None
    pos_lons, pos_lats = [], [] # None, None # to plot specific locations
    verbose = False # True
    saliency_maps = np.load(save_root_path+'myCNN_{}_{}_saliency_maps.npy'.format(predictor,predictand)) # [Nmonth,Ngcm,Nlat,Nlon] 
    
    
    # saliency_maps = np.load('../results/myCNN/lag_0/Amazon/2021-06-14_21.59.38.060129_GCM_masked_Amazon_region1/Saliency/myCNN_GCM_Amazon_saliency_maps.npy')
    # #cmap = 'bwr'
    # verbose = True
    # print('saliency_maps.min,max=[{},{}]'.format(np.min(saliency_maps),np.max(saliency_maps)))


    #print('original saliency_maps.shape={}'.format(saliency_maps.shape))
#    for gcm in range(len(gcm_names)):
#        print('processing {}: {}-th GCM'.format(method,gcm+1))
#        gcm_name = str(gcm+1)+'_'+gcm_names[gcm].split('_')[4]
#        savepath = savepath_prefix+gcm_name+'/'
#        saliencymaps = plots.get_saliency(abs(saliency_maps[:,gcm,:,:]),fillnan=fillnan,threshold=threshold)
        
    #%% 
    savepath = None if not save_root_path else save_root_path+'SaliencyMaps/{}/'.format(method) # None #
    #savepath = None if not save_root_path else save_root_path+'SaliencyMaps2/{}/'.format(method) # None # 
    #savepath = save_root_path+'SaliencyMaps_std/{}/'.format(method) # None #
    saliencymaps = plots.get_saliency(saliency_maps,fillnan=fillnan,threshold=threshold) # [Nmonth,Nlat,Nlon]
    #print('cyclical saliency_maps.shape={}'.format(saliencymaps.shape))
    if savepath and not os.path.exists(savepath):
        os.makedirs(savepath)
    months = plots.get_months(method)  
    for i,month in enumerate(months):
        #if i>0: break
        title,savename = plots.get_title_savename(method,i,predictor,predictand)
        img = np.mean(saliencymaps[month,:,:],axis=0)
        #img[img==0] = np.nan
        
        #img[mask!=0] = np.nan # mask out ocean
        img[mask==0] = np.nan # mask out land
        

        #title = ''



        plots.plot_map(img,title=title,savepath=savepath,savename=savename,cmap=cmap,alpha=alpha,
                        lonlat=lonlat,projection=projection,resolution=resolution,area_thresh=area_thresh,
                        parallels=parallels,meridians=meridians,pos_lons=pos_lons, pos_lats=pos_lats,clim=clim,
                        watercolor=watercolor,verbose=verbose)
    
#%%
def cal_saliency_maps(hyparas=None,paras=None,test_loader=None,folder='test'):
    #hyparas = paras = None
    #### model parameters
    # if not hyparas: 
    #     hyparas = {}
    #     hyparas['input_channels'] = input_channels = 1 # 32 # 3 # 4 # 12 # 24
    #     hyparas['output_channels'] = output_channels = 1
    #     hyparas['hidden_channels'] = hidden_channels = [32,32,64] # [8,16,32,64,128,256] #[8,8,16,16,32,32,64,46,128,128,256,256]
    #     hyparas['fc_channels'] = fc_channels = [128,64]
    #     hyparas['drop_rate'] = drop_rate = 0 # 0.5
    #     hyparas['activation'] = activation = 'ReLU' # 'LeakyReLU' # 'RReLU' #     
    # #### other parameters
    # if not paras:
    #     ngcm = 0 # 'all' # 
    #     window = 3 # moving window
    #     predictand = 'gcm1nino3' # 'nino34' # 'amazon' # 'nino3' # 'ppt'
    #     predictor = 'gcm_masked'
    #     model_name = 'myCNN'
    #     model_saved_date = '2021-01-25_23.00.54.863878'
    #     #model_saved_path = '../results/myCNN/lag_0/{}/{}/'.format(predictand,model_saved_date)
    #     model_saved_path = '../results/myCNN/lag_0/Amazon/2021-01-25_23.00.54.863878_gcm1_gcm1nino3/'
    #     save_root_path = model_saved_path+'{}_Saliency/'.format(model_saved_date)
    #     noise_std = 0.0
    if paras:
        #ngcm = paras['ngcm']
        model_name = paras['model_name']
        model_saved_path = paras['savepath']
        #window = paras['window']
        predictand = paras['predictand']
        predictor = paras['predictor']
        #noise_std = paras['noise_std']
        seed = paras['seed'] # 123

    save_root_path = model_saved_path+'Saliency/'
    #checkpoint_pathname = model_saved_path+'myCNN_epoch_best.pth'
    checkpoint_pathname = model_saved_path+'{}_epoch_best.pth'.format(model_name)
    #seed = paras['seed'] # 123
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True # true if input size not vary else false
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('using device {}'.format(device))
    #nGPU = torch.cuda.device_count() # number of GPU used, 0,1,..

    # #%%
    #test_dataset = datasets.myDataset_CNN(fold='test',window=window,noise_std=noise_std,ngcm=ngcm)
    #test_loader = DataLoader(dataset=test_dataset,batch_size=1,shuffle=False)
    #%%
    model = models.CNN(**hyparas)
    checkpoint = torch.load(checkpoint_pathname, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    if save_root_path and not os.path.exists(save_root_path):
        os.makedirs(save_root_path)
    saliency_maps = []
    for inputs,y in test_loader:
        #inputs # [batch,channel,height,width]
        inputs = inputs.to(device)
        inputs.requires_grad_()
        y_pred = model(inputs) # [1,1]
        y_pred = y_pred.squeeze() # [1,1] --> []
        y_pred.backward()
        gradient_map = inputs.grad.data.abs()
        saliency_maps.append(gradient_map)

    saliency_maps = torch.cat(saliency_maps,dim=0)
    saliency_maps = saliency_maps.cpu().detach().numpy()
    print('saliency_maps.shape()={}'.format(saliency_maps.shape))

    if save_root_path:
        savename = '{}_{}_{}_saliency_maps'.format(model_name,predictor,predictand) 
        np.save(save_root_path+savename+'.npy',saliency_maps)
    
    #### plot saliency maps
    #predictand,predictor = 'amazon','gcm_masked'
    #save_root_path = '../results/myCNN/lag_0/Amazon/2021-02-17_23.06.48.428983_gcm_masked_amazon_region1/2021-02-17_23.06.48.428983_Saliency/'
    for method in ['All','Years','Seasons']: # ['Years','Seasons','NaturalMonths','Months']: # ['Years']: # ['Seasons','NaturalMonths','Months']: # ['Seasons']: # ['Seasons','NaturalMonths','Months']: # 
        plot_saliency_maps(method=method,predictand=predictand,predictor=predictor,save_root_path=save_root_path)
    
        
#for i in range(7,32): # [7]: #
#    cal_saliency_maps(hyparas=None,paras=None)
   
#cal_saliency_maps(hyparas=None,paras=None)     

#%% ### plot saliency maps
# predictand = 'Amazon'
# predictor = 'GCM'
# for predictand in ['Amazon']: #['Amazon','Congo']:
#     for predictor in ['Reanalysis']: #['GCM','Reanalysis']:
#         if predictor=='GCM' and predictand== 'Amazon':
#             save_root_path = '../results/myCNN/lag_0/Amazon/2021-06-14_21.59.38.060129_GCM_masked_Amazon_region1/Saliency/'
#         elif predictor=='Reanalysis' and predictand== 'Amazon':
#             save_root_path = '../results/myCNN/lag_0/Amazon/2021-06-15_00.23.44.969560_Reanalysis_masked_Amazon_region1/Saliency/'
#         elif predictor=='GCM' and predictand== 'Congo':
#             save_root_path = '../results/myCNN/lag_0/Congo/2021-06-15_00.42.20.427649_GCM_masked_Congo_region1/Saliency/'
#         elif predictor=='Reanalysis' and predictand== 'Congo':
#             save_root_path = '../results/myCNN/lag_0/Congo/2021-06-14_23.51.47.085179_Reanalysis_masked_Congo_region1/Saliency/'

#         for method in ['Years']: # ['All']: # ['All','Years','Seasons']: # ['Years','Seasons','NaturalMonths','Months']: # 
#             plot_saliency_maps(method=method,predictand=predictand,predictor=predictor,save_root_path=save_root_path)
