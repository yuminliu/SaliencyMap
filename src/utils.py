class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


import torch
from tqdm import tqdm
def train_one_epoch(model,optimizer,criterion,train_loader,epoch,device,num_epochs):
    model.train()
    total_losses = AverageMeter()
    with tqdm(total=len(train_loader)) as _tqdm:
        _tqdm.set_description('train epoch: {}/{}'.format(epoch,num_epochs))
        for data in train_loader:
            inputs, labels = data
            labels = labels.to(device)
            if isinstance(inputs,list): # multiple inputs
                #print('len(inputs)={},inputs[0].size()={},inputs[1].size()={},labels.size()={}'.format(len(inputs),inputs[0].size(),inputs[1].size(),labels.size()))
                inputs = [e.to(device) for e in inputs]
                preds = model(*inputs)
            else: # single input
                #print('inputs.size()={},labels.size()={}'.format(inputs.size(),labels.size()))
                inputs = inputs.to(device)
                preds = model(inputs)
                #preds,_ = model(inputs)
            #print('inputs.size()={},preds.size()={},labels.size()={}'.format(inputs.size(),preds.size(),labels.size()))
            #inputs, labels = inputs.to(device), labels.to(device)
            #preds = model(inputs)
            loss = criterion(preds, labels)
            if isinstance(inputs,list):
                total_losses.update(loss.item(), len(inputs[0]))
            else:
                total_losses.update(loss.item(), len(inputs))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _tqdm.set_postfix(loss='{:.6f}'.format(total_losses.avg))
            _tqdm.update(1)
    return total_losses.avg

@torch.no_grad()
def validate(model,criterion,valid_loader,device):
    model.eval()
    total_losses = AverageMeter()
    with tqdm(total=len(valid_loader)) as _tqdm:
        _tqdm.set_description('valid progress: ')
        for data in valid_loader:
            inputs, labels = data
            labels = labels.to(device)
            
            if isinstance(inputs,list):
                inputs = [e.to(device) for e in inputs] 
                preds = model(*inputs)
            else:
                inputs = inputs.to(device)
                preds = model(inputs)
                #preds,_ = model(inputs)
            
            loss = criterion(preds, labels)
            if isinstance(inputs,list):
                total_losses.update(loss.item(), len(inputs[0]))
            else:
                total_losses.update(loss.item(), len(inputs))
            _tqdm.set_postfix(loss='{:.6f}'.format(total_losses.avg))
            _tqdm.update(1)
    return total_losses.avg


import os
import datetime
def create_savepath(rootpath='../results/',is_debug=True):
    timestamp = str(datetime.datetime.now()).replace(' ','_').replace(':','.')
    if is_debug:
        savepath = rootpath+timestamp+'_debug/'
    else:
        savepath = rootpath+timestamp+'/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    return savepath


#import torch
def save_checkpoint(savepath,epoch,model,optimizer,train_losses,valid_losses,lr,lr_patience,model_name='YNet30',nGPU=0):
    #### save models
    if nGPU>1:
        model_state_dict = model.module.state_dict() # multi GPU version
    else:
        model_state_dict = model.state_dict() # single GPU / CPU version
        
    torch.save({'epoch':epoch,
                'model_state_dict':model_state_dict,
                'optimizer_state_dict':optimizer.state_dict(),
                'train_losses':train_losses,
                'valid_losses':valid_losses,
                'lr':lr,
                'lr_patience':lr_patience}, 
                os.path.join(savepath, '{}_epoch_{}.pth'.format(model_name,epoch)))
    #torch.save({'train_losses':train_losses,'valid_losses':valid_losses},savepath+'losses_epoch_{}.pkl'.format(epoch))

import collections
def load_checkpoint(checkpoint_path,checkpoint_name,model,optimizer,device,nGPU):
    #checkpoint_path = '../results/ImageNet/saved/2019-11-14_21.43.07.504184/'
    #checkpoint_name = 'YNet30_epoch_21.pth'
    #checkpoint = torch.load(checkpoint_path+checkpoint_name)
    
    #checkpoint_path = '../results/ImageNet/2019-11-21_17.22.46.173887_debug/'
    #checkpoint_name = 'YNet30_epoch_3.pth'
    checkpoint = torch.load(checkpoint_path+checkpoint_name,map_location=device)
    model_state_dict = collections.OrderedDict()
    if nGPU>1: # multi GPU version
        for key in checkpoint['model_state_dict']:
            model_state_dict['module.'+key] = checkpoint['model_state_dict'][key]
    else: # single GPU / CPU version
        model_state_dict = checkpoint['model_state_dict']
                   
    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_start = checkpoint['epoch']+1
    train_losses = checkpoint['train_losses']
    valid_losses = checkpoint['valid_losses']
    
    res = {'model':model,'optimizer':optimizer,'epoch_start':epoch_start,
           'train_losses':train_losses,'valid_losses':valid_losses}
    
    return res


#%%
import numpy as np
import pandas as pd
def read_enso():
    years = np.load('../data/Climate/Nino/processed/Nino12_187001-201912_array.npz')['years']
    nino12 = pd.read_csv('../data/Climate/Nino/processed/Nino12_187001-201912_series.csv',index_col=0)
    nino3 = pd.read_csv('../data/Climate/Nino/processed/Nino3_187001-201912_series.csv',index_col=0)
    nino34 = pd.read_csv('../data/Climate/Nino/processed/Nino34_187001-201912_series.csv',index_col=0)
    nino4 = pd.read_csv('../data/Climate/Nino/processed/Nino4_187001-201912_series.csv',index_col=0)
    nino12_anom = pd.read_csv('../data/Climate/Nino/processed/Nino12_anom_187001-201912_series.csv',index_col=0)
    nino3_anom = pd.read_csv('../data/Climate/Nino/processed/Nino3_anom_187001-201912_series.csv',index_col=0)
    nino34_anom = pd.read_csv('../data/Climate/Nino/processed/Nino34_anom_187001-201912_series.csv',index_col=0)
    nino4_anom = pd.read_csv('../data/Climate/Nino/processed/Nino4_anom_187001-201912_series.csv',index_col=0)
    tni = pd.read_csv('../data/Climate/Nino/processed/Trans_Nino_index_hadISST_187001-201912_series.csv',index_col=0)
    soi = pd.read_csv('../data/Climate/Nino/processed/soi_186601-201912_series.csv',index_col=0)
    soi = soi.loc[187001:] # 187001 to 201912
    
    indices = pd.concat((nino12,nino3,nino34,nino4,nino12_anom,nino3_anom,nino34_anom,nino4_anom,tni,soi),axis=1) # 187001 to 201912
    return indices, years    

## https://stackoverflow.com/questions/15736995/how-can-i-quickly-estimate-the-distance-between-two-latitude-longitude-points
from math import radians, cos, sin, asin, sqrt
def gps2distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km


def get_X_y(predictor='GCM',predictand='Amazon',detrend=True,ismasked=True,standardized=False):
    #%% tas
    import numpy as np
    Ntrain = 600 # int(Ntotal*0.8)
    Nvalid = 36 # int(Ntotal*0.1)
    left,right = 50,350
    top,bottom = 50,130 # region 1
    if predictor=='GCM':
        imgpath = '/scratch/wang.zife/YuminLiu/DATA/GCM/GCMdata/tas/processeddata/32gcms_tas_monthly_1by1_195001-200512_World.npy'
        lats = np.load('/scratch/wang.zife/YuminLiu/DATA/GCM/GCMdata/tas/processeddata/lats_gcm.npy')
        lons = np.load('/scratch/wang.zife/YuminLiu/DATA/GCM/GCMdata/tas/processeddata/lons_gcm.npy')
        data = np.load(imgpath) # 87.5N to -87.5N by 1, 
    elif predictor=='Reanalysis':
        imgpath = '../data/Climate/Reanalysis/AlignedData/sst_cobe_hadley_noaa_uod_195001-200512_1by1_world.npy'
        lats = np.load('../data/Climate/Reanalysis/AlignedData/lats.npy')
        lons = np.load('../data/Climate/Reanalysis/AlignedData/lons.npy')
        data = np.load(imgpath) # 89.5N to -89.5N by 1, 
        data = data[:,0:3,:,:] # exclude uod  
    data = data[:,:,top:bottom,left:right] # 50.5E to 349.5E, 37.5N to -41.5N, # [672,32,80,300]
    lats = lats[top:bottom]
    lons = lons[left:right]
    cobe = np.load('/scratch/wang.zife/YuminLiu/DATA/COBE/processed/AlignedData/sst.mon.mean_185001-201912_1by1.npy')
    mask = np.mean(cobe,axis=0)
    mask = mask[top:bottom,left:right]
    mask = np.nan_to_num(mask,nan=1000)
    if ismasked:    
        for mon in range(len(data)):
            for d in range(data.shape[1]):
                data[mon,d,:,:][mask==1000] = 0      
    if detrend:### detrend
        mean_m,std_m = {},{}
        for d in range(data.shape[1]):
            for i in range(data.shape[2]):
                for j in range(data.shape[3]):
                    data_n = data[:,d,i,j]
                    for m in range(12):
                        mean_m[m] = np.mean(data_n[m::12])
                        std_m[m] = np.std(data_n[m::12],ddof=1)
                        data[m::12,d,i,j] = (data_n[m::12]-mean_m[m])/(std_m[m]+1e-15)
        
    #%% river flow 
    if predictand in ['Amazon','Congo']:
        column = {'Amazon':'0','Congo':'1'}
        riverflow_df = pd.read_csv('../data/Climate/RiverFlow/processed/riverflow.csv',index_col=0,header=0)
        targets = riverflow_df[[column[predictand]]].iloc[600:1272].to_numpy().reshape((-1,1)) # from 195001 to 200512
    if detrend:#### detrend y
        mean_m,std_m = {},{}
        for m in range(12):
            mean_m[m] = np.mean(targets[m::12,0])
            std_m[m] = np.std(targets[m::12,0],ddof=1)
            targets[m::12,0] = (targets[m::12,0]-mean_m[m])/(std_m[m]+1e-15)

    X = data
    y = targets # 

    ## standardize
    if standardized:
        data_mean = np.mean(X[:Ntrain],axis=0)
        data_std = np.std(X[:Ntrain],axis=0)+1e-5
        for t in range(len(X)):
            X[t,:,:,:] = (X[t,:,:,:]-data_mean)/data_std
        y_mean = np.mean(y[:Ntrain],axis=0)
        y_std = np.std(y[:Ntrain],axis=0)
        y = (y-y_mean)/y_std

    return X, y, lats,lons,mask