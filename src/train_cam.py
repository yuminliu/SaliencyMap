#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 21:49:03 2021

@author: wang.zife
"""

#%%
#def main(lag,month):
import time
import json
import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torchsummary import summary
import datasets
import models
#import convlstm_model_2 as models
import utils
from inference_cam import myinference


#%%
start_time = time.time()
#### model parameters
input_channels = 1 # 3 # 4 # 32 # 12 # 24
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
is_debug = False # True # 
tstep = 0 #12 # time lag in predictor
window = 3 # moving window
num_epochs = 200 # 100 # 500 # 500#300#
batch_size = 64 #128
lr = 5e-5#5e-6
lr_patience = 5
weight_decay = 1e-4
num_workers = 8
model_name = 'CAM' # 'ConvLSTM'
data_name = '/scratch/wang.zife/YuminLiu/DATA/GCM/GCMdata/tas/processeddata/32gcms_tas_monthly_1by1_195001-200512_World.npy' # 
#data_name = '../data/Climate/SST/sst_cobe_hadley_noaa_uod_195001-200512_1by1_world.npy' # 'UoD_NOAA_ExtendedReconstructed_RiverFlow' # 'CPC' #
variable = 'amazon' # 'nino3' #'ppt'
verbose = True # False
datapath = '../data/'
save_root_path = '../results/'
save_root_path += model_name+'/lag_{}/'.format(tstep)
seed = 123
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = True # true if input size not vary else false
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('using device {}'.format(device))
nGPU = torch.cuda.device_count() # number of GPU used, 0,1,..
if is_debug: num_epochs = 12
paras = {}
paras['window'] = window
paras['tstep'] = tstep
paras['is_debug'] = is_debug
paras['num_epochs'] = num_epochs
paras['batch_size'] = batch_size
paras['lr'] = lr
paras['lr_patience'] = lr_patience
paras['weight_decay'] = weight_decay
paras['num_workers'] = num_workers
paras['model_name'] = model_name
paras['data_name'] = data_name
#paras['dataname'] = dataname
paras['verbose'] = verbose
paras['datapath'] = datapath
paras['save_root_path'] = save_root_path
paras['device'] = device
paras['nGPU'] = nGPU
paras['variable'] = variable

#%%
train_dataset = datasets.myDataset_CNN(fold='train',window=window)
valid_dataset = datasets.myDataset_CNN(fold='valid',window=window)
test_dataset = datasets.myDataset_CNN(fold='test',window=window)
print('len(train_dataset)={}'.format(len(train_dataset)))
print('len(valid_dataset)={}'.format(len(valid_dataset)))
train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
valid_loader = DataLoader(dataset=valid_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
test_loader = DataLoader(dataset=test_dataset,batch_size=1,shuffle=False,num_workers=num_workers)
input_size = next(iter(train_loader))[0][0].shape # [batch,channel,height,width]
print('input_size={}'.format(input_size))

#%%
model = models.CAM(**hyparas)
#print('model=\n{}'.format(model))
if nGPU>1:
    print('Using {} GPUs'.format(nGPU))
    model = torch.nn.DataParallel(model)
model = model.to(device)
#summary(model,input_size)
optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
criterion = torch.nn.MSELoss() # utils.extreme_enhence_MSELoss #
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=lr_patience)

#%% create save path
savepath = utils.create_savepath(rootpath=save_root_path,is_debug=is_debug)
paras['savepath'] = savepath
train_losses, valid_losses,best_epoch = [], [], 0
for epoch in range(1,num_epochs+1):
    train_loss = utils.train_one_epoch(model,optimizer,criterion,train_loader,epoch,device,num_epochs)
    train_losses.append(train_loss)
    valid_loss = utils.validate(model,criterion,valid_loader,device)
    valid_losses.append(valid_loss)
    lr_scheduler.step(valid_loss)
    
    ## save checkpoint model
    if epoch and epoch%100==0:
        utils.save_checkpoint(savepath=savepath,epoch=epoch,model=model,optimizer=optimizer,
                              train_losses=train_losses,valid_losses=valid_losses,
                              lr=lr,lr_patience=lr_patience,model_name=model_name,nGPU=nGPU)
    ## save best model
    if epoch==1 or valid_losses[-1]<valid_losses[-2]:
        utils.save_checkpoint(savepath=savepath,epoch='best',model=model,optimizer=optimizer,
                              train_losses=train_losses,valid_losses=valid_losses,
                              lr=lr,lr_patience=lr_patience,model_name=model_name,nGPU=nGPU)
        best_epoch = epoch
        
paras['checkpoint_name'] = '{}_epoch_{}.pth'.format(model_name,'best')

#%% plot losses
if verbose:
    import matplotlib
    #matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xx = range(1,len(train_losses)+1)
    fig = plt.figure()
    plt.plot(xx,train_losses,'b--',label='Train_losses')
    plt.plot(xx,valid_losses,'r-',label='Valid_losses')
    plt.legend()
    plt.title('Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss(avgerage MSE)')
    savename = 'losses'
    if savepath:
        plt.savefig(savepath+savename+'.png',dpi=1200,bbox_inches='tight')
    #plt.show()
    #plt.close()

#%% inference
mae,mse,rmse = myinference(hyparas,paras,test_loader)
run_time = (time.time()-start_time)/60 # minutes
paras['device'] = str(device)
results = {}
results['mae'] = mae
results['mse'] = mse
results['rmse'] = rmse
results['best_epoch'] = best_epoch
results['run_time'] = str(run_time)+' minutes'
configs = {**hyparas,**paras,**results}
with open(savepath+'configs.txt', 'w') as file:
     file.write(json.dumps(configs,indent=0)) # use `json.loads` to do the reverse

data_mean,data_std = train_dataset.data_mean,train_dataset.data_std#.cpu().detach().numpy()
y_mean,y_std = train_dataset.y_mean,train_dataset.y_std#.cpu().detach().numpy()
if savepath:
    np.savez(savepath+'data_mean_data_std_y_mean_y_std.npz',data_mean=data_mean,data_std=data_std,y_mean=y_mean,y_std=y_std)

torch.cuda.empty_cache()
print('{}: Job done! total running time: {} minutes!\n'.format(variable,run_time))




