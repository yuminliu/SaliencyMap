#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:27:21 2021

@author: liu.yum
"""

#%%
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
from inference_cnn import myinference
import main_cnn
     
#%%
#def train_cnn(ngcm=0):
start_time = time.time()
is_debug = True # False # 
noise_std = 0.0 # 0.05 # std for additive Gaussian noise
#ngcm = 'all' # 10 # 'None' # ## which GCM to select
#predictor = 'GCM{}'.format(ngcm) # 
predictor = 'GCM' # 'Reanalysis' # 'sst_masked' # 'gcm' # 
predictand = 'Amazon' # 'Congo' # 'mississippi' #'gcmnino34' # 'gcmnino3' # 'nino3' # 'nino34_anom' # 'dmi' # 'ppt'
sst = True # mask out land if true, only use sea surface temperature
lr = 10*5e-6 # 5e-6 # 
seed = 123 # np.random.randint(low=0,high=2**63-1) # 
#### model parameters
input_channels = 32 # 1 # 3 # 4 # 12 # 24
output_channels = 1
hidden_channels = [32,32,64] # [8,16,32,64,128,256] #[8,8,16,16,32,32,64,46,128,128,256,256]
fc_channels = [128,64]
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
# is_debug = True # False # 
# noise_std = 0.0 # 0.05 # std for additive Gaussian noise
ngcm = 'all' # 0 # which GCM to select
tstep = 0 # 12 # time lag in predictor
window = 3 # moving window
num_epochs = 200 # 100 # 500 # 500#300#
batch_size = 64 #128
# lr = 5e-6 # 10*5e-6
lr_patience = 5
weight_decay = 1e-4
num_workers = 8
model_name = 'myCNN' # 'ConvLSTM'
if 'GCM' in predictor: # predictor=='GCM':
    datapath = '/scratch/wang.zife/YuminLiu/DATA/GCM/GCMdata/tas/processeddata/32gcms_tas_monthly_1by1_195001-200512_World.npy' # 
elif 'Reanalysis' in predictor: # predictor=='Reanalysis':
    datapath = '../data/Climate/Reanalysis/sst_cobe_hadley_noaa_uod_195001-200512_1by1_world.npy'
#data_name = 'UoD_NOAA_ExtendedReconstructed_RiverFlow' # 'CPC' #
#gcm_names = np.load('/scratch/wang.zife/YuminLiu/DATA/GCM/GCMdata/tas/processeddata/gcm_names.npy')
riverpath = '../data/Climate/RiverFlow/processed/riverflow.csv'
#predictand = 'nino34' # 'nino34_anom' # 'dmi' # 'amazon' # 'nino3' #'ppt'
verbose = True # False
#datapath = '../data/'
save_root_path = '../results/'
#save_root_path += model_name+'/lag_{}/'.format(tstep)
#save_root_path += model_name+'/lag_{}/{}/'.format(tstep,predictand)
save_root_path += model_name+'/lag_{}/{}/SingleGCM/EachGCM/'.format(tstep,predictand) 

# seed = 123 # np.random.randint(low=0,high=2**63-1) # 
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = True # true if input size not vary else false
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('using device {}'.format(device))
nGPU = torch.cuda.device_count() # number of GPU used, 0,1,..
if is_debug: num_epochs = 2
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
#paras['data_name'] = data_name
paras['verbose'] = verbose
paras['datapath'] = datapath
paras['riverpath'] = riverpath
paras['save_root_path'] = save_root_path
paras['device'] = device
paras['nGPU'] = nGPU
paras['predictor'] = predictor
paras['predictand'] = predictand
paras['sst'] = sst
paras['noise_std'] = noise_std
paras['ngcm'] = ngcm
#paras['gcm_name'] = gcm_names[ngcm]
paras['seed'] = seed

#%%
train_dataset = datasets.myDataset_CNN(predictor=predictor,predictand=predictand,imgpath=datapath,sst=sst,fold='train',window=window,noise_std=noise_std,ngcm=ngcm)
valid_dataset = datasets.myDataset_CNN(predictor=predictor,predictand=predictand,imgpath=datapath,sst=sst,fold='valid',window=window,noise_std=noise_std,ngcm=ngcm)
test_dataset = datasets.myDataset_CNN(predictor=predictor,predictand=predictand,imgpath=datapath,sst=sst,fold='test',window=window,noise_std=noise_std,ngcm=ngcm)
print('len(train_dataset)={}'.format(len(train_dataset)))
print('len(valid_dataset)={}'.format(len(valid_dataset)))
train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
valid_loader = DataLoader(dataset=valid_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
test_loader = DataLoader(dataset=test_dataset,batch_size=1,shuffle=False,num_workers=num_workers)
input_size = next(iter(train_loader))[0][0].shape # [batch,channel,height,width]
print('input_size={}'.format(input_size))

#%%
model = models.CNN(**hyparas)
print('model=\n\n{}'.format(model))
if nGPU>1:
    print('Using {} GPUs'.format(nGPU))
    model = torch.nn.DataParallel(model)
model = model.to(device)
summary(model,input_size)
optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
criterion = torch.nn.MSELoss() # utils.extreme_enhence_MSELoss #
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=lr_patience)

#%% create save path
#savepath = utils.create_savepath(rootpath=save_root_path,is_debug=is_debug)   
savepath = save_root_path+'{}/'.format(ngcm)
import os
if not os.path.exists(savepath): os.makedirs(savepath)
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
    matplotlib.use('Agg')
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
mae,mse,rmse = myinference(hyparas,paras,test_loader,folder='test')
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

main_cnn.cal_saliency_maps(hyparas=hyparas,paras=paras,test_loader=test_loader,folder='test')    


# all_dataset = datasets.myDataset_CNN(fold='all',window=window,noise_std=noise_std,ngcm=ngcm)
# train_loader = DataLoader(dataset=train_dataset,batch_size=1,shuffle=False,num_workers=num_workers)
# valid_loader = DataLoader(dataset=valid_dataset,batch_size=1,shuffle=False,num_workers=num_workers)
# test_loader = DataLoader(dataset=test_dataset,batch_size=1,shuffle=False,num_workers=num_workers)
# all_loader = DataLoader(dataset=all_dataset,batch_size=1,shuffle=False,num_workers=num_workers)    
# #paras['checkpoint_name'] = '{}_epoch_{}.pth'.format(model_name,'best')
# #paras['savepath'] = '../results/myCNN/lag_0/nino34/2021-03-23_17.56.37.146542_gcm_masked_nino34/'
# _,_,_ = myinference(hyparas,paras,train_loader,folder='train')
# _,_,_ = myinference(hyparas,paras,valid_loader,folder='valid')
# _,_,_ = myinference(hyparas,paras,test_loader,folder='test')
# _,_,_ = myinference(hyparas,paras,all_loader,folder='all')

torch.cuda.empty_cache()
#print('{}: Job done! total running time: {} minutes!\n'.format(predictand,run_time))
print('{}: Job done! total running time: {} minutes!\n'.format(ngcm,run_time))




#%%

#     return 

# for i in range(13,16): # range(29,32): # [0,1]: # 
#    train_cnn(ngcm=i)

