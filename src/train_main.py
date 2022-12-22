#%%
#def main(variable,resolution):
import time
import json
#import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import datasets
import models
import utils
from inference import inference
#%%
start_time = time.time()

is_debug = False # True # 
variable = 'tmax' # 'tmin' # 'ppt' # 
resolution = 0.125 # 0.25 # 0.5 # 
num_epochs = 300 # 100 # 

#### model parameter setting
if variable=='ppt':
    input_channels = 35 #
elif variable=='tmax' or variable=='tmin':
    input_channels = 33 # 35 #
output_channels = 1
hidden_channels = 64 # number of feature maps for hidden layers
num_layers = 15 # number of Conv/Deconv layer pairs
scale = int(1/resolution) # 8 # 2 # downscaling factor
use_climatology = True
hyparas = {}
hyparas['input_channels'] = input_channels
hyparas['output_channels'] = output_channels
hyparas['hidden_channels'] = hidden_channels
hyparas['num_layers'] = num_layers
hyparas['scale'] = scale
hyparas['use_climatology'] = use_climatology

#### other parameters setting
batch_size = 32
lr = 1e-4
lr_patience = 5
num_workers = 8
model_name = 'YNet'
datapath = '../data/Climate/PRISM_GCMdata/{}/{}by{}/'.format(variable,resolution,resolution)
train_datapath = datapath+'train/'
valid_datapath = datapath+'val/'
test_datapath = datapath+'test/'
save_root_path = '../results/Climate/PRISM_GCM/{}/{}/scale{}/'.format(model_name,variable,scale)
verbose = True # bool, plot figures or not
seed = 123
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = True # true if input size not vary else false
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nGPU = torch.cuda.device_count() # number of GPU used, 0,1,..
print('using device {} with {} GPUs'.format(device,nGPU))
if is_debug: num_epochs = 5

paras = {}
paras['is_debug'] = is_debug
paras['num_epochs'] = num_epochs
paras['batch_size'] = batch_size
paras['lr'] = lr
paras['lr_patience'] = lr_patience
paras['num_workers'] = num_workers
paras['resolution'] = resolution
paras['variable'] = variable
paras['train_datapath'] = train_datapath
paras['valid_datapath'] = valid_datapath
paras['test_datapath'] = test_datapath
paras['save_root_path'] = save_root_path
paras['model_name'] = model_name
paras['device'] = device
paras['nGPU'] = nGPU
paras['verbose'] = verbose
#%%
train_dataset = datasets.myDataset(datapath=train_datapath)
valid_dataset = datasets.myDataset(datapath=valid_datapath)
print('len(train_dataset)={}'.format(len(train_dataset)))
print('len(valid_dataset)={}'.format(len(valid_dataset)))
train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
valid_loader = DataLoader(dataset=valid_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)

#%% create model
model = models.YNet(input_channels=input_channels,output_channels=output_channels,
                    hidden_channels=hidden_channels,num_layers=num_layers,
                    scale=scale,use_climatology=use_climatology)
#print('model=\n{}'.format(model))
if nGPU>1: model = torch.nn.DataParallel(model)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.MSELoss()
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=lr_patience)
#%% create save path
savepath = utils.create_savepath(rootpath=save_root_path,is_debug=is_debug)
paras['savepath'] = savepath
#%% begin training model
train_losses, valid_losses, best_epoch = [], [], 0
for epoch in range(num_epochs):
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
    if epoch==0 or valid_losses[-1]<valid_losses[-2]:
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
    plt.ylabel('Loss')
    savename = 'losses'
    if savepath:
        plt.savefig(savepath+savename+'.png',dpi=1200,bbox_inches='tight')
    #plt.show()

#%% inference
mae,mse,rmse = inference(hyparas,paras)
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

torch.cuda.empty_cache()
print('{} {}: Job done! total running time: {} minutes!\n'.format(variable,resolution,run_time))

#for variable in ['tmin']: #['ppt','tmax','tmin']:
#    for resolution in [0.5]: #[0.5,0.25,0.125]:
#        main(variable,resolution)
