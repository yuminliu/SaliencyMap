#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 18:37:57 2020

@author: wang.zife
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 13:07:33 2020

@author: wang.zife
"""


#%%
def dnn_main(hyparas,paras):
    import time
    import json
    import numpy as np
    import torch
    from torch.utils.data.dataloader import DataLoader
    from torchsummary import summary
    import mydatasets
    import models
    #import convlstm_model_2 as models
    import utils
    from dnn_inference import myinference
    
    #%%
    #### model parameters
#    hyparas = {}
#    hyparas['input_channels'] = input_channels = 120 # 6 # 8 # 8*lag # 7 #
#    hyparas['output_channels'] = output_channels # 1
#    hyparas['hidden_channels'] = hidden_channels = [20,10,5] # len(hidden_channels)
    ### other parameters
    start_time = time.time()
    data = paras['data']
    Ntrain = paras['Ntrain']
    Nvalid = paras['Nvalid']
    Ntest = paras['Ntest'] 
    is_debug = paras['is_debug']
    num_epochs = paras['num_epochs']
    batch_size = paras['batch_size'] 
    lr = paras['lr'] 
    lr_patience = paras['lr_patience'] 
    weight_decay = paras['weight_decay'] 
    num_workers = paras['num_workers'] 
    model_name = paras['model_name'] 
    #paras['data_name'] = data_name
    #paras['dataname'] = dataname
    verbose = paras['verbose'] 
    #paras['datapath'] = datapath
    save_root_path = paras['save_root_path'] 
    #device = paras['device']
    #nGPU = paras['nGPU'] 
    predictor = paras['predictor']
    predictand = paras['predictand'] 
    tstep = paras['tstep']
    seed = paras['seed']

    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True # true if input size not vary else false
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('using device {}'.format(device))
    nGPU = torch.cuda.device_count() # number of GPU used, 0,1,..
    paras['device'] = str(device)
    paras['nGPU'] = nGPU
    if is_debug: num_epochs = 2
    
    X,y = data['X'],data['y']
    train_dataset = mydatasets.ARDataset(X=X,y=y,fold='train',Ntrain=Ntrain,Nvalid=Nvalid)
    valid_dataset = mydatasets.ARDataset(X=X,y=y,fold='valid',Ntrain=Ntrain,Nvalid=Nvalid)
    test_dataset = mydatasets.ARDataset(X=X,y=y,fold='test',Ntrain=Ntrain,Nvalid=Nvalid)
    print('len(train_dataset)={}'.format(len(train_dataset)))
    print('len(valid_dataset)={}'.format(len(valid_dataset)))
    train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers)
    valid_loader = DataLoader(dataset=valid_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset,batch_size=1,shuffle=False,num_workers=num_workers)
    input_size = next(iter(train_loader))[0][0].shape # [batch,channel,height,width]
    print('input_size={}'.format(input_size))
    #%%
    model = models.DNN(**hyparas)
    #print('model=\n{}'.format(model))
    if nGPU>1:
        print('Using {} GPUs'.format(nGPU))
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    ##summary(model,input_size[1:])
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
    res_train = myinference(hyparas,paras,train_loader,datasettype='train')
    res_valid = myinference(hyparas,paras,valid_loader,datasettype='valid')
    
    y_pred_train_valid = np.concatenate((res_train['y_pred'],res_valid['y_pred']),axis=0)
    y_train_valid = np.concatenate((res_train['y_test'],res_valid['y_test']),axis=0)
        
    res = myinference(hyparas,paras,test_loader,datasettype='test')
    mae,mse,rmse,rae_avg = res['mae'],res['mse'],res['rmse'],res['rae_avg']
    run_time = (time.time()-start_time)/60 # minutes
    paras['data'] = ''
    results = {}
    results['mae'] = float(mae)
    results['mse'] = float(mse)
    results['rmse'] = float(rmse)
    results['rae_avg'] = float(rae_avg)
    results['best_epoch'] = best_epoch
    results['run_time'] = str(run_time)+' minutes'
    configs = {**hyparas,**paras,**results}
    with open(savepath+'configs.txt', 'w') as file:
         file.write(json.dumps(configs,indent=0)) # use `json.loads` to do the reverse
    
    torch.cuda.empty_cache()
    print('{} to {}: Job done! total running time: {} minutes!\n'.format(predictor,predictand,run_time))
    
    return res['y_pred'], res['y_test'], y_pred_train_valid, y_train_valid
    #from dnn_inference import myinference
    #_ = myinference(hyparas,paras,train_loader)
    #
    #dnn = np.load('../results/analysis/Regression//Amazon/Amazon_dnnRegression_train.npz')
    #dnn_pred = dnn['y_pred']
    #y_test = dnn['y_test']


