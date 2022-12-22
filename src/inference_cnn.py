#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:43:50 2021

@author: wang.zife
"""

#%%
#import os
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import models
#import mydatasets

#%%
#def myinfernece(savepath,test_loader,lag,month,hyparas):
def myinference(hyparas,paras,test_loader,folder='test'):
    
    predictand = paras['predictand'] #= "amazon"
    savepath = paras['savepath'] #= "../results/myCNN/lag_0/Amazon/2021-02-24_20.23.01.635288_sst_masked_amazon_region1/"
    #device = paras['device']
    checkpoint_name = paras['checkpoint_name'] #= "myCNN_epoch_best.pth" 
    checkpoint_pathname = savepath+checkpoint_name
    verbose = paras['verbose'] #= True

    predictor = paras['predictor'] #= "sst_masked" # "gcm_masked"
    #model_name = paras['model_name']
    #save_name_prefix = '{}_{}_{}'.format(model_name,predictor,predictand)
    save_name_prefix = '{}_{}'.format(predictor,predictand)


    #input_size = next(iter(test_dataset))[0][0].shape # [Tstep,channel,height,width]
    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #%%
    model = models.CNN(**hyparas)
    checkpoint = torch.load(checkpoint_pathname, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    mse,nsamples = 0,0
    preds,ys = [],[]
    for inputs,y in test_loader:
        #inputs # [batch,channel,height,width]
        # y [batch,]
        if isinstance(inputs,list):
            inputs = [e.to(device) for e in inputs]
            with torch.no_grad():
                pred = model(*inputs) # [1,1]
        else:
            inputs = inputs.to(device)
            with torch.no_grad():
                pred = model(inputs) # [1,1]
        #print('pred.shape={}'.format(pred.shape))
        pred = np.squeeze(pred.float().cpu().numpy()) # scalar
        y = np.squeeze(y.float().cpu().numpy()) # scalar?
        mse += np.sum((pred-y)**2)
        nsamples += 1
        
        preds.append(pred)
        ys.append(y)

    preds = np.stack(preds,axis=0) #[Ntest,], unit: mm/day
    ys = np.stack(ys,axis=0) #[Ntest,]

    ## scale back
    #preds = preds*300000
    #ys = ys*300000
    
    mse = mse/nsamples
    rmse = np.sqrt(mse)
    mae = np.sum(np.abs(ys-preds))/nsamples
    rmae = np.sum(np.abs(ys-preds)/(np.mean(np.abs(ys))+1e-5))/nsamples
    print('test RMSE:{}, MAE={}, RMAE={}'.format(rmse,mae,rmae))
    mse2 = np.mean((ys-preds)**2)
    rmse2 = np.sqrt(mse2)
    print('test RMSE2:{}'.format(rmse2))

    #%%
    if savepath:
        #np.savez(savepath+'{}_pred_results_RMSE{}_{}.npz'.format(save_name_prefix,rmse,folder),rmse=rmse,mae=mae,rmae=rmae,preds=preds,ytest=ys)
        np.savez(savepath+'{}_pred_results_{}.npz'.format(save_name_prefix,folder),rmse=rmse,mae=mae,rmae=rmae,preds=preds,ytest=ys)


    # import numpy as np
    # data = np.load('../results/myCNN/lag_0/amazon/2021-06-01_12.46.17.327893_gcm_masked_amazon_region1/gcm_masked_amazon_pred_results_RMSE0.2644236593943959_test.npz',allow_pickle=True)
    # data.files
    # folder = 'test'
    #ylabel = 'Standardized {} River Flow'.format(predictand)
    ylabel = 'Standardized River Flow'
    #%% plot figures
    year_months = np.load('../data/Climate/year_months_195001-200512.npy')
    interval = 12
    if folder=='test':
        year_months = year_months[636:]
        interval = 6
    elif folder=='valid':
        year_months = year_months[600:636]
        interval = 6
    elif folder=='train':
        year_months = year_months[0:600]
        interval = 12*5
    elif folder=='all':
        interval = 12*5
    if verbose:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig = plt.figure()
        #fig = plt.figure(figsize=(12,5))
        plt.plot(ys,'--k',label='Groundtruth')
        plt.plot(preds,'-r',label='Prediction')
        plt.xticks(ticks=range(0,len(ys),interval), labels=year_months[::interval])
        plt.title('Groundtruth vs Prediction')
        plt.xlabel('Month')
        plt.ylabel(ylabel)
        plt.legend()
        if savepath:
            plt.savefig(savepath+'{}_pred_vs_time_{}.png'.format(save_name_prefix,folder),dpi=1200,bbox_inches='tight')

        fig = plt.figure()
        #fig = plt.figure(figsize=(12,5))
        diff_y_pred = ys-preds
        plt.plot(diff_y_pred)
        plt.title('Difference of Groundtruth and Prediction')
        plt.xticks(ticks=range(0,len(ys),interval), labels=year_months[::interval])#,rotation=90)
        plt.xlabel('Month')
        #plt.yticks([], [])
        plt.ylabel(ylabel)
        if savepath:
            plt.savefig(savepath+'{}_diff_y_pred_{}.png'.format(save_name_prefix,folder),dpi=1200,bbox_inches='tight')


    return mae,mse,rmse


