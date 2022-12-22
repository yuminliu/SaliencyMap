#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 21:52:07 2021

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
def myinference(hyparas,paras,test_loader):
    
    savepath = paras['savepath']
    device = paras['device']
    checkpoint_name = paras['checkpoint_name']    
    checkpoint_pathname = savepath+checkpoint_name
    verbose = paras['verbose']
    #input_size = next(iter(test_dataset))[0][0].shape # [Tstep,channel,height,width]
    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #%%
    model = models.CAM(**hyparas)
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
                pred,_ = model(inputs) # [1,1]
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
    rmae = np.sum(np.abs(ys-preds)/(np.mean(ys)+1e-5))/nsamples
    print('test RMSE:{}, MAE={}, RMAE={}'.format(rmse,mae,rmae))
    mse2 = np.mean((ys-preds)**2)
    rmse2 = np.sqrt(mse2)
    print('test RMSE2:{}'.format(rmse2))

    #%%
    if savepath:
        np.savez(savepath+'pred_results_RMSE{}.npz'.format(rmse),rmse=rmse,mae=mae,rmae=rmae,preds=preds)
        #np.savez('../results/PRISM/ConvLSTM/lag_by_month/lag_{}/pred_results_lag_{}_month_{}.npz'.format(lag,lag,month),rmse=rmse,mae=mae,preds=preds)

    #%% plot figures
    if verbose:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig = plt.figure()
        plt.plot(ys,'--k',label='Groundtruth')
        plt.plot(preds,'-r',label='Prediction')
        plt.title('Groundtruth vs Prediction')
        plt.xlabel('Month')
        plt.ylabel('River flow')
        plt.legend()
        if savepath:
            plt.savefig(savepath+'pred_vs_time.png',dpi=1200,bbox_inches='tight')

        fig = plt.figure()
        diff_y_pred = ys-preds
        plt.plot(diff_y_pred)
        plt.title('Groundtruth-prediction')
        #plt.xticks([], [])
        plt.xlabel('Month')
        #plt.yticks([], [])
        plt.ylabel('River flow')
        if savepath:
            plt.savefig(savepath+'diff_y_pred.png',dpi=1200,bbox_inches='tight')


    return mae,mse,rmse


