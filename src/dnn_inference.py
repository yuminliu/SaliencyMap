#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 18:38:28 2020

@author: wang.zife
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 00:15:31 2020

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
def myinference(hyparas,paras,test_loader,datasettype=None):
    
    savepath = paras['savepath']
    device = paras['device']
    checkpoint_name = paras['checkpoint_name']    
    checkpoint_pathname = savepath+checkpoint_name
    verbose = paras['verbose']
    #input_size = next(iter(test_dataset))[0][0].shape # [Tstep,channel,height,width]
    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #%%
    model = models.DNN(**hyparas)
    checkpoint = torch.load(checkpoint_pathname, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    #mse,nsamples = 0,0
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
        #mse += np.sum((pred-y)**2)
        #nsamples += 1

        preds.append(pred.reshape((-1,1)))
        ys.append(y.reshape((-1,1)))
    
    preds = np.concatenate(preds,axis=0) #[Ntest,], unit: mm/day
    ys = np.concatenate(ys,axis=0) #[Ntest,]
    
    print('preds.shape={},ys.shape={}'.format(preds.shape,ys.shape))

    ## scale back
    #preds = preds*300000
    #ys = ys*300000
    
    #mse = mse/nsamples
    #rmse = np.sqrt(mse)
    #mae = np.sum(np.abs(ys-preds))/nsamples
    #rmae = np.sum(np.abs(ys-preds)/(np.mean(ys)+1e-5))/nsamples
    #print('{} RMSE:{}, MAE={}, RMAE={}'.format(datasettype,rmse,mae,rmae))
    mse = np.mean((ys-preds)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(ys-preds))
    rmae = np.mean(np.abs(ys-preds)/(np.abs(ys)+1e-5))
    
    print('{} RMSE2:{}'.format(datasettype,rmse))
    rae = np.abs(preds-ys)/np.abs(ys)
    rae_avg = np.mean(rae)
    rae_std = np.std(rae)
    #print('relative_error={}'.format(rae))
    print('relative_error_avg={}'.format(rae_avg))
    print('relative_error_std={}'.format(rae_std))

    #%%
    if savepath:
        np.savez(savepath+'pred_results_RMSE{}_{}.npz'.format(rmse,datasettype),rmse=rmse,mae=mae,rmae=rmae,y_pred=preds,y_test=ys,
                 rae=rae,rae_avg=rae_avg)
#        np.savez('../results/analysis/Regression/{}/{}_DNNRegression_{}.npz'.format(paras['rivername'],paras['rivername'],datasettype),
#                 rmse=rmse,rae_avg=rae_avg,rae_std=rae_std,y_pred=preds,y_test=ys)

    #%% plot figures
    if verbose and datasettype=='test':
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig = plt.figure()
        plt.plot(ys,'--k',label='Groundtruth')
        plt.plot(preds,'-r',label='Prediction')
        plt.title('Anomaly Groundtruth vs Prediction')
        plt.xlabel('Year')
        plt.ylabel('River flow anomaly')
        plt.legend()
        if savepath:
            plt.savefig(savepath+'pred_vs_time.png',dpi=1200,bbox_inches='tight')

        fig = plt.figure()
        diff_y_pred = ys-preds
        plt.plot(diff_y_pred)
        plt.title('Groundtruth-prediction')
        #plt.xticks([], [])
        plt.xlabel('Year')
        #plt.yticks([], [])
        plt.ylabel('River flow diff')
        if savepath:
            plt.savefig(savepath+'diff_y_pred.png',dpi=1200,bbox_inches='tight')


    res = {}
    res['mae'] = mae
    res['mse'] = mse
    res['rmse'] = rmse
    res['rae_avg'] = rae_avg
    res['y_pred'] = preds.flatten()
    res['y_test'] = ys.flatten()
    return res # mae,mse,rmse,rae_avg


