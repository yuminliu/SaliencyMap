#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 18:44:22 2020

@author: wang.zife
"""
#import os
#import json
import numpy as np
import pandas as pd
#import scipy.stats as stats
#import matplotlib
#matplotlib.use('Agg')
#matplotlib.use('Qt5Agg')
#import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
import utils
def regressions(X_train,y_train, X_valid,y_valid, X_test,y_test, savepath, kwargs): 
    X_train_valid = np.concatenate((X_train,X_valid),axis=0) # 
    y_train_valid = np.concatenate((y_train,y_valid),axis=0) # 
    X = np.concatenate((X_train,X_valid,X_test),axis=0)
    y = np.concatenate((y_train,y_valid,y_test),axis=0)
    station_ind = kwargs['station_ind']
    rivername = kwargs['rivername'].replace(' ','_')
    verbose = kwargs['verbose']
    Ntrain = kwargs['Ntrain']
    Nvalid = kwargs['Nvalid']
    Ntest = kwargs['Ntest']
    txtsavepath = kwargs['txtsavepath']
    mode = kwargs['mode']
    scaleback = kwargs['scaleback']
    if scaleback and savepath:
        X_scaler_mean = kwargs['X_scaler_mean'] 
        X_scaler_std = kwargs['X_scaler_std'] 
        y_scaler_mean = kwargs['y_scaler_mean'] 
        y_scaler_std = kwargs['y_scaler_std'] 
        np.savez(savepath+'station_ind_{}_scaler.npz'.format(station_ind),
                 X_scaler_mean=X_scaler_mean,X_scaler_std=X_scaler_std,y_scaler_mean=y_scaler_mean,y_scaler_std=y_scaler_std)
    
    #%% linear regression
    linear = LinearRegression().fit(X_train_valid, y_train_valid)
    y_pred_train_valid = linear.predict(X_train_valid).reshape((-1,1))
    y_pred = linear.predict(X_test).reshape((-1,1))
    rmse,corr,p,r2,rae_avg,rae_std = utils.cal_metrics(y_pred,y_test)
        
    if savepath:
        np.savez(savepath+'station_ind_{}_linearRegression.npz'.format(station_ind),rmse=rmse,rae_avg=rae_avg,rae_std=rae_std,
                 y_pred=y_pred,y_test=y_test,corr=corr,p=p,r2=r2,y_pred_train=y_pred_train_valid,y_train=y_train_valid)
    if verbose:
        print('linear: rmse={};corr={};p={};r2={};rae_avg={};rae_std={};train rmse={}\n'.format(rmse,corr,p,r2,rae_avg,rae_std,np.sqrt(np.mean(y_pred_train_valid-y_train_valid)**2)))
    res = {}
    res['linear'] = [rmse,corr,p,r2,rae_avg,rae_std]
    
    with open(txtsavepath+'linear_{}.txt'.format(mode),'a') as file:
        file.write('{} {} {} {} {} {} {} {}\n'.format(station_ind,rivername,rmse,corr,p,r2,rae_avg,rae_std))
    file.close()

    
    #%% Ridge regression
    #param_grid = {'alpha':[1.0e-5,1e-3,1e-1,1,10]}
    param_grid = {'alpha':[1.0e-7,1.0e-6,1.0e-5,1e-3,1e-1]}
    ridge, paras = utils.modelsearch(Ridge(),param_grid,X_train_valid,y_train_valid.flatten(),scoring='neg_mean_squared_error',cv=5)
    y_pred_train_valid = ridge.predict(X_train_valid).reshape((-1,1))
    y_pred = ridge.predict(X_test).reshape((-1,1))
    rmse,corr,p,r2,rae_avg,rae_std = utils.cal_metrics(y_pred,y_test)
    if savepath:
        np.savez(savepath+'station_ind_{}_ridgeRegression.npz'.format(station_ind),rmse=rmse,rae_avg=rae_avg,rae_std=rae_std,
                 y_pred=y_pred,y_test=y_test,corr=corr,p=p,r2=r2,y_pred_train=y_pred_train_valid,y_train=y_train_valid)
    if verbose:
        print('ridge: rmse={};corr={};p={};r2={};rae_avg={};rae_std={}\n'.format(rmse,corr,p,r2,rae_avg,rae_std))
    res['ridge'] = [rmse,corr,p,r2,rae_avg,rae_std]
    hyparas = {}
    hyparas['ridge_hyparas'] = paras
    
    with open(txtsavepath+'ridge_{}.txt'.format(mode),'a') as file:
        file.write('{} {} {} {} {} {} {} {}\n'.format(station_ind,rivername,rmse,corr,p,r2,rae_avg,rae_std))
    file.close()
    
    #%% Lasso regression
    #param_grid = {'alpha':[1.0e-5,1e-3,1e-1,1,10]}
    param_grid = {'alpha':[1.0e-7,1.0e-6,1.0e-5,1e-3,1e-1]}
    lasso, paras = utils.modelsearch(Lasso(),param_grid,X_train_valid,y_train_valid.flatten(),scoring='neg_mean_squared_error',cv=5)
    y_pred_train_valid = lasso.predict(X_train_valid).reshape((-1,1))
    y_pred = lasso.predict(X_test).reshape((-1,1))
    rmse,corr,p,r2,rae_avg,rae_std = utils.cal_metrics(y_pred,y_test)
    if savepath:
        np.savez(savepath+'station_ind_{}_lassoRegression.npz'.format(station_ind),rmse=rmse,rae_avg=rae_avg,rae_std=rae_std,
                 y_pred=y_pred,y_test=y_test,corr=corr,p=p,r2=r2,y_pred_train=y_pred_train_valid,y_train=y_train_valid)
    if verbose:
        print('lasso: rmse={};corr={};p={};r2={};rae_avg={};rae_std={}\n'.format(rmse,corr,p,r2,rae_avg,rae_std))
    res['lasso'] = [rmse,corr,p,r2,rae_avg,rae_std]
    hyparas['lasso_hyparas'] = paras

    with open(txtsavepath+'lasso_{}.txt'.format(mode),'a') as file:
        file.write('{} {} {} {} {} {} {} {}\n'.format(station_ind,rivername,rmse,corr,p,r2,rae_avg,rae_std))
    file.close()
    
    #%% Elastic Net
    #param_grid = {'alpha':[1.0e-5,1e-3,1e-1,1,10],'l1_ratio':[0.1,0.3,0.5,0.7,0.9]}
    param_grid = {'alpha':[1.0e-7,1.0e-6,1.0e-5,1e-3,1e-1],'l1_ratio':[0.001,0.01,0.1,0.3,0.5]}
    elastic, paras = utils.modelsearch(ElasticNet(),param_grid,X_train_valid,y_train_valid.flatten(),scoring='neg_mean_squared_error',cv=5)
    y_pred_train_valid = elastic.predict(X_train_valid).reshape((-1,1))
    y_pred = elastic.predict(X_test).reshape((-1,1))
    rmse,corr,p,r2,rae_avg,rae_std = utils.cal_metrics(y_pred,y_test)
    if savepath:
        np.savez(savepath+'station_ind_{}_elasticNetRegression.npz'.format(station_ind),rmse=rmse,rae_avg=rae_avg,rae_std=rae_std,
                 y_pred=y_pred,y_test=y_test,corr=corr,p=p,r2=r2,y_pred_train=y_pred_train_valid,y_train=y_train_valid)
    if verbose:
        print('elastic: rmse={};corr={};p={};r2={};rae_avg={};rae_std={}\n'.format(rmse,corr,p,r2,rae_avg,rae_std))
    res['elastic'] = [rmse,corr,p,r2,rae_avg,rae_std]
    hyparas['elastic_hyparas'] = paras

    with open(txtsavepath+'elasticNet_{}.txt'.format(mode),'a') as file:
        file.write('{} {} {} {} {} {} {} {}\n'.format(station_ind,rivername,rmse,corr,p,r2,rae_avg,rae_std))
    file.close()
    
    #%% Random Forest
    from sklearn.ensemble import RandomForestRegressor
    param_grid = {'n_estimators':[50,100,300],'max_depth':[None,10,30],'max_features': [0.5,'auto'],
                  'min_samples_leaf': [3,5],'min_samples_split': [2,8]}
    rfr, paras = utils.modelsearch(RandomForestRegressor(),param_grid,X_train_valid,y_train_valid.flatten(),scoring='neg_mean_squared_error',cv=5)
    y_pred_train_valid = rfr.predict(X_train_valid).reshape((-1,1))
    y_pred = rfr.predict(X_test).reshape((-1,1))
    rmse,corr,p,r2,rae_avg,rae_std = utils.cal_metrics(y_pred,y_test)
    if savepath:
        np.savez(savepath+'station_ind_{}_randomForestRegression.npz'.format(station_ind),rmse=rmse,rae_avg=rae_avg,rae_std=rae_std,
                 y_pred=y_pred,y_test=y_test,corr=corr,p=p,r2=r2,y_pred_train=y_pred_train_valid,y_train=y_train_valid)
    if verbose:
        print('randomforest: rmse={};corr={};p={};r2={};rae_avg={};rae_std={}\n'.format(rmse,corr,p,r2,rae_avg,rae_std))
    res['randomforest'] = [rmse,corr,p,r2,rae_avg,rae_std]
    hyparas['randomforest_hyparas'] = paras

    with open(txtsavepath+'randomForest_{}.txt'.format(mode),'a') as file:
        file.write('{} {} {} {} {} {} {} {}\n'.format(station_ind,rivername,rmse,corr,p,r2,rae_avg,rae_std))
    file.close()
    
    #%% DNN regression
    from dnn_main import dnn_main
    input_channels = X_train.shape[1] # 120 #36
    hidden_channels = [20,10,5]
    output_channels = 1
    drop_rate = 0.0
    dnn_hyparas = {}
    dnn_hyparas['input_channels'] = input_channels # 6 # 8 # 8*lag # 7 #
    dnn_hyparas['output_channels'] = output_channels # 1
    dnn_hyparas['hidden_channels'] = hidden_channels # len(hidden_channels)
    dnn_hyparas['drop_rate'] = drop_rate
    
    y_pred,y_test_dnn, y_pred_train_valid, y_train_valid_dnn = dnn_main(dnn_hyparas,data={'X':X,'y':y},Ntrain=Ntrain,Nvalid=Nvalid,Ntest=Ntest)
    y_pred = y_pred.reshape((-1,1))
    y_pred_train_valid = y_pred_train_valid.reshape((-1,1))
    rmse,corr,p,r2,rae_avg,rae_std = utils.cal_metrics(y_pred,y_test)
    if savepath:
        np.savez(savepath+'station_ind_{}_DNNRegression.npz'.format(station_ind),rmse=rmse,rae_avg=rae_avg,rae_std=rae_std,
                 y_pred=y_pred,y_test=y_test,corr=corr,p=p,r2=r2,y_pred_train=y_pred_train_valid,y_train=y_train_valid)
    if verbose:
        print('DNN: rmse={};corr={};p={};r2={};rae_avg={};rae_std={}\n'.format(rmse,corr,p,r2,rae_avg,rae_std))
    res['DNN'] = [rmse,corr,p,r2,rae_avg,rae_std]
    hyparas['DNN_hyparas'] = dnn_hyparas

    with open(txtsavepath+'DNN_{}.txt'.format(mode),'a') as file:
        file.write('{} {} {} {} {} {} {} {}\n'.format(station_ind,rivername,rmse,corr,p,r2,rae_avg,rae_std))
    file.close()
    
    #%%
    res_df = pd.DataFrame.from_dict(res,orient='index',dtype=np.float,columns=['rmse','corr','p','r2','rae_avg','rae_std'])
    res_df.to_csv(savepath+'results_AR_station_ind_{}.csv'.format(station_ind))
    
    return hyparas, res