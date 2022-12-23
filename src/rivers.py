#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 18:27:31 2020

@author: Yumin Liu
"""


#%%
#def main(column,nstep=1):
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from scipy.stats import pearsonr as pearsonr
import utils



predictor = 'Nino34_anom' # 'Nino34' # 'Reanalysis_Nino34' #'Reanalysis_Mean_Nino34' # 'GCM_Nino34' # 'GCM_Mean_Nino34' #  
predictand = 'Amazon' # 'Congo' # 
savepath = '../results/Regression/{}_{}/'.format(predictor,predictand)
Ntrain,Nvalid,Ntest = 600,36,36
#%% read ENSO index
indices_df, _ = utils.read_enso() # 187001 to 201912
# nino34 = indices_df[['Nino34']].iloc[960:1632].to_numpy().reshape((-1,1)) # from 195001 to 200512
#nino34 = indices_df[['Nino34_anom']].iloc[960:1632].to_numpy().reshape((-1,1)) # from 195001 to 200512
nino34 = indices_df[[predictor]].iloc[960:1632].to_numpy().reshape((-1,1)) # from 195001 to 200512

#%% Reanalysis Nino34: average between area in 5N-5S, 170W-120W
# datapath = '../data/Climate/Reanalysis/sst_cobe_hadley_noaa_uod_195001-200512_1by1_world.npy'
# data = np.load(datapath)[:,0:3,2:-2,:] # 
# print('data.shape={}'.format(data.shape))
# #nino34 = np.mean(data[:,:,82:94,190:240],axis=(1,2,3)).reshape((-1,1)) # from 195001 to 200512
# nino34 = np.mean(data[:,:,82:94,190:240],axis=(2,3)) # from 195001 to 200512
# print('nino34.shape={}'.format(nino34.shape))
# lats = np.load('../data/Climate/Reanalysis/lats.npy')[2:-2]
# lons = np.load('../data/Climate/Reanalysis/lons.npy')
# print('lats[82]={},lats[94]={}'.format(lats[82],lats[94]))
# print('lons[190]={},lons[240]={}'.format(lons[190],lons[240]))

#%% GCMNino34: average between area in 5N-5S, 170W-120W
# gcmpath = '../data/Climate/GCM/GCMdata/tas/processeddata/32gcms_tas_monthly_1by1_195001-200512_World.npy'
# #nino34 = np.mean(np.load(gcmpath)[:,:,82:94,190:240],axis=(1,2,3)).reshape((-1,1)) # from 195001 to 200512
# nino34 = np.mean(np.load(gcmpath)[:,:,82:94,190:240],axis=(2,3))#.reshape((-1,)) # from 195001 to 200512
# print('gcm_nino34.shape={}'.format(nino34.shape))
# lons = np.load('../data/Climate/GCM/GCMdata/tas/processeddata/lons_gcm.npy')
# print('lons[190]={},lons[240]={}'.format(lons[190],lons[240]))
# lats = np.load('../data/Climate/GCM/GCMdata/tas/processeddata/lats_gcm.npy')
# print('lats[82]={},lats[94]={}'.format(lats[82],lats[94]))
# time = np.load('../data/Climate/GCM/GCMdata/tas/processeddata/time_gcm.npy')

#%% river flow 
window = 3
riverpath = '../data/Climate/RiverFlow/processed/riverflow.csv'
riverflow_df = pd.read_csv(riverpath,index_col=0,header=0)

## amazon
if predictand=='Amazon':
    amazon = riverflow_df[['0']].iloc[600-window+1:1272].to_numpy().reshape((-1,)) # from 1950-window to 200512
    if predictor == 'Nino34_anom': # Nino3.4 anomaly to predict river anomaly and the added back historical mean
        amazon_season = riverflow_df[['0']].iloc[600:1272].to_numpy().reshape((-1,12))#.iloc[600-window+1:1272]#.to_numpy().reshape((-1,)) # from 1950-window to 200512
        amazon_season = amazon_season[:50,:]
        amazon_season = np.mean(amazon_season,axis=0)
        amazon_season = np.tile(amazon_season,reps=56)
        amazon_season = np.concatenate((amazon_season[1-window:],amazon_season),axis=0)
        amazon_deseason = amazon - amazon_season
        amazon_deseason = np.array([np.mean(amazon_deseason[i:i+window]) for i in range(len(amazon_deseason)-window+1)]).reshape((-1,))
        amazon_season_avg = np.array([np.mean(amazon_season[i:i+window]) for i in range(len(amazon_season)-window+1)]).reshape((-1,))  
    ## moving average, result in 195001 to200512
    amazon = np.array([np.mean(amazon[i:i+window]) for i in range(len(amazon)-window+1)]).reshape((-1,))
elif predictand=='Congo':
    ## congo
    congo = riverflow_df[['1']].iloc[600-window+1:1272].to_numpy().reshape((-1,)) # from 1950-window to 200512
    if predictor == 'Nino34_anom':
        congo_season = riverflow_df[['1']].iloc[600:1272].to_numpy().reshape((-1,12))#.iloc[600-window+1:1272]#.to_numpy().reshape((-1,)) # from 1950-window to 200512
        congo_season = congo_season[:50,:]
        congo_season = np.mean(congo_season,axis=0)
        congo_season = np.tile(congo_season,reps=56)
        congo_season = np.concatenate((congo_season[1-window:],congo_season),axis=0)
        congo_deseason = congo - congo_season
        congo_deseason = np.array([np.mean(congo_deseason[i:i+window]) for i in range(len(congo_deseason)-window+1)]).reshape((-1,))
        congo_season_avg = np.array([np.mean(congo_season[i:i+window]) for i in range(len(congo_season)-window+1)]).reshape((-1,))
    ## moving average, result in 195001 to200512
    congo = np.array([np.mean(congo[i:i+window]) for i in range(len(congo)-window+1)]).reshape((-1,))

X = nino34 # gcm_nino34 # [N,D]
#y = congo #amazon #  [N,]
y = amazon if predictand=='Amazon' else congo
if predictor == 'Nino34_anom':
    if predictand=='Amazon':
        y = amazon_deseason
    elif predictand=='Congo':
        y = congo_deseason # 

## standardize
data_mean = np.mean(X[:Ntrain],axis=0)
data_std = np.std(X[:Ntrain],axis=0)+1e-15
print('data_mean.shape={},data_std.shape={}'.format(data_mean.shape,data_std.shape))
for t in range(len(X)):
    X[t,:] = (X[t,:]-data_mean)/data_std
y_mean = np.mean(y[:Ntrain],axis=0)
y_std = np.std(y[:Ntrain],axis=0)
y = (y-y_mean)/y_std

X_train,X_valid,X_test = X[0:Ntrain,:],X[Ntrain:Ntrain+Nvalid,:],X[Ntrain+Nvalid:,:]
y_train,y_valid,y_test = y[0:Ntrain],y[Ntrain:Ntrain+Nvalid],y[Ntrain+Nvalid:]
X_train_valid, y_train_valid = X[0:Ntrain+Nvalid,:],y[0:Ntrain+Nvalid]

if not os.path.exists(savepath):
    os.makedirs(savepath)
if predictor == 'Nino34_anom':
    if predictand=='Amazon':
        amazon_season_avg_mean, amazon_season_avg_std = np.mean(amazon_season_avg[:Ntrain]), np.std(amazon_season_avg[:Ntrain])
        amazon_season_avg_demean = (amazon_season_avg-amazon_season_avg_mean)/amazon_season_avg_std
        np.savez(savepath+f'season_avg.npz',season_avg=amazon_season_avg,season_avg_demean=amazon_season_avg_demean,Ntrain=Ntrain,Nvalid=Nvalid,Ntest=Ntest)
    elif predictand=='Congo':
        congo_season_avg_mean, congo_season_avg_std = np.mean(congo_season_avg[:Ntrain]), np.std(congo_season_avg[:Ntrain])
        congo_season_avg_demean = (congo_season_avg-congo_season_avg_mean)/congo_season_avg_std
        np.savez(savepath+f'season_avg.npz',season_avg=congo_season_avg,season_avg_demean=congo_season_avg_demean,Ntrain=Ntrain,Nvalid=Nvalid,Ntest=Ntest)

#%% scale to zero mean, standard std
# from sklearn.preprocessing import StandardScaler as Scaler
# #X_scaler = Scaler(feature_range=(-1, 1)).fit(X_train)
# X_scaler = Scaler().fit(X_train)
# X_train = X_scaler.transform(X_train)
# X_valid = X_scaler.transform(X_valid)
# X_test = X_scaler.transform(X_test)
# #y_scaler = Scaler(feature_range=(-1, 1)).fit(y_train)
# y_scaler = Scaler().fit(y_train)
# y_train = y_scaler.transform(y_train)
# y_valid = y_scaler.transform(y_valid)
# y_test = y_scaler.transform(y_test)
print('X.shape={},y.shape={}'.format(X.shape,y.shape))
if X.shape[1]==1:
    corr_all,p_all = pearsonr(X.flatten(),y)
    fig = plt.figure(figsize=(12,5))
    plt.plot(X.flatten(),'k--',label=predictor)
    plt.plot(y,'r-',label=predictand)
    plt.legend()
    plt.text(x=0,y=plt.gca().get_ylim()[0]+0.1,s='correlation={:.3f}'.format(corr_all))
    title = 'Standardized {} and {}'.format(predictor,predictand)
    plt.title(title)
    plt.xlabel('Month from 195001')
    plt.savefig(savepath+title.replace(' ','_')+'_all.png',dpi=1200,bbox_tight='inches')

    corr_test,p_test = pearsonr(X_test.flatten(),y_test)
    fig = plt.figure(figsize=(12,5))
    plt.plot(X_test.flatten(),'k--',label=predictor)
    plt.plot(y_test,'r-',label=predictand)
    plt.legend(loc='upper left')
    plt.text(x=0,y=plt.gca().get_ylim()[0]+0.1,s='correlation={:.3f}'.format(corr_test))
    title = 'Standardized {} and {}'.format(predictor,predictand)
    plt.title(title)
    plt.xlabel('Month from 200301')
    plt.savefig(savepath+title.replace(' ','_')+'_test.png',dpi=1200,bbox_tight='inches')
elif X.shape[1]>1:
    ## multiple feature dimensions
    X_mean_all = np.mean(X,axis=1)
    X_std_all = np.std(X,axis=1)
    corr_all = np.array([pearsonr(X[:,d],y)[0] for d in range(X.shape[1])])
    corr_all_mean = np.mean(corr_all)
    corr_all_std = np.std(corr_all)
    fig = plt.figure(figsize=(12,5))
    plt.plot(X_mean_all,'k--',label='{} mean'.format(predictor))
    plt.fill_between(np.arange(len(X_mean_all)),y1=X_mean_all-X_std_all,y2=X_mean_all+X_std_all,alpha=0.2)
    plt.plot(y,'r-',label=predictand)
    plt.legend(loc='upper right')
    plt.text(x=0,y=plt.gca().get_ylim()[0]+0.1,s='correlation={:.3f}({:.3f})'.format(corr_all_mean,corr_all_std))
    title = 'Standardized {} and {}'.format(predictor,predictand)
    plt.title(title)
    plt.xlabel('Month from 195001')
    plt.savefig(savepath+title.replace(' ','_')+'_all.png',dpi=1200,bbox_tight='inches')

    X_mean_test = np.mean(X_test,axis=1)
    X_std_test = np.std(X_test,axis=1)
    corr_test = np.array([pearsonr(X_test[:,d],y_test)[0] for d in range(X_test.shape[1])])
    corr_test_mean = np.mean(corr_test)
    corr_test_std = np.std(corr_test)
    fig = plt.figure(figsize=(12,5))
    plt.plot(X_mean_test,'k--',label='{} mean'.format(predictor))
    plt.fill_between(np.arange(len(X_mean_test)),y1=X_mean_test-X_std_test,y2=X_mean_test+X_std_test,alpha=0.2)
    plt.plot(y_test,'r-',label=predictand)
    plt.legend(loc='upper left')
    plt.text(x=0,y=plt.gca().get_ylim()[0]+0.1,s='correlation={:.3f}({:.3f})'.format(corr_test_mean,corr_test_std))
    title = 'Standardized {} and {}'.format(predictor,predictand)
    plt.title(title)
    plt.xlabel('Month from 200301')
    plt.savefig(savepath+title.replace(' ','_')+'_test.png',dpi=1200,bbox_tight='inches')


#%% define helper functions
from sklearn.model_selection import GridSearchCV
def modelsearch(estimator,param_grid,X,y,scoring='neg_mean_squared_error',cv=5):
    grid = GridSearchCV(estimator=estimator, param_grid=param_grid,scoring=scoring, cv=cv)
    grid.fit(X,y)
    best_model = grid.best_estimator_
    best_params = grid.best_params_
    
    return best_model, best_params

def cal_metrics(y_pred,y_test):
    assert y_pred.shape==y_test.shape, 'ERROR: y_pred.shape!=y_test.shape'
    mse = np.mean((y_pred-y_test)**2)
    rmse = np.sqrt(mse)
    rae = np.abs(y_pred-y_test)/np.abs(y_test)
    rae_avg = np.mean(rae)
    rae_std = np.std(rae)
    corr,p = pearsonr(y_pred,y_test)
    r2 = 1-np.sum((y_test-y_pred)**2)/np.sum((y_test-np.mean(y_test))**2)
    
    return rmse,corr,p,r2,rae_avg,rae_std

np.savez(savepath+'y_gt.npz',y=np.concatenate((y_train_valid,y_test),axis=0),Ntrain=Ntrain,Nvalid=Nvalid,Ntest=Ntest)
#%% linear regression
linear = LinearRegression().fit(X_train_valid, y_train_valid)
y_pred_train_valid = linear.predict(X_train_valid)
y_pred = linear.predict(X_test)
rmse,corr,p,r2,rae_avg,rae_std = cal_metrics(y_pred,y_test)
np.save(savepath+'linear.npy',np.concatenate((y_pred_train_valid,y_pred),axis=0))
res = {}
res['linear'] = [rmse,corr,p,r2,rae_avg,rae_std]
print('linear: rmse={};corr={};p={};r2={};rae_avg={};rae_std={};train rmse={}\n'.format(rmse,corr,p,r2,rae_avg,rae_std,np.sqrt(np.mean((y_pred_train_valid-y_train_valid)**2))))


#%% Ridge regression
param_grid = {'alpha':[1.0e-5,1e-3,1e-1,1,10]}
ridge, paras = modelsearch(Ridge(),param_grid,X_train_valid,y_train_valid.flatten(),scoring='neg_mean_squared_error',cv=5)
y_pred_train_valid = ridge.predict(X_train_valid)
y_pred = ridge.predict(X_test)
rmse,corr,p,r2,rae_avg,rae_std = cal_metrics(y_pred,y_test)
np.save(savepath+'ridge.npy',np.concatenate((y_pred_train_valid,y_pred),axis=0))
res['ridge'] = [rmse,corr,p,r2,rae_avg,rae_std]
hyparas = {}
hyparas['ridge_hyparas'] = paras
print('ridge: rmse={};corr={};p={};r2={};rae_avg={};rae_std={}\n'.format(rmse,corr,p,r2,rae_avg,rae_std))

#%% Lasso regression
param_grid = {'alpha':[1.0e-5,1e-3,1e-1,1,10]}
lasso, paras = modelsearch(Lasso(),param_grid,X_train_valid,y_train_valid.flatten(),scoring='neg_mean_squared_error',cv=5)
y_pred_train_valid = lasso.predict(X_train_valid)
y_pred = lasso.predict(X_test)
rmse,corr,p,r2,rae_avg,rae_std = cal_metrics(y_pred,y_test)
np.save(savepath+'lasso.npy',np.concatenate((y_pred_train_valid,y_pred),axis=0))
print('lasso: rmse={};corr={};p={};r2={};rae_avg={};rae_std={}\n'.format(rmse,corr,p,r2,rae_avg,rae_std))
res['lasso'] = [rmse,corr,p,r2,rae_avg,rae_std]
hyparas['lasso_hyparas'] = paras

#%% Elastic Net
param_grid = {'alpha':[1.0e-5,1e-3,1e-1,1,10],'l1_ratio':[0.1,0.3,0.5,0.7,0.9]}
elastic, paras = modelsearch(ElasticNet(),param_grid,X_train_valid,y_train_valid.flatten(),scoring='neg_mean_squared_error',cv=5)
y_pred_train_valid = elastic.predict(X_train_valid)
y_pred = elastic.predict(X_test)
rmse,corr,p,r2,rae_avg,rae_std = cal_metrics(y_pred,y_test)
np.save(savepath+'elasticNet.npy',np.concatenate((y_pred_train_valid,y_pred),axis=0))
print('elastic: rmse={};corr={};p={};r2={};rae_avg={};rae_std={}\n'.format(rmse,corr,p,r2,rae_avg,rae_std))
res['elastic'] = [rmse,corr,p,r2,rae_avg,rae_std]
hyparas['elastic_hyparas'] = paras

#%% Random Forest
from sklearn.ensemble import RandomForestRegressor
param_grid = {'n_estimators':[50,100,300],'max_depth':[None,10,30],'max_features': [0.5,'auto'],
            'min_samples_leaf': [1,3,5],'min_samples_split': [2,8]}
rfr, paras = modelsearch(RandomForestRegressor(),param_grid,X_train_valid,y_train_valid.flatten(),scoring='neg_mean_squared_error',cv=5)
y_pred_train_valid = rfr.predict(X_train_valid)
y_pred = rfr.predict(X_test)
rmse,corr,p,r2,rae_avg,rae_std = cal_metrics(y_pred,y_test)
np.save(savepath+'randomForest.npy',np.concatenate((y_pred_train_valid,y_pred),axis=0))
print('randomforest: rmse={};corr={};p={};r2={};rae_avg={};rae_std={}\n'.format(rmse,corr,p,r2,rae_avg,rae_std))
res['randomforest'] = [rmse,corr,p,r2,rae_avg,rae_std]
hyparas['randomforest_hyparas'] = paras


#%% DNN regression
from dnn_main import dnn_main
## network hyperparameters
dnn_hyparas = {}
dnn_hyparas['input_channels'] = input_channels = X_train.shape[1] # 120 #36 # 6 # 8 # 8*lag # 7 #
dnn_hyparas['output_channels'] = output_channels =  1
dnn_hyparas['hidden_channels'] = hidden_channels = [40,20,10,5] # len(hidden_channels)
dnn_hyparas['drop_rate'] = drop_rate = 0.0
#### other parameters
paras = {}
paras['is_debug'] = is_debug = False # True # 
paras['num_epochs'] = num_epochs = 200 # 50 # 100 # 
paras['batch_size'] = batch_size = 4 #128
paras['lr'] = lr = 5e-5
paras['lr_patience'] = lr_patience = 5
paras['weight_decay'] = weight_decay = 1e-4
paras['num_workers'] = num_workers = 8
paras['model_name'] = model_name = 'DNN' # 'ConvLSTM'
paras['verbose'] = verbose = True # False
paras['tstep'] = tstep = 0 #36 # 12 # months
paras['seed'] = seed = 123
paras['predictor'] = predictor
paras['predictand'] = predictand
paras['save_root_path'] = save_root_path = '../results/Regression/{}_{}/{}/lag_{}/'.format(predictor,predictand,model_name,tstep)
paras['data'] = {'X':X,'y':y}
paras['Ntrain'] = Ntrain
paras['Nvalid'] = Nvalid
paras['Ntest']  = Ntest

y_pred,y_test_dnn, y_pred_train_valid, y_train_valid_dnn = dnn_main(dnn_hyparas,paras)
rmse,corr,p,r2,rae_avg,rae_std = cal_metrics(y_pred,y_test)
np.save(savepath+'DNN.npy',np.concatenate((y_pred_train_valid,y_pred),axis=0))
print('DNN: rmse={};corr={};p={};r2={};rae_avg={};rae_std={}\n'.format(rmse,corr,p,r2,rae_avg,rae_std))
res['DNN'] = [rmse,corr,p,r2,rae_avg,rae_std]
hyparas['DNN_hyparas'] = dnn_hyparas


#%%
res_df = pd.DataFrame.from_dict(res,orient='index',dtype=float,columns=['rmse','corr','p','r2','rae_avg','rae_std'])
savename = '{}_{}_metrics_results'.format(predictor.lower(),predictand.lower())
if savepath and savename:
    res_df.to_csv(savepath+savename+'.csv')


#%% plot
verbose = True # False # 
if verbose:
    #import numpy as np
    #import matplotlib
    #matplotlib.use('Agg')
    #import matplotlib.pyplot as plt
    
    datapath = savepath #'../results/Regression/'
    gt = np.load(datapath+'y_gt.npz')
    y_all,Ntrain,Nvalid,Ntest = gt['y'],gt['Ntrain'],gt['Nvalid'],gt['Ntest']
    y_train,y_valid,y_test = y_all[0:Ntrain],y_all[Ntrain:Ntrain+Nvalid],y_all[Ntrain+Nvalid:]

    linear_pred_all = np.load(datapath+'linear.npy')
    linear_pred = linear_pred_all[Ntrain+Nvalid:]

    ridge_pred_all = np.load(datapath+'ridge.npy')
    ridge_pred = ridge_pred_all[Ntrain+Nvalid:]

    lasso_pred_all = np.load(datapath+'lasso.npy')
    lasso_pred = lasso_pred_all[Ntrain+Nvalid:]

    elastic_pred_all = np.load(datapath+'elasticNet.npy')
    elastic_pred = elastic_pred_all[Ntrain+Nvalid:]

    randomforest_pred_all = np.load(datapath+'randomForest.npy')
    randomforest_pred = randomforest_pred_all[Ntrain+Nvalid:]

    dnn_pred_all = np.load(datapath+'DNN.npy')
    dnn_pred = dnn_pred_all[Ntrain+Nvalid:]
            
    fig = plt.figure(figsize=(12,5))
    years = np.arange(len(y_test))
    line0 = plt.plot(years,y_test,'k--',label='Groundtruth')
    line1 = plt.plot(years,linear_pred,'b-',label='Linear')
    line2 = plt.plot(years,ridge_pred,'g-.',label='Ridge')
    line3 = plt.plot(years,lasso_pred,'r:',label='Lasso')
    line4 = plt.plot(years,elastic_pred,'c-',label='ElasticNet')
    line5 = plt.plot(years,randomforest_pred,'m-.',label='RandomForest')
    line6 = plt.plot(years,dnn_pred,'y:',label='DNN')
    plt.legend()
    plt.grid('on')
    plt.xlabel('Month')
    plt.ylabel('Flow (m3/s)')
    title = '{} River Flow'.format(predictand)
    plt.title(title)
    plt.savefig(savepath+title.replace(' ','_')+'.png',dpi=1200)
    
    fig = plt.figure(figsize=(12,5))
    years_all = np.arange(len(y_all))
    line0 = plt.plot(years_all,y_all,'k--',label='Groundtruth')
    line1 = plt.plot(years_all,linear_pred_all,'b-',label='Linear')
    line2 = plt.plot(years_all,ridge_pred_all,'g-.',label='Ridge')
    line3 = plt.plot(years_all,lasso_pred_all,'r:',label='Lasso')
    line4 = plt.plot(years_all,elastic_pred_all,'c-',label='ElasticNet')
    line5 = plt.plot(years_all,randomforest_pred_all,'m-.',label='RandomForest')
    line6 = plt.plot(years_all,dnn_pred_all,'y:',label='DNN')
    linev = plt.axvline(x=635.5,ymin=-1,ymax=1,color='r')
    plt.legend()
    plt.grid('on')
    plt.xlabel('Month')
    plt.ylabel('Flow (m3/s)')
    title = '{} River Flow'.format(predictand)
    plt.title(title)
    plt.savefig(savepath+title.replace(' ','_')+'_all.png',dpi=1200)

