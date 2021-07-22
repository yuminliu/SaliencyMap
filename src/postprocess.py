#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:52:52 2020

@author: yumin
"""






#%% compare persistence prediction with CNN prediction
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

predictand = 'Congo' # 'Amazon' # 
rootpath = '../results/Regression_saved/'
#folderpath = rootpath+'GCM_Mean_Nino34_Amazon/'
if predictand=='Amazon':
    y_pred_gcm = np.load('../results/myCNN/lag_0/Amazon/2021-06-14_21.59.38.060129_GCM_masked_Amazon_region1/GCM_Amazon_pred_results_RMSE0.26488437282175314_test.npz')['preds']
    y_pred_reanalysis = np.load('../results/myCNN/lag_0/Amazon/2021-06-15_00.23.44.969560_Reanalysis_masked_Amazon_region1/Reanalysis_Amazon_pred_results_RMSE0.31252981619007986_test.npz')['preds']
    gt = np.load('../results/Regression_saved/GCM_Mean_Nino34_Amazon/y_gt.npz',allow_pickle=True)
elif predictand=='Congo':
    y_pred_gcm = np.load('../results/myCNN/lag_0/Congo/2021-06-15_00.42.20.427649_GCM_masked_Congo_region1/GCM_Congo_pred_results_RMSE0.5976172958058634_test.npz')['preds']
    y_pred_reanalysis = np.load('../results/myCNN/lag_0/Congo/2021-06-14_23.51.47.085179_Reanalysis_masked_Congo_region1/Reanalysis_Congo_pred_results_RMSE0.4853571058879003_test.npz')['preds']
    gt = np.load('../results/Regression_saved/GCM_Mean_Nino34_Congo/y_gt.npz',allow_pickle=True)
y_persist = np.load('../results/Regression/Persistence/{}_y_persist_y_persist_std.npz'.format(predictand),allow_pickle=True)['y_persist']

y_gt_test = gt['y'][-gt['Ntest']:]
year_months = np.load('../data/Climate/year_months_195001-200512.npy')[-36:]
interval = 4
fig = plt.figure()
plt.plot(y_pred_gcm)
plt.plot(y_pred_reanalysis)
plt.plot(y_persist)
plt.plot(y_gt_test)
plt.xlabel('Months')
plt.ylabel('Stadardized River Flow')
plt.xticks(ticks=range(0,len(y_gt_test),interval), labels=year_months[::interval])
bottom, top = plt.ylim()  # return the current ylim
plt.ylim(bottom,top+0.8)
plt.xlabel('Months')
plt.ylabel('Stadardized River Flow')
plt.legend(loc='upper center',ncol=4)
plt.show()

## corrleation
correlation = {}
correlation['Persistence'] = stats.pearsonr(y_gt_test,y_persist)[0]
correlation['GCM'] = stats.pearsonr(y_gt_test,y_pred_gcm)[0]
correlation['Reanalysis'] = stats.pearsonr(y_gt_test,y_pred_reanalysis)[0]
print('correlation={}'.format(correlation))
#### seasonal rmse
def cal_seasonal_rmse(y_gt,y_pred):
    error = (y_pred-y_gt)**2
    spring = np.sqrt(np.mean(np.array(error[[0,1,11,12,13,23,24,25,35]])))
    summer = np.sqrt(np.mean(np.array(error[[2,3,4,14,15,16,26,27,28]])))
    autumn = np.sqrt(np.mean(np.array(error[[5,6,7,17,18,19,29,30,31]])))
    winter = np.sqrt(np.mean(np.array(error[[8,9,10,20,21,22,32,33,34]])))
    return spring,summer,autumn,winter
seasonal_rmse = {}
seasonal_rmse['Persistence'] = cal_seasonal_rmse(y_gt_test,y_persist)
seasonal_rmse['GCM'] = cal_seasonal_rmse(y_gt_test,y_pred_gcm)
seasonal_rmse['Reanalysis'] = cal_seasonal_rmse(y_gt_test,y_pred_reanalysis)
print('seasonal_rmse={}'.format(seasonal_rmse))
#### yearly rmse
def cal_yearly_rmse(y_gt,y_pred):
    error = (y_pred-y_gt)**2
    year2003 = np.sqrt(np.mean(error[0:12]))
    year2004 = np.sqrt(np.mean(error[12:24]))
    year2005 = np.sqrt(np.mean(error[24:36]))
    return year2003,year2004,year2005
yearly_rmse = {}
yearly_rmse['Persistence'] = cal_yearly_rmse(y_gt_test,y_persist)
yearly_rmse['GCM'] = cal_yearly_rmse(y_gt_test,y_pred_gcm)
yearly_rmse['Reanalysis'] = cal_yearly_rmse(y_gt_test,y_pred_reanalysis)
print('yearly_rmse={}'.format(yearly_rmse))
#### extreme and not extreme rmse
def cal_extreme_rmse(y_gt,y_pred,sigma=2):
    thred = {1:0.6827,2:0.9545,3:0.9973,4:0.9999,5:1}
    einds = abs(y_pred)>=thred[sigma]
    error = (y_pred-y_gt)**2
    extreme = np.sqrt(np.mean(error[einds]))
    notextreme = np.sqrt(np.mean(error[~einds]))
    return extreme,notextreme
extreme_rmse = {}
extreme_rmse['Persistence'] = cal_extreme_rmse(y_gt_test,y_persist)
extreme_rmse['GCM'] = cal_extreme_rmse(y_gt_test,y_pred_gcm)
extreme_rmse['Reanalysis'] = cal_extreme_rmse(y_gt_test,y_pred_reanalysis)
print('extreme_rmse={}'.format(extreme_rmse))
#### enso rmse
def cal_enso_rmse(y_gt,y_pred):
    error = (y_pred-y_gt)**2
    warm = np.sqrt(np.mean(np.array(error[[0,1,18,19,20,21,22,23,24,25]])))
    cool = np.sqrt(np.mean(np.array(error[[34,35]])))
    neutral = np.sqrt(np.mean(np.array(error[[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,26,27,28,29,30,31,32,33]])))
    return warm,cool,neutral
enso_rmse = {}
enso_rmse['Persistence'] = cal_enso_rmse(y_gt_test,y_persist)
enso_rmse['GCM'] = cal_enso_rmse(y_gt_test,y_pred_gcm)
enso_rmse['Reanalysis'] = cal_enso_rmse(y_gt_test,y_pred_reanalysis)
print('enso_rmse={}'.format(enso_rmse))
#### mean absolute error
def cal_mae(y_gt,y_pred):
    return np.mean(abs(y_gt-y_pred))
mae = {}
mae['Persistence'] = cal_mae(y_gt_test,y_persist)
mae['GCM'] = cal_mae(y_gt_test,y_pred_gcm)
mae['Reanalysis'] = cal_mae(y_gt_test,y_pred_reanalysis)
print('mae={}'.format(mae))
#### nash sutcliffe efficiency
import hydroeval as he
def cal_nash_sutcliffe(y_gt,y_pred):
    return he.evaluator(he.nse, y_pred, y_gt)
nash_sutcliffe = {}
nash_sutcliffe['Persistence'] = cal_nash_sutcliffe(y_gt_test,y_persist)[0]
nash_sutcliffe['GCM'] = cal_nash_sutcliffe(y_gt_test,y_pred_gcm)[0]
nash_sutcliffe['Reanalysis'] = cal_nash_sutcliffe(y_gt_test,y_pred_reanalysis)[0]
print('nash_sutcliffe={}'.format(nash_sutcliffe))

rows = []
rows.extend([correlation,seasonal_rmse,yearly_rmse,extreme_rmse,enso_rmse,mae,nash_sutcliffe])

metrics_df = pd.DataFrame(rows,index=['correlation','seasonal_rmse','yearly_rmse','extreme_rmse','enso_rmse','mae','nash_sucliffe'],columns=['Persistence','GCM','Reanalysis'])
print('metrics_df.shape={}'.format(metrics_df.shape))
savepath = '../results/Regression/Persistence/'
savename = '{}_metrics_comparison'.format(predictand)
metrics_df.to_csv(savepath+savename+'.csv')




#%% plot intermediate layer output for CNN
def plot_featuremap():
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    data = np.load('../results/myCNN/lag_0/Amazon/all/intermediate_outputs_0.npy',allow_pickle=True).item()
    savepath = '../results/featuremaps/'
    if savepath and not os.path.exists(savepath): os.makedirs(savepath)
    ngcm = 0
    verbose = False # True


    def plot_cnn_featuremap(img,savepathname=None,verbose=True,cmap='bwr',alpha=1):
        fig = plt.figure()
        plt.imshow(img,cmap=cmap,alpha=alpha)
        plt.xticks([])
        plt.yticks([])
        if verbose:
            plt.show()
        if savepathname:
            plt.savefig(savepathname+'.png',dpi=1200,bbox_inches='tight')
            plt.close()

    for key,values in data.items():
        if len(values.shape)!=3: continue
        for ngcm in [0,-1]:
            print('ploting {}_{}'.format(key,ngcm))
            img = values[ngcm]
            savepathname = savepath+'{}_{}'.format(key,abs(ngcm))
            plot_cnn_featuremap(img=img,savepathname=savepathname,verbose=verbose,cmap='bwr',alpha=1)

    print('Job done!')



def cal_corrs():
    import numpy as np
    import pandas as pd
    from scipy import stats
    from sklearn.feature_selection import mutual_info_regression
    import utils

    #predictor = 'Nino34' # 'Reanalysis_Nino34' #'Reanalysis_Mean_Nino34' # 'GCM_Nino34' # 'GCM_Mean_Nino34' #  
    #predictand = 'Amazon' # 'Congo' #
    detrend = True # False # 
    standardized = True
    if detrend:
        savepath = '../results/Correlations/Indices_vs_Rivers/'
    else:
        savepath = '../results/Correlations/Indices_vs_Rivers/not_detrend/'
    savename = 'correlations_indices_riverflows'
    Ntrain,Nvalid,Ntest = 600,36,36

    def get_data(variable,detrend=True,standardized=True,Ntrain=600):
        #%% read ENSO index
        if variable=='Nino34':
            indices_df, _ = utils.read_enso() # 187001 to 201912
            data = indices_df[['Nino34']].iloc[960:1632].to_numpy().reshape((-1,1)) # from 195001 to 200512
        #%% Reanalysis Nino34: average between area in 5N-5S, 170W-120W
        elif variable=='Reanalysis_Nino34':
            data = np.load('../data/Climate/Reanalysis/sst_cobe_hadley_noaa_uod_195001-200512_1by1_world.npy')[:,0:3,2:-2,:] # 
            data = np.mean(data[:,:,82:94,190:240],axis=(2,3))#.reshape((-1,1)) # from 195001 to 200512
        elif variable=='Reanalysis_Mean_Nino34':
            data = np.load('../data/Climate/Reanalysis/sst_cobe_hadley_noaa_uod_195001-200512_1by1_world.npy')[:,0:3,2:-2,:] # 
            data = np.mean(data[:,:,82:94,190:240],axis=(1,2,3)).reshape((-1,1)) # from 195001 to 200512
        #%% GCMNino34: average between area in 5N-5S, 170W-120W
        elif variable=='GCM_Nino34':
            data = np.load('/scratch/wang.zife/YuminLiu/DATA/GCM/GCMdata/tas/processeddata/32gcms_tas_monthly_1by1_195001-200512_World.npy')
            #nino34 = np.mean(np.load(gcmpath)[:,:,82:94,190:240],axis=(1,2,3)).reshape((-1,1)) # from 195001 to 200512
            data = np.mean(data[:,:,82:94,190:240],axis=(2,3))#.reshape((-1,1)) # from 195001 to 200512
        elif variable=='GCM_Mean_Nino34':
            data = np.load('/scratch/wang.zife/YuminLiu/DATA/GCM/GCMdata/tas/processeddata/32gcms_tas_monthly_1by1_195001-200512_World.npy')
            data = np.mean(data[:,:,82:94,190:240],axis=(1,2,3)).reshape((-1,1)) # from 195001 to 200512
        #%% river flow 
        elif variable in ['Amazon','Congo']:
            column = {'Amazon':'0','Congo':'1'}
            riverflow_df = pd.read_csv('../data/Climate/RiverFlow/processed/riverflow.csv',index_col=0,header=0)
            data = riverflow_df[[column[variable]]].iloc[600:1272].to_numpy().reshape((-1,1)) # from 195001 to 200512
        #### detrend data
        if detrend:
            mean_m,std_m = {},{}
            for m in range(12):
                mean_m[m] = np.mean(data[m:Ntrain:12,0])
                std_m[m] = np.std(data[m:Ntrain:12,0],ddof=1)
                data[m::12,0] = (data[m::12,0]-mean_m[m])/(std_m[m]+1e-15)
        ## standardize
        if standardized:
            data_mean = np.mean(data[:Ntrain],axis=0)
            data_std = np.std(data[:Ntrain],axis=0)+1e-5
            data = (data-data_mean)/data_std

        return data

    nino34 = get_data('Nino34',detrend=detrend,standardized=standardized,Ntrain=Ntrain)
    reanalysis_nino34 = get_data('Reanalysis_Nino34',detrend=detrend,standardized=standardized,Ntrain=Ntrain)
    reanalysis_mean_nino34 = get_data('Reanalysis_Mean_Nino34',detrend=detrend,standardized=standardized,Ntrain=Ntrain)
    gcm_nino34 = get_data('GCM_Nino34',detrend=detrend,standardized=standardized,Ntrain=Ntrain)
    gcm_mean_nino34 = get_data('GCM_Mean_Nino34',detrend=detrend,standardized=standardized,Ntrain=Ntrain)
    amazon = get_data('Amazon',detrend=detrend,standardized=standardized,Ntrain=Ntrain).reshape((-1,))
    congo = get_data('Congo',detrend=detrend,standardized=standardized,Ntrain=Ntrain).reshape((-1,))
    # print('nino34.shape={}'.format(nino34.shape))
    # print('reanalysis_nino34.shape={}'.format(reanalysis_nino34.shape))
    # print('reanalysis_mean_nino34.shape={}'.format(reanalysis_mean_nino34.shape))
    # print('gcm_nino34.shape={}'.format(gcm_nino34.shape))
    # print('gcm_mean_nino34.shape={}'.format(gcm_mean_nino34.shape))
    # print('amazon.shape={}'.format(amazon.shape))
    # print('congo.shape={}'.format(congo.shape))
    data = np.concatenate((nino34,reanalysis_mean_nino34,gcm_mean_nino34,reanalysis_nino34,gcm_nino34),axis=1)
    print('data.shape={}'.format(data.shape))

    corrs = np.zeros((data.shape[1],8))
    for d in range(data.shape[1]):
        corrs[d,0] = stats.pearsonr(data[:,d],amazon)[0]
        corrs[d,1] = stats.spearmanr(data[:,d],amazon)[0]
        corrs[d,2] = stats.kendalltau(data[:,d],amazon)[0]
        corrs[d,3] = mutual_info_regression(data[:,[d]],amazon)[0]

        corrs[d,4] = stats.pearsonr(data[:,d],congo)[0]
        corrs[d,5] = stats.spearmanr(data[:,d],congo)[0]
        corrs[d,6] = stats.kendalltau(data[:,d],congo)[0]
        corrs[d,7] = mutual_info_regression(data[:,[d]],congo)[0]
    reanalysis = np.mean(corrs[3:6,:],axis=0,keepdims=True)
    gcm = np.mean(corrs[6:,:],axis=0,keepdims=True)
    corrs = np.concatenate((corrs,reanalysis,gcm),axis=0)
    columns = ['pearson_Amazon','spearman_Amazon','kendalltau_Amazon','mutualinfo_Amazon',
                'pearson_Congo','spearman_Congo','kendalltau_Congo','mutualinfo_Congo']
    rows = ['Nino34','Reanalysis_Mean','GCM_Mean','Cobe','Hadley','Noaa']
    rows.extend(['GCM{}'.format(n) for n in range(1,gcm_nino34.shape[1]+1)])
    rows.extend(['Reanalysis','GCM'])
    print('corrs.shape={}'.format(corrs.shape))
    print('len(rows)={},len(columns)={}'.format(len(rows),len(columns)))
    corrs_df = pd.DataFrame(data=corrs,index=rows,columns=columns,dtype=float)
    corrs_df.head()
    corrs_df.to_csv(savepath+savename+'.csv',header=True,index=True)
    # pearson_corr_p[d,i,j] = stats.pearsonr(X[:,d,i,j],y) # pearson correlation
    # spearman_corr_p[d,i,j] = stats.spearmanr(X[:,d,i,j],y) # spearman correlation
    # kendalltau_corr_p[d,i,j] = stats.kendalltau(X[:,d,i,j],y) # spearman correlation
    # mutualinfo_corr[d,i,j] = mutual_info_regression(X[:,d,i,j].reshape((-1,1)),y)[0] # mutual information










#%% plot ground truth vs prediction with std
def plot_gt_vs_pred_mean_std():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    predictand = 'Amazon' # 'Congo' # 
    rootpath = '../results/Regression_saved/'
    #folderpath = rootpath+'GCM_Mean_Nino34_Amazon/'

    if predictand=='Amazon':
        y_pred_gcm = np.load('../results/myCNN/lag_0/Amazon/2021-06-14_21.59.38.060129_GCM_masked_Amazon_region1/GCM_Amazon_pred_results_RMSE0.26488437282175314_test.npz')['preds']
        y_pred_reanalysis = np.load('../results/myCNN/lag_0/Amazon/2021-06-15_00.23.44.969560_Reanalysis_masked_Amazon_region1/Reanalysis_Amazon_pred_results_RMSE0.31252981619007986_test.npz')['preds']
    elif predictand=='Congo':
        y_pred_gcm = np.load('../results/myCNN/lag_0/Congo/2021-06-15_00.42.20.427649_GCM_masked_Congo_region1/GCM_Congo_pred_results_RMSE0.5976172958058634_test.npz')['preds']
        y_pred_reanalysis = np.load('../results/myCNN/lag_0/Congo/2021-06-14_23.51.47.085179_Reanalysis_masked_Congo_region1/Reanalysis_Congo_pred_results_RMSE0.4853571058879003_test.npz')['preds']

    def get_y_persist(predictand):
        window = 3
        riverpath = '../data/Climate/RiverFlow/processed/riverflow.csv'
        riverflow_df = pd.read_csv(riverpath,index_col=0,header=0)
        #%% Amazon river flow 
        if predictand=='Amazon':
            targets = riverflow_df[['0']].iloc[600-window+1:1272].to_numpy().reshape((-1,)) # from 1950-window to 200512
        #%% Congo river flow 
        elif predictand=='Congo':
            targets = riverflow_df[['1']].iloc[600-window+1:1272].to_numpy().reshape((-1,)) # from 1950-window to 200512
        ## moving average, result in 195001 to200512
        targets = np.array([np.mean(targets[i:i+window]) for i in range(len(targets)-window+1)]).reshape((-1,))        
        ## standardize
        targets_mean = np.mean(targets[:600],axis=0)
        targets_std = np.std(targets[:600],axis=0)
        targets = (targets-targets_mean)/targets_std
        ## persistence
        month_mean = np.array([np.mean(targets[i:600:12]) for i in range(12)])
        month_std = np.array([np.std(targets[i:600:12]) for i in range(12)])
        y_persist = np.tile(month_mean,reps=3)
        y_persist_std = np.tile(month_std,reps=3)

        return y_persist,y_persist_std

    y_persist,y_persist_std = get_y_persist(predictand=predictand)
    # fig = plt.figure(figsize=(12,5))
    # plt.plot(y_persist)
    # plt.show()
    #np.savez('../results/Regression/Persistence/{}_y_persist_y_persist_std.npz'.format(predictand),y_persist=y_persist,y_persist_std=y_persist_std)

    def get_ys(folderpath):
        gt = np.load(folderpath+'y_gt.npz',allow_pickle=True)
        Ntest = gt['Ntest']
        y_gt_test = gt['y'][-Ntest:]
        linear = np.load(folderpath+'linear.npy',allow_pickle=True)[-Ntest:]
        lasso = np.load(folderpath+'lasso.npy',allow_pickle=True)[-Ntest:]
        ridge = np.load(folderpath+'ridge.npy',allow_pickle=True)[-Ntest:]
        elasticnet = linear = np.load(folderpath+'elasticNet.npy',allow_pickle=True)[-Ntest:]
        randomforest = np.load(folderpath+'randomForest.npy',allow_pickle=True)[-Ntest:]
        dnn = np.load(folderpath+'DNN.npy',allow_pickle=True)[-Ntest:]
        #print('y_gt_test.shape={},linear.shape={}'.format(y_gt_test.shape,linear.shape))
        y_pred_test = np.stack((linear,lasso,ridge,elasticnet,randomforest,dnn),axis=1)
        #print('y_pred_test.shape={}'.format(y_pred_test.shape))
        y_pred_test_mean = np.mean(y_pred_test,axis=1)
        y_pred_test_std = np.std(y_pred_test,axis=1)
        #print('y_pred_test_mean.shape={},y_pred_test_std.shape={}'.format(y_pred_test_mean.shape,y_pred_test_std.shape))
        return y_pred_test_mean,y_pred_test_std,y_gt_test

    y_pred_mean_1,y_pred_std_1,y_gt = get_ys(rootpath+'GCM_Mean_Nino34_{}/'.format(predictand))
    y_pred_mean_2,y_pred_std_2,_ = get_ys(rootpath+'GCM_Nino34_{}/'.format(predictand))
    y_pred_mean_3,y_pred_std_3,_ = get_ys(rootpath+'Nino34_{}/'.format(predictand))
    y_pred_mean_4,y_pred_std_4,_ = get_ys(rootpath+'Reanalysis_Mean_Nino34_{}/'.format(predictand))
    y_pred_mean_5,y_pred_std_5,_ = get_ys(rootpath+'Reanalysis_Nino34_{}/'.format(predictand))

    print('y_pred_mean_1.shape={},y_gt.shape={}'.format(y_pred_mean_1.shape,y_gt.shape))

    #%% calculate RMSE
    rmse = {}
    rmse['GCM_Mean_Nino34'] = np.sqrt(np.mean((y_pred_mean_1-y_gt)**2))
    rmse['GCM_Nino34'] = np.sqrt(np.mean((y_pred_mean_2-y_gt)**2))
    rmse['Nino34'] = np.sqrt(np.mean((y_pred_mean_3-y_gt)**2))
    rmse['Reanalysis_Mean_Nino34'] = np.sqrt(np.mean((y_pred_mean_4-y_gt)**2))
    rmse['Reanalysis_Nino34'] = np.sqrt(np.mean((y_pred_mean_5-y_gt)**2))
    rmse['Persistence'] = np.sqrt(np.mean((y_persist-y_gt)**2))
    print('RMSE={}'.format(rmse))

    #%% plot prediction
    year_months = np.load('../data/Climate/year_months_195001-200512.npy')[-36:]
    interval = 4
    alpha = 0.3
    fig = plt.figure(figsize=(12,5))
    plt.plot(y_pred_mean_1,'-o',label='ESM Mean Nino 3.4')
    plt.fill_between(x=range(len(y_gt)),y1=y_pred_mean_1-y_pred_std_1,y2=y_pred_mean_1+y_pred_std_1,alpha=alpha)
    plt.plot(y_pred_mean_2,'-o',label='ESM Nino 3.4')
    plt.fill_between(x=range(len(y_gt)),y1=y_pred_mean_2-y_pred_std_2,y2=y_pred_mean_2+y_pred_std_2,alpha=alpha)
    plt.plot(y_pred_mean_3,'-o',label='Nino 3.4')
    plt.fill_between(x=range(len(y_gt)),y1=y_pred_mean_3-y_pred_std_3,y2=y_pred_mean_3+y_pred_std_3,alpha=alpha)
    plt.plot(y_pred_mean_4,'-o',label='Reanalysis Mean Nino 3.4')
    plt.fill_between(x=range(len(y_gt)),y1=y_pred_mean_4-y_pred_std_4,y2=y_pred_mean_4+y_pred_std_4,alpha=alpha)
    plt.plot(y_pred_mean_5,'-o',label='Reanalysis Nino 3.4')
    plt.fill_between(x=range(len(y_gt)),y1=y_pred_mean_5-y_pred_std_5,y2=y_pred_mean_5+y_pred_std_5,alpha=alpha)
    plt.plot(y_persist,'-o',label='Persistence')
    plt.fill_between(x=range(len(y_gt)),y1=y_persist-y_persist_std,y2=y_persist+y_persist_std,alpha=alpha)
    plt.plot(y_pred_gcm,'-o',label='ESM SST')
    plt.plot(y_pred_reanalysis,'-o',label='Reanalysis SST')
    plt.plot(y_gt,'--o',color='k',label='Observation')
    bottom, top = plt.ylim()  # return the current ylim
    plt.ylim(bottom,top+0.8)
    plt.xlabel('Months')
    plt.ylabel('Stadardized River Flow')
    plt.xticks(ticks=range(0,len(y_gt),interval), labels=year_months[::interval])
    plt.legend(loc='upper center',ncol=4)

    savepath = rootpath
    savename = '{}_prediction_comparison'.format(predictand)
    plt.savefig(savepath+savename+'.png',dpi=1200,bbox_inches='tight')
    plt.show()






#%%
# def plot_gt_vs_pred():
#     import numpy as np
#     import matplotlib.pyplot as plt

#     predictor = 'GCM_Reanalysis'
#     predictand = 'Congo' # 'Amazon'
#     folder = 'test'
#     save_name_prefix = '{}_{}'.format(predictor,predictand)
#     savepath = '../results/myCNN/lag_0/Congo/2021-06-14_23.51.47.085179_Reanalysis_masked_Congo_region1/'

#     data1 = np.load('../results/myCNN/lag_0/Congo/2021-06-14_23.51.47.085179_Reanalysis_masked_Congo_region1/Reanalysis_Congo_pred_results_RMSE0.4853571058879003_test.npz',allow_pickle=True)
#     data2 = np.load('../results/myCNN/lag_0/Congo/2021-06-15_00.42.20.427649_GCM_masked_Congo_region1/GCM_Congo_pred_results_RMSE0.5976172958058634_test.npz',allow_pickle=True)

#     preds_gcm,y_test = data1['preds'],data1['ytest']
#     preds_reanalysis = data2['preds']

#     ylabel = 'Standardized River Flow'
#     #%% plot figures
#     year_months = np.load('../data/Climate/year_months_195001-200512.npy')
#     year_months = year_months[636:]
#     interval = 6

#     fig = plt.figure(figsize=(12,5))
#     plt.plot(y_test,'--k',label='Observation')
#     plt.plot(preds_gcm,'-vr',label='GCM')
#     plt.plot(preds_reanalysis,'-ob',label='Reanalysis')
#     plt.xticks(ticks=range(0,len(y_test),interval), labels=year_months[::interval])
#     #plt.title('Groundtruth vs Prediction')
#     plt.xlabel('Month')
#     plt.ylabel(ylabel)
#     plt.legend(loc='upper center',ncol=3)
#     if savepath:
#         plt.savefig(savepath+'{}_pred_vs_time_{}.png'.format(save_name_prefix,folder),dpi=1200,bbox_inches='tight')
#     plt.show()








#%% plot
def plot_corr_map2(name,predictor,predictand,lag=0):
    import numpy as np
    import os
    #os.environ['PROJ_LIB'] = 'C:\\WIN10ProgramFiles\\anaconda3\\pkgs\\basemap-1.3.0-py37ha7665c8_0\\Library\\share\\basemap\\'
    from mpl_toolkits.basemap import Basemap
    import matplotlib.pyplot as plt
    import plots

    ## gcm region 1
    lonlat = [50.5,-41.5,349.5,37.5] # [50.5,-42.5,349.5,37.5] # [-124.5,24.5,-66.5,49.5] # map area, [left,bottom,right,top]
    parallels = np.arange(-40.0,40.0,10.0)
    meridians = np.arange(60.0,350.0,30.0) # label lons, 60E to -9.5W, or 60E to 350E
    ## region 1 mask
    masks = np.load('../data/Climate/Reanalysis/masks_cobe_hadley_noaa_uod_1by_world.npy')
    mask = masks[0,52:132,50:350] # cobe region 1

    watercolor = 'white' # '#46bcec'
    cmap = 'bwr' # 'YlOrRd' # 'rainbow' # 'Accent' #'YlGn' #'hsv' #'seismic' # 
    alpha = 1.0 # 0.7
    projection = 'merc' # 'cyl' # 
    resolution = 'i' # 'h'
    area_thresh = 10000
    clim = None # [0.0,1.0] # 
    pos_lons, pos_lats = [], [] # None, None # to plot specific locations
    verbose = False # True # 

    # name = 'Pearson' # 'Spearman' # 'Kendalltau' # 'MutualInfo'
    # predictor = 'GCM'
    # predictand = 'Amazon'
    # lag = 0
    #savepath = None

    savepath = '../results/Correlations/{}_{}_lag_{}/Figs2/'.format(predictor,predictand,lag)
    if savepath and not os.path.exists(savepath): os.makedirs(savepath)
    if name in ['Pearson','Spearman','Kendalltau']:
        filename = '{}_{}_lag_{}/{}_corr_p_{}_{}_lag{}'.format(predictor,predictand,lag,name.lower(),predictor,predictand,lag)
        maps_corr_p = np.load('../results/Correlations/{}.npy'.format(filename),allow_pickle=True)[:,:,:,0] # [Nch,Nlat,Nlon]
    # if name=='Pearson':
    #     pearson_corr_p = np.load('../results/Correlations/GCM_Amazon_lag_0/pearson_corr_p_GCM_Amazon_lag0.npy',allow_pickle=True)
    #     maps_corr_p = pearson_corr_p[:,:,:,0] # [Nch,Nlat,Nlon]
    # elif name=='Spearman':
    #     spearman_corr_p = np.load('../results/Correlations/GCM_Amazon_lag_0/spearman_corr_p_GCM_Amazon_lag0.npy',allow_pickle=True)
    #     maps_corr_p = spearman_corr_p[:,:,:,0] # [Nch,Nlat,Nlon]
    # elif name=='Kendalltau':
    #     kendalltau_corr_p = np.load('../results/Correlations/GCM_Amazon_lag_0/kendalltau_corr_p_GCM_Amazon_lag0.npy',allow_pickle=True)
    #     maps_corr_p = kendalltau_corr_p[:,:,:,0] # [Nch,Nlat,Nlon]
    elif name=='MutualInfo':
        filename = '{}_{}_lag_{}/mutualinfo_corr_{}_{}_lag{}'.format(predictor,predictand,lag,predictor,predictand,lag)
        maps_corr_p = np.load('../results/Correlations/{}.npy'.format(filename),allow_pickle=True)
        #mutualinfo_corr = np.load('../results/Correlations/GCM_Amazon_lag_0/mutualinfo_corr_GCM_Amazon_lag0.npy',allow_pickle=True)
        #maps_corr_p = mutualinfo_corr # [Nch,Nlat,Nlon]
    #### average
    img = np.mean(maps_corr_p,axis=0)
    img[mask==0] = np.nan
    savename = '{}_corrmap_{}_{}_lag{}_avg'.format(name,predictor,predictand,lag)
    title = 'Averaged {} Correlation between {} and {} Lag {}'.format(name,predictor,predictand,lag)
    title = title.replace('GCM','ESM')
    plots.plot_map(img,title=title,savepath=savepath,savename=savename,cmap=cmap,alpha=alpha,
                    lonlat=lonlat,projection=projection,resolution=resolution,area_thresh=area_thresh,
                    parallels=parallels,meridians=meridians,pos_lons=pos_lons, pos_lats=pos_lats,clim=clim,
                    watercolor=watercolor,verbose=verbose)
    #### std
    img = np.std(maps_corr_p,axis=0) 
    img[mask==0] = np.nan
    savename = '{}_corrmap_{}_{}_lag{}_std'.format(name,predictor,predictand,lag)
    title = 'std {} Correlation between {} and {} Lag {}'.format(name,predictor,predictand,lag)
    title = title.replace('GCM','ESM')
    plots.plot_map(img,title=title,savepath=savepath,savename=savename,cmap=cmap,alpha=alpha,
                    lonlat=lonlat,projection=projection,resolution=resolution,area_thresh=area_thresh,
                    parallels=parallels,meridians=meridians,pos_lons=pos_lons, pos_lats=pos_lats,clim=clim,
                    watercolor=watercolor,verbose=verbose)
    #### each GCM or Reanalysis SST
    # gcm_names = np.load('/scratch/wang.zife/YuminLiu/DATA/GCM/GCMdata/tas/processeddata/gcm_names.npy')
    # predictor_names = np.array([gcm_names[i][24:-35] for i in range(len(gcm_names))])
    # if predictor=='Reanalysis': predictor_names = np.array(['COBE','Hadley','NOAA'])
    # for d in range(len(predictor_names)):
    #     img = maps_corr_p[d,:,:]
    #     img[mask==0] = np.nan
    #     title = '{} Correlation between {} {} and {}'.format(name,predictor,predictor_names[d],predictand)
    #     savepath = '../data/Climate/{}/Correlation/Figs/'.format(predictor)
    #     savename = 'corr_{}_{}_ch{}_2.png'.format(predictor,predictand,d+1)
    #     plots.plot_map(img,title=title,savepath=savepath,savename=savename,cmap=cmap,alpha=alpha,
    #                 lonlat=lonlat,projection=projection,resolution=resolution,area_thresh=area_thresh,
    #                 parallels=parallels,meridians=meridians,pos_lons=pos_lons, pos_lats=pos_lats,clim=clim,
    #                 watercolor=watercolor,verbose=verbose)
# lag = 0
# for name in ['Pearson','MutualInfo']: # ['Pearson','Spearman','Kendalltau','MutualInfo']:
#     for predictor in ['GCM']: #['GCM','Reanalysis']:
#         for predictand in ['Amazon','Congo']:
#             print('processing {} {} {} {}'.format(name,predictor,predictand,lag))
#             plot_corr_map2(name,predictor,predictand)





#%% Calculate Correlation between river flows with SST
def cal_corr():
    import os
    import time
    import numpy as np
    from scipy import stats
    from sklearn.feature_selection import mutual_info_regression
    # import matplotlib
    # matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    from utils import get_X_y

    start_time = time.time()
    predictor = 'GCM' # 'Reanalysis' # 
    predictand = 'Amazon' # 'Congo' # 
    lag = 0
    detrend = True # False
    ismasked = False
    standardized = False # True # 
    # X,y,lats,lons,mask = get_X_y(predictor=predictor,predictand=predictand,detrend=detrend,ismasked=ismasked,standardized=standardized)
    # y = y.reshape((-1,))
    # if lag>0: X,y = X[:-lag,:,:,:],y[lag:]
    # print('X.shape={},y.shape={}'.format(X.shape,y.shape))

    # #import matplotlib.pyplot as plt
    # fig = plt.figure(figsize=(12,5))
    # plt.plot(X[:,0,40,150],label='location1')
    # plt.plot(X[:,10,45,100],label='location2')
    # plt.show()

    # # x = np.random.rand(6720,1).reshape((-1,))
    # # y = np.random.rand(6720,1).reshape((-1,))
    # # x = -y
    # # print('pearsonr(x, y)={}'.format(stats.pearsonr(x, y)))
    # # print('spearmanr(x, y)={}'.format(stats.spearmanr(x, y)))
    # # print('kendalltau(x, y)={}'.format(stats.kendalltau(x, y)))
    # # print('mutual_info_regression(x, y)={}'.format(mutual_info_regression(x.reshape((-1,1)), y)))
    # Nmon,D,Nlat,Nlon = X.shape
    # pearson_corr_p = np.zeros((D,Nlat,Nlon,2))
    # spearman_corr_p = np.zeros((D,Nlat,Nlon,2))
    # kendalltau_corr_p = np.zeros((D,Nlat,Nlon,2))
    # mutualinfo_corr = np.zeros((D,Nlat,Nlon))
    # for i in range(Nlat):
    #     print('processing {}-th latitude'.format(i+1))
    #     for j in range(Nlon):
    #         if mask[i,j]==1000: continue
    #         for d in range(D):
    #             if np.isnan(X[:,d,i,j]).any() or np.std(X[:,d,i,j])==0: continue
    #             pearson_corr_p[d,i,j] = stats.pearsonr(X[:,d,i,j],y) # pearson correlation
    #             spearman_corr_p[d,i,j] = stats.spearmanr(X[:,d,i,j],y) # spearman correlation
    #             kendalltau_corr_p[d,i,j] = stats.kendalltau(X[:,d,i,j],y) # spearman correlation
    #             mutualinfo_corr[d,i,j] = mutual_info_regression(X[:,d,i,j].reshape((-1,1)),y)[0] # mutual information
    #         #break
    # savepath = '../results/Correlations/{}_{}_lag_{}/'.format(predictor,predictand,lag)
    # if not os.path.exists(savepath): os.makedirs(savepath)
    # savename1 = 'pearson_corr_p_{}_{}_lag{}'.format(predictor,predictand,lag)
    # np.save(savepath+savename1+'.npy',pearson_corr_p)
    # savename2 = 'spearman_corr_p_{}_{}_lag{}'.format(predictor,predictand,lag)
    # np.save(savepath+savename2+'.npy',spearman_corr_p)
    # savename3 = 'kendalltau_corr_p_{}_{}_lag{}'.format(predictor,predictand,lag)
    # np.save(savepath+savename3+'.npy',kendalltau_corr_p)
    # savename4 = 'mutualinfo_corr_{}_{}_lag{}'.format(predictor,predictand,lag)
    # np.save(savepath+savename4+'.npy',mutualinfo_corr)
    # with open(savepath+'config.txt',mode='w') as f:
    #     f.write('lag={}\npredictor:{}\npredictand={}\ndetrend={}\nismasked={}\nstandardized={}\n'.format(lag,predictor,predictand,detrend,ismasked,standardized))

    # fig = plt.figure(figsize=(12,5))
    # plt.plot(y,label='{}'.format(predictand))
    # plt.plot(X[:,1,40,50],label='{} sample at ({},{})'.format(predictor,lats[40],lons[50]))
    # plt.plot(X[:,1,40,180],label='{} sample at ({},{})'.format(predictor,lats[40],lons[180]))
    # #plt.plot(y,label='{}'.format(predictand))
    # plt.legend()
    # title = '{} and {} Time Series Samples'.format(predictor,predictand)
    # plt.title(title)
    # #plt.savefig('../data/Climate/{}/Correlation/Figs/{}.png'.format(predictor,title.lower().replace(' ','_')),dpi=1200,bbox_inches='tight')
    # plt.show()

    #%% plot
    def plot_corr_map(corr_maps,name='',predictor='GCM',predictand='Amazon',lag=0):
        savepath = '../results/Correlations/{}_{}_lag_{}/Figs/'.format(predictor,predictand,lag)
        if not os.path.exists(savepath): os.makedirs(savepath)
        #### each GCM or Reanalysis SST
        # gcm_names = np.load('/scratch/wang.zife/YuminLiu/DATA/GCM/GCMdata/tas/processeddata/gcm_names.npy')
        # predictor_names = np.array([gcm_names[i][24:-35] for i in range(len(gcm_names))])
        # if predictor=='Reanalysis': predictor_names = np.array(['COBE','Hadley','NOAA'])
        # for d in range(D):
        #     fig = plt.figure(figsize=(12,5))
        #     img = corr_maps[d,:,:]
        #     img[img=0] = np.nan
        #     plt.imshow(img)
        #     plt.colorbar(fraction=0.01)
        #     plt.xticks([])
        #     plt.yticks([])
        #     plt.xlabel('Longitude')
        #     plt.ylabel('Latitude')
        #     plt.title('Pearson Correlation between {} {} and {}'.format(predictor,predictor_names[D],predictand))
        #     plt.savefig('../data/Climate/{}/Correlation/Figs/corr_{}_{}_ch{}.png'.format(predictor,predictor,predictand,d+1),dpi=1200,bbox_inches='tight')
        #     plt.close()
        #### average
        fig = plt.figure(figsize=(12,5))
        img = np.mean(corr_maps,axis=0)
        img[img==0] = np.nan
        plt.imshow(img)
        plt.colorbar(fraction=0.01)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Averaged {} Correlation between {} and {} Lag {}'.format(name,predictor,predictand,lag))
        #savepath = '../results/Correlations/{}_{}/'.format(predictor,predictand)
        savename = '{}_corrmap_{}_{}_lag{}_avg_2'.format(name,predictor,predictand,lag)
        plt.savefig(savepath+savename+'.png',dpi=1200,bbox_inches='tight')
        plt.close()
        #### std
        fig = plt.figure(figsize=(12,5))
        img = np.std(corr_maps,axis=0)
        img[img==0] = np.nan
        plt.imshow(img)
        plt.colorbar(fraction=0.01)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('std {} Correlation between {} and {} Lag {}'.format(name,predictor,predictand,lag))
        #savepath = '../results/Correlations/{}_{}/'.format(predictor,predictand)
        savename = '{}_corrmap_{}_{}_lag{}_std'.format(name,predictor,predictand,lag)
        plt.savefig(savepath+savename+'.png',dpi=1200,bbox_inches='tight')
        plt.close()

    pearson_corr_p = np.load('../results/Correlations/GCM_Amazon_lag_0/pearson_corr_p_GCM_Amazon_lag0.npy',allow_pickle=True)
    # spearman_corr_p = np.load('../results/Correlations/GCM_Amazon_lag_0/spearman_corr_p_GCM_Amazon_lag0.npy',allow_pickle=True)
    # kendalltau_corr_p = np.load('../results/Correlations/GCM_Amazon_lag_0/kendalltau_corr_p_GCM_Amazon_lag0.npy',allow_pickle=True)
    # mutualinfo_corr = np.load('../results/Correlations/GCM_Amazon_lag_0/mutualinfo_corr_GCM_Amazon_lag0.npy',allow_pickle=True)
    plot_corr_map(pearson_corr_p[:,:,:,0],name='Pearson',predictor=predictor,predictand=predictand,lag=lag)
    # plot_corr_map(spearman_corr_p[:,:,:,0],name='Spearman',predictor=predictor,predictand=predictand,lag=lag)
    # plot_corr_map(kendalltau_corr_p[:,:,:,0],name='Kendalltau',predictor=predictor,predictand=predictand,lag=lag)
    # plot_corr_map(mutualinfo_corr,name='Mutual_Information',predictor=predictor,predictand=predictand,lag=lag)
    end_time = time.time()
    print('used time: {} minutes'.format((end_time-start_time)/60))






#%%
def plot_worldmap2(year):
    import os
    #os.environ['PROJ_LIB'] = 'C:/WIN10ProgramFiles/anaconda3/pkgs/basemap-1.3.0-py37ha7665c8_0/Library/share/basemap/'
    from mpl_toolkits.basemap import Basemap
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    #import matplotlib
    from matplotlib import pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection 
    import numpy as np

    lons = np.load('../data/Climate/Reanalysis/lons.npy',allow_pickle=True)
    lats = np.load('../data/Climate/Reanalysis/lats.npy',allow_pickle=True)
    data = np.load('../data/Climate/Reanalysis/sst_cobe_hadley_noaa_190001-201912_1by1_world.npy',allow_pickle=True)
    cobe,hadley,noaa = data[:,0,:,:],data[:,1,:,:],data[:,2,:,:]




    def get_diff_cobe(year):
        # ### 2000 anomaly
        # if year==2000:
        #     cobe_avg = np.mean(cobe[972:1332,:,:],axis=0) # sst in year [1981 to 2010]
        #     cobe_year = np.mean(cobe[1200:1212,:,:],axis=0) # sst in year 2000
        ### 2001 anomaly
        if year==2001:
            cobe_avg = np.mean(cobe[1032:1392,:,:],axis=0) # sst in year [1986 to 2015]
            cobe_year = np.mean(cobe[1212:1224,:,:],axis=0) # sst in year 2002
        ### 2002 anomaly
        elif year==2002:
            cobe_avg = np.mean(cobe[1032:1392,:,:],axis=0) # sst in year [1986 to 2015]
            cobe_year = np.mean(cobe[1224:1236,:,:],axis=0) # sst in year 2002
        ### 2003 anomaly
        elif year==2003:
            cobe_avg = np.mean(cobe[1032:1392,:,:],axis=0) # sst in year [1986 to 2015]
            cobe_year = np.mean(cobe[1236:1248,:,:],axis=0) # sst in year 2003
        ### 2004 anomaly
        elif year==2004:
            cobe_avg = np.mean(cobe[1032:1392,:,:],axis=0) # sst in year [1986 to 2015]
            cobe_year = np.mean(cobe[1248:1260,:,:],axis=0) # sst in year 2004
        ### 2005 anomaly
        elif year==2005:
            cobe_avg = np.mean(cobe[1032:1392,:,:],axis=0) # sst in year [1986 to 2015]
            cobe_year = np.mean(cobe[1260:1272,:,:],axis=0) # sst in year 2005
        ### 2006 anomaly
        elif year==2006:
            cobe_avg = np.mean(cobe[1080:1440,:,:],axis=0) # sst in year [1990 to 2019]
            cobe_year = np.mean(cobe[1272:1284,:,:],axis=0) # sst in year 2006
        ### 2007 anomaly
        elif year==2007:
            cobe_avg = np.mean(cobe[1080:1440,:,:],axis=0) # sst in year [1990 to 2019]
            cobe_year = np.mean(cobe[1284:1296,:,:],axis=0) # sst in year 2007
        ### 2008 anomaly
        elif year==2008:
            cobe_avg = np.mean(cobe[1080:1440,:,:],axis=0) # sst in year [1990 to 2019]
            cobe_year = np.mean(cobe[1296:1308,:,:],axis=0) # sst in year 2008
        ### 2009 anomaly
        elif year==2009:
            cobe_avg = np.mean(cobe[1080:1440,:,:],axis=0) # sst in year [1990 to 2019]
            cobe_year = np.mean(cobe[1308:1320,:,:],axis=0) # sst in year 2009
        ### 2010 anomaly
        elif year==2010:
            cobe_avg = np.mean(cobe[1080:1440,:,:],axis=0) # sst in year [1990 to 2019]
            cobe_year = np.mean(cobe[1320:1332,:,:],axis=0) # sst in year 2010
        cobe_diff = cobe_year-cobe_avg
        cobe_diff[cobe_diff==0] = np.nan
        return cobe_diff

    #year = 2001
    cobe_diff = get_diff_cobe(year=year)
    # fig = plt.figure()
    # plt.imshow(cobe_diff)
    # plt.show()


    #lonlat = [235.5,24.5,293.5,49.5]
    lonlat = [0.5,-89.5,359.5,89.5] # [-20,-80,330,80] # [lonmin,latmin,lonmax,latmax]
    watercolor = 'white' # '#46bcec'
    cmap = 'bwr' # 'rainbow' # 'YlOrRd' # 'Accent' #'YlGn' #'hsv' #'seismic' # 
    alpha = 1 # 0.7
    projection = 'cyl' # 'merc' # 
    resolution = 'i' # 'l' # 'f' # 'h' # 
    area_thresh =  10000 # None # 
    clim = None
    parallels = np.arange(-80.0,80.0,10.0)
    meridians = np.arange(-20.0,330.0,30.0) # label lons, 60E to -9.5W, or 60E to 350E
        
    #%% draw world map
    #fig = plt.figure(figsize=(18,8))
    fig = plt.figure()
    fig.set_size_inches([18,8])
    mainax = fig.add_subplot(111) 
    m = Basemap(llcrnrlon=lonlat[0],llcrnrlat=lonlat[1],urcrnrlon=lonlat[2],urcrnrlat=lonlat[3],
                projection=projection,resolution=resolution,area_thresh=area_thresh)
    m.drawcoastlines(linewidth=1.0,color='k')
    #m.drawcountries(linewidth=1.0,color='k')
    #m.drawstates(linewidth=0.2,color='k')
    #m.drawrivers(color='dodgerblue',linewidth=1.0,zorder=1)
    #m.fillcontinents(color='w',alpha=0.1)
    #m.drawmapboundary(fill_color=watercolor)
    m.fillcontinents(color='gray',alpha=0.3,lake_color=watercolor,zorder=0)
    m.drawparallels(parallels,labels=[True,False,False,False],dashes=[1,2])
    m.drawmeridians(meridians,labels=[False,False,False,True],dashes=[1,2])
    img = np.flipud(cobe_diff)
    m.imshow(img,cmap=cmap,alpha=alpha,zorder=0)
    cbar = m.colorbar(fraction=0.02,location='bottom',extend='both',pad=0.5)
    cbar.set_label(label='Annual SST anomaly in {} (in '.format(year)+r'$^{\circ}C$'+')')


    #%% draw Congo waterway
    ### m.readshapefile('../data/Shapefiles/hotosm_cod_waterways_lines/hotosm_cod_waterways_lines',name='CongoWaterWay',
    #                 drawbounds=True,zorder=None,linewidth=0.5,color='blue')
    # m.readshapefile('../data/Shapefiles/hotosm_cod_waterways_lines/hotosm_cod_waterways_lines', 'CongoWaterWay', drawbounds=False)
    # for info, shape in zip(m.CongoWaterWay_info, m.CongoWaterWay):
    #     #print('info={},len(shape)={}'.format(info,len(shape)))
    #     shape = np.array(shape)
    #     lons,lats = shape[:,0],shape[:,1]
    #     m.plot(*m(lons, lats), linewidth=0.5, color='blue', zorder=18)
    # plt.show()
    #%% draw Amazon waterway
    # # m.readshapefile('../data/Shapefiles/AmazonDrainage/AmazonDrainage',name='AmazonDrainage',
    # #                 drawbounds=True)#,zorder=None,linewidth=0.5,color='blue')
    # m.readshapefile('../data/Shapefiles/AmazonDrainage/AmazonDrainage',name='AmazonDrainage',drawbounds=False)
    # for info, shape in zip(m.AmazonDrainage_info, m.AmazonDrainage):
    #     #print('info={},len(shape)={}'.format(info,len(shape)))
    #     shape = np.array(shape)
    #     lons,lats = shape[:,0]+360,shape[:,1]
    #     m.plot(*m(lons, lats), linewidth=0.5, color='blue', zorder=18)
    # plt.show()


    #%% plot Amazon River basin
    #m.readshapefile('../data/Shapefiles/AmazonBasin/amapoly_ivb', 'AmazonBasin', drawbounds=False)
    m.readshapefile('../data/Climate/Shapefiles/AmazonBasinLimits-master/amazon_sensulatissimo_gmm_v1', 'AmazonBasin', drawbounds=True)
    patches_amazon = []
    for info, shape in zip(m.AmazonBasin_info, m.AmazonBasin):
        shape = np.array(shape)
        shape[:,0] += 360 # transform negative longitude to positive
        patches_amazon.append(Polygon(xy=shape, closed=True))
    mainax.add_collection(PatchCollection(patches_amazon, facecolor='g', edgecolor='g', alpha=0.5))
    pos_lons,pos_lats = [304.49,15.3],[-1.95,-4.3]
    lonss, latss = m(pos_lons, pos_lats)
    m.scatter(lonss, latss, marker = 'o', color='k', zorder=2,s=10)
    #%% draw Congo River Basin
    m.readshapefile('../data/Climate/Shapefiles/congo_basin_polyline/congo_basin_polyline', 'CongoBasin', drawbounds=False)
    patches_congo,shapes = [],[]
    for info, shape in zip(m.CongoBasin_info, m.CongoBasin):
        #print('info={},len(shape)={}'.format(info,len(shape)))
        shapes.append(np.array(shape))
    shapes = [shapes[2],shapes[0],shapes[4],shapes[3],shapes[1]]
    shapes = np.concatenate(shapes,axis=0)
    patches_congo.append(Polygon(xy=shapes, closed=True))
    mainax.add_collection(PatchCollection(patches_congo, facecolor='lime', edgecolor='lime', alpha=0.5))

    #%%
    def draw_rectangle(lats, lons, m, facecolor='red', alpha=0.5, edgecolor='k',fill=False,**kwargs):
        x, y = m(lons, lats)
        xy = zip(x,y)
        rect = Polygon(list(xy),facecolor=facecolor,alpha=alpha,edgecolor=edgecolor,fill=fill,**kwargs)
        plt.gca().add_patch(rect)
    ## plot enso regions
    nino12_lats,nino12_lons = [-10,0,0,-10],[270,270,280,280]
    nino3_lats,nino3_lons = [-5,5,5,-5],[210,210,270,270]
    nino34_lats,nino34_lons = [-5,5,5,-5],[190,190,240,240]
    #oni_lats,oni_lons = nino34_lats,nino34_lons
    nino4_lats,nino4_lons = [-5,5,5,-5],[160,160,210,210]
    draw_rectangle(nino4_lats,nino4_lons,m,facecolor='c',alpha=1.0,edgecolor='c',fill=False,linewidth=4,zorder=5)
    draw_rectangle(nino3_lats,nino3_lons,m,facecolor='y',alpha=1.0,edgecolor='y',fill=False,linewidth=4,zorder=5)
    draw_rectangle(nino34_lats,nino34_lons,m,edgecolor='k',fill=False,linewidth=4,hatch='/',zorder=5)
    draw_rectangle(nino12_lats,nino12_lons,m,facecolor='w',alpha=0.8,edgecolor='w',fill=False,linewidth=4,zorder=5)
    #%% plot IOD regions
    dmi_lats_west,dmi_lons_west = [-10,10,10,-10],[50,50,70,70]
    dmi_lats_east,dmi_lons_east = [-10,0,0,-10],[90,90,110,110]
    draw_rectangle(dmi_lats_west,dmi_lons_west,m,facecolor='orange',alpha=1,edgecolor='orange',fill=False,linewidth=4,zorder=5)
    draw_rectangle(dmi_lats_east,dmi_lons_east,m,facecolor='m',alpha=1,edgecolor='m',fill=False,linewidth=4,zorder=5)
    ## add text
    def add_text(lat, lon, m, text,fontsize=12,**kwargs):
        x, y = m(lon, lat)
        plt.text(x,y,text,**kwargs)
    add_text(lat=-10,lon=165,m=m,text='Nino4',color='c')
    add_text(lat=-10,lon=245,m=m,text='Nino3',color='y')
    add_text(lat=-10,lon=210,m=m,text='Nino3.4',color='k')
    add_text(lat=-15,lon=260,m=m,text='Nino1+2',color='w')
    add_text(lat=-15,lon=55,m=m,text='IOD west',color='orange')
    add_text(lat=-15,lon=95,m=m,text='IOD east',color='m')
    add_text(lat=5,lon=310,m=m,text='Amazon Basin',color='g')
    add_text(lat=10,lon=10,m=m,text='Congo Basin',color='lime')



    #%% draw zoom in sub figures
    def add_subplot_axes(ax,pos,facecolor='w'):
        fig = plt.gcf()
        box = ax.get_position()
        width = box.width
        height = box.height
        inax_position  = ax.transAxes.transform(pos[0:2])
        transFigure = fig.transFigure.inverted()
        infig_position = transFigure.transform(inax_position)    
        x = infig_position[0]
        y = infig_position[1]
        width *= pos[2]
        height *= pos[3]  # <= Typo was here
        subax = fig.add_axes([x,y,width,height],facecolor=facecolor)
        subax.set_xticks([])
        subax.set_yticks([])
        # x_labelsize = subax.get_xticklabels()[0].get_size()
        # y_labelsize = subax.get_yticklabels()[0].get_size()
        # x_labelsize *= rect[2]**0.5
        # y_labelsize *= rect[3]**0.5
        # subax.xaxis.set_tick_params(labelsize=x_labelsize)
        # subax.yaxis.set_tick_params(labelsize=y_labelsize)
        return subax

    #%% draw Amazon River zoomed in basin
    subpos = [0.6,0.65,0.3,0.3] # [left, bottom, width, height]
    subax3 = add_subplot_axes(mainax,subpos)

    # subax3 = zoomed_inset_axes(parent_axes=mainax,zoom=2,loc='upper right')
    # subax3.set_xlim(280,316)
    # subax3.set_ylim(-21,10)
    # subax3.set_xticks([])
    # subax3.set_yticks([])
        
    m3 = Basemap(llcrnrlon=280,llcrnrlat=-21,urcrnrlon=316,urcrnrlat=10,resolution='i',ax=subax3,area_thresh=8000)
    m3.shadedrelief()

    #mark_inset(mainax, subax3, loc1=2, loc2=4, fc="none", ec="0.5")

    # m3.drawcoastlines(linewidth=1.0,color='k')
    # m3.fillcontinents(color='gray',alpha=0.3,lake_color=watercolor,zorder=0)
    # #m3.drawcountries(color='k', linewidth=1)
    # m3.drawmapboundary(fill_color=watercolor)
    # #m3.drawrivers(color='dodgerblue',linewidth=0.5,zorder=1)
    lonss,latss = m3(15.3,-4.3)
    m3.scatter(lonss, latss, marker = 'o', color='k', zorder=2,s=15)
    # m3.readshapefile('../data/Shapefiles/AmazonBasinLimits-master/amazon_sensulatissimo_gmm_v1', 'AmazonBasin', drawbounds=True)
    # patches_amazon = []
    # for info, shape in zip(m3.AmazonBasin_info, m3.AmazonBasin):
    #     shape = np.array(shape)
    #     shape[:,0] += 360 # transform negative longitude to positive
    #     patches_amazon.append(Polygon(xy=shape, closed=True,fill=False))
    # subax3.add_collection(PatchCollection(patches_amazon,edgecolor='r',alpha=0.2))
    m3.readshapefile('../data/Climate/Shapefiles/AmazonDrainage/AmazonDrainage',name='AmazonDrainage',drawbounds=False)
    for info, shape in zip(m3.AmazonDrainage_info, m3.AmazonDrainage):
        #print('info={},len(shape)={}'.format(info,len(shape)))
        shape = np.array(shape)
        lons,lats = shape[:,0]+360,shape[:,1]
        m3.plot(*m3(lons, lats), linewidth=0.2, color='blue', zorder=1)
    #%% draw Congo River zoomed in basin
    subpos = [0.08,0.65,0.3,0.3] # [left, bottom, width, height]
    subax2 = add_subplot_axes(mainax,subpos)
    m2 = Basemap(llcrnrlon=10,llcrnrlat=-15,urcrnrlon=35,urcrnrlat=10,resolution='i',ax=subax2,area_thresh=8000)
    m2.shadedrelief()
    # m2.drawcoastlines(linewidth=1.0,color='k')
    # m2.fillcontinents(color='gray',alpha=0.3,lake_color=watercolor,zorder=0)
    # #m2.drawcountries(color='k', linewidth=1)
    # m2.drawmapboundary(fill_color=watercolor)
    # #m2.drawrivers(color='dodgerblue',linewidth=0.5,zorder=1)
    lonss,latss = m2(304.49,-1.95)
    m2.scatter(lonss, latss, marker = 'o', color='k', zorder=2,s=15)
    # m2.readshapefile('../data/Shapefiles/congo_basin_polyline/congo_basin_polyline', 'CongoBasin', drawbounds=False)
    # patches_congo,shapes = [],[]
    # for info, shape in zip(m2.CongoBasin_info, m2.CongoBasin):
    #     #print('info={},len(shape)={}'.format(info,len(shape)))
    #     shapes.append(np.array(shape))
    # shapes = [shapes[2],shapes[0],shapes[4],shapes[3],shapes[1]]
    # shapes = np.concatenate(shapes,axis=0)
    # patches_congo.append(Polygon(xy=shapes, closed=True,fill=False))
    # subax2.add_collection(PatchCollection(patches_congo,edgecolor='m',alpha=0.2))
    m2.readshapefile('../data/Climate/Shapefiles/hotosm_cod_waterways_lines/hotosm_cod_waterways_lines',name='CongoWaterWay',
                    drawbounds=True,zorder=1,linewidth=0.2,color='blue')
    # m2.readshapefile('../data/Shapefiles/hotosm_cod_waterways_lines/hotosm_cod_waterways_lines', 'CongoWaterWay', drawbounds=False)
    # for info, shape in zip(m2.CongoWaterWay_info, m2.CongoWaterWay):
    #     #print('info={},len(shape)={}'.format(info,len(shape)))
    #     shape = np.array(shape)
    #     lons,lats = shape[:,0],shape[:,1]
    #     m2.plot(*m2(lons, lats), linewidth=0.5, color='blue', zorder=1)


    savepath = '../data/'
    savename = 'worldmap_{}'.format(year)
    plt.savefig(savepath+savename+'.png',dpi=1200,bbox='tight')
    #plt.show()
    plt.close()

# for year in range(2001,2011):
#     plot_worldmap2(year)


# def plot_worldmap():
#     import os
#     os.environ['PROJ_LIB'] = 'C:\\WIN10ProgramFiles\\anaconda3\\pkgs\\basemap-1.3.0-py37ha7665c8_0\\Library\\share\\basemap\\'
#     from mpl_toolkits.basemap import Basemap
#     from matplotlib import pyplot as plt
#     from matplotlib.patches import Polygon
#     from matplotlib.collections import PatchCollection 
#     import numpy as np


#     #lonlat = [235.5,24.5,293.5,49.5]
#     lonlat = [-20,-80,330,80]
#     watercolor = 'white' # '#46bcec'
#     cmap = 'YlOrRd' # 'rainbow' # 'Accent' #'YlGn' #'hsv' #'seismic' # 
#     alpha = 0.7
#     projection = 'cyl' # 'merc' # 
#     resolution = 'i' # 'h' # 'l' #
#     area_thresh = 10000
#     clim = None
#     parallels = np.arange(-80.0,80.0,10.0)
#     meridians = np.arange(-20.0,330.0,30.0) # label lons, 60E to -9.5W, or 60E to 350E
        
#     #fig = plt.figure(figsize=(18,8))
#     fig = plt.figure()
#     fig.set_size_inches([18,8])
#     ax = fig.add_subplot(111) 
#     m = Basemap(llcrnrlon=lonlat[0],llcrnrlat=lonlat[1],urcrnrlon=lonlat[2],urcrnrlat=lonlat[3],
#                 projection=projection,resolution=resolution,area_thresh=area_thresh)
#     m.drawcoastlines(linewidth=1.0,color='k')
#     #m.drawcountries(linewidth=1.0,color='k')
#     #m.drawstates(linewidth=0.2,color='k')
#     #m.drawrivers(color='dodgerblue',linewidth=1.0,zorder=1)
#     #m.fillcontinents(color='w',alpha=0.1)
#     #m.drawmapboundary(fill_color=watercolor)
#     m.fillcontinents(color = 'gray',alpha=1.0,lake_color=watercolor)
#     m.drawparallels(parallels,labels=[True,False,False,False],dashes=[1,2])
#     m.drawmeridians(meridians,labels=[False,False,False,True],dashes=[1,2])
#     # img = np.flipud(img)
#     # m.imshow(img,cmap=cmap,alpha=alpha,zorder=1)
#     # m.colorbar(fraction=0.02)
#     # plot Amazon River basin
#     #m.readshapefile('..\\data\\Shapefiles\\AmazonBasin\\amapoly_ivb', 'AmazonBasin', drawbounds=False)
#     m.readshapefile('..\\data\\Shapefiles\\AmazonBasinLimits-master\\amazon_sensulatissimo_gmm_v1', 'AmazonBasin', drawbounds=True)
#     patches = []
#     for info, shape in zip(m.AmazonBasin_info, m.AmazonBasin):
#         shape = np.array(shape)
#         shape[:,0] += 360 # transform negative longitude to positive
#         patches.append(Polygon(xy=shape, closed=True))
#         # if info['area']>10:
#         #     x,y = zip(*shape)
#         #     x = tuple(map(lambda e:e+360,x))
#         #     m.plot(x,y,marker=None,color='m')
#     ax.add_collection(PatchCollection(patches, facecolor='r', edgecolor='r', alpha=0.5))


#     m.readshapefile('..\\data\\Shapefiles\\congo_basin_polyline\\congo_basin_polyline', 'CongoBasin', drawbounds=False)
#     patches,shapes = [],[]
#     for info, shape in zip(m.CongoBasin_info, m.CongoBasin):
#         print('info={},len(shape)={}'.format(info,len(shape)))
#         shapes.append(np.array(shape))
#     shapes = [shapes[2],shapes[0],shapes[4],shapes[3],shapes[1]]
#     shapes = np.concatenate(shapes,axis=0)
#     patches.append(Polygon(xy=shapes, closed=True))
#     ax.add_collection(PatchCollection(patches, facecolor='m', edgecolor='m', alpha=0.5))

#     def draw_rectangle(lats, lons, m, facecolor='red', alpha=0.5, edgecolor='k',fill=False,linewidth=1):
#         x, y = m(lons, lats)
#         xy = zip(x,y)
#         rect = Polygon(list(xy),facecolor=facecolor,alpha=alpha,edgecolor=edgecolor,fill=fill)
#         plt.gca().add_patch(rect)
#     ## plot enso regions
#     nino12_lats,nino12_lons = [-10,0,0,-10],[270,270,280,280]
#     nino3_lats,nino3_lons = [-5,5,5,-5],[210,210,270,270]
#     nino34_lats,nino34_lons = [-5,5,5,-5],[190,190,240,240]
#     #oni_lats,oni_lons = nino34_lats,nino34_lons
#     nino4_lats,nino4_lons = [-5,5,5,-5],[160,160,210,210]

#     draw_rectangle(nino4_lats,nino4_lons,m,facecolor='g',alpha=0.8,edgecolor='g',fill=True)
#     draw_rectangle(nino3_lats,nino3_lons,m,facecolor='b',alpha=0.8,edgecolor='b',fill=True)
#     draw_rectangle(nino34_lats,nino34_lons,m,edgecolor='k',fill=False,linewidth=4)
#     draw_rectangle(nino12_lats,nino12_lons,m,facecolor='r',alpha=0.8,edgecolor='r',fill=True)

#     ## add text
#     def add_text(lat, lon, m, text,fontsize=12,**kwargs):
#         x, y = m(lon, lat)
#         plt.text(x,y,text,**kwargs)
#     add_text(lat=0,lon=165,m=m,text='Nino4')
#     add_text(lat=0,lon=245,m=m,text='Nino3')
#     add_text(lat=10,lon=215,m=m,text='Nino3.4')
#     add_text(lat=-15,lon=270,m=m,text='Nino1+2')
#     add_text(lat=5,lon=40,m=m,text='Congo Basin')
#     add_text(lat=5,lon=310,m=m,text='Amazon Basin')

#     savepath = '../data/'
#     savename = 'worldmap'
#     #plt.savefig(savepath+savename+'.png',dpi=1200,bbox='tight')
#     plt.show()

#     # fig = plt.figure()
#     # fig.set_size_inches([18,8])
#     # ax = fig.add_subplot(111)
    
#     # ## plot basemap, rivers and countries
#     # #m = Basemap(llcrnrlat=19.5, urcrnrlat=26.0, llcrnrlon=99.6, urcrnrlon=107.5, resolution='h')
#     # m = Basemap(llcrnrlat=-80, urcrnrlat=80.0, llcrnrlon=-20, urcrnrlon=330, resolution='h')
#     # m.arcgisimage(service='World_Shaded_Relief')
#     # m.drawrivers(color='dodgerblue',linewidth=1.0,zorder=1)
#     # m.drawcountries(color='k',linewidth=1.25)



#### save as gif
# def savegif():
#     import imageio
    
#     imgpath = 'C:\\Users\\YM\\Desktop\\Months\\'
#     prefix = 'saliency_gcms_tmin_month_'
#     months = [str(n) for n in range(0,36)]
    
#     imgpath = 'C:\\Users\\YM\\Desktop\\NaturalMonths\\'
#     prefix = 'saliency_gcms_tmin_month_'
#     months = ['January','February','March','April','May','June','July','August','September','October','November','December']
    
#     imgpath = 'C:\\Users\\YM\\Desktop\\Seasons\\'
#     prefix = 'saliency_gcms_tmin_season_'
#     months = ['Spring(MAM)','Summer(JJA)','Autumn(SON)','Winter(DJF)']
    
#     imgpath = 'C:\\Users\\YM\\Desktop\\Years\\'
#     prefix = 'saliency_gcms_tmin_year_'
#     months = ['2003','2004','2005']
    
#     duration = 1
#     images = []
    
#     for month in months:
#         filename = imgpath+prefix+month+'.png'
#         images.append(imageio.imread(filename))
        
#     savepath = imgpath
#     savename = prefix+'evolution'
#     imageio.mimsave(savepath+savename+'.gif',images,duration=duration)



#%% plot river flow
# def plot_river_flow_enso_index():
#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt
    
    
#     # def read_enso():
#     #     years = np.load('../data/Nino/processed/Nino12_187001-201912_array.npz')['years']
#     #     nino12 = pd.read_csv('../data/Nino/processed/Nino12_187001-201912_series.csv',index_col=0)
#     #     nino3 = pd.read_csv('../data/Nino/processed/Nino3_187001-201912_series.csv',index_col=0)
#     #     nino34 = pd.read_csv('../data/Nino/processed/Nino34_187001-201912_series.csv',index_col=0)
#     #     nino4 = pd.read_csv('../data/Nino/processed/Nino4_187001-201912_series.csv',index_col=0)
#     #     nino12_anom = pd.read_csv('../data/Nino/processed/Nino12_anom_187001-201912_series.csv',index_col=0)
#     #     nino3_anom = pd.read_csv('../data/Nino/processed/Nino3_anom_187001-201912_series.csv',index_col=0)
#     #     nino34_anom = pd.read_csv('../data/Nino/processed/Nino34_anom_187001-201912_series.csv',index_col=0)
#     #     nino4_anom = pd.read_csv('../data/Nino/processed/Nino4_anom_187001-201912_series.csv',index_col=0)
#     #     tni = pd.read_csv('../data/Nino/processed/Trans_Nino_index_hadISST_187001-201912_series.csv',index_col=0)
#     #     soi = pd.read_csv('../data/Nino/processed/soi_186601-201912_series.csv',index_col=0)
#     #     soi = soi.loc[187001:] # 187001 to 201912
        
#     #     indices = pd.concat((nino12,nino3,nino34,nino4,nino12_anom,nino3_anom,nino34_anom,nino4_anom,tni,soi),axis=1) # 187001 to 201912
#     #     return indices, years  
    
#     # indices_df, _ = read_enso() # 187001 to 201912
    
#     # nino34_anom_df = indices_df['Nino34_anom'].loc[195001:201012]
#     # nino34_anom = nino34_anom_df.to_numpy().reshape((-1,12))
    
#     def get_colors(heights):
#         colors = []
#         for h in heights:
#             if h>=0.5:
#                 c = 'r'
#             elif h<=-0.5:
#                 c = 'b'
#             else:
#                 c = 'gray'
#             colors.append(c)
#         return colors
    
#     oni_df = pd.read_csv('../data/ONI/raw/oni_nino34_anomaly_new.csv',index_col=0)
#     oni = oni_df.loc[1950:2010].to_numpy()
#     oni = oni[-10:,:]
    
#     riverpath = 'C:\\Users\\YM\\OneDrive - Northeastern University\\myProgramFiles\\myPythonFiles\\RiverFlow\\data\\RiverFlow\\processed\\riverflow.csv'
#     riverflow_df = pd.read_csv(riverpath,index_col=0,header=0)
#     amazon = riverflow_df[['0']].loc[195001:201012]#.to_numpy().reshape((-1,))
#     congo = riverflow_df[['1']].loc[195001:201012]#.to_numpy().reshape((-1,)) # from 1950 to 201012
#     years = np.array(list(range(1950,2011)))
#     amazon_annual = np.sum(amazon.to_numpy().reshape((-1,12)),axis=1)
#     congo_annual = np.sum(congo.to_numpy().reshape((-1,12)),axis=1)
#     amazon_annual_10 = amazon_annual[-10:]
#     congo_annual_10 = congo_annual[-10:]
#     years = years[-10:]
#     amazon_mean,amazon_std = np.mean(amazon_annual_10),np.std(amazon_annual_10)
#     amazon_annual_10 = (amazon_annual_10-amazon_mean)/amazon_std
#     congo_mean,congo_std = np.mean(congo_annual_10),np.std(congo_annual_10)
#     congo_annual_10 = (congo_annual_10-congo_mean)/congo_std
    
#     #fig = plt.figure(figsize=(9,4))
#     fig,(ax1,ax2) = plt.subplots(nrows=2,ncols=1,figsize=(9,4),sharex=True)
#     suptitle = 'Standardized Annual Discharge Anomaly and ONI Index'
#     plt.suptitle(suptitle)
#     ax1.plot(years,amazon_annual_10,'g',linewidth=2,marker='o',markersize=5,label='Amazon')
#     ax1.plot(years,congo_annual_10,'--',color='lime',linewidth=2,marker='s',markersize=5,label='Congo')
#     ax1.set_ylabel('Standardized Anomaly')
#     ax1.grid()
#     ax1.legend(loc='upper left')
    
#     w = 1.0/12
#     rects = {}
#     xx = years[0]+np.arange(len(oni))
#     for month in range(12):
#         barcenters = xx+(month-5.5)*w
#         heights = oni[:,month]
#         colors = get_colors(heights)
#         rects[month] = ax2.bar(x=barcenters,height=heights,width=w,label=oni_df.columns[month],color=colors)
#     ax2.axhline(y=0.5,xmin=0,xmax=10,color='r',linestyle='--',linewidth=0.5)
#     ax2.axhline(y=-0.5,xmin=0,xmax=10,color='b',linestyle='--',linewidth=0.5)
#     ax2.set_ylabel('ONI Index')
#     ax2.set_xlabel('Year')
#     ax2.set_xticks(years)
#     ax2.set_xticklabels(years)
#     ax2.grid()
#     plt.show()
    
#     savepath = '../data/'
#     savename = suptitle.lower().replace(' ','_')
#     plt.savefig(savepath+savename+'.png',dpi=1200,bbox='tight')



# def plot_motif():
    # import stumpy
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from matplotlib.patches import Rectangle
    # from mpl_toolkits.basemap import Basemap

    # predictor = 'Reanalysis' # 'GCM' # 
    # winlen = 12
    # thred = 0.1
    # left,right = 50,350
    # top,bottom = 50,130 # region 1
    # #top,bottom = 67,109 #[20.5,-20.5], region 4

    # if predictor=='Reanalysis':
    #     data = np.load('../data/Climate/Reanalysis/sst_cobe_hadley_noaa_uod_195001-200512_1by1_world.npy')[:,0:3,2:-2,:] # 87.5N to -87.5N by 1, 
    # elif predictor=='GCM':
    #     data = np.load('/scratch/wang.zife/YuminLiu/DATA/GCM/GCMdata/tas/processeddata/32gcms_tas_monthly_1by1_195001-200512_World.npy')
    # print('data.shape={}'.format(data.shape))
    # lats = np.load('/scratch/wang.zife/YuminLiu/DATA/GCM/GCMdata/tas/processeddata/lats_gcm.npy')
    # lons = np.load('/scratch/wang.zife/YuminLiu/DATA/GCM/GCMdata/tas/processeddata/lons_gcm.npy')
    # data = data[:,:,top:bottom,left:right] # 50.5E to 349.5E, 37.5N to -41.5N, # [672,32,80,300]

    # cobe = np.load('/scratch/wang.zife/YuminLiu/DATA/COBE/processed/sst.mon.mean_185001-201912_1by1.npy')[:,2:-2,:]
    # mask = np.mean(cobe,axis=0)
    # mask = mask[top:bottom,left:right]
    # mask = np.nan_to_num(mask,nan=1000)
    # for mon in range(len(data)):
    #     for gcm in range(data.shape[1]):
    #         data[mon,gcm,:,:][mask==1000] = 0
    # data = data[:,0,:,:] # 1st GCM
    # print('data.shape={}'.format(data.shape))
    # Nmon,Nlat,Nlon = data.shape
    # motif_map = np.zeros((Nlat,Nlon))


    # # for i in range(Nlat):
    # #     for j in range(Nlon):
    # #         if mask[i][j]==1000: continue
    # #         mp = stumpy.stump(data[:,i,j],winlen)
    # #         #print('mp.shape={}'.format(mp.shape))
    # #         # motif_idx = np.argsort(mp[:, 0])[0]
    # #         # print(f"The motif is located at index {motif_idx}")
    # #         # nearest_neighbor_idx = mp[motif_idx, 1]
    # #         # print(f"The nearest neighbor is located at index {nearest_neighbor_idx}")

    # #         #motif_ids = np.argsort(mp[:, 0])[0:20]
    # #         motif_ids = np.where(mp[:,0]<(1.0+thred)*np.min(mp[:,0]))[0]
    # #         motif_ids_new = []
    # #         for ind in sorted(motif_ids):
    # #             if len(motif_ids_new)==0 or ind-motif_ids_new[-1]>=winlen:
    # #                 motif_ids_new.append(ind)
    # #         #print('motif_ids={}'.format(motif_ids))
    # #         #print('motif_ids_new={}'.format(motif_ids_new))
    # #         motif_ids = np.array(motif_ids_new)
    # #         nearest_neighbor_ids = mp[motif_ids, 1]
    # #         motif_map[i][j] = len(motif_ids)
    # # motif_map[motif_map==0] = np.nan

    # # def plot_map_2(img,title,savepath,savename,verbose):
    # #     lonlat = [50.5,-41.5,349.5,37.5] #
    # #     parallels = np.arange(-40.0,40.0,10.0)
    # #     meridians = np.arange(60.0,350.0,30.0) # label lons, 60E to -9.5W, or 60E to 350E
    # #     watercolor = 'white' # '#46bcec'
    # #     cmap = 'viridis' #'YlGn' #'YlOrRd' # 'rainbow' # 'Accent' #'hsv' #'seismic' # 
    # #     alpha = 1 # 0.7
    # #     projection = 'merc' # 'cyl' # 
    # #     resolution = 'i' # 'h'
    # #     area_thresh = 10000

    # #     fig = plt.figure(figsize=(15,4)) 
    # #     m = Basemap(llcrnrlon=lonlat[0],llcrnrlat=lonlat[1],urcrnrlon=lonlat[2],urcrnrlat=lonlat[3],
    # #                 projection=projection,resolution=resolution,area_thresh=area_thresh)
    # #     m.drawcoastlines(linewidth=1.0,color='k')
    # #     m.drawcountries(linewidth=1.0,color='k')
    # #     m.drawstates(linewidth=0.2,color='k')
    # #     #m.fillcontinents(color='w',alpha=0.1)
    # #     m.drawmapboundary(fill_color=watercolor)
    # #     m.fillcontinents(color = 'white',alpha=1.0,lake_color=watercolor)
    # #     m.drawparallels(parallels,labels=[True,False,False,False],dashes=[1,2])
    # #     m.drawmeridians(meridians,labels=[False,False,False,True],dashes=[1,2])
    # #     img = np.flipud(img)
    # #     m.imshow(img,cmap=cmap,alpha=alpha,zorder=1)
    # #     m.colorbar(fraction=0.02)

    # #     plt.title(title)
    # #     if savepath and savename:
    # #         plt.savefig(savepath+savename+'.png',dpi=1200,bbox_inches='tight')
    # #     #plt.show()
    # #     #plt.close()

    # # img = motif_map
    # # title = 'Motif Map for {} Data'.format(predictor)
    # # savepath = '../data/Climate/{}/Motifs/'.format(predictor)
    # # savename = title.lower().replace(' ','_')+'_w{}_m{}'.format(winlen,thred)
    # # verbose = True
    # # plot_map_2(img,title,savepath,savename,verbose)
    # # np.save(savepath+savename+'.npy',motif_map)

    # # winlen = 36
    # # thred = 0.1
    # # gcm_motif = np.load('../data/Climate/GCM/Motifs/motif_map_for_gcm_data_w{}_m0.1.npy'.format(winlen),allow_pickle=True)
    # # reanalysis_motif = np.load('../data/Climate/Reanalysis/Motifs/motif_map_for_reanalysis_data_w{}_m0.1.npy'.format(winlen),allow_pickle=True)
    # # img = gcm_motif-reanalysis_motif

    # # img2 = np.nan_to_num(gcm_motif-reanalysis_motif,0)
    # # P = len(np.where(img2.flatten()>0)[0])
    # # N = len(np.where(img2.flatten()<0)[0])

    # # title = 'Difference of Motif Map for GCM and Reanalysis Data P{} N{}'.format(P,N)
    # # savepath = '../data/Climate/Reanalysis/Motifs/'
    # # savename = title.lower().replace(' ','_')+'_w{}_m{}'.format(winlen,thred)
    # # verbose = True
    # # plot_map_2(img,title,savepath,savename,verbose)
    # # #np.save(savepath+savename+'.npy',motif_map)

    # i,j = Nlat//2,Nlon//2
    # if mask[i][j]==1000: print('ERROR! Invalid location!')
    # mp = stumpy.stump(data[:,i,j],winlen)
    # #print('mp.shape={}'.format(mp.shape))
    # # motif_idx = np.argsort(mp[:, 0])[0]
    # # print(f"The motif is located at index {motif_idx}")
    # # nearest_neighbor_idx = mp[motif_idx, 1]
    # # print(f"The nearest neighbor is located at index {nearest_neighbor_idx}")

    # #motif_ids = np.argsort(mp[:, 0])[0:20]
    # motif_ids = np.where(mp[:,0]<(1.0+thred)*np.min(mp[:,0]))[0]
    # motif_ids_new = []
    # for ind in sorted(motif_ids):
    #     if len(motif_ids_new)==0 or ind-motif_ids_new[-1]>=winlen:
    #         motif_ids_new.append(ind)
    # #print('motif_ids={}'.format(motif_ids))
    # #print('motif_ids_new={}'.format(motif_ids_new))
    # motif_ids = np.array(motif_ids_new)
    # nearest_neighbor_ids = mp[motif_ids, 1]
    # fig, axs = plt.subplots(3,figsize=(12,9))
    # title = 'Motif Discovery for {}'.format(predictor)
    # plt.suptitle(title)
    # axs[0].plot(data[:,i,j])
    # axs[0].set_ylabel('Temperature')
    # axs[1].plot(mp[:, 0])
    # for k in range(len(motif_ids)):
    #     motif_idx,nearest_neighbor_idx = motif_ids[k],nearest_neighbor_ids[k]
    #     rect = Rectangle((motif_idx, 0), winlen, 40, facecolor='C{}'.format(k),alpha=0.1)
    #     axs[0].add_patch(rect)
    #     rect = Rectangle((nearest_neighbor_idx, 0), winlen, 40, facecolor='C{}'.format(k),alpha=0.1)
    #     axs[0].add_patch(rect)

    #     axs[1].axvline(x=motif_idx, linestyle="dashed",c='C{}'.format(k))
    #     axs[1].axvline(x=nearest_neighbor_idx, linestyle="dashed",c='C{}'.format(k))

    #     axs[2].plot(data[motif_idx:motif_idx+winlen,i,j],c='C{}'.format(k))
    #     axs[2].plot(data[nearest_neighbor_idx:nearest_neighbor_idx+winlen,i,j],c='C{}'.format(k))
    # axs[1].set_ylabel('Matrix Profile')
    # axs[2].set_xlabel('Month')
    # axs[2].set_ylabel("Temperature")
    # savepath = '../data/Climate/{}/Motifs/'.format(predictor)
    # savename = title.lower().replace(' ','_')
    # if savepath and savename:
    #     plt.savefig(savepath+savename+'.png',dpi=1200,bbox_inches='tight')
    # plt.show()


#%% calculate dmi-enso histogram with diferent correlation threshold
# def cal_dmi_enso_hist():
#     import numpy as np
#     import matplotlib.pyplot as plt
#     #c_th = 0.5
#     for c_th in [0.6,0.7,0.8,0.9,0.93,0.95]:
#         predictor = 'GCM'
#         #bins,bin_range,density = 50,[80,19950],False
#         bins,bin_range,density = 50,[10000,19950],False
#         datapath = '../data/Climate/{}/Autocorrelation/'.format(predictor)
#         dataname = 'dmi_enso_dis_corr_p_loc1_loc2_lat50-130'
#         savepath = datapath+'Figs/'
#         dis_corr_p = np.load(datapath+dataname+'.npy',allow_pickle=True)


#         dis_total,corr_total,p_total = dis_corr_p[:,0],dis_corr_p[:,1],dis_corr_p[:,2]
#         dis_total_hist,total_bin_edges = np.histogram(dis_total,bins=bins,range=bin_range)
#         # print('dis_total.shape={},corr_total.shape={},p_total.shape={}'.format(dis_total.shape,corr_total.shape,p_total.shape))
#         # print('dis_total.min={},dis_total.max={}'.format(np.min(dis_total),np.max(dis_total)))
#         # print('dis_total_hist[0:5]={},total_bin_edges[0:5]={}'.format(dis_total_hist[0:5],total_bin_edges[0:5]))
#         # print('dis_total_hist[-5:]={},total_bin_edges[-5:]={}'.format(dis_total_hist[-5:],total_bin_edges[-5:]))

#         inds = abs(dis_corr_p[:,1])>c_th
#         print('len(inds)={}'.format(len(inds)))
#         dis,corr,p = dis_corr_p[inds,0],dis_corr_p[inds,1],dis_corr_p[inds,2]
#         dis_hist,bin_edges = np.histogram(dis,bins=bins,range=bin_range)
#         # print('dis.shape={},corr.shape={},p.shape={}'.format(dis.shape,corr.shape,p.shape))
#         # print('dis.min={},dis.max={}'.format(np.min(dis),np.max(dis)))
#         # print('dis_hist[0:5]={},bin_edges[0:5]={}'.format(dis_hist[0:5],bin_edges[0:5]))
#         # print('dis_hist[-5:]={},bin_edges[-5:]={}'.format(dis_hist[-5:],bin_edges[-5:]))

#         dis_hist_normalized = np.divide(dis_hist,dis_total_hist+1e-15)
#         xticks = np.array(bin_edges)
#         xticks = np.array([(xticks[i]+xticks[i+1])//2 for i in range(len(xticks)-1)])
#         fig = plt.figure(figsize=(15,4))
#         xx = range(len(dis_hist_normalized))
#         plt.bar(x=xx,height=dis_hist_normalized)
#         plt.xticks(ticks=xx[::5],labels=xticks[::5])
#         plt.xlabel('Distance (km)')
#         plt.ylabel('Num of Edges / Total Num of Dis. Counts')
#         title = 'Proportional Histogram {} between Indian and Pacific Ocean'.format(predictor)
#         plt.title(title)
#         savename = title.lower().replace(' ','_')+'_c{}'.format(c_th)
#         plt.savefig(savepath+savename+'.png',dpi=1200,bbox_inches='tight')
#         plt.close()

#         fig = plt.figure(figsize=(15,4))
#         plt.hist(dis_total,bins=bins,range=bin_range,density=density)
#         plt.xlabel('Distance (km)')
#         plt.ylabel('Total Num of Dis. Counts')
#         title = 'Distance Histogram {} between Indian and Pacific Ocean'.format(predictor)
#         plt.title(title)
#         savename = title.lower().replace(' ','_')+'_c{}'.format(c_th)
#         plt.savefig(savepath+savename+'.png',dpi=1200,bbox_inches='tight')
#         plt.close()

#         fig = plt.figure(figsize=(15,4))
#         plt.hist(dis,bins=bins,range=bin_range,density=density)
#         plt.xlabel('Distance (km)')
#         plt.ylabel('Num of Edges')
#         title = 'Edges Histogram {} between Indian and Pacific Ocean'.format(predictor)
#         plt.title(title)
#         savename = title.lower().replace(' ','_')+'_c{}'.format(c_th)
#         plt.savefig(savepath+savename+'.png',dpi=1200,bbox_inches='tight')
#         plt.close()





# def get_valid_dis_corr_p_loc1_loc2():
#     import numpy as np
#     import pandas as pd
#     datapath = '../data/Climate/GCM/Autocorrelation/'
#     dataname = 'dis_corr_p_loc1_loc2_lat50-130'
#     savename = dataname+'_valid'
#     data = np.load(datapath+dataname+'.npy',allow_pickle=True)
#     valid_inds = np.array([i for i in range(len(data)) if type(data[i,0])==float])
#     data = data[valid_inds,:]
#     np.save(datapath+savename+'.npy',data)
#     data_df = pd.DataFrame(data=data,columns=['dis','corr','p','lat1','lon1','lat2','lon2'])
#     data_df.to_csv(datapath+savename+'.csv',index=False)



#%% calculate teleconnection between India Ocean and ENSO area
def cal_dmi_enso_teleconnections(predictor,threshold):
    import numpy as np
    import pandas as pd
    import matplotlib
    #matplotlib.use('Agg')
    #matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    from matplotlib.patches import Polygon

    def plot_map_2(img,title,savepath,savename,verbose,lon1,lat1,lon2,lat2,cbar_label='Number of degrees'):
        lonlat = [50.5,-41.5,349.5,37.5] #
        parallels = np.arange(-40.0,40.0,10.0)
        meridians = np.arange(60.0,350.0,30.0) # label lons, 60E to -9.5W, or 60E to 350E
        watercolor = 'white' # '#46bcec'
        cmap = 'viridis' #'YlGn' #'YlOrRd' # 'rainbow' # 'Accent' #'hsv' #'seismic' # 
        alpha = 1 # 0.7
        projection = 'merc' # 'cyl' # 
        resolution = 'i' # 'h'
        area_thresh = 10000

        fig = plt.figure(figsize=(15,4)) 
        m = Basemap(llcrnrlon=lonlat[0],llcrnrlat=lonlat[1],urcrnrlon=lonlat[2],urcrnrlat=lonlat[3],
                    projection=projection,resolution=resolution,area_thresh=area_thresh)
        m.drawcoastlines(linewidth=1.0,color='k')
        #m.drawcountries(linewidth=1.0,color='k')
        #m.drawstates(linewidth=0.2,color='k')
        #m.fillcontinents(color='w',alpha=0.1)
        m.drawmapboundary(fill_color=watercolor)
        #m.fillcontinents(color = 'white',alpha=1.0,lake_color=watercolor)
        m.fillcontinents(color='gray',alpha=0.3,lake_color=watercolor)
        m.drawparallels(parallels,labels=[True,False,False,False],dashes=[1,2])
        m.drawmeridians(meridians,labels=[False,False,False,True],dashes=[1,2])
        img = np.flipud(img)
        m.imshow(img,cmap=cmap,alpha=alpha,zorder=1)
        #m.colorbar(fraction=0.02)
        cbar = m.colorbar(fraction=0.015,location='bottom',extend='both',pad=0.3)
        cbar.formatter.set_powerlimits((0, 0))
        if cbar_label: cbar.set_label(label=cbar_label)
        

        # def draw_rectangle(lats, lons, m):
        #     lons, lats = m(lons, lats)
        #     for i in range(-1,len(lats)-1):
        #         m.plot([lons[i],lons[i+1]],[lats[i],lats[i+1]], 'r-', linewidth=3,alpha=1)

        # pos_lons = [60,60,90,90] # [indian['left'],indian['left'],indian['right'],indian['right']]
        # pos_lats =  [-10,10,10,-10] # [indian['bottom'],indian['top'],indian['top'],indian['bottom']]
        # draw_rectangle(pos_lats, pos_lons, m)
        # pos_lons = [180,180,280,280] # [ENSO['left'],ENSO['left'],ENSO['right'],ENSO['right']]
        # pos_lats = [-10,10,10,-10]
        # draw_rectangle(pos_lats, pos_lons, m)
        # #lonss, latss = m(indian['top'], indian['bottom'])# convert lat and lon to map projection coordinates
        # #m.plot(lonss, latss, 'r-', linewidth=1,alpha=0.01) 

        #lon1,lat1,lon2,lat2 = kwargs['lon1'],kwargs['lat1'],kwargs['lon2'],kwargs['lat2']
        print('len(lat1)={}'.format(len(lat1)))
        for i in range(len(lat1)):
            #print('drawing {}-th teleconnection'.format(i+1))
            pos_lons = [lon1[i],lon2[i]]
            pos_lats = [lat1[i],lat2[i]]
            lonss, latss = m(pos_lons, pos_lats)# convert lat and lon to map projection coordinates
            m.plot(lonss, latss, 'r-', linewidth=1,alpha=0.01) 
        plt.title(title)
        if savepath and savename:
            plt.savefig(savepath+savename+'.png',dpi=1200,bbox_inches='tight')
        #plt.show()
        plt.close()

    print('loading data...')
    #predictor = 'GCM' # 'Reanalysis' # 
    # c_th = 0.5
    # c_th2 = 0.9
    d_th = 15000 # 0 # 19000 # 
    width = 300

    if predictor=='Reanalysis':
        #c_th,c_th2 = 0.5,0.5 # 
        #c_th,c_th2 = 0.9,0.9
        #c_th,c_th2 = 0.5,0.6
        c_th = c_th2 = threshold # 0.6
        top = 52
        bottom = 132
        left,right = 50,350
        filepath = '../data/Climate/Reanalysis/Autocorrelation/'
        filename = 'dis_corr_p_loc1_loc2_lat{}-{}_valid'.format(top,bottom)
        lats = np.load('../data/Climate/Reanalysis/lats.npy')
        lons = np.load('../data/Climate/Reanalysis/lons.npy')
        #lons = np.array([lons[i] if lons[i]<=180 else lons[i]-360 for i in range(len(lons))])
        lats = lats[top:bottom]
        lons = lons[left:right]
        # indian = {'top':18,'bottom':58,'left':10,'right':50} # {'top':10,'bottom':-10,'left':60,'right':90} # 
        # ENSO = {'top':18,'bottom':58,'left':160,'right':230} # {'top':10,'bottom':-10,'left':180,'right':280} # 
    elif predictor=='GCM':
        #c_th = 0.8 # 0.9
        #c_th2 = 0.9 # 0 #0.8 # 0.93
        c_th = c_th2 = threshold # 0.6
        top = 50
        bottom = 130
        left,right = 50,350
        filepath = '../data/Climate/GCM/Autocorrelation/'
        filename = 'dis_corr_p_loc1_loc2_lat{}-{}_valid'.format(top,bottom)
        lats = np.load('/scratch/wang.zife/YuminLiu/DATA/GCM/GCMdata/tas/processeddata/lats_gcm.npy')
        lons = np.load('/scratch/wang.zife/YuminLiu/DATA/GCM/GCMdata/tas/processeddata/lons_gcm.npy')
        lats = lats[top:bottom]
        lons = lons[left:right]
        # indian = {'top':18,'bottom':58,'left':10,'right':50} # {'top':10,'bottom':-10,'left':60,'right':90} # 
        # ENSO = {'top':18,'bottom':58,'left':160,'right':230} # {'top':10,'bottom':-10,'left':180,'right':280} # 

    indian = {'top':28,'bottom':48,'left':10,'right':50} # {'top':10,'bottom':-10,'left':60,'right':90} # 
    ENSO = {'top':28,'bottom':48,'left':160,'right':230} # {'top':10,'bottom':-10,'left':180,'right':280} # 
    savepath = None # filepath+'Figs/'
    dis_corr_p = np.load(filepath+filename+'.npy',allow_pickle=True)

    ## get correlation map
    print('getting correlation map...')
    inds = abs(dis_corr_p[:,1])>c_th
    corr = dis_corr_p[inds,1]
    latlons_inds = dis_corr_p[inds,3:].astype(int)
    corr_map = np.zeros((bottom-top,width))
    corr_map_weighted = np.zeros((bottom-top,width))
    for i in range(len(latlons_inds)):
        la1,lo1,la2,lo2 = latlons_inds[i,:]
        corr_map[la1,lo1] += 1
        corr_map[la2,lo2] += 1
        corr_map_weighted[la1,lo1] += abs(corr[i])
        corr_map_weighted[la2,lo2] += abs(corr[i])
    corr_map[corr_map==0] = np.nan 
    corr_map_weighted[corr_map_weighted==0] = np.nan
    # savepathname = filepath+'corr_map_weighted_corr_map_lat{}-{}_c{}.npz'.format(top,bottom,c_th)
    # np.savez(savepathname,corr_map=corr_map,corr_map_weighted=corr_map_weighted)

    # data = np.load('../data/Climate/Reanalysis/Autocorrelation/corr_map_weighted_corr_map_lat52-132_c0.5.npz')
    # corr_map,corr_map_weighted = data['corr_map'],data['corr_map_weighted']

    ## get teleconnection locations
    print('getting teleconnections...')
    lat1,lon1,lat2,lon2 = [],[],[],[]

    condition0 = (dis_corr_p[:,0]>d_th) & (abs(dis_corr_p[:,1])>c_th2)

    indian_lat1 = (indian['top']<=dis_corr_p[:,3]) & (dis_corr_p[:,3]<indian['bottom'])
    indian_lon1 = (indian['left']<=dis_corr_p[:,4]) & (dis_corr_p[:,4]<indian['right'])
    ENSO_lat1 = (ENSO['top']<=dis_corr_p[:,5]) & (dis_corr_p[:,5]<ENSO['bottom'])
    ENSO_lon1 = (ENSO['left']<=dis_corr_p[:,6]) & (dis_corr_p[:,6]<ENSO['right'])
    condition1 = indian_lat1 & indian_lon1 & ENSO_lat1 & ENSO_lon1
    #del indian_lat1, indian_lon1, ENSO_lat1, ENSO_lon1
    ENSO_lat2 = (ENSO['top']<=dis_corr_p[:,3]) & (dis_corr_p[:,3]<ENSO['bottom'])
    ENSO_lon2 = (ENSO['left']<=dis_corr_p[:,4]) & (dis_corr_p[:,4]<ENSO['right'])
    indian_lat2 = (indian['top']<=dis_corr_p[:,5]) & (dis_corr_p[:,5]<indian['bottom'])
    indian_lon2 = (indian['left']<=dis_corr_p[:,6]) & (dis_corr_p[:,6]<indian['right'])
    condition2 = indian_lat2 & indian_lon2 & ENSO_lat2 & ENSO_lon2 
    #del indian_lat2, indian_lon2, ENSO_lat2, ENSO_lon2 

    #inds2 = condition0 & (condition1 | condition2)
    #inds2 = condition1 | condition2
    inds2 = condition0
    print('len(inds2)={}'.format(len(inds2)))

    #dmi_enso_dis_corr_p = dis_corr_p[inds2,:]
    #np.save('../data/Climate/{}/Autocorrelation/dmi_enso_dis_corr_p_loc1_loc2_lat{}-{}.npy'.format(predictor,top,bottom),dmi_enso_dis_corr_p)

    if len(inds2)>0:
        latlons_inds2 = dis_corr_p[inds2,3:].astype(int)
        lat1 = np.array([lats[latlons_inds2[i,0]] for i in range(len(latlons_inds2))])
        lon1 = np.array([lons[latlons_inds2[i,1]] for i in range(len(latlons_inds2))])
        lat2 = np.array([lats[latlons_inds2[i,2]] for i in range(len(latlons_inds2))])
        lon2 = np.array([lons[latlons_inds2[i,3]] for i in range(len(latlons_inds2))])
    for i in range(len(lon1)):
        if lon1[i]<0: lon1[i] += 360
        if lon2[i]<0: lon2[i] += 360

    # loc1_loc2 = np.stack((lat1,lon1,lat2,lon2),axis=1)
    # np.save('../data/Climate/{}/Autocorrelation/dmi_enso_loc1_loc2_c2{}_d{}.npy'.format(predictor,c_th2,d_th),loc1_loc2)

    # data = np.load('../data/Climate/Reanalysis/Autocorrelation/dmi_enso_loc1_loc2_c20.5_d15000.npy')
    # lat1,lon1,lat2,lon2 = data[:,0],data[:,1],data[:,2],data[:,3]

    ## plot map
    print('plotting degree maps...')
    verbose = True
    img = corr_map
    title = 'Degree Map of {} with Teleconnections'.format(predictor)
    savepath = '../data/Climate/{}/Autocorrelation/Figs2/'.format(predictor) # None # 
    savename = title.lower().replace(' ','_')+'_lat{}-{}_c{}_c2{}_d{}'.format(top,bottom,c_th,c_th2,d_th)
    plot_map_2(img,title,savepath,savename,verbose,lon1,lat1,lon2,lat2)
    # img = corr_map_weighted
    # title = 'Weighted Degree Map of {} with Teleconnections'.format(predictor)
    # #savepath = '../data/Climate/{}/Autocorrelation/Figs/'.format(predictor)
    # savename = title.lower().replace(' ','_')+'_lat{}-{}_c{}_c2{}_d{}'.format(top,bottom,c_th,c_th2,d_th)
    # plot_map_2(img,title,savepath,savename,verbose,lon1,lat1,lon2,lat2)

# for predictor in ['GCM']: # ['Reanalysis','GCM']:
#     for threshold in [0.6]: # [0.6,0.9,0.8]:
#         print('processing {}-{}'.format(predictor,threshold))
#         cal_dmi_enso_teleconnections(predictor,threshold)


#%% get proportional histogram of edges
# def get_predictor_prop_hist():
#     import numpy as np
#     import pandas as pd
#     import matplotlib
#     matplotlib.use('Agg')
#     #matplotlib.use('Qt5Agg')
#     import matplotlib.pyplot as plt
#     from mpl_toolkits.basemap import Basemap

#     predictor = 'GCM' # 'Reanalysis' # 
#     # c_th = 0.5
#     # c_th2 = 0.9
#     d_th = 15000 # 19000 # 
#     width = 300
#     bins,bin_range,density = 50,[80,19950],False
#     if predictor=='Reanalysis':
#         #c_th,c_th2 = 0.5,0.5 # 
#         c_th,c_th2 = 0.9,0.9
#         #c_th,c_th2 = 0.5,0.6
#         top = 52
#         bottom = 132
#         left,right = 50,350
#         filepath = '../data/Climate/Reanalysis/Autocorrelation/'
#         filename = 'dis_corr_p_loc1_loc2_lat{}-{}'.format(top,bottom)
#         lats = np.load('../data/Climate/Reanalysis/lats.npy')
#         lons = np.load('../data/Climate/Reanalysis/lons.npy')
#         #lons = np.array([lons[i] if lons[i]<=180 else lons[i]-360 for i in range(len(lons))])
#         lats = lats[top:bottom]
#         lons = lons[left:right]
#     elif predictor=='GCM':
#         c_th = 0.9
#         c_th2 = 0.93
#         top = 50
#         bottom = 130
#         left,right = 50,350
#         filepath = '../data/Climate/GCM/Autocorrelation/'
#         filename = 'dis_corr_p_loc1_loc2_lat{}-{}'.format(top,bottom)
#         lats = np.load('/scratch/wang.zife/YuminLiu/DATA/GCM/GCMdata/tas/processeddata/lats_gcm.npy')
#         lons = np.load('/scratch/wang.zife/YuminLiu/DATA/GCM/GCMdata/tas/processeddata/lons_gcm.npy')
#         lats = lats[top:bottom]
#         lons = lons[left:right]
#     savepath = filepath+'Figs/'
#     dis_corr_p = np.load(filepath+filename+'.npy',allow_pickle=True)
#     #dis_corr_p = dis_corr_p[:10000]
#     inds = np.array([i for i in range(len(dis_corr_p)) if type(dis_corr_p[i,0])==float and abs(dis_corr_p[i,1])>c_th])
#     corr = dis_corr_p[inds,1]
#     latlons_inds = dis_corr_p[inds,3:].astype(int)
#     ## get correlation map
#     corr_map = np.zeros((bottom-top,width))
#     corr_map_weighted = np.zeros((bottom-top,width))
#     for i in range(len(latlons_inds)):
#         la1,lo1,la2,lo2 = latlons_inds[i,:]
#         corr_map[la1,lo1] += 1
#         corr_map[la2,lo2] += 1
#         corr_map_weighted[la1,lo1] += abs(corr[i])
#         corr_map_weighted[la2,lo2] += abs(corr[i])
#     corr_map[corr_map==0] = np.nan 
#     corr_map_weighted[corr_map_weighted==0] = np.nan
#     ## get teleconnection locations
#     lon1,lat1,lon2,lat2 = [],[],[],[]
#     inds2 = np.array([i for i in range(len(dis_corr_p)) if type(dis_corr_p[i,0])==float and dis_corr_p[i,0]>d_th and abs(dis_corr_p[i,1])>c_th2])
#     if len(inds2)>0:
#         latlons_inds2 = dis_corr_p[inds2,3:].astype(int)
#         lon1 = np.array([lons[latlons_inds2[i,1]] for i in range(len(latlons_inds2))])
#         lat1 = np.array([lats[latlons_inds2[i,0]] for i in range(len(latlons_inds2))])
#         lon2 = np.array([lons[latlons_inds2[i,3]] for i in range(len(latlons_inds2))])
#         lat2 = np.array([lats[latlons_inds2[i,2]] for i in range(len(latlons_inds2))])
#     for i in range(len(lon1)):
#         if lon1[i]<0: lon1[i] += 360
#         if lon2[i]<0: lon2[i] += 360
#     ## plot map
#     lonlat = [50.5,-41.5,349.5,37.5] #
#     parallels = np.arange(-40.0,40.0,10.0)
#     meridians = np.arange(60.0,350.0,30.0) # label lons, 60E to -9.5W, or 60E to 350E
#     watercolor = 'white' # '#46bcec'
#     cmap = 'viridis' #'YlGn' #'YlOrRd' # 'rainbow' # 'Accent' #'hsv' #'seismic' # 
#     alpha = 1 # 0.7
#     projection = 'merc' # 'cyl' # 
#     resolution = 'i' # 'h'
#     area_thresh = 10000

#     img = corr_map
#     img = np.flipud(img)
#     fig = plt.figure(figsize=(15,4)) 
#     m = Basemap(llcrnrlon=lonlat[0],llcrnrlat=lonlat[1],urcrnrlon=lonlat[2],urcrnrlat=lonlat[3],
#                 projection=projection,resolution=resolution,area_thresh=area_thresh)
#     m.drawcoastlines(linewidth=1.0,color='k')
#     m.drawcountries(linewidth=1.0,color='k')
#     m.drawstates(linewidth=0.2,color='k')
#     #m.fillcontinents(color='w',alpha=0.1)
#     m.drawmapboundary(fill_color=watercolor)
#     m.fillcontinents(color = 'white',alpha=1.0,lake_color=watercolor)
#     m.drawparallels(parallels,labels=[True,False,False,False],dashes=[1,2])
#     m.drawmeridians(meridians,labels=[False,False,False,True],dashes=[1,2])
#     m.imshow(img,cmap=cmap,alpha=alpha,zorder=1)
#     m.colorbar(fraction=0.02)
#     for i in range(len(lat1)):
#         pos_lons = [lon1[i],lon2[i]]
#         pos_lats = [lat1[i],lat2[i]]
#         lonss, latss = m(pos_lons, pos_lats)# convert lat and lon to map projection coordinates
#         m.plot(lonss, latss, 'r-', linewidth=1,alpha=0.01) 
#     title = 'Degree Map of {}'.format(predictor)
#     plt.title(title)
#     savename = title.lower().replace(' ','_')+'_lat{}-{}_c{}_c2{}_d{}'.format(top,bottom,c_th,c_th2,d_th)
#     plt.savefig('../data/Climate/{}/Autocorrelation/Figs/{}.png'.format(predictor,savename),dpi=1200,bbox_inches='tight')
#     #plt.show()
#     plt.close()

#     img = corr_map_weighted
#     img = np.flipud(img)
#     fig = plt.figure(figsize=(15,4)) 
#     m = Basemap(llcrnrlon=lonlat[0],llcrnrlat=lonlat[1],urcrnrlon=lonlat[2],urcrnrlat=lonlat[3],
#                 projection=projection,resolution=resolution,area_thresh=area_thresh)
#     m.drawcoastlines(linewidth=1.0,color='k')
#     m.drawcountries(linewidth=1.0,color='k')
#     m.drawstates(linewidth=0.2,color='k')
#     #m.fillcontinents(color='w',alpha=0.1)
#     m.drawmapboundary(fill_color=watercolor)
#     m.fillcontinents(color = 'white',alpha=1.0,lake_color=watercolor)
#     m.drawparallels(parallels,labels=[True,False,False,False],dashes=[1,2])
#     m.drawmeridians(meridians,labels=[False,False,False,True],dashes=[1,2])
#     m.imshow(img,cmap=cmap,alpha=alpha,zorder=1)
#     m.colorbar(fraction=0.02)
#     for i in range(len(lat1)):
#         pos_lons = [lon1[i],lon2[i]]
#         pos_lats = [lat1[i],lat2[i]]
#         lonss, latss = m(pos_lons, pos_lats)# convert lat and lon to map projection coordinates
#         m.plot(lonss, latss, 'r-', linewidth=1,alpha=0.01) 
#     title = 'Weighted Degree Map of {}'.format(predictor)
#     plt.title(title)
#     savename = title.lower().replace(' ','_')+'_lat{}-{}_c{}_c2{}_d{}'.format(top,bottom,c_th,c_th2,d_th)
#     plt.savefig('../data/Climate/{}/Autocorrelation/Figs/{}.png'.format(predictor,savename),dpi=1200,bbox_inches='tight')
#     #plt.show()
#     #plt.close()

#     #%%
#     # corr_map = np.zeros((bottom-top,width))
#     # corr_map_weighted = np.zeros((bottom-top,width))
#     # for i in range(len(latlons_inds)):
#     #     lat1,lon1,lat2,lon2 = latlons_inds[i,:]
#     #     corr_map[lat1,lon1] += 1
#     #     corr_map[lat2,lon2] += 1
#     #     corr_map_weighted[lat1,lon1] += abs(corr[i])
#     #     corr_map_weighted[lat2,lon2] += abs(corr[i])

#     # corr_map[corr_map==0] = np.nan 
#     # corr_map_weighted[corr_map_weighted==0] = np.nan

#     # fig = plt.figure(figsize=(12,5))
#     # plt.imshow(corr_map)
#     # plt.colorbar(fraction=0.01)
#     # plt.xticks([])
#     # plt.yticks([])
#     # plt.xlabel('Longitude')
#     # plt.ylabel('Latitude')
#     # title = 'Degree Map of {}'.format(predictor)
#     # plt.title(title)
#     # savename = title.lower().replace(' ','_')+'_lat{}-{}_c{}'.format(top,bottom,c_th)
#     # plt.savefig('../data/Climate/{}/Autocorrelation/Figs/{}.png'.format(predictor,savename),dpi=1200,bbox_inches='tight')
#     # plt.close()

#     # fig = plt.figure(figsize=(12,5))
#     # plt.imshow(corr_map_weighted)
#     # plt.colorbar(fraction=0.01)
#     # plt.xticks([])
#     # plt.yticks([])
#     # plt.xlabel('Longitude')
#     # plt.ylabel('Latitude')
#     # title = 'Weighted Degree Map of {}'.format(predictor)
#     # plt.title(title)
#     # savename = title.lower().replace(' ','_')+'_lat{}-{}_c{}'.format(top,bottom,c_th)
#     # plt.savefig('../data/Climate/{}/Autocorrelation/Figs/{}.png'.format(predictor,savename),dpi=1200,bbox_inches='tight')
#     # plt.close()


#     #%%
#     # total_inds = np.array([i for i in range(len(dis_corr_p)) if type(dis_corr_p[i,0])==float])
#     # dis_total,corr_total,p_total = dis_corr_p[total_inds,0],dis_corr_p[total_inds,1],dis_corr_p[total_inds,2]
#     # dis_total_hist,total_bin_edges = np.histogram(dis_total,bins=bins,range=bin_range)
#     # print('dis_total.shape={},corr_total.shape={},p_total.shape={}'.format(dis_total.shape,corr_total.shape,p_total.shape))
#     # print('dis_total.min={},dis_total.max={}'.format(np.min(dis_total),np.max(dis_total)))
#     # print('dis_total_hist[0:5]={},total_bin_edges[0:5]={}'.format(dis_total_hist[0:5],total_bin_edges[0:5]))
#     # print('dis_total_hist[-5:]={},total_bin_edges[-5:]={}'.format(dis_total_hist[-5:],total_bin_edges[-5:]))

#     # #inds = np.array([i for i in range(len(dis_corr_p)) if type(dis_corr_p[i,0])==float and abs(dis_corr_p[i,1])>c_th])
#     # dis,corr,p = dis_corr_p[inds,0],dis_corr_p[inds,1],dis_corr_p[inds,2]
#     # dis_hist,bin_edges = np.histogram(dis,bins=bins,range=bin_range)
#     # print('dis.shape={},corr.shape={},p.shape={}'.format(dis.shape,corr.shape,p.shape))
#     # print('dis.min={},dis.max={}'.format(np.min(dis),np.max(dis)))
#     # print('dis_hist[0:5]={},bin_edges[0:5]={}'.format(dis_hist[0:5],bin_edges[0:5]))
#     # print('dis_hist[-5:]={},bin_edges[-5:]={}'.format(dis_hist[-5:],bin_edges[-5:]))

#     # dis_hist_normalized = np.divide(dis_hist,dis_total_hist+1e-15)
#     # xticks = np.array(bin_edges)
#     # xticks = np.array([(xticks[i]+xticks[i+1])//2 for i in range(len(xticks)-1)])
#     # fig = plt.figure(figsize=(12,5))
#     # xx = range(len(dis_hist_normalized))
#     # plt.bar(x=xx,height=dis_hist_normalized)
#     # plt.xticks(ticks=xx[::5],labels=xticks[::5])
#     # plt.xlabel('Distance (km)')
#     # plt.ylabel('Num of Edges / Total Num of Dis. Counts')
#     # title = 'Proportional Histogram {}'.format(predictor)
#     # plt.title(title)
#     # savename = title.lower().replace(' ','_')+'_lat{}-{}_c{}'.format(top,bottom,c_th)
#     # plt.savefig(savepath+savename+'.png',dpi=1200,bbox_inches='tight')
#     # #plt.close()


#     # fig = plt.figure(figsize=(12,5))
#     # plt.hist(dis_total,bins=bins,range=bin_range,density=density)
#     # plt.xlabel('Distance (km)')
#     # plt.ylabel('Total Num of Dis. Counts')
#     # title = 'Distance Histogram {}'.format(predictor)
#     # plt.title(title)
#     # savename = title.lower().replace(' ','_')+'_lat{}-{}_c{}'.format(top,bottom,c_th)
#     # plt.savefig(savepath+savename+'.png',dpi=1200,bbox_inches='tight')
#     # #plt.close()

#     # fig = plt.figure(figsize=(12,5))
#     # plt.hist(dis,bins=bins,range=bin_range,density=density)
#     # plt.xlabel('Distance (km)')
#     # plt.ylabel('Num of Edges')
#     # title = 'Edges Histogram {}'.format(predictor)
#     # plt.title(title)
#     # savename = title.lower().replace(' ','_')+'_lat{}-{}_c{}'.format(top,bottom,c_th)
#     # plt.savefig(savepath+savename+'.png',dpi=1200,bbox_inches='tight')
#     # #plt.close()



#%% Calculate Correlation between Predictor and Predictand
# import time
# import numpy as np
# from scipy import stats
# from test import get_X_y
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# #plt.switch_backend('Agg')

# start_time = time.time()
# #var = 'SST' # 'GCM'
# predictor = 'GCM' # 'Reanalysis' # 
# predictand = 'SingleGCMNino34' # 'Nino34' # 'GCMNino34' # 'Nino34_anom' # 
# #%% calculate
# X,y,lats,lons = get_X_y(predictor=predictor,predictand=predictand)
# #X = X[:,0:2,:,:]
# y = y.reshape((-1,))
# print('X.shape={},y.shape={}'.format(X.shape,y.shape))

# Nmon,Ngcm,Nlat,Nlon = X.shape
# corr_p = np.zeros((Ngcm,Nlat,Nlon,2))
# for i in range(Nlat):
#     for j in range(Nlon):
#         for gcm in range(Ngcm):
#             corr_p[gcm,i,j] = stats.pearsonr(X[:,gcm,i,j],y)
#         #break
# np.save('../data/Climate/{}/Correlation/corr_p_{}_{}.npy'.format(predictor,predictand,predictor),corr_p)
# # corr = np.load('../data/Climate/GCM/Correlation/corr_p_{}_{}.npy'.format(predictand,predictor))
# # Ngcm,Nlat,Nlon,_ = corr.shape


# fig = plt.figure(figsize=(12,5))
# plt.plot(y,label='{}'.format(predictand))
# plt.plot(X[:,1,40,50],label='{} sample at ({},{})'.format(predictor,lats[40],lons[50]))
# plt.plot(X[:,1,40,180],label='{} sample at ({},{})'.format(predictor,lats[40],lons[180]))
# plt.legend()
# title = '{} and {} Time Series Samples'.format(predictor,predictand)
# plt.title(title)
# plt.savefig('../data/Climate/{}/Correlation/Figs/{}.png'.format(predictor,title.lower().replace(' ','_')),dpi=1200,bbox_inches='tight')
# plt.show()



# #%% plot
# gcm_names = np.load('/scratch/wang.zife/YuminLiu/DATA/GCM/GCMdata/tas/processeddata/gcm_names.npy')
# predictor_names = np.array([gcm_names[i][24:-35] for i in range(len(gcm_names))])
# if predictor=='Reanalysis': predictor_names = np.array(['COBE','Hadley','NOAA'])
# for gcm in range(Ngcm):
#     fig = plt.figure(figsize=(12,5))
#     plt.imshow(corr_p[gcm,:,:,0])
#     plt.colorbar(fraction=0.01)
#     plt.xticks([])
#     plt.yticks([])
#     plt.xlabel('Longitude')
#     plt.ylabel('Latitude')
#     plt.title('Pearson Correlation between {} {} and {}'.format(predictor,predictor_names[gcm],predictand))
#     plt.savefig('../data/Climate/{}/Correlation/Figs/corr_{}_{}_ch{}.png'.format(predictor,predictor,predictand,gcm+1),dpi=1200,bbox_inches='tight')
#     plt.close()
# #### average
# fig = plt.figure(figsize=(12,5))
# plt.imshow(np.mean(corr_p[:,:,:,0],axis=0))
# plt.colorbar(fraction=0.01)
# plt.xticks([])
# plt.yticks([])
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.title('Averaged Pearson Correlation between {} and {}'.format(predictor,predictand))
# plt.savefig('../data/Climate/{}/Correlation/Figs/corr_{}_{}_avg.png'.format(predictor,predictor,predictand),dpi=1200,bbox_inches='tight')
# plt.close()
# #### std
# fig = plt.figure(figsize=(12,5))
# plt.imshow(np.std(corr_p[:,:,:,0],axis=0))
# plt.colorbar(fraction=0.01)
# plt.xticks([])
# plt.yticks([])
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.title('std Pearson Correlation between {} and {}'.format(predictor,predictand))
# plt.savefig('../data/Climate/{}/Correlation/Figs/corr_{}_{}_std.png'.format(predictor,predictor,predictand),dpi=1200,bbox_inches='tight')
# plt.close()

# end_time = time.time()
# print('used time: {} minutes'.format((end_time-start_time)/60))


# # a = stats.pearsonr(np.ones((50,)),np.zeros((50,)))
# # print('a={}'.format(a))












#%%
# import numpy as np

# c_th = 0.5
# p_th = 1e-10

# predictor = 'SST' # 'GCM'
# lat,lon = 78,98
# if predictor=='GCM':
#     filepath = '../data/Climate/GCM/SingleGCM/'
# filename = 'dis_corr_p_lat{}-{}'.format(lat,lon)
# dis_corr_p = []
# for ch in range(32):
#     data = np.load(filepath+'GCM{}/'.format(ch+1)+filename+'_gcm{}'.format(ch+1)+'.npy',allow_pickle=True)
#     dis_corr_p.append(data)
# dis_corr_p = np.stack(dis_corr_p,axis=2)
# print('dis_corr_p.shape={}'.format(dis_corr_p.shape))
# savepath = filepath + 'processed/'
# savename = filename+'_avg'

# dis_corr_p[dis_corr_p==None] = np.nan
# dis_corr_p = np.mean(dis_corr_p,axis=2)

# valid_inds = np.array([i for i in range(len(dis_corr_p)) if type(dis_corr_p[i,0])==float and abs(dis_corr_p[i,1])>c_th])

# print('len(valid_inds)={}'.format(len(valid_inds)))
# print('valid_inds[:20]={}'.format(valid_inds[:20]))

# dis,corr,p = dis_corr_p[valid_inds,0],dis_corr_p[valid_inds,1],dis_corr_p[valid_inds,2]

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(12,5))
# plt.hist(dis,bins=50)
# plt.savefig(savepath+savename+'_dis_c0.5.png',dpi=1200,bbox_inches='tight')
# plt.close()

# fig = plt.figure(figsize=(12,5))
# plt.hist(corr,bins=50)
# plt.savefig(savepath+savename+'_corr_c0.5.png',dpi=1200,bbox_inches='tight')
# plt.close()

# fig = plt.figure(figsize=(12,5))
# plt.hist(p,bins=5)
# plt.savefig(savepath+savename+'_p_c0.5.png',dpi=1200,bbox_inches='tight')
# #plt.show()
# plt.close()

# # # from vincenty import vincenty
# # # from geopy import distance
# # # import time
# # # start_time = time.time()
# # # dis = distance.distance((-41.32, 174.81),(40.96, -5.50)).km
# # # print(dis)
# # # for i in range(10):
# # #     #dis1 = distance.distance(loc1,loc2).km
# # #     corr_1,p_1 = stats.pearsonr(time_series1,time_series2)
# # # time_end1 = time.time()
# # # print('geopy use {} minutes'.format((time_end1-start_time)/60))
# # # for i in range(10):
# # #     #dis2 = vincenty(loc1,loc2)
# # #     corr_2,p_2 = stats.pearsonr(time_series1,time_series2)
# # # time_end2 = time.time()
# # # print('vincenty use {} minutes'.format((time_end2-time_end1)/60))








#%%
# def get_dis_corr_p():
#     import numpy as np
#     from vincenty import vincenty
#     from geopy import distance
#     #import geopy
#     from scipy import stats
#     import time
#     import sys
#     # import matplotlib
#     # matplotlib.use('Agg')
#     import matplotlib.pyplot as plt
#     import os

#     #ch = 0
#     var = 'GCM' # 'Reanalysis' # 
#     for ch in [2]: #range(3):
#         if var=='GCM':
#             imgpath = '/scratch/wang.zife/YuminLiu/DATA/GCM/GCMdata/tas/processeddata/32gcms_tas_monthly_1by1_195001-200512_World.npy'
#             left,right = 50,350
#             ## 37.5N to -41.5N
#             top,bottom = 50,130
#             ## 19.5N to -19.5N
#             #top,bottom = 68,108
#             ## 9.5N to -9.5N
#             #top,bottom = 78,98

#             data = np.load(imgpath) # 87.5N to -87.5N by 1,
#             data = data[:,:,top:bottom,left:right] # 50.5E to 349.5E, 37.5N to -41.5N, # [672,32,80,300]
#             data = np.mean(data,axis=1) # [Nmon,Nlat,Nlon]
#             #data = data[:,ch,:,:] # [Nmon,Nlat,Nlon]

#             lats = np.load('/scratch/wang.zife/YuminLiu/DATA/GCM/GCMdata/tas/processeddata/lats_gcm.npy')
#             lons = np.load('/scratch/wang.zife/YuminLiu/DATA/GCM/GCMdata/tas/processeddata/lons_gcm.npy')
#             lons = np.array([lons[i] if lons[i]<=180 else lons[i]-360 for i in range(len(lons))])
#             lats = lats[top:bottom]
#             lons = lons[left:right]

#             cobe = np.load('/scratch/wang.zife/YuminLiu/DATA/COBE/processed/sst.mon.mean_185001-201912_1by1.npy')
#             mask = np.mean(cobe,axis=0)
#             mask = mask[top+2:bottom+2,left:right]
#             mask = np.nan_to_num(mask,nan=1000)



#             # fig = plt.figure(figsize=(12,5))
#             # plt.plot(data[:,10,150])
#             # plt.title('Not De-trended')
#             # plt.savefig('../data/Climate/{}/not_detrended.png'.format(var),dpi=1200,bbox_inches='tight')

#             ## detrend
#             mean_m,std_m = {},{}
#             for i in range(data.shape[1]):
#                 for j in range(data.shape[2]):
#                     data_n = data[:,i,j]
#                     for m in range(12):
#                         mean_m[m] = np.mean(data_n[m::12])
#                         std_m[m] = np.std(data_n[m::12],ddof=1)
#                         data[m::12,i,j] = (data_n[m::12]-mean_m[m])/std_m[m]

#             # fig = plt.figure(figsize=(12,5))
#             # plt.plot(data[:,10,150])
#             # plt.title('De-trended')
#             # plt.savefig('../data/Climate/{}/detrended.png'.format(var),dpi=1200,bbox_inches='tight')


#         elif var=='Reanalysis':
#             pass
#             # imgpath = '../data/Climate/Reanalysis/sst_cobe_hadley_noaa_uod_195001-200512_1by1_world.npy'
#             # left,right = 50,350
#             # ## 37.5N to -41.5N
#             # top,bottom = 52,132 #
#             # ## 19.5N to -19.5N
#             # #top,bottom = 70,110 #
#             # ## 9.5N to -9.5N
#             # #top,bottom = 80,100 #

#             # data = np.load(imgpath) # 89.5N to -89.5N by 1,

#             # #mask = np.sum(data,axis=(0,1))

#             # data = data[:,0:3,top:bottom,left:right] # 50.5E to 349.5E, 37.5N to -41.5N, [672,3,80,300], exclude uod
#             # data = np.mean(data,axis=1) # [Nmon,Nlat,Nlon]
#             # #data = data[:,ch,:,:] # [Nmon,Nlat,Nlon]
            

#             # lats = np.load('../data/Climate/Reanalysis/lats.npy')
#             # lons = np.load('../data/Climate/Reanalysis/lons.npy')
#             # lons = np.array([lons[i] if lons[i]<=180 else lons[i]-360 for i in range(len(lons))])
#             # lats = lats[top:bottom]
#             # lons = lons[left:right]

#             # cobe = np.load('/scratch/wang.zife/YuminLiu/DATA/COBE/processed/sst.mon.mean_185001-201912_1by1.npy')
#             # #lats_mask = np.load('/scratch/wang.zife/YuminLiu/DATA/COBE/processed/lats.npy')[top:bottom]
#             # #lons_mask = np.load('/scratch/wang.zife/YuminLiu/DATA/COBE/processed/lons.npy')[left:right]
#             # mask = np.mean(cobe,axis=0)
#             # mask = mask[top:bottom,left:right]
#             # mask = np.nan_to_num(mask,nan=1000)



#             # # fig = plt.figure(figsize=(12,5))
#             # # plt.plot(data[:,10,150])
#             # # plt.title('Not De-trended')
#             # # plt.savefig('../data/Climate/{}/not_detrended.png'.format(var),dpi=1200,bbox_inches='tight')

#             # ## detrend
#             # mean_m,std_m = {},{}
#             # for i in range(data.shape[1]):
#             #     for j in range(data.shape[2]):
#             #         data_n = data[:,i,j]
#             #         for m in range(12):
#             #             mean_m[m] = np.mean(data_n[m::12])
#             #             std_m[m] = np.std(data_n[m::12],ddof=1)
#             #             data[m::12,i,j] = (data_n[m::12]-mean_m[m])/(std_m[m]+1e-10)

#             # # fig = plt.figure(figsize=(12,5))
#             # # #plt.plot(data[:,10,150])
#             # # #plt.title('De-trended')
#             # # #plt.savefig('../data/Climate/{}/detrended.png'.format(var),dpi=1200,bbox_inches='tight')
#             # # plt.imshow(data[0,:,:])
#             # # plt.show()

#             # # fig = plt.figure(figsize=(12,5))
#             # # plt.imshow(mask)
#             # # plt.show()




#         locations = [(i,j) for i in range(len(lats)) for j in range(len(lons)) if mask[i][j]!=1000]
#         #locations = locations[:24]
#         print('len(locations)={}'.format(len(locations)))
#         #print('locations[:30]={}'.format(locations[:30]))
        
#         dis_corr_p = []
#         start_time = time.time()
#         for i in range(len(locations)):
#             lat1,lon1 = locations[i]
#             print('processing {}-th location'.format(i))
#             for j in range(i+1,len(locations)):
#                 lat2,lon2 = locations[j]
#                 #dis = distance.distance(loc1,loc2).km
#                 #print('lats[lat1]={},lons[lon1]={}'.format(lats[lat1],lons[lon1]))
#                 #print('lats[lat2]={},lons[lon2]={}'.format(lats[lat2],lons[lon2]))
#                 #print('data[:10,lat1,lon1]={}'.format(data[:10,lat1,lon1]))
#                 #print('data[:10,lat2,lon2]={}'.format(data[:10,lat2,lon2]))
#                 dis = vincenty((lats[lat1],lons[lon1]),(lats[lat2],lons[lon2]))
#                 corr,p = stats.pearsonr(data[:,lat1,lon1],data[:,lat2,lon2])
#                 #print('corr={},p={}'.format(corr,p))
#                 dis_corr_p.append((dis,corr,p,lat1,lon1,lat2,lon2))
#                 #if j>=10: break
#             #if i>=2: break
#         print('processes use {} minutes'.format((time.time()-start_time)/60))

#         dis_corr_p = np.stack(dis_corr_p,axis=0)
#         savepath = '../data/Climate/{}/Autocorrelation/'.format(var)
#         #savepath = '../data/Climate/{}/SingleGCM/GCM{}/'.format(var,ch+1)
#         #savepath = '../data/Climate/{}/SingleReanalysis/Reanalysis{}/'.format(var,ch+1)
#         #savepath = '../data/Climate/{}/Not_trended/SingleReanalysis/Reanalysis{}/'.format(var,ch+1)
#         if not os.path.exists(savepath):
#             os.makedirs(savepath)
#         #savename = 'dis_corr_p_lat{}-{}_gcm{}'.format(top,bottom,ch+1)
#         #savename = 'dis_corr_p_lat{}-{}_reanalysis{}'.format(top,bottom,ch+1)
#         savename = 'dis_corr_p_loc1_loc2_lat{}-{}'.format(top,bottom)
#         np.save(savepath+savename+'.npy',dis_corr_p)
#         #np.save('../data/Climate/{}/dis_corr_p_lat{}-{}.npy'.format(var,top,bottom),dis_corr_p)

#         #np.save(savepath+'locations_lat{}-{}.npy'.format(top,bottom),locations)



#         #%%
#         import numpy as np

#         c_th = 0.5
#         p_th = 1e-10

#         filepath = '../data/Climate/SST/'
#         filename = 'dis_corr_p_lat52-132'
#         data = np.load(filepath+filename+'.npy',allow_pickle=True)

#         # filepath = savepath
#         # filename = savename
#         # data = dis_corr_p

#         data = np.load('../data/Climate/SST/Not_trended/SingleReanalysis/Reanalysis3/dis_corr_p_lat80-100_reanalysis3.npy',allow_pickle=True)
#         valid_inds = np.array([i for i in range(len(data)) if type(data[i,0])==float and abs(data[i,1])>c_th])

#         print('len(valid_inds)={}'.format(len(valid_inds)))
#         print('valid_inds[:20]={}'.format(valid_inds[:20]))

#         dis,corr,p = data[valid_inds,0],data[valid_inds,1],data[valid_inds,2]

#         import matplotlib
#         matplotlib.use('Agg')
#         import matplotlib.pyplot as plt
#         fig = plt.figure(figsize=(12,5))
#         plt.hist(dis,bins=50)
#         plt.savefig(filepath+filename+'_dis_c0.5.png',dpi=1200,bbox_inches='tight')
#         plt.close()

#         fig = plt.figure(figsize=(12,5))
#         plt.hist(corr,bins=50)
#         plt.savefig(filepath+filename+'_corr_c0.5.png',dpi=1200,bbox_inches='tight')
#         plt.close()

#         fig = plt.figure(figsize=(12,5))
#         plt.hist(p,bins=5)
#         plt.savefig(filepath+filename+'_p_c0.5.png',dpi=1200,bbox_inches='tight')
#         #plt.show()
#         plt.close()

#         # # from vincenty import vincenty
#         # # from geopy import distance
#         # # import time
#         # # start_time = time.time()
#         # # dis = distance.distance((-41.32, 174.81),(40.96, -5.50)).km
#         # # print(dis)
#         # # for i in range(10):
#         # #     #dis1 = distance.distance(loc1,loc2).km
#         # #     corr_1,p_1 = stats.pearsonr(time_series1,time_series2)
#         # # time_end1 = time.time()
#         # # print('geopy use {} minutes'.format((time_end1-start_time)/60))
#         # # for i in range(10):
#         # #     #dis2 = vincenty(loc1,loc2)
#         # #     corr_2,p_2 = stats.pearsonr(time_series1,time_series2)
#         # # time_end2 = time.time()
#         # # print('vincenty use {} minutes'.format((time_end2-time_end1)/60))




#%%

# def plot_rmses():
#     import os
#     import numpy as np
#     import datasets

#     path = '../results/myCNN/lag_0/SingleGCM/'
#     test_dataset = datasets.myDataset_CNN(fold='test',window=3,noise_std=0.0)
#     ys = []
#     for y in test_dataset:
#         ys.append(y[1].numpy()[0])
#     ys = np.array(ys)

#     train_dataset = datasets.myDataset_CNN(fold='train',window=3,noise_std=0.0)
#     ys_train = []
#     for y in train_dataset:
#         ys_train.append(y[1].numpy()[0])
#     ys_train = np.array(ys_train)

#     #thresholds = [np.percentile(ys_train,q) for q in [5,15,50,85,95]]
#     thresholds = [np.percentile(ys_train,q) for q in [10,25,50,75,90]]

#     month_name_test = []
#     for year in range(2003,2006):
#         for month in range(1,13):
#             if month<10:
#                 month_name_test.append('{}0{}'.format(year,month))
#             else:
#                 month_name_test.append('{}{}'.format(year,month))
#     month_name_test = np.array(month_name_test)
#     preds_mgcm = np.load('../results/myCNN/lag_0/2021-02-17_23.06.48.428983_gcm_masked_amazon_region1/pred_results_RMSE0.2839257769608812.npz')['preds']

#     preds_sgcm = []
#     rmses_sgcm = []
#     for ngcm in range(32):
#         filepath = path+str(ngcm)+'/'
#         for file in os.listdir(filepath):
#             if file.startswith('pred_results_RMSE'):
#                 res = np.load(filepath+file)
#                 preds_sgcm.append(res['preds'])
#                 rmses_sgcm.append(res['rmse'])

#     preds_sgcm = np.stack(preds_sgcm,axis=1)
#     preds_sgcm_mean = np.mean(preds_sgcm,axis=1)
#     preds_sgcm_std = np.std(preds_sgcm,axis=1)

#     rmses_sgcm = np.stack(rmses_sgcm,axis=0)
#     rmses_sgcm_mean = np.mean(rmses_sgcm)


#     ys_extend = np.dot(ys.reshape((36,1)),np.ones((1,32)))
#     rmse_permonth_sgcm = np.sqrt((preds_sgcm-ys_extend)**2)
#     rmse_permonth_sgcm_mean = np.mean(rmse_permonth_sgcm,axis=1)
#     rmse_permonth_sgcm_std = np.std(rmse_permonth_sgcm,axis=1)

#     rmse_permonth_mgcm = np.sqrt((preds_mgcm-ys)**2)

#     # rmse_percentile10 = np.stack([rmse_permonth_sgcm[i,:] for i in range(len(ys)) if ys[i]<=thresholds[0]],axis=0)
#     # rmse_percentile25 = np.stack([rmse_permonth_sgcm[i,:] for i in range(len(ys)) if thresholds[0]<ys[i]<=thresholds[1]],axis=0)
#     # rmse_percentile50 = np.stack([rmse_permonth_sgcm[i,:] for i in range(len(ys)) if thresholds[1]<=ys[i]<thresholds[2]],axis=0)
#     # rmse_percentile75 = np.stack([rmse_permonth_sgcm[i,:] for i in range(len(ys)) if thresholds[2]<=ys[i]<thresholds[3]],axis=0)
#     # rmse_percentile90 = np.stack([rmse_permonth_sgcm[i,:] for i in range(len(ys)) if thresholds[3]<=ys[i]<thresholds[4]],axis=0)
#     # rmse_percentile100 = np.stack([rmse_permonth_sgcm[i,:] for i in range(len(ys)) if thresholds[4]<=ys[i]],axis=0)

#     # rmse_percentile10_mean = np.mean(rmse_percentile10)
#     # rmse_percentile10_std = np.mean(np.std(rmse_percentile10,axis=1))
#     # rmse_percentile25_mean = np.mean(rmse_percentile25)
#     # rmse_percentile25_std = np.mean(np.std(rmse_percentile25,axis=1))
#     # rmse_percentile50_mean = np.mean(rmse_percentile50)
#     # rmse_percentile50_std = np.mean(np.std(rmse_percentile50,axis=1))
#     # rmse_percentile75_mean = np.mean(rmse_percentile75)
#     # rmse_percentile75_std = np.mean(np.std(rmse_percentile75,axis=1))
#     # rmse_percentile90_mean = np.mean(rmse_percentile90)
#     # rmse_percentile90_std = np.mean(np.std(rmse_percentile90,axis=1))
#     # rmse_percentile100_mean = np.mean(rmse_percentile100)
#     # rmse_percentile100_std = np.mean(np.std(rmse_percentile100,axis=1))

#     import matplotlib
#     matplotlib.use('Agg')
#     import matplotlib.pyplot as plt

#     # fig = plt.figure()
#     # xx = np.array(['$0-10$','$10-25$','$25-50$','$50-75$','$75-90$','$90-100$'])
#     # yy = np.array([rmse_percentile10_mean,rmse_percentile25_mean,rmse_percentile50_mean,rmse_percentile75_mean,rmse_percentile90_mean,rmse_percentile100_mean])
#     # yy_std = np.array([rmse_percentile10_std,rmse_percentile25_std,rmse_percentile50_std,rmse_percentile75_std,rmse_percentile90_std,rmse_percentile100_std])
#     # yerr = yy_std
#     # #plt.errorbar(xx, yy, yerr=yerr, fmt='o')
#     # plt.bar(xx, height=yy, yerr=yerr, capsize=6)
#     # #plt.grid()
#     # plt.xlabel('Percentile')
#     # plt.ylabel('RMSE and std')
#     # plt.title('RMSE for Different Percentiles')
#     # plt.show()
#     # savepath = '../results/myCNN/lag_0/SingleGCM/postprocessed/'
#     # savename = 'percentile_rmse_2'
#     # plt.savefig(savepath+savename+'.png',dpi=1200,bbox_inches='tight')

# def double_end_percentile():
#     rmse_percentile5 = np.stack([rmse_permonth_sgcm[i,:] for i in range(len(ys)) if ys[i]<=thresholds[0] or ys[i]>=thresholds[-1]],axis=0)
#     rmse_percentile25 = np.stack([rmse_permonth_sgcm[i,:] for i in range(len(ys)) if thresholds[0]<ys[i]<=thresholds[1] or thresholds[-2]<=ys[i]<thresholds[-1]],axis=0)
#     rmse_percentile50 = np.stack([rmse_permonth_sgcm[i,:] for i in range(len(ys)) if thresholds[1]<=ys[i]<thresholds[-2]],axis=0)

#     rmse_percentile5_mean = np.mean(rmse_percentile5)
#     rmse_percentile5_std = np.mean(np.std(rmse_percentile5,axis=1))
#     rmse_percentile25_mean = np.mean(rmse_percentile25)
#     rmse_percentile25_std = np.mean(np.std(rmse_percentile25,axis=1))
#     rmse_percentile50_mean = np.mean(rmse_percentile50)
#     rmse_percentile50_std = np.mean(np.std(rmse_percentile50,axis=1))

#     import matplotlib.pyplot as plt
#     #plt.switch_backend('Qt5Agg')
#     fig = plt.figure()
#     xx = np.array(['$\pm5$','$\pm15$','$15-85$'])
#     yy = np.array([rmse_percentile5_mean,rmse_percentile25_mean,rmse_percentile50_mean])
#     yy_std = np.array([rmse_percentile5_std,rmse_percentile25_std,rmse_percentile50_std])
#     yerr = yy_std
#     #plt.errorbar(xx, yy, yerr=yerr, fmt='o')
#     plt.bar(xx, height=yy, yerr=yerr, capsize=6)
#     #plt.grid()
#     plt.xlabel('Percentile')
#     plt.ylabel('RMSE and std')
#     plt.title('RMSE for Different Percentiles')
#     plt.show()
#     savepath = '../results/myCNN/lag_0/SingleGCM/postprocessed/'
#     savename = 'percentile_rmse'
#     plt.savefig(savepath+savename+'.png',dpi=1200,bbox_inches='tight')

#     fig = plt.figure(figsize=(12,5))
#     #plt.plot(ys,'--b',label='Groundtruth')
#     plt.plot(month_name_test,rmse_permonth_mgcm,'-r',label='MultiGCM Prediction')
#     plt.plot(month_name_test,rmse_permonth_sgcm_mean,'-g',label='SingleGCM Prediction')
#     plt.fill_between(month_name_test,y1=rmse_permonth_sgcm_mean+rmse_permonth_sgcm_std,y2= rmse_permonth_sgcm_mean-rmse_permonth_sgcm_std,alpha=0.3,color='g',label='SingleGCM 1 std')
#     plt.legend()
#     plt.xticks(rotation=45,fontsize=8)
#     plt.xlabel('Month')
#     plt.ylabel('Absolute Error')
#     plt.title('Absolute Error per Month for Predictions')
#     plt.show()
#     savepath = '../results/myCNN/lag_0/SingleGCM/'
#     savename = 'abs_diff_vs_time'
#     plt.savefig(savepath+savename+'.png',dpi=1200,bbox_inches='tight')



#     fig = plt.figure(figsize=(12,5))
#     plt.plot(ys,'--b',label='Groundtruth')
#     plt.plot(preds_mgcm,'-r',label='MultiGCM Prediction')
#     plt.plot(preds_sgcm_mean,'-g',label='SingleGCM Prediction')
#     plt.fill_between(range(len(ys)),y1=preds_sgcm_mean+preds_sgcm_std,y2= preds_sgcm_mean-preds_sgcm_std,alpha=0.3,color='g',label='SingleGCM 1 std')
#     plt.legend()
#     plt.xlabel('Month')
#     plt.ylabel('River flow')
#     plt.title('Groundtruth vs Predictions')
#     plt.show()
#     #savepath = '../results/myCNN/lag_0/SingleGCM/'
#     #savename = 'pred_vs_time'
#     #plt.savefig(savepath+savename+'.png',dpi=1200,bbox_inches='tight')

#     fig = plt.figure(figsize=(12,5))
#     plt.plot(ys,'--b',label='Groundtruth')
#     plt.plot(preds_mgcm,'-r',label='MultiGCM Prediction')
#     plt.plot(preds_sgcm[:,13],'-g',label='GCM_13 Prediction')
#     plt.plot(preds_sgcm[:,15],'-y',label='GCM_15 Prediction')
#     plt.legend()
#     plt.xlabel('Month')
#     plt.ylabel('River flow')
#     plt.title('Groundtruth vs Predictions')
#     plt.show()
#     #savepath = '../results/myCNN/lag_0/SingleGCM/'
#     #savename = 'pred_vs_time_sample_gcms'
#     #plt.savefig(savepath+savename+'.png',dpi=1200,bbox_inches='tight')


#     gcm_names = np.load('/scratch/wang.zife/YuminLiu/DATA/GCM/GCMdata/tas/processeddata/gcm_names.npy')
#     xx = np.array([gcm_names[i][24:-35] for i in range(len(gcm_names))])
#     fig = plt.figure(figsize=(12,5))
#     plt.axhline(y=0.2839257769608812,xmin=0,xmax=32,color='r',linestyle='-',label='MultiGCM')
#     plt.axhline(y=rmses_sgcm_mean,xmin=0,xmax=32,color='g',linestyle='-',label='SingleGCM Mean')
#     #plt.scatter(x=xx,y=rmses_sgcm,c='g',label='SingleGCM')
#     plt.bar(x=xx,height=rmses_sgcm,label='SingleGCM')
#     plt.title('RMSE When Using Single GCM or Multiple GCMs')
#     plt.xticks(rotation=90,fontsize=8)
#     plt.xlabel('GCM')
#     plt.ylabel('RMSE')
#     plt.legend()
#     #plt.show()
#     savepath = '../results/myCNN/lag_0/SingleGCM/postprocessed/'
#     savename = 'rmses_3'
#     plt.savefig(savepath+savename+'.png',dpi=1200,bbox_inches='tight')


# def cal_rmse():
#     import numpy as np
#     import pandas as pd
#     import plots

#     path = '../results/myCNN/lag_0/SingleGCM/'
#     filename = 'myCNN_amazon_saliency_maps'
#     column_name = ['GCM','All']+['Year2003','Year2004','Year2005']+['SeasonSpring','SeasonSummer','SeasonAutumn','SeasonWinter']+['NaturalMonth{}'.format(m) for m in range(1,13)]+['Month{}'.format(m) for m in range(36)]

#     salencymap_mgcm = np.load('../results/myCNN/lag_0/2021-02-17_23.06.48.428983_gcm_masked_amazon_region1/2021-02-17_23.06.48.428983_Saliency/myCNN_amazon_saliency_maps.npy')
#     salencymap_mgcm = plots.get_saliency(salencymap_mgcm,fillnan=0.0,threshold=0.15) # [Nmonth,Nlat,Nlon]

#     rmses = []
#     for ngcm in range(32):
#         saliencymap_sgcm = np.load(path+'{}/{}_Saliency/'.format(ngcm,ngcm)+filename+'.npy')
#         saliencymap_sgcm = plots.get_saliency(saliencymap_sgcm,fillnan=0.0,threshold=0.15) # [Nmonth,Nlat,Nlon]
#         rmse = [ngcm]
#         for method in ['All','Years','Seasons','NaturalMonths','Months']:
#             months = plots.get_months(method)
#             for i,month in enumerate(months):
#                 mean_mgcm = np.mean(salencymap_mgcm[month,:,:],axis=0)
#                 mean_sgcm = np.mean(saliencymap_sgcm[month,:,:],axis=0)
#                 rmse.append(np.sqrt(np.mean((mean_mgcm-mean_sgcm)**2)))
#         rmses.append(rmse)
#     rmses = np.array(rmses)#.reshape((1,-1))

#     df = pd.DataFrame(data=rmses,columns=column_name)
#     df.to_csv(path+'postprocessed/difference.csv',index=False,index_label=column_name)
#     def plotrmse(df,column_name,start,end,savename,legend=True):
#         import matplotlib.pyplot as plt
#         fig = plt.figure(figsize=(12,5))
#         for col_name in column_name[start:end]:
#             plt.plot(df[col_name],label=col_name)
#         if legend:
#             plt.legend()
#         plt.xlabel('GCM')
#         plt.ylabel('Difference')
#         plt.title('Difference between Single GCM and Multiple GCMs')
#         plt.show()
#         plt.savefig(path+'postprocessed/'+savename+'.png',dpi=1200,bbox_inches='tight')

#     def plotrmse_months2(df,column_name,start,end,savename):
#         import matplotlib.pyplot as plt
#         fig = plt.figure(figsize=(12,5))
#         rmse_months = df.iloc[:,start:end].to_numpy()
#         rmse_months_mean = np.mean(rmse_months,axis=0)
#         rmse_months_std = np.std(rmse_months,axis=0)
#         plt.plot(rmse_months_mean,label='Mean')
#         plt.fill_between(range(36),y1=rmse_months_mean+rmse_months_std,y2=rmse_months_mean-rmse_months_std,alpha=0.3,label='1 std')
#         plt.legend()
#         plt.xlabel('GCM')
#         plt.ylabel('Difference')
#         plt.title('Single Month Difference between Single GCM and Multiple GCMs')
#         plt.show()
#         plt.savefig(path+'postprocessed/'+savename+'_2.png',dpi=1200,bbox_inches='tight')

#     #plotrmse(df,column_name,1,5,'diff_all_years')
#     #plotrmse(df,column_name,5,9,'diff_seasons')
#     #plotrmse(df,column_name,9,21,'diff_naturalmonths')
#     #plotrmse(df,column_name,21,57,'diff_months',False)
#     #plotrmse_months2(df,column_name,21,57,'diff_months')


# def plot_saliencymap_std():
#     import numpy as np
#     import pandas as pd
#     import plots
#     import os

#     path = '../results/myCNN/lag_0/SingleGCM/'
#     filename = 'myCNN_amazon_saliency_maps'
#     #column_name = ['GCM','All']+['Year2003','Year2004','Year2005']+['SeasonSpring','SeasonSummer','SeasonAutumn','SeasonWinter']+['NaturalMonth{}'.format(m) for m in range(1,13)]+['Month{}'.format(m) for m in range(36)]

#     salencymap_mgcm = np.load('../results/myCNN/lag_0/2021-02-17_23.06.48.428983_gcm_masked_amazon_region1/2021-02-17_23.06.48.428983_Saliency/myCNN_amazon_saliency_maps.npy')
#     salencymap_mgcm = plots.get_saliency(salencymap_mgcm,fillnan=0.0,threshold=0.15) # [Nmonth,Nlat,Nlon]

#     saliencymap_sgcms = []
#     for ngcm in range(32):
#         saliencymap_sgcm = np.load(path+'{}/{}_Saliency/'.format(ngcm,ngcm)+filename+'.npy')
#         saliencymap_sgcm = plots.get_saliency(saliencymap_sgcm,fillnan=0.0,threshold=0.15) # [Nmonth,Nlat,Nlon]
#         saliencymap_sgcms.append(saliencymap_sgcm)
#     varname = 'saliency_gcms_masked'
#     variable = 'amazon' # 'nino3' #
#     savepath_root = path+'postprocessed/std/'
#     ## gcm region 1
#     lonlat = [50.5,-41.5,349.5,37.5] # [50.5,-42.5,349.5,37.5] # [-124.5,24.5,-66.5,49.5] # map area, [left,bottom,right,top]
#     parallels = np.arange(-40.0,40.0,10.0)
#     meridians = np.arange(60.0,350.0,30.0) # label lons, 60E to -9.5W, or 60E to 350E
#     watercolor = 'white' # '#46bcec'
#     cmap = 'YlOrRd' # 'rainbow' # 'Accent' #'YlGn' #'hsv' #'seismic' #
#     alpha = 0.7
#     projection = 'merc' # 'cyl' #
#     resolution = 'i' # 'h'
#     area_thresh = 10000
#     clim = None
#     pos_lons, pos_lats = [], [] # None, None # to plot specific locations
#     verbose = False # True #
#     for method in ['All','Years','Seasons','NaturalMonths','Months']: # ['Years']: #
#         savepath = savepath_root+method+'/'
#         if savepath and not os.path.exists(savepath):
#             os.makedirs(savepath)
#         months = plots.get_months(method)
#         for i,month in enumerate(months):
#             #if i>0: break
#             saliencymaps = []
#             for ngcm in range(32):
#                 saliencymaps_n = np.mean(saliencymap_sgcms[ngcm][month,:,:],axis=0)
#                 saliencymaps.append(saliencymaps_n)
#             saliencymaps = np.stack(saliencymaps)
#             saliencymaps_std = np.std(saliencymaps,axis=0)
#             title,savename = plots.get_title_savename(method,i,varname,variable)
#             #img = np.mean(saliencymaps[month,:,:],axis=0)
#             img = saliencymaps_std
#             img[img==0] = np.nan

#             #img[mask!=0] = np.nan # mask out ocean
#             #img[mask==0] = np.nan # mask out land

#             plots.plot_map(img,title=title,savepath=savepath,savename=savename,cmap=cmap,alpha=alpha,
#                          lonlat=lonlat,projection=projection,resolution=resolution,area_thresh=area_thresh,
#                          parallels=parallels,meridians=meridians,pos_lons=pos_lons, pos_lats=pos_lats,clim=clim,
#                          watercolor=watercolor,verbose=verbose)


# def cal_correlationss():
#     import numpy as np
#     import pandas as pd
#     import plots
#     from scipy.stats import pearsonr as pearsonr

#     path = '../results/myCNN/lag_0/SingleGCM/'
#     filename = 'myCNN_amazon_saliency_maps'
#     column_name = ['GCM','All']+['Year2003','Year2004','Year2005']+['SeasonSpring','SeasonSummer','SeasonAutumn','SeasonWinter']+['NaturalMonth{}'.format(m) for m in range(1,13)]+['Month{}'.format(m) for m in range(36)]

#     salencymap_mgcm = np.load('../results/myCNN/lag_0/2021-02-17_23.06.48.428983_gcm_masked_amazon_region1/2021-02-17_23.06.48.428983_Saliency/myCNN_amazon_saliency_maps.npy')
#     salencymap_mgcm = plots.get_saliency(salencymap_mgcm,fillnan=0.0,threshold=0.15) # [Nmonth,Nlat,Nlon]

#     corrs = []
#     for ngcm in range(32):
#         saliencymap_sgcm = np.load(path+'{}/{}_Saliency/'.format(ngcm,ngcm)+filename+'.npy')
#         saliencymap_sgcm = plots.get_saliency(saliencymap_sgcm,fillnan=0.0,threshold=0.15) # [Nmonth,Nlat,Nlon]
#         corr = [ngcm]
#         for method in ['All','Years','Seasons','NaturalMonths','Months']:
#             months = plots.get_months(method)
#             for i,month in enumerate(months):
#                 mean_mgcm = np.mean(salencymap_mgcm[month,:,:],axis=0)
#                 mean_sgcm = np.mean(saliencymap_sgcm[month,:,:],axis=0)
#                 #corr.append(np.sum(mean_mgcm*mean_sgcm)/np.sqrt(np.sum(mean_mgcm))*np.sqrt(np.sum(mean_sgcm)))
#                 corr.append(pearsonr(mean_mgcm.reshape((-1,)),mean_sgcm.reshape((-1,)))[0])
#         corrs.append(corr)
#     corrs = np.array(corrs)#.reshape((1,-1))

#     df = pd.DataFrame(data=corrs,columns=column_name)
#     #df.to_csv(path+'postprocessed/pearson_corrs.csv',index=False,index_label=column_name)
#     def plotcorr(df,column_name,start,end,savename,legend=True):
#         import matplotlib.pyplot as plt
#         fig = plt.figure(figsize=(12,5))
#         for col_name in column_name[start:end]:
#             plt.plot(df[col_name],label=col_name)
#         if legend:
#             plt.legend()
#         plt.xlabel('GCM')
#         plt.ylabel('Correlation')
#         plt.title('Correlation between Single GCM and Multiple GCMs')
#         plt.show()
#         plt.savefig(path+'postprocessed/'+savename+'.png',dpi=1200,bbox_inches='tight')

#     def plotcorr_months2(df,column_name,start,end,savename):
#         import matplotlib.pyplot as plt
#         fig = plt.figure(figsize=(12,5))
#         corr_months = df.iloc[:,start:end].to_numpy()
#         corr_months_mean = np.mean(corr_months,axis=0)
#         corr_months_std = np.std(corr_months,axis=0)
#         plt.plot(corr_months_mean,label='Mean')
#         plt.fill_between(range(36),y1=corr_months_mean+corr_months_std,y2=corr_months_mean-corr_months_std,alpha=0.3,label='1 std')
#         plt.legend()
#         plt.xlabel('GCM')
#         plt.ylabel('Correlation')
#         plt.title('Single Month Correlation between Single GCM and Multiple GCMs')
#         plt.show()
#         plt.savefig(path+'postprocessed/'+savename+'_2.png',dpi=1200,bbox_inches='tight')

#     plotcorr(df,column_name,1,5,'correlation_all_years')
#     plotcorr(df,column_name,5,9,'correlation_seasons')
#     plotcorr(df,column_name,9,21,'correlation_naturalmonths')
#     plotcorr(df,column_name,21,57,'correlation_months',False)
#     plotcorr_months2(df,column_name,21,57,'correlation_months')


# def cal_ssims():
#     import numpy as np
#     import pandas as pd
#     import plots
#     from skimage.measure import compare_ssim as skissim

#     path = '../results/myCNN/lag_0/SingleGCM/'
#     filename = 'myCNN_amazon_saliency_maps'
#     column_name = ['GCM','All']+['Year2003','Year2004','Year2005']+['SeasonSpring','SeasonSummer','SeasonAutumn','SeasonWinter']+['NaturalMonth{}'.format(m) for m in range(1,13)]+['Month{}'.format(m) for m in range(36)]

#     salencymap_mgcm = np.load('../results/myCNN/lag_0/2021-02-17_23.06.48.428983_gcm_masked_amazon_region1/2021-02-17_23.06.48.428983_Saliency/myCNN_amazon_saliency_maps.npy')
#     salencymap_mgcm = plots.get_saliency(salencymap_mgcm,fillnan=0.0,threshold=0.15) # [Nmonth,Nlat,Nlon]

#     ssims = []
#     for ngcm in range(32):
#         saliencymap_sgcm = np.load(path+'{}/{}_Saliency/'.format(ngcm,ngcm)+filename+'.npy')
#         saliencymap_sgcm = plots.get_saliency(saliencymap_sgcm,fillnan=0.0,threshold=0.15) # [Nmonth,Nlat,Nlon]
#         ssim = [ngcm]
#         for method in ['All','Years','Seasons','NaturalMonths','Months']:
#             months = plots.get_months(method)
#             for i,month in enumerate(months):
#                 ssim.append(skissim(np.mean(salencymap_mgcm[month,:,:],axis=0),np.mean(saliencymap_sgcm[month,:,:],axis=0)))
#         ssims.append(ssim)
#     ssims = np.array(ssims)#.reshape((1,-1))

#     df = pd.DataFrame(data=ssims,columns=column_name)
#     #df.to_csv(path+'postprocessed/ssims.csv',index=False,index_label=column_name)
#     ##a = pd.read_csv(path+'postprocessed/ssims.csv',header=0)

#     import matplotlib.pyplot as plt
#     fig = plt.figure(figsize=(12,5))
#     for col_name in column_name[1:5]:
#         plt.plot(df[col_name],label=col_name)
#     plt.legend()
#     plt.xlabel('GCM')
#     plt.ylabel('SSIM')
#     plt.title('SSIM between Single GCM and Multiple GCMs')
#     plt.show()
#     plt.savefig(path+'postprocessed/ssims_all_years.png',dpi=1200,bbox_inches='tight')

#     fig = plt.figure(figsize=(12,5))
#     for col_name in column_name[5:9]:
#         plt.plot(df[col_name],label=col_name)
#     plt.legend()
#     plt.xlabel('GCM')
#     plt.ylabel('SSIM')
#     plt.title('SSIM between Single GCM and Multiple GCMs')
#     plt.show()
#     plt.savefig(path+'postprocessed/ssims_seasons.png',dpi=1200,bbox_inches='tight')

#     fig = plt.figure(figsize=(12,5))
#     for col_name in column_name[9:21]:
#         plt.plot(df[col_name],label=col_name)
#     plt.legend()
#     plt.xlabel('GCM')
#     plt.ylabel('SSIM')
#     plt.title('SSIM between Single GCM and Multiple GCMs')
#     plt.show()
#     plt.savefig(path+'postprocessed/ssims_naturalmonths.png',dpi=1200,bbox_inches='tight')

#     fig = plt.figure(figsize=(12,5))
#     for col_name in column_name[21:57]:
#         plt.plot(df[col_name],label=col_name)
#     #plt.legend()
#     plt.xlabel('GCM')
#     plt.ylabel('SSIM')
#     plt.title('Single Month SSIM between Single GCM and Multiple GCMs')
#     plt.show()
#     plt.savefig(path+'postprocessed/ssims_months.png',dpi=1200,bbox_inches='tight')

#     fig = plt.figure(figsize=(12,5))
#     ssim_months = df.iloc[:,21:57].to_numpy()
#     ssim_months_mean = np.mean(ssim_months,axis=0)
#     ssim_months_std = np.std(ssim_months,axis=0)
#     plt.plot(ssim_months_mean,label='Mean')
#     plt.fill_between(range(36),y1=ssim_months_mean+ssim_months_std,y2= ssim_months_mean-ssim_months_std,alpha=0.3,label='1 std')
#     plt.legend()
#     plt.xlabel('GCM')
#     plt.ylabel('SSIM')
#     plt.title('Single Month SSIM between Single GCM and Multiple GCMs')
#     plt.show()
#     plt.savefig(path+'postprocessed/ssims_months_2.png',dpi=1200,bbox_inches='tight')


# def plot_saliencymap_diff(ngcm=0):
#     import numpy as np
#     import plots

#     path = '../results/myCNN/lag_0/SingleGCM/'
#     filename = 'myCNN_amazon_saliency_maps'

#     salencymap_mgcm = np.load('../results/myCNN/lag_0/2021-02-17_23.06.48.428983_gcm_masked_amazon_region1/2021-02-17_23.06.48.428983_Saliency/myCNN_amazon_saliency_maps.npy')

#     salencymap_mgcm = np.mean(salencymap_mgcm,axis=(0,1))

#     saliencymaps_sgcm = []
#     for ngcm in range(32): # [13,15]:#
#         filepath = path+'{}/{}_Saliency/'.format(ngcm,ngcm)
#         saliencymap = np.load(filepath+filename+'.npy')
#         saliencymap = plots.get_saliency(saliencymap,fillnan=0.0,threshold=0.15) # [Nmonth,Nlat,Nlon]
#         #saliencymap = np.mean(saliencymap,axis=0)
#         saliencymap = saliencymap[0]
#         saliencymaps_sgcm.append(saliencymap)

#     #ngcm = 13
#     saliencymap_sgcm = saliencymaps_sgcm[ngcm]


#     import matplotlib
#     matplotlib.use('Qt5Agg')
#     ## gcm region 1
#     lonlat = [50.5,-41.5,349.5,37.5] # [50.5,-42.5,349.5,37.5] # [-124.5,24.5,-66.5,49.5] # map area, [left,bottom,right,top]
#     parallels = np.arange(-40.0,40.0,10.0)
#     meridians = np.arange(60.0,350.0,30.0) # label lons, 60E to -9.5W, or 60E to 350E

#     watercolor = 'white' # '#46bcec'
#     cmap = 'rainbow' # 'YlOrRd' # 'Accent' #'YlGn' #'hsv' #'seismic' #
#     alpha = 0.7
#     projection = 'merc' # 'cyl' #
#     resolution = 'i' # 'h'
#     area_thresh = 10000
#     clim = None
#     pos_lons, pos_lats = [], [] # None, None # to plot specific locations
#     verbose = True # False #


#     img = (saliencymap_sgcm-salencymap_mgcm)
#     #title = 'Saliency Map Difference GCM {} to MultiGCM'.format(ngcm)
#     title = 'Single Month {} Saliency Map Difference GCM {} to MultiGCM'.format(0,ngcm)
#     savepath = '../results/myCNN/lag_0/SingleGCM/postprocessed/' # None #
#     savename = title.lower().replace(' ','_') # None
#     plots.plot_map(img,title=title,savepath=savepath,savename=savename,cmap=cmap,alpha=alpha,
#                      lonlat=lonlat,projection=projection,resolution=resolution,area_thresh=area_thresh,
#                      parallels=parallels,meridians=meridians,pos_lons=pos_lons, pos_lats=pos_lats,clim=clim,
#                      watercolor=watercolor,verbose=verbose)


# def plot_singlegcm_preds_rmse():
#     import os
#     import numpy as np
#     import datasets

#     path = '../results/myCNN/lag_0/SingleGCM/'
#     test_dataset = datasets.myDataset_CNN(fold='test',window=3,noise_std=0.0)
#     ys = []
#     for y in test_dataset:
#         ys.append(y[1].numpy()[0])
#     ys = np.array(ys)

#     preds_mgcm = np.load('../results/myCNN/lag_0/2021-02-17_23.06.48.428983_gcm_masked_amazon_region1/pred_results_RMSE0.2839257769608812.npz')['preds']

#     preds_sgcm = []
#     rmses_sgcm = []
#     for ngcm in range(32):
#         filepath = path+str(ngcm)+'/'
#         for file in os.listdir(filepath):
#             if file.startswith('pred_results_RMSE'):
#                 res = np.load(filepath+file)
#                 preds_sgcm.append(res['preds'])
#                 rmses_sgcm.append(res['rmse'])

#     preds_sgcm = np.stack(preds_sgcm,axis=1)
#     preds_sgcm_mean = np.mean(preds_sgcm,axis=1)
#     preds_sgcm_std = np.std(preds_sgcm,axis=1)

#     rmses_sgcm = np.stack(rmses_sgcm,axis=0)
#     rmses_sgcm_mean = np.mean(rmses_sgcm)

#     import matplotlib.pyplot as plt

#     fig = plt.figure(figsize=(12,5))
#     plt.plot(ys,'--b',label='Groundtruth')
#     plt.plot(preds_mgcm,'-r',label='MultiGCM Prediction')
#     plt.plot(preds_sgcm_mean,'-g',label='SingleGCM Prediction')
#     plt.fill_between(range(len(ys)),y1=preds_sgcm_mean+preds_sgcm_std,y2= preds_sgcm_mean-preds_sgcm_std,alpha=0.3,color='g',label='SingleGCM 1 std')
#     plt.legend()
#     plt.xlabel('Month')
#     plt.ylabel('River flow')
#     plt.title('Groundtruth vs Predictions')
#     plt.show()
#     savepath = '../results/myCNN/lag_0/SingleGCM/'
#     savename = 'pred_vs_time'
#     plt.savefig(savepath+savename+'.png',dpi=1200,bbox_inches='tight')

#     fig = plt.figure(figsize=(12,5))
#     plt.plot(ys,'--b',label='Groundtruth')
#     plt.plot(preds_mgcm,'-r',label='MultiGCM Prediction')
#     plt.plot(preds_sgcm[:,13],'-g',label='GCM_13 Prediction')
#     plt.plot(preds_sgcm[:,15],'-y',label='GCM_15 Prediction')
#     plt.legend()
#     plt.xlabel('Month')
#     plt.ylabel('River flow')
#     plt.title('Groundtruth vs Predictions')
#     plt.show()
#     savepath = '../results/myCNN/lag_0/SingleGCM/'
#     savename = 'pred_vs_time_sample_gcms'
#     plt.savefig(savepath+savename+'.png',dpi=1200,bbox_inches='tight')



#     fig = plt.figure(figsize=(12,5))
#     plt.axhline(y=0.2839257769608812,xmin=0,xmax=32,color='r',linestyle='-',label='MultiGCM')
#     plt.axhline(y=rmses_sgcm_mean,xmin=0,xmax=32,color='g',linestyle='-',label='SingleGCM Mean')
#     plt.scatter(x=range(len(rmses_sgcm)),y=rmses_sgcm,c='g',label='SingleGCM')
#     plt.title('RMSE When Using Single GCM or Multiple GCMs')
#     plt.xlabel('GCM')
#     plt.ylabel('RMSE')
#     plt.legend()
#     plt.show()
#     savepath = '../results/myCNN/lag_0/SingleGCM/'
#     savename = 'rmses'
#     plt.savefig(savepath+savename+'.png',dpi=1200,bbox_inches='tight')


# def plot_elevation():
#     import numpy as np
#     from plots import plot_map
#     elevation = np.load('/home/yumin/DS/DATA/PRISM/PRISMdata/elevation/processeddata/prism_elevation_0.125by0.125_USA.npy')
#     data = np.load('../data/Climate/PRISM_GCMdata/ppt/0.125by0.125/prism_USAmask_0.125by0.125.npz')
#     USAmask_HR = data['USAmask_HR']
#     elevation[USAmask_HR==0] = np.nan
#     img = np.flipud(elevation)
#     title = 'Elevation of CONUS'
#     savepath = '../results/Climate/PRISM_GCM/Figures/' #None
#     savename = 'elevation_of_CONUS_0.125by0.125'
#     plot_map(img,title=title,savepath=savepath,savename=savename,cmap='Reds',clim=None)


# def _plot_pdf_temperature(trainpath):
#     import os
#     import numpy as np
#     from skimage.transform import resize
#     #import seaborn as sns
#     #trainpath = '../data/Climate/PRISM_GCMdata/tmax/0.125by0.125/train/'
#     trainnames = [f for f in os.listdir(trainpath) if f.endswith('.npz')]
#     trainnames = sorted(trainnames)
#     prism_train = []
#     gcm_train = []
#     for name in trainnames:
#         data = np.load(trainpath+name)
#         prism_train.append(np.squeeze(data['prism']))
#         gcm_train.append(data['gcms'])
#     prism_train = np.stack(prism_train,axis=0)
#     gcm_train = np.stack(gcm_train,axis=1)
#     prism_train_ori = prism_train*50.0
#     gcm_train_ori = gcm_train*50.0
#     USAmask_HR = np.sum(prism_train,axis=0)
#     USAmask_LR = resize(USAmask_HR,gcm_train.shape[2:],order=1,preserve_range=True)

#     #USAmask_HR[USAmask_HR<1.0] = 0
#     #USAmask_LR[USAmask_LR<1.0] = 0

#     for i in range(len(prism_train)):
#         prism_train[i,:,:][USAmask_HR==0] = np.nan
#         prism_train_ori[i,:,:][USAmask_HR==0] = np.nan
#         for j in range(len(gcm_train)):
#             gcm_train[j,i,:,:][USAmask_LR==0] = np.nan
#             gcm_train_ori[j,i,:,:][USAmask_LR==0] = np.nan

#     prism_train_ori_flat = prism_train_ori.flatten()
#     prism_train_flat = prism_train.flatten()
#     gcm_train_ori_flat = gcm_train_ori.flatten()
#     gcm_train_flat = gcm_train.flatten()

#     prism_train_ori_flat = prism_train_ori_flat[~np.isnan(prism_train_ori_flat)]
#     prism_train_flat = prism_train_flat[~np.isnan(prism_train_flat)]
#     gcm_train_ori_flat = gcm_train_ori_flat[~np.isnan(gcm_train_ori_flat)]
#     gcm_train_flat = gcm_train_flat[~np.isnan(gcm_train_flat)]

#     return prism_train_ori_flat,gcm_train_ori_flat





#     #%%

# def plot_pdf_temperature():
#     import matplotlib.pyplot as plt

#     trainpath_tmax = '../data/Climate/PRISM_GCMdata/tmax/0.125by0.125/train/'
#     trainpath_tmin = '../data/Climate/PRISM_GCMdata/tmin/0.125by0.125/train/'
#     prism_train_ori_flat_tmax,gcm_train_ori_flat_tmax = _plot_pdf_temperature(trainpath_tmax)
#     prism_train_ori_flat_tmin,gcm_train_ori_flat_tmin = _plot_pdf_temperature(trainpath_tmin)

#     bins = 1000
#     #fig = plt.figure()

#     plt.subplot(211)
#     plt.hist(prism_train_ori_flat_tmax,bins=bins,density=True,facecolor='r',alpha=0.8)
#     plt.hist(prism_train_ori_flat_tmin,bins=bins,density=True,facecolor='b',alpha=0.6)
#     plt.legend(['tmax','tmin'])
#     plt.xlim(-30,50)
#     plt.ylim(0,0.08)
#     plt.title('PRISM temperature',fontsize=5)
#     plt.ylabel('Density')
#     plt.grid(True)
#     #plt.show()

#     plt.subplot(212)
#     plt.hist(gcm_train_ori_flat_tmax,bins=bins,density=True,facecolor='r',alpha=0.8)
#     plt.hist(gcm_train_ori_flat_tmin,bins=bins,density=True,facecolor='b',alpha=0.6)
#     plt.legend(['tmax','tmin'])
#     plt.xlim(-30,50)
#     plt.ylim(0,0.08)
#     plt.title('GCM temperature',fontsize=5)
#     plt.ylabel('Density')
#     plt.grid(True)

#     savepath = '../data/Climate/PRISM_GCMdata/'
#     savename = 'prism_gcm_pdf_train_temperature'
#     plt.savefig(savepath+savename+'.jpg',dpi=1200,bbox_inches='tight')
#     plt.show()

#     #sns.distplot(gcm_train_ori_flat,bins=bins,hist=False,kde=True,color='red')
#     ##plt.subplot(212)
#     #sns.distplot(gcm_train_flat,bins=bins,hist=False,kde=True,color='blue')
#     #plt.legend(['original','after log1p transform'])
#     #plt.show()

# #plot_pdf_temperature()

# def plot_pdf_precipitation():
#     import os
#     import numpy as np
#     from skimage.transform import resize
#     import seaborn as sns
#     is_precipitation = True
#     if is_precipitation:
#         trainpath = '../data/Climate/PRISM_GCMdata/ppt/0.125by0.125/train/'
#     else:
#         trainpath = '../data/Climate/PRISM_GCMdata/tmax/0.125by0.125/train/'
#     trainnames = [f for f in os.listdir(trainpath) if f.endswith('.npz')]
#     trainnames = sorted(trainnames)
#     prism_train = []
#     gcm_train = []
#     for name in trainnames:
#         data = np.load(trainpath+name)
#         prism_train.append(np.squeeze(data['prism']))
#         gcm_train.append(data['gcms'])
#     prism_train = np.stack(prism_train,axis=0)
#     gcm_train = np.stack(gcm_train,axis=1)
#     if is_precipitation:
#         prism_train_ori = np.expm1(prism_train)
#         gcm_train_ori = np.expm1(gcm_train)
#     else:
#         prism_train_ori = prism_train*50.0
#         gcm_train_ori = gcm_train*50.0
#     USAmask_HR = np.sum(prism_train,axis=0)
#     USAmask_LR = resize(USAmask_HR,gcm_train.shape[2:],order=1,preserve_range=True)

#     #USAmask_HR[USAmask_HR<1.0] = 0
#     #USAmask_LR[USAmask_LR<1.0] = 0

#     for i in range(len(prism_train)):
#         prism_train[i,:,:][USAmask_HR==0] = np.nan
#         prism_train_ori[i,:,:][USAmask_HR==0] = np.nan
#         for j in range(len(gcm_train)):
#             gcm_train[j,i,:,:][USAmask_LR==0] = np.nan
#             gcm_train_ori[j,i,:,:][USAmask_LR==0] = np.nan

#     prism_train_ori_flat = prism_train_ori.flatten()
#     prism_train_flat = prism_train.flatten()
#     gcm_train_ori_flat = gcm_train_ori.flatten()
#     gcm_train_flat = gcm_train.flatten()

#     prism_train_ori_flat = prism_train_ori_flat[~np.isnan(prism_train_ori_flat)]
#     prism_train_flat = prism_train_flat[~np.isnan(prism_train_flat)]
#     gcm_train_ori_flat = gcm_train_ori_flat[~np.isnan(gcm_train_ori_flat)]
#     gcm_train_flat = gcm_train_flat[~np.isnan(gcm_train_flat)]


#     prism_train_ori_flat = prism_train_ori_flat[prism_train_ori_flat>0.0]
#     prism_train_flat = prism_train_flat[prism_train_flat>0.0]
#     gcm_train_ori_flat = gcm_train_ori_flat[gcm_train_ori_flat>0.0]
#     gcm_train_flat = gcm_train_flat[gcm_train_flat>0.0]


#     #%%
#     import matplotlib.pyplot as plt
#     if is_precipitation:
#         #fig = plt.subplots(nrows=2,ncols=1)
#         #bins = 500
#         bins = 1000
#         fig = plt.figure()

#         plt.subplot(211)
#         plt.hist(prism_train_ori_flat,bins=bins,density=True,facecolor='r',alpha=0.8)
#         plt.hist(prism_train_flat,bins=bins,density=True,facecolor='b',alpha=0.6)
#         plt.legend(['original','after log1p transform'])
#         plt.xlim(-1,12)
#         plt.ylim(0,0.8)
#         plt.title('PRISM ppt',fontsize=5)
#         plt.ylabel('Density')
#         plt.grid(True)
#         #plt.show()

#         plt.subplot(212)
#         plt.hist(gcm_train_ori_flat,bins=bins,density=True,facecolor='r',alpha=0.8)
#         plt.hist(gcm_train_flat,bins=bins,density=True,facecolor='b',alpha=0.6)
#         plt.legend(['original','after log1p transform'])
#         plt.xlim(-1,12)
#         plt.ylim(0,0.8)
#         plt.title('GCM ppt',fontsize=5)
#         plt.ylabel('Density')
#         plt.grid(True)
#         savepath = '../data/Climate/PRISM_GCMdata/ppt/'
#         savename = 'prism_gcm_pdf_train_ppt'
#         #plt.savefig(savepath+savename+'.png',dpi=1200,bbox_inches='tight')
#         plt.show()

#         #sns.distplot(gcm_train_ori_flat,bins=bins,hist=False,kde=True,color='red')
#         ##plt.subplot(212)
#         #sns.distplot(gcm_train_flat,bins=bins,hist=False,kde=True,color='blue')
#         #plt.legend(['original','after log1p transform'])
#         #plt.show()


# #%%
# def plot_ynet():
#     import os
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from plots import plot_map
#     from paths import folders, prednames

#     variable = 'tmin' # 'tmax' # 'ppt' #
#     scale = 2

#     modelname = 'YNet'
#     month = 0 # month to be plot
#     ngcm = 0 # gcm number to be plot
#     clim = None #[0,25]
#     clim_diff = [0,5.0] #[0,3.5] # None #[0,18]

#     resolution = 1/scale
#     if variable=='ppt':
#         is_precipitation = True # False #
#     elif variable=='tmax' or variable=='tmin':
#         is_precipitation = False #
#     #%% predict result
#     folder = folders[variable]['YNet'][scale]
#     predname = prednames[variable]['YNet'][scale]

#     #folder = '2020-01-08_11.03.16.421380_debug'
#     #predname = 'pred_results_MSE1.4542109436459012' #'gcms_bcsd.npz'
#     predpath = '../results/Climate/PRISM_GCM/YNet30/{}/scale{}/{}/'.format(variable,scale,folder)
#     preds = np.load(predpath+predname+'.npy')
#     savepath = '../results/Climate/PRISM_GCM/Figures/abs_diff/' # None
#     #names = {8:'pred_results_MSE1.5432003852393892'}
#     #folders = {8:'2020-01-09_12.22.56.720490'}
#     #%% USA mask
#     datapath = '../data/Climate/PRISM_GCMdata/{}/{}by{}/'.format(variable,resolution,resolution)
#     maskdatapath = datapath+'prism_USAmask_{}by{}.npz'.format(resolution,resolution)
#     maskdata = np.load(maskdatapath)
#     USAmask_HR,USAmask_LR = maskdata['USAmask_HR'],maskdata['USAmask_LR']
#     USAmask_HR,USAmask_LR = abs(USAmask_HR),abs(USAmask_LR)
#     USAmask_HR[USAmask_HR<1] = 0.0
#     USAmask_LR[USAmask_LR<1] = 0.0

#     #%% test data
#     test_filenames = [f for f in os.listdir(datapath+'test/') if f.endswith('.npz')]
#     test_filenames = sorted(test_filenames)
#     gcms_test = []
#     prism_test = []
#     for filename in test_filenames:
#         data = np.load(datapath+'test/'+filename)
#         gcms_test.append(data['gcms']) # [Ngcm,Nlat,Nlon]
#         prism_test.append(np.squeeze(data['prism'])) # [1,Nlat,Nlon] --> [Nlat,Nlon]
#     gcms_test = np.stack(gcms_test,axis=1) #[Ngcm,Nlat,Nlon] --> [Ngcm,Nmon,Nlat,Nlon]
#     prism_test = np.stack(prism_test,axis=0) # [Nlat,Nlon] --> [Nmon,Nlat,Nlon]
#     #print('gcms_test.shape={}\nprism_test.shape={}'.format(gcms_test.shape,prism_test.shape))
#     #print('gcms_test.max={}\nprism_test.max={}\npreds.max={}'.format(np.amax(gcms_test),np.amax(prism_test),np.amax(preds)))

#     if is_precipitation:
#         gcms_test = np.expm1(gcms_test)
#         prism_test = np.expm1(prism_test)
#     else:
#         gcms_test = gcms_test*50.0
#         prism_test = prism_test*50.0
#     print('YNet:\ngcms_test=[{},{}]\nprism_test=[{},{}]\npreds=[{},{}]\n\n'.format(np.amin(gcms_test),np.amax(gcms_test),
#           np.amin(prism_test),np.amax(prism_test),np.amin(preds),np.amax(preds)))

# # =============================================================================
# #     MSE_DJF = np.mean((prism_test[[-1,0,1,11,12,13,23,24,25],:,:]-preds[[-1,0,1,11,12,13,23,24,25],:,:])**2)
# #     MSE_MAM = np.mean((prism_test[[2,3,4,14,15,16,26,27,28],:,:]-preds[[2,3,4,14,15,16,26,27,28],:,:])**2)
# #     MSE_JJA = np.mean((prism_test[[5,6,7,17,18,19,29,30,31],:,:]-preds[[5,6,7,17,18,19,29,30,31],:,:])**2)
# #     MSE_SON = np.mean((prism_test[[8,9,10,20,21,22,32,33,34],:,:]-preds[[8,9,10,20,21,22,32,33,34],:,:])**2)
# #     print('MSE_DJF={}\nMSE_MAM={}\nMSE_JJA={}\nMSE_SON={}\n'.format(MSE_DJF,MSE_MAM,MSE_JJA,MSE_SON))
# #     np.savez(predpath+'MSE_seasonal.npz',MSE_DJF=MSE_DJF,MSE_MAM=MSE_MAM,MSE_JJA=MSE_JJA,MSE_SON=MSE_SON)
# # =============================================================================

#     bias = np.mean(preds-prism_test)
#     corr = np.corrcoef(preds.flatten(),prism_test.flatten())
#     print('bias={}'.format(bias))
#     print('corr={}'.format(corr))

# # =============================================================================
# #     bias_DJF = np.mean(prism_test[[-1,0,1,11,12,13,23,24,25],:,:]-preds[[-1,0,1,11,12,13,23,24,25],:,:])
# #     bias_MAM = np.mean(prism_test[[2,3,4,14,15,16,26,27,28],:,:]-preds[[2,3,4,14,15,16,26,27,28],:,:])
# #     bias_JJA = np.mean(prism_test[[5,6,7,17,18,19,29,30,31],:,:]-preds[[5,6,7,17,18,19,29,30,31],:,:])
# #     bias_SON = np.mean(prism_test[[8,9,10,20,21,22,32,33,34],:,:]-preds[[8,9,10,20,21,22,32,33,34],:,:])
# #     print('bias_DJF={}\nbias_MAM={}\nbias_JJA={}\nbias_SON={}\n'.format(bias_DJF,bias_MAM,bias_JJA,bias_SON))
# #     #np.savez(predpath+'bias_seasonal.npz',bias_DJF=bias_DJF,bias_MAM=bias_MAM,bias_JJA=bias_JJA,bias_SON=bias_SON)
# #     corr_DJF = np.corrcoef(prism_test[[-1,0,1,11,12,13,23,24,25],:,:]-preds[[-1,0,1,11,12,13,23,24,25],:,:])
# #     corr_MAM = np.corrcoef(prism_test[[2,3,4,14,15,16,26,27,28],:,:]-preds[[2,3,4,14,15,16,26,27,28],:,:])
# #     corr_JJA = np.corrcoef(prism_test[[5,6,7,17,18,19,29,30,31],:,:]-preds[[5,6,7,17,18,19,29,30,31],:,:])
# #     corr_SON = np.corrcoef(prism_test[[8,9,10,20,21,22,32,33,34],:,:]-preds[[8,9,10,20,21,22,32,33,34],:,:])
# #     print('corr_DJF={}\ncorr_MAM={}\ncorr_JJA={}\ncorr_SON={}\n'.format(corr_DJF,corr_MAM,corr_JJA,corr_SON))
# #     #np.savez(predpath+'corr_seasonal.npz',corr_DJF=corr_DJF,corr_MAM=corr_MAM,corr_JJA=corr_JJA,corr_SON=corr_SON)
# # =============================================================================

#     pred = preds[month,:,:]
#     pred[USAmask_HR==0] = np.nan
#     #pred[abs(pred)<=0.0001] = np.nan

#     #gcm = gcms_test[ngcm,month,:,:]
#     gcm = np.mean(gcms_test,axis=0)
#     gcm = gcm[month,:,:]
#     gcm[USAmask_LR==0] = np.nan
#     #gcm[abs(gcm)<=0.0001] = np.nan

#     prism = prism_test[month,:,:]
#     prism[USAmask_HR==0] = np.nan
#     #prism[abs(prism)<=0.0001] = np.nan

#     absdiff = abs(preds[month,:,:]-prism_test[month,:,:])
#     absdiff[USAmask_HR==0] = np.nan
#     #absdiff[absdiff<=0.0001] = np.nan

#     absdiff_avg = np.mean(abs(preds-prism_test),axis=0)
#     absdiff_avg[USAmask_HR==0] = np.nan
#     #absdiff_avg[absdiff_avg<=0.0001] = np.nan

#     diff_avg = np.mean(preds-prism_test,axis=0)
#     diff_avg[USAmask_HR==0] = np.nan
#     #diff_avg[diff_avg<=0.0001] = np.nan



#     #%% plot figures
#     img = np.flipud(prism)
#     title = 'ground truth {} '.format(variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
#     savename = 'groundtruth_{}_{}by{}_month{}'.format(variable,resolution,resolution,month+1)
#     plot_map(img,title=title,savepath=savepath,savename=savename,clim=clim)

#     img = np.flipud(gcm)
#     #title = 'input GCM mean ppt '+str(1)+'$^{\circ}$x'+str(1)+'$^{\circ}$'
#     title = 'input GCM mean {}'.format(variable)
#     savename = 'input_gcm_mean_{}_month{}'.format(variable,month+1)
#     plot_map(img,title=title,savepath=savepath,savename=savename,clim=clim)

#     img = np.flipud(pred)
#     title = '{} pred {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
#     savename = '{}_pred_result_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
#     plot_map(img,title=title,savepath=savepath,savename=savename,clim=clim)
#     #plot_map(img,title=None,savepath=None,savename=None,cmap='YlOrRd',
#     #             lonlat=[235,24.125,293.458,49.917],resolution='i',area_thresh=10000)

#     img = np.flipud(absdiff)
#     title = '{} GT absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
#     savename = '{}_abs_diff_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
#     plot_map(img,title=title,savepath=savepath,savename=savename,clim=clim_diff)

#     img = np.flipud(absdiff_avg)
#     #title = '{} GT absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
#     #savename = '{}_abs_diff_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
#     title = '{} GT mean absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
#     savename = '{}_avg_abs_diff_{}_{}by{}'.format(modelname,variable,resolution,resolution)
#     plot_map(img,title=title,savepath=savepath,savename=savename,clim=clim_diff)

#     img = np.flipud(diff_avg)
#     #title = '{} GT absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
#     #savename = '{}_abs_diff_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
#     title = '{} GT mean error {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
#     savename = '{}_avg_diff_{}_{}by{}'.format(modelname,variable,resolution,resolution)
#     plot_map(img,title=title,savepath=savepath,savename=savename,cmap='cool',clim=None)

# #plot_ynet()

# #%%
# def plot_rednet():
#     import os
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from plots import plot_map
#     from paths import folders, prednames

#     variable = 'tmin' # 'tmax' # 'ppt' #
#     scale = 2
#     #resolution = 0.125
#     #is_precipitation = True # False #
#     modelname = 'REDNet'
#     month = 0 # month to be plot
#     ngcm = 0 # gcm number to be plot
#     clim_diff = [0,5.0] #[0,3.5] # None #[0,18]

#     resolution = 1/scale
#     if variable=='ppt':
#         is_precipitation = True # False #
#     elif variable=='tmax' or variable=='tmin':
#         is_precipitation = False #
#     #%% predict result
#     folder = folders[variable]['REDNet'][scale]
#     predname = prednames[variable]['REDNet'][scale]
#     #folder = '2020-01-09_12.22.56.720490'
#     #predname = 'pred_results_MSE1.5432003852393892' #'gcms_bcsd.npz'
#     predpath = '../results/Climate/PRISM_GCM/REDNet30/{}/scale{}/{}/'.format(variable,scale,folder)
#     preds = np.load(predpath+predname+'.npy')
#     savepath = '../results/Climate/PRISM_GCM/Figures/abs_diff/'
#     #names = {8:'pred_results_MSE1.5432003852393892'}
#     #folders = {8:'2020-01-09_12.22.56.720490'}
#     #%% USA mask
#     datapath = '../data/Climate/PRISM_GCMdata/{}/{}by{}/'.format(variable,resolution,resolution)
#     maskdatapath = datapath+'prism_USAmask_{}by{}.npz'.format(resolution,resolution)
#     maskdata = np.load(maskdatapath)
#     USAmask_HR,USAmask_LR = maskdata['USAmask_HR'],maskdata['USAmask_LR']
#     USAmask_HR,USAmask_LR = abs(USAmask_HR),abs(USAmask_LR)
#     USAmask_HR[USAmask_HR<1] = 0.0
#     USAmask_LR[USAmask_LR<1] = 0.0

#     #%% test data
#     test_filenames = [f for f in os.listdir(datapath+'test/') if f.endswith('.npz')]
#     test_filenames = sorted(test_filenames)
#     gcms_test = []
#     prism_test = []
#     for filename in test_filenames:
#         data = np.load(datapath+'test/'+filename)
#         gcms_test.append(data['gcms']) # [Ngcm,Nlat,Nlon]
#         prism_test.append(np.squeeze(data['prism'])) # [1,Nlat,Nlon] --> [Nlat,Nlon]
#     gcms_test = np.stack(gcms_test,axis=1) #[Ngcm,Nlat,Nlon] --> [Ngcm,Nmon,Nlat,Nlon]
#     prism_test = np.stack(prism_test,axis=0) # [Nlat,Nlon] --> [Nmon,Nlat,Nlon]
#     #print('gcms_test.shape={}\nprism_test.shape={}'.format(gcms_test.shape,prism_test.shape))
#     #print('gcms_test.max={}\nprism_test.max={}\npreds.max={}'.format(np.amax(gcms_test),np.amax(prism_test),np.amax(preds)))

#     if is_precipitation:
#         gcms_test = np.expm1(gcms_test)
#         prism_test = np.expm1(prism_test)
#     else:
#         gcms_test = gcms_test*50.0
#         prism_test = prism_test*50.0
#     print('REDNet:\ngcms_test=[{},{}]\nprism_test=[{},{}]\npreds=[{},{}]\n\n'.format(np.amin(gcms_test),np.amax(gcms_test),
#           np.amin(prism_test),np.amax(prism_test),np.amin(preds),np.amax(preds)))

# # =============================================================================
# #     MSE_DJF = np.mean((prism_test[[-1,0,1,11,12,13,23,24,25],:,:]-preds[[-1,0,1,11,12,13,23,24,25],:,:])**2)
# #     MSE_MAM = np.mean((prism_test[[2,3,4,14,15,16,26,27,28],:,:]-preds[[2,3,4,14,15,16,26,27,28],:,:])**2)
# #     MSE_JJA = np.mean((prism_test[[5,6,7,17,18,19,29,30,31],:,:]-preds[[5,6,7,17,18,19,29,30,31],:,:])**2)
# #     MSE_SON = np.mean((prism_test[[8,9,10,20,21,22,32,33,34],:,:]-preds[[8,9,10,20,21,22,32,33,34],:,:])**2)
# #     print('MSE_DJF={}\nMSE_MAM={}\nMSE_JJA={}\nMSE_SON={}\n'.format(MSE_DJF,MSE_MAM,MSE_JJA,MSE_SON))
# #     np.savez(predpath+'MSE_seasonal.npz',MSE_DJF=MSE_DJF,MSE_MAM=MSE_MAM,MSE_JJA=MSE_JJA,MSE_SON=MSE_SON)
# # =============================================================================

#     bias = np.mean(preds-prism_test)
#     corr = np.corrcoef(preds.flatten(),prism_test.flatten())
#     print('bias={}'.format(bias))
#     print('corr={}'.format(corr))

# # =============================================================================
# #     bias_DJF = np.mean(prism_test[[-1,0,1,11,12,13,23,24,25],:,:]-preds[[-1,0,1,11,12,13,23,24,25],:,:])
# #     bias_MAM = np.mean(prism_test[[2,3,4,14,15,16,26,27,28],:,:]-preds[[2,3,4,14,15,16,26,27,28],:,:])
# #     bias_JJA = np.mean(prism_test[[5,6,7,17,18,19,29,30,31],:,:]-preds[[5,6,7,17,18,19,29,30,31],:,:])
# #     bias_SON = np.mean(prism_test[[8,9,10,20,21,22,32,33,34],:,:]-preds[[8,9,10,20,21,22,32,33,34],:,:])
# #     print('bias_DJF={}\nbias_MAM={}\nbias_JJA={}\nbias_SON={}\n'.format(bias_DJF,bias_MAM,bias_JJA,bias_SON))
# #     #np.savez(predpath+'bias_seasonal.npz',bias_DJF=bias_DJF,bias_MAM=bias_MAM,bias_JJA=bias_JJA,bias_SON=bias_SON)
# #     corr_DJF = np.corrcoef(prism_test[[-1,0,1,11,12,13,23,24,25],:,:]-preds[[-1,0,1,11,12,13,23,24,25],:,:])
# #     corr_MAM = np.corrcoef(prism_test[[2,3,4,14,15,16,26,27,28],:,:]-preds[[2,3,4,14,15,16,26,27,28],:,:])
# #     corr_JJA = np.corrcoef(prism_test[[5,6,7,17,18,19,29,30,31],:,:]-preds[[5,6,7,17,18,19,29,30,31],:,:])
# #     corr_SON = np.corrcoef(prism_test[[8,9,10,20,21,22,32,33,34],:,:]-preds[[8,9,10,20,21,22,32,33,34],:,:])
# #     print('corr_DJF={}\ncorr_MAM={}\ncorr_JJA={}\ncorr_SON={}\n'.format(corr_DJF,corr_MAM,corr_JJA,corr_SON))
# #     #np.savez(predpath+'corr_seasonal.npz',corr_DJF=corr_DJF,corr_MAM=corr_MAM,corr_JJA=corr_JJA,corr_SON=corr_SON)
# # =============================================================================

#     pred = preds[month,:,:]
#     pred[USAmask_HR==0] = np.nan
#     #pred[abs(pred)<=0.0001] = np.nan

#     #gcm = gcms_test[ngcm,month,:,:]
#     gcm = np.mean(gcms_test,axis=0)
#     gcm = gcm[month,:,:]
#     gcm[USAmask_LR==0] = np.nan
#     #gcm[abs(gcm)<=0.0001] = np.nan

#     prism = prism_test[month,:,:]
#     prism[USAmask_HR==0] = np.nan
#     #prism[abs(prism)<=0.0001] = np.nan

#     absdiff = abs(preds[month,:,:]-prism_test[month,:,:])
#     absdiff[USAmask_HR==0] = np.nan
#     #absdiff[absdiff<=0.0001] = np.nan

#     absdiff_avg = np.mean(abs(preds-prism_test),axis=0)
#     absdiff_avg[USAmask_HR==0] = np.nan
#     #absdiff_avg[absdiff_avg<=0.0001] = np.nan

#     diff_avg = np.mean(preds-prism_test,axis=0)
#     diff_avg[USAmask_HR==0] = np.nan
#     #diff_avg[diff_avg<=0.0001] = np.nan

#     #%% plot figures
# # =============================================================================
# #     img = np.flipud(prism)
# #     title = 'ground truth {} '.format(variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
# #     savename = 'groundtruth_{}_{}by{}_month{}'.format(variable,resolution,resolution,month+1)
# #     plot_map(img,title=title,savepath=savepath,savename=savename)
# #
# #     img = np.flipud(gcm)
# #     #title = 'input GCM mean ppt '+str(1)+'$^{\circ}$x'+str(1)+'$^{\circ}$'
# #     title = 'input GCM mean {}'.format(variable)
# #     savename = 'input_gcm_mean_{}_month{}'.format(variable,month+1)
# #     plot_map(img,title=title,savepath=savepath,savename=savename)
# #
# #     img = np.flipud(pred)
# #     title = '{} pred {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
# #     savename = '{}_pred_result_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
# #     plot_map(img,title=title,savepath=savepath,savename=savename)
# #     #plot_map(img,title=None,savepath=None,savename=None,cmap='YlOrRd',
# #     #             lonlat=[235,24.125,293.458,49.917],resolution='i',area_thresh=10000)
# #
# #     img = np.flipud(absdiff)
# #     title = '{} GT absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
# #     savename = '{}_abs_diff_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
# #     plot_map(img,title=title,savepath=savepath,savename=savename,clim=clim_diff)
# #
# #     img = np.flipud(absdiff_avg)
# #     #title = '{} GT absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
# #     #savename = '{}_abs_diff_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
# #     title = '{} GT mean absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
# #     savename = '{}_avg_abs_diff_{}_{}by{}'.format(modelname,variable,resolution,resolution)
# #     plot_map(img,title=title,savepath=savepath,savename=savename,clim=clim_diff)
# #
# #     img = np.flipud(diff_avg)
# #     #title = '{} GT absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
# #     #savename = '{}_abs_diff_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
# #     title = '{} GT mean error {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
# #     savename = '{}_avg_diff_{}_{}by{}'.format(modelname,variable,resolution,resolution)
# #     plot_map(img,title=title,savepath=savepath,savename=savename,cmap='cool',clim=None)
# # =============================================================================

# #plot_rednet()

# def plot_espcn():
#     import os
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from plots import plot_map
#     from paths import folders, prednames
#     variable = 'tmin' # 'tmax' # 'ppt' #
#     scale = 2
#     #resolution = 0.125
#     #is_precipitation = True # False #
#     modelname = 'ESPCN'
#     month = 0 # month to be plot
#     ngcm = 0 # gcm number to be plot
#     clim_diff = [0,5.0] #[0,3.5] #None # [0,15] #

#     resolution = 1/scale
#     if variable=='ppt':
#         is_precipitation = True # False #
#     elif variable=='tmax' or variable=='tmin':
#         is_precipitation = False #
#     #%% predict result
#     folder = folders[variable]['ESPCN'][scale]
#     predname = prednames[variable]['ESPCN'][scale]
#     #folder = '2020-01-08_18.18.38.602588'
#     #predname = 'pred_results_MSE2.4984338548448353' #'gcms_bcsd.npz'
#     predpath = '../results/Climate/PRISM_GCM/ESPCN/{}/scale{}/{}/'.format(variable,scale,folder)
#     preds = np.load(predpath+predname+'.npy')
#     savepath = '../results/Climate/PRISM_GCM/Figures/abs_diff/' # None #

#     #%% USA mask
#     datapath = '../data/Climate/PRISM_GCMdata/{}/{}by{}/'.format(variable,resolution,resolution)
#     maskdatapath = datapath+'prism_USAmask_{}by{}.npz'.format(resolution,resolution)
#     maskdata = np.load(maskdatapath)
#     USAmask_HR,USAmask_LR = maskdata['USAmask_HR'],maskdata['USAmask_LR']
#     USAmask_HR,USAmask_LR = abs(USAmask_HR),abs(USAmask_LR)

#     USAmask_HR[USAmask_HR<1] = 0.0
#     USAmask_LR[USAmask_LR<1] = 0.0
#     #%% test data
#     test_filenames = [f for f in os.listdir(datapath+'test/') if f.endswith('.npz')]
#     test_filenames = sorted(test_filenames)
#     gcms_test = []
#     prism_test = []
#     for filename in test_filenames:
#         data = np.load(datapath+'test/'+filename)
#         gcms_test.append(data['gcms']) # [Ngcm,Nlat,Nlon]
#         prism_test.append(np.squeeze(data['prism'])) # [1,Nlat,Nlon] --> [Nlat,Nlon]
#     gcms_test = np.stack(gcms_test,axis=1) #[Ngcm,Nlat,Nlon] --> [Ngcm,Nmon,Nlat,Nlon]
#     prism_test = np.stack(prism_test,axis=0) # [Nlat,Nlon] --> [Nmon,Nlat,Nlon]
#     #print('gcms_test.shape={}\nprism_test.shape={}'.format(gcms_test.shape,prism_test.shape))
#     #print('gcms_test.max={}\nprism_test.max={}\npreds.max={}\n'.format(np.amax(gcms_test),np.amax(prism_test),np.amax(preds)))

#     if is_precipitation:
#         gcms_test = np.expm1(gcms_test)
#         prism_test = np.expm1(prism_test)
#     else:
#         gcms_test = gcms_test*50.0
#         prism_test = prism_test*50.0
#     print('ESPCN:\ngcms_test=[{},{}]\nprism_test=[{},{}]\npreds=[{},{}]\n\n'.format(np.amin(gcms_test),np.amax(gcms_test),
#           np.amin(prism_test),np.amax(prism_test),np.amin(preds),np.amax(preds)))

# # =============================================================================
# #     MSE_DJF = np.mean((prism_test[[-1,0,1,11,12,13,23,24,25],:,:]-preds[[-1,0,1,11,12,13,23,24,25],:,:])**2)
# #     MSE_MAM = np.mean((prism_test[[2,3,4,14,15,16,26,27,28],:,:]-preds[[2,3,4,14,15,16,26,27,28],:,:])**2)
# #     MSE_JJA = np.mean((prism_test[[5,6,7,17,18,19,29,30,31],:,:]-preds[[5,6,7,17,18,19,29,30,31],:,:])**2)
# #     MSE_SON = np.mean((prism_test[[8,9,10,20,21,22,32,33,34],:,:]-preds[[8,9,10,20,21,22,32,33,34],:,:])**2)
# #     print('MSE_DJF={}\nMSE_MAM={}\nMSE_JJA={}\nMSE_SON={}\n'.format(MSE_DJF,MSE_MAM,MSE_JJA,MSE_SON))
# #     np.savez(predpath+'MSE_seasonal.npz',MSE_DJF=MSE_DJF,MSE_MAM=MSE_MAM,MSE_JJA=MSE_JJA,MSE_SON=MSE_SON)
# # =============================================================================

#     bias = np.mean(preds-prism_test)
#     corr = np.corrcoef(preds.flatten(),prism_test.flatten())
#     print('bias={}'.format(bias))
#     print('corr={}'.format(corr))

# # =============================================================================
# #     bias_DJF = np.mean(prism_test[[-1,0,1,11,12,13,23,24,25],:,:]-preds[[-1,0,1,11,12,13,23,24,25],:,:])
# #     bias_MAM = np.mean(prism_test[[2,3,4,14,15,16,26,27,28],:,:]-preds[[2,3,4,14,15,16,26,27,28],:,:])
# #     bias_JJA = np.mean(prism_test[[5,6,7,17,18,19,29,30,31],:,:]-preds[[5,6,7,17,18,19,29,30,31],:,:])
# #     bias_SON = np.mean(prism_test[[8,9,10,20,21,22,32,33,34],:,:]-preds[[8,9,10,20,21,22,32,33,34],:,:])
# #     print('bias_DJF={}\nbias_MAM={}\nbias_JJA={}\nbias_SON={}\n'.format(bias_DJF,bias_MAM,bias_JJA,bias_SON))
# #     #np.savez(predpath+'bias_seasonal.npz',bias_DJF=bias_DJF,bias_MAM=bias_MAM,bias_JJA=bias_JJA,bias_SON=bias_SON)
# #     corr_DJF = np.corrcoef(prism_test[[-1,0,1,11,12,13,23,24,25],:,:]-preds[[-1,0,1,11,12,13,23,24,25],:,:])
# #     corr_MAM = np.corrcoef(prism_test[[2,3,4,14,15,16,26,27,28],:,:]-preds[[2,3,4,14,15,16,26,27,28],:,:])
# #     corr_JJA = np.corrcoef(prism_test[[5,6,7,17,18,19,29,30,31],:,:]-preds[[5,6,7,17,18,19,29,30,31],:,:])
# #     corr_SON = np.corrcoef(prism_test[[8,9,10,20,21,22,32,33,34],:,:]-preds[[8,9,10,20,21,22,32,33,34],:,:])
# #     print('corr_DJF={}\ncorr_MAM={}\ncorr_JJA={}\ncorr_SON={}\n'.format(corr_DJF,corr_MAM,corr_JJA,corr_SON))
# #     #np.savez(predpath+'corr_seasonal.npz',corr_DJF=corr_DJF,corr_MAM=corr_MAM,corr_JJA=corr_JJA,corr_SON=corr_SON)
# # =============================================================================

#     pred = preds[month,:,:]
#     pred[USAmask_HR==0] = np.nan
#     #pred[abs(pred)<=0.0001] = np.nan

#     #gcm = gcms_test[ngcm,month,:,:]
#     gcm = np.mean(gcms_test,axis=0)
#     gcm = gcm[month,:,:]
#     gcm[USAmask_LR==0] = np.nan
#     #gcm[abs(gcm)<=0.0001] = np.nan

#     prism = prism_test[month,:,:]
#     prism[USAmask_HR==0] = np.nan
#     #prism[abs(prism)<=0.0001] = np.nan

#     absdiff = abs(preds[month,:,:]-prism_test[month,:,:])
#     absdiff[USAmask_HR==0] = np.nan
#     #absdiff[absdiff<=0.0001] = np.nan

#     absdiff_avg = np.mean(abs(preds-prism_test),axis=0)
#     absdiff_avg[USAmask_HR==0] = np.nan
#     #absdiff_avg[absdiff_avg<=0.0001] = np.nan

#     diff_avg = np.mean(preds-prism_test,axis=0)
#     diff_avg[USAmask_HR==0] = np.nan
#     #diff_avg[diff_avg<=0.0001] = np.nan

#     #%% plot figures
# # =============================================================================
# #     img = np.flipud(prism)
# #     title = 'ground truth {} '.format(variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
# #     savename = 'groundtruth_{}_{}by{}_month{}'.format(variable,resolution,resolution,month+1)
# #     plot_map(img,title=title,savepath=savepath,savename=savename)
# #
# #     img = np.flipud(gcm)
# #     #title = 'input GCM mean ppt '+str(1)+'$^{\circ}$x'+str(1)+'$^{\circ}$'
# #     title = 'input GCM mean {}'.format(variable)
# #     savename = 'input_gcm_mean_{}_month{}'.format(variable,month+1)
# #     plot_map(img,title=title,savepath=savepath,savename=savename)
# #
# #     img = np.flipud(pred)
# #     title = '{} pred {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
# #     savename = '{}_pred_result_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
# #     plot_map(img,title=title,savepath=savepath,savename=savename)
# #     #plot_map(img,title=title,savepath=savepath,savename=savename,cmap='YlOrRd')
# #     #plot_map(img,title=None,savepath=None,savename=None,cmap='YlOrRd',
# #     #             lonlat=[235,24.125,293.458,49.917],resolution='i',area_thresh=10000)
# #
# #     img = np.flipud(absdiff)
# #     title = '{} GT absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
# #     savename = '{}_abs_diff_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
# #     plot_map(img,title=title,savepath=savepath,savename=savename,clim=clim_diff)
# #
# #     img = np.flipud(absdiff_avg)
# #     #title = '{} GT absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
# #     #savename = '{}_abs_diff_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
# #     title = '{} GT mean absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
# #     savename = '{}_avg_abs_diff_{}_{}by{}'.format(modelname,variable,resolution,resolution)
# #     plot_map(img,title=title,savepath=savepath,savename=savename,clim=clim_diff)
# #
# #     img = np.flipud(diff_avg)
# #     #title = '{} GT absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
# #     #savename = '{}_abs_diff_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
# #     title = '{} GT mean error {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
# #     savename = '{}_avg_diff_{}_{}by{}'.format(modelname,variable,resolution,resolution)
# #     plot_map(img,title=title,savepath=savepath,savename=savename,cmap='cool',clim=None)
# #
# # =============================================================================
# #plot_espcn()

# #%%
# def plot_deepsd():
#     import os
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from plots import plot_map
#     from paths import folders, prednames
#     variable = 'tmin' # 'tmax' # 'ppt' #
#     scale = 2
#     #resolution = 0.125
#     #is_precipitation = True # False #
#     modelname = 'DeepSD'
#     month = 0 # month to be plot
#     ngcm = 0 # gcm number to be plot
#     clim_diff = [0,5.0] #[0,3.5] #None #[0,18]

#     resolution = 1/scale
#     if variable=='ppt':
#         is_precipitation = True # False #
#     elif variable=='tmax' or variable=='tmin':
#         is_precipitation = False #
#     folder = folders[variable]['DeepSD'][scale]
#     predname = prednames[variable]['DeepSD'][scale]
#     #%% predict result
#     #folder = '2020-01-08_16.06.29.813876'
#     #predname = 'pred_results_MSE1.7008076906204224' #'gcms_bcsd.npz'
#     predpath = '../results/Climate/PRISM_GCM/DeepSD/{}/train_together/{}by{}/{}/'.format(variable,resolution,resolution,folder)
#     preds = np.load(predpath+predname+'.npy')
#     savepath = '../results/Climate/PRISM_GCM/Figures/abs_diff/'

#     #%% USA mask
#     datapath = '../data/Climate/PRISM_GCMdata/{}/{}by{}/'.format(variable,resolution,resolution)
#     maskdatapath = datapath+'prism_USAmask_{}by{}.npz'.format(resolution,resolution)
#     maskdata = np.load(maskdatapath)
#     USAmask_HR,USAmask_LR = maskdata['USAmask_HR'],maskdata['USAmask_LR']
#     USAmask_HR,USAmask_LR = abs(USAmask_HR),abs(USAmask_LR)

#     USAmask_HR[USAmask_HR<1] = 0.0
#     USAmask_LR[USAmask_LR<1] = 0.0
#     #%% test data
#     test_filenames = [f for f in os.listdir(datapath+'test/') if f.endswith('.npz')]
#     test_filenames = sorted(test_filenames)
#     gcms_test = []
#     prism_test = []
#     for filename in test_filenames:
#         data = np.load(datapath+'test/'+filename)
#         gcms_test.append(data['gcms']) # [Ngcm,Nlat,Nlon]
#         prism_test.append(np.squeeze(data['prism'])) # [1,Nlat,Nlon] --> [Nlat,Nlon]
#     gcms_test = np.stack(gcms_test,axis=1) #[Ngcm,Nlat,Nlon] --> [Ngcm,Nmon,Nlat,Nlon]
#     prism_test = np.stack(prism_test,axis=0) # [Nlat,Nlon] --> [Nmon,Nlat,Nlon]
#     #print('gcms_test.shape={}\nprism_test.shape={}'.format(gcms_test.shape,prism_test.shape))
#     #print('gcms_test.max={}\n,prism_test.max={}'.format(np.amax(gcms_test),np.amax(prism_test)))

#     if is_precipitation:
#         gcms_test = np.expm1(gcms_test)
#         prism_test = np.expm1(prism_test)
#     else:
#         gcms_test = gcms_test*50.0
#         prism_test = prism_test*50.0
#     print('DeepSD:\ngcms_test=[{},{}]\nprism_test=[{},{}]\npreds=[{},{}]\n\n'.format(np.amin(gcms_test),np.amax(gcms_test),
#           np.amin(prism_test),np.amax(prism_test),np.amin(preds),np.amax(preds)))

# # =============================================================================
# #     MSE_DJF = np.mean((prism_test[[-1,0,1,11,12,13,23,24,25],:,:]-preds[[-1,0,1,11,12,13,23,24,25],:,:])**2)
# #     MSE_MAM = np.mean((prism_test[[2,3,4,14,15,16,26,27,28],:,:]-preds[[2,3,4,14,15,16,26,27,28],:,:])**2)
# #     MSE_JJA = np.mean((prism_test[[5,6,7,17,18,19,29,30,31],:,:]-preds[[5,6,7,17,18,19,29,30,31],:,:])**2)
# #     MSE_SON = np.mean((prism_test[[8,9,10,20,21,22,32,33,34],:,:]-preds[[8,9,10,20,21,22,32,33,34],:,:])**2)
# #     print('MSE_DJF={}\nMSE_MAM={}\nMSE_JJA={}\nMSE_SON={}\n'.format(MSE_DJF,MSE_MAM,MSE_JJA,MSE_SON))
# #     np.savez(predpath+'MSE_seasonal.npz',MSE_DJF=MSE_DJF,MSE_MAM=MSE_MAM,MSE_JJA=MSE_JJA,MSE_SON=MSE_SON)
# # =============================================================================

#     bias = np.mean(preds-prism_test)
#     corr = np.corrcoef(preds.flatten(),prism_test.flatten())
#     print('bias={}'.format(bias))
#     print('corr={}'.format(corr))

# # =============================================================================
# #     bias_DJF = np.mean(prism_test[[-1,0,1,11,12,13,23,24,25],:,:]-preds[[-1,0,1,11,12,13,23,24,25],:,:])
# #     bias_MAM = np.mean(prism_test[[2,3,4,14,15,16,26,27,28],:,:]-preds[[2,3,4,14,15,16,26,27,28],:,:])
# #     bias_JJA = np.mean(prism_test[[5,6,7,17,18,19,29,30,31],:,:]-preds[[5,6,7,17,18,19,29,30,31],:,:])
# #     bias_SON = np.mean(prism_test[[8,9,10,20,21,22,32,33,34],:,:]-preds[[8,9,10,20,21,22,32,33,34],:,:])
# #     print('bias_DJF={}\nbias_MAM={}\nbias_JJA={}\nbias_SON={}\n'.format(bias_DJF,bias_MAM,bias_JJA,bias_SON))
# #     #np.savez(predpath+'bias_seasonal.npz',bias_DJF=bias_DJF,bias_MAM=bias_MAM,bias_JJA=bias_JJA,bias_SON=bias_SON)
# #     corr_DJF = np.corrcoef(prism_test[[-1,0,1,11,12,13,23,24,25],:,:]-preds[[-1,0,1,11,12,13,23,24,25],:,:])
# #     corr_MAM = np.corrcoef(prism_test[[2,3,4,14,15,16,26,27,28],:,:]-preds[[2,3,4,14,15,16,26,27,28],:,:])
# #     corr_JJA = np.corrcoef(prism_test[[5,6,7,17,18,19,29,30,31],:,:]-preds[[5,6,7,17,18,19,29,30,31],:,:])
# #     corr_SON = np.corrcoef(prism_test[[8,9,10,20,21,22,32,33,34],:,:]-preds[[8,9,10,20,21,22,32,33,34],:,:])
# #     print('corr_DJF={}\ncorr_MAM={}\ncorr_JJA={}\ncorr_SON={}\n'.format(corr_DJF,corr_MAM,corr_JJA,corr_SON))
# #     #np.savez(predpath+'corr_seasonal.npz',corr_DJF=corr_DJF,corr_MAM=corr_MAM,corr_JJA=corr_JJA,corr_SON=corr_SON)
# # =============================================================================

#     pred = preds[month,:,:]
#     pred[USAmask_HR==0] = np.nan
#     #pred[abs(pred)<=0.0001] = np.nan

#     #gcm = gcms_test[ngcm,month,:,:]
#     gcm = np.mean(gcms_test,axis=0)
#     gcm = gcm[month,:,:]
#     gcm[USAmask_LR==0] = np.nan
#     #gcm[abs(gcm)<=0.0001] = np.nan

#     prism = prism_test[month,:,:]
#     prism[USAmask_HR==0] = np.nan
#     #prism[abs(prism)<=0.0001] = np.nan

#     absdiff = abs(preds[month,:,:]-prism_test[month,:,:])
#     absdiff[USAmask_HR==0] = np.nan
#     #absdiff[absdiff<=0.0001] = np.nan

#     absdiff_avg = np.mean(abs(preds-prism_test),axis=0)
#     absdiff_avg[USAmask_HR==0] = np.nan
#     #absdiff_avg[absdiff_avg<=0.0001] = np.nan

#     diff_avg = np.mean(preds-prism_test,axis=0)
#     diff_avg[USAmask_HR==0] = np.nan
#     #diff_avg[diff_avg<=0.0001] = np.nan

#     #%% plot figures
# # =============================================================================
# #     img = np.flipud(prism)
# #     title = 'ground truth {} '.format(variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
# #     savename = 'groundtruth_{}_{}by{}_month{}'.format(variable,resolution,resolution,month+1)
# #     plot_map(img,title=title,savepath=savepath,savename=savename)
# #
# #     img = np.flipud(gcm)
# #     #title = 'input GCM mean ppt '+str(1)+'$^{\circ}$x'+str(1)+'$^{\circ}$'
# #     title = 'input GCM mean {}'.format(variable)
# #     savename = 'input_gcm_mean_{}_month{}'.format(variable,month+1)
# #     plot_map(img,title=title,savepath=savepath,savename=savename)
# #
# #     img = np.flipud(pred)
# #     title = '{} pred {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
# #     savename = '{}_pred_result_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
# #     plot_map(img,title=title,savepath=savepath,savename=savename)
# #     #plot_map(img,title=None,savepath=None,savename=None,cmap='YlOrRd',
# #     #             lonlat=[235,24.125,293.458,49.917],resolution='i',area_thresh=10000)
# #
# #     img = np.flipud(absdiff)
# #     title = '{} GT absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
# #     savename = '{}_abs_diff_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
# #     plot_map(img,title=title,savepath=savepath,savename=savename,clim=clim_diff)
# #
# #
# #     img = np.flipud(absdiff_avg)
# #     #title = '{} GT absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
# #     #savename = '{}_abs_diff_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
# #     title = '{} GT mean absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
# #     savename = '{}_avg_abs_diff_{}_{}by{}'.format(modelname,variable,resolution,resolution)
# #     plot_map(img,title=title,savepath=savepath,savename=savename,clim=clim_diff)
# #
# #     img = np.flipud(diff_avg)
# #     #title = '{} GT absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
# #     #savename = '{}_abs_diff_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
# #     title = '{} GT mean error {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
# #     savename = '{}_avg_diff_{}_{}by{}'.format(modelname,variable,resolution,resolution)
# #     plot_map(img,title=title,savepath=savepath,savename=savename,cmap='cool',clim=None)
# # =============================================================================

# #plot_deepsd()

# #%%
# def plot_bcsd():
#     import os
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from plots import plot_map
#     from paths import folders, prednames
#     variable = 'ppt' # 'tmin' # 'tmax' #
#     scale = 8
#     #resolution = 0.125
#     #is_precipitation = True # False #
#     modelname = 'BCSD'
#     month = 0 # month to be plot
#     ngcm = 0 # gcm number to be plot
#     clim_diff = [0,5.0] #[0,3.5] # None #[0,18]

#     resolution = 1/scale
#     if variable=='ppt':
#         is_precipitation = True # False #
#     elif variable=='tmax' or variable=='tmin':
#         is_precipitation = False #
#     #%% predict result
#     #folder = folders[variable]['BCSD'][scale]
#     predname = prednames[variable]['BCSD'][scale]
#     #predname = 'gcms_bcsd_pred_MSE1.7627197220364208' #'gcms_bcsd.npz'
#     predpath = '../results/Climate/PRISM_GCM/BCSD/{}/{}by{}/'.format(variable,resolution,resolution)
#     preddata = np.load(predpath+predname+'.npz')
#     preds = preddata['gcms_bcsd']
#     #MSE = preddata['MSE']
#     savepath = '../results/Climate/PRISM_GCM/Figures/abs_diff/'

#     #%% USA mask
#     datapath = '../data/Climate/PRISM_GCMdata/{}/{}by{}/'.format(variable,resolution,resolution)
#     maskdatapath = datapath+'prism_USAmask_{}by{}.npz'.format(resolution,resolution)
#     maskdata = np.load(maskdatapath)
#     USAmask_HR,USAmask_LR = maskdata['USAmask_HR'],maskdata['USAmask_LR']
#     USAmask_HR,USAmask_LR = abs(USAmask_HR),abs(USAmask_LR)

#     USAmask_HR[USAmask_HR<1] = 0.0
#     USAmask_LR[USAmask_LR<1] = 0.0
#     #%% test data
#     test_filenames = [f for f in os.listdir(datapath+'test/') if f.endswith('.npz')]
#     test_filenames = sorted(test_filenames)
#     gcms_test = []
#     prism_test = []
#     for filename in test_filenames:
#         data = np.load(datapath+'test/'+filename)
#         gcms_test.append(data['gcms']) # [Ngcm,Nlat,Nlon]
#         prism_test.append(np.squeeze(data['prism'])) # [1,Nlat,Nlon] --> [Nlat,Nlon]
#     gcms_test = np.stack(gcms_test,axis=1) #[Ngcm,Nlat,Nlon] --> [Ngcm,Nmon,Nlat,Nlon]
#     prism_test = np.stack(prism_test,axis=0) # [Nlat,Nlon] --> [Nmon,Nlat,Nlon]
#     #print('gcms_test.shape={}\nprism_test.shape={}'.format(gcms_test.shape,prism_test.shape))
#     #print('gcms_test.max={}\nprism_test.max={}\n'.format(np.amax(gcms_test),np.amax(prism_test)))

#     if is_precipitation:
#         gcms_test = np.expm1(gcms_test)
#         prism_test = np.expm1(prism_test)
#     else:
#         gcms_test = gcms_test*50.0
#         prism_test = prism_test*50.0
#     print('BCSD:\ngcms_test=[{},{}]\nprism_test=[{},{}]\npreds=[{},{}]\n\n'.format(np.amin(gcms_test),np.amax(gcms_test),
#           np.amin(prism_test),np.amax(prism_test),np.amin(preds),np.amax(preds)))
#     # =============================================================================
#     #     MSE_DJF = np.mean((prism_test[[-1,0,1,11,12,13,23,24,25],:,:]-preds[[-1,0,1,11,12,13,23,24,25],:,:])**2)
#     #     MSE_MAM = np.mean((prism_test[[2,3,4,14,15,16,26,27,28],:,:]-preds[[2,3,4,14,15,16,26,27,28],:,:])**2)
#     #     MSE_JJA = np.mean((prism_test[[5,6,7,17,18,19,29,30,31],:,:]-preds[[5,6,7,17,18,19,29,30,31],:,:])**2)
#     #     MSE_SON = np.mean((prism_test[[8,9,10,20,21,22,32,33,34],:,:]-preds[[8,9,10,20,21,22,32,33,34],:,:])**2)
#     #     print('MSE_DJF={}\nMSE_MAM={}\nMSE_JJA={}\nMSE_SON={}\n'.format(MSE_DJF,MSE_MAM,MSE_JJA,MSE_SON))
#     #     np.savez(predpath+'MSE_seasonal.npz',MSE_DJF=MSE_DJF,MSE_MAM=MSE_MAM,MSE_JJA=MSE_JJA,MSE_SON=MSE_SON)
#     # =============================================================================

#     bias = np.mean(preds-prism_test)
#     corr = np.corrcoef(preds.flatten(),prism_test.flatten())
#     print('bias={}'.format(bias))
#     print('corr={}'.format(corr))

# # =============================================================================
# #     bias_DJF = np.mean(prism_test[[-1,0,1,11,12,13,23,24,25],:,:]-preds[[-1,0,1,11,12,13,23,24,25],:,:])
# #     bias_MAM = np.mean(prism_test[[2,3,4,14,15,16,26,27,28],:,:]-preds[[2,3,4,14,15,16,26,27,28],:,:])
# #     bias_JJA = np.mean(prism_test[[5,6,7,17,18,19,29,30,31],:,:]-preds[[5,6,7,17,18,19,29,30,31],:,:])
# #     bias_SON = np.mean(prism_test[[8,9,10,20,21,22,32,33,34],:,:]-preds[[8,9,10,20,21,22,32,33,34],:,:])
# #     print('bias_DJF={}\nbias_MAM={}\nbias_JJA={}\nbias_SON={}\n'.format(bias_DJF,bias_MAM,bias_JJA,bias_SON))
# #     #np.savez(predpath+'bias_seasonal.npz',bias_DJF=bias_DJF,bias_MAM=bias_MAM,bias_JJA=bias_JJA,bias_SON=bias_SON)
# #     corr_DJF = np.corrcoef(prism_test[[-1,0,1,11,12,13,23,24,25],:,:]-preds[[-1,0,1,11,12,13,23,24,25],:,:])
# #     corr_MAM = np.corrcoef(prism_test[[2,3,4,14,15,16,26,27,28],:,:]-preds[[2,3,4,14,15,16,26,27,28],:,:])
# #     corr_JJA = np.corrcoef(prism_test[[5,6,7,17,18,19,29,30,31],:,:]-preds[[5,6,7,17,18,19,29,30,31],:,:])
# #     corr_SON = np.corrcoef(prism_test[[8,9,10,20,21,22,32,33,34],:,:]-preds[[8,9,10,20,21,22,32,33,34],:,:])
# #     print('corr_DJF={}\ncorr_MAM={}\ncorr_JJA={}\ncorr_SON={}\n'.format(corr_DJF,corr_MAM,corr_JJA,corr_SON))
# #     #np.savez(predpath+'corr_seasonal.npz',corr_DJF=corr_DJF,corr_MAM=corr_MAM,corr_JJA=corr_JJA,corr_SON=corr_SON)
# # =============================================================================

#     pred = preds[month,:,:]
#     pred[USAmask_HR==0] = np.nan
#     #pred[abs(pred)<=0.0001] = np.nan

#     #gcm = gcms_test[ngcm,month,:,:]
#     gcm = np.mean(gcms_test,axis=0)
#     gcm = gcm[month,:,:]
#     gcm[USAmask_LR==0] = np.nan
#     #gcm[abs(gcm)<=0.0001] = np.nan

#     prism = prism_test[month,:,:]
#     prism[USAmask_HR==0] = np.nan
#     #prism[abs(prism)<=0.0001] = np.nan

#     absdiff = abs(preds[month,:,:]-prism_test[month,:,:])
#     absdiff[USAmask_HR==0] = np.nan
#     #absdiff[absdiff<=0.0001] = np.nan

#     absdiff_avg = np.mean(abs(preds-prism_test),axis=0)
#     print('[absdiff_avg.min,absdiff_avg.max]=[{},{}]'.format(np.amin(absdiff_avg),np.amax(absdiff_avg)))
#     absdiff_avg[USAmask_HR==0] = np.nan
#     #absdiff_avg[absdiff_avg<=0.0001] = np.nan

#     diff_avg = np.mean(preds-prism_test,axis=0)
#     diff_avg[USAmask_HR==0] = np.nan
#     #diff_avg[diff_avg<=0.0001] = np.nan


#     prism_avg = np.mean(prism_test,axis=0)
#     prism_avg[USAmask_HR==0] = np.nan


#     # =============================================================================
#     #     winter = np.sum(prism_test[[-1,0,1,11,12,13,23,24,25],:,:],axis=0)
#     #     spring = np.sum(prism_test[[2,3,4,14,15,16,26,27,28],:,:],axis=0)
#     #     summer = np.sum(prism_test[[5,6,7,17,18,19,29,30,31],:,:],axis=0)
#     #     autumn = np.sum(prism_test[[8,9,10,20,21,22,32,33,34],:,:],axis=0)
#     #
#     #     winter[USAmask_HR==0] = np.nan
#     #     spring[USAmask_HR==0] = np.nan
#     #     summer[USAmask_HR==0] = np.nan
#     #     autumn[USAmask_HR==0] = np.nan
#     #
#     #
#     #     import matplotlib.pyplot as plt
#     #     fig = plt.figure()
#     #     plt.imshow(winter,cmap='YlOrRd')
#     #     plt.title('winter ppt')
#     #     plt.colorbar()
#     #     plt.show()
#     #
#     #     fig = plt.figure()
#     #     plt.imshow(spring,cmap='YlOrRd')
#     #     plt.title('spring ppt')
#     #     plt.colorbar()
#     #     plt.show()
#     #
#     #     fig = plt.figure()
#     #     plt.imshow(summer,cmap='YlOrRd')
#     #     plt.title('summer ppt')
#     #     plt.colorbar()
#     #     plt.show()
#     #
#     #     fig = plt.figure()
#     #     plt.imshow(autumn,cmap='YlOrRd')
#     #     plt.title('autumn ppt')
#     #     plt.colorbar()
#     #     plt.show()
#     # =============================================================================

#     #%% plot figures
#     img = np.flipud(prism)
#     title = 'ground truth {} '.format(variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
#     savename = 'groundtruth_{}_{}by{}_month{}'.format(variable,resolution,resolution,month+1)
#     plot_map(img,title=title,savepath=savepath,savename=savename)

# # =============================================================================
# #     img = np.flipud(gcm)
# #     #title = 'input GCM mean ppt '+str(1)+'$^{\circ}$x'+str(1)+'$^{\circ}$'
# #     title = 'input GCM mean {}'.format(variable)
# #     savename = 'input_gcm_mean_{}_month{}'.format(variable,month+1)
# #     plot_map(img,title=title,savepath=savepath,savename=savename)
# # =============================================================================

# # =============================================================================
# #     img = np.flipud(pred)
# #     title = '{} pred {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
# #     savename = '{}_pred_result_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
# #     plot_map(img,title=title,savepath=savepath,savename=savename)
# #     #plot_map(img,title=None,savepath=None,savename=None,cmap='YlOrRd',
# #     #             lonlat=[235,24.125,293.458,49.917],resolution='i',area_thresh=10000)
# #
# #     img = np.flipud(absdiff)
# #     title = '{} GT absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
# #     savename = '{}_abs_diff_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
# #     plot_map(img,title=title,savepath=savepath,savename=savename,clim=clim_diff)
# # =============================================================================

# # =============================================================================
# #     img = np.flipud(absdiff_avg)
# #     #title = '{} GT absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
# #     #savename = '{}_abs_diff_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
# #     title = '{} GT mean absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
# #     savename = '{}_avg_abs_diff_{}_{}by{}'.format(modelname,variable,resolution,resolution)
# #     plot_map(img,title=title,savepath=savepath,savename=savename,clim=clim_diff)
# #
# #     img = np.flipud(diff_avg)
# #     #title = '{} GT absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
# #     #savename = '{}_abs_diff_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
# #     title = '{} GT mean error {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
# #     savename = '{}_avg_diff_{}_{}by{}'.format(modelname,variable,resolution,resolution)
# #     plot_map(img,title=title,savepath=savepath,savename=savename,cmap='cool',clim=None)
# #
# #     img = np.flipud(prism_avg)
# #     #title = '{} GT absolute difference {} '.format(modelname,variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
# #     #savename = '{}_abs_diff_{}_{}by{}_month{}'.format(modelname,variable,resolution,resolution,month+1)
# #     title = 'GT mean {} over 3 years '.format(variable)+str(resolution)+'$^{\circ}$x'+str(resolution)+'$^{\circ}$'
# #     savename = 'GT_avg_{}_{}by{}_2'.format(variable,resolution,resolution)
# #     plot_map(img,title=title,savepath=savepath,savename=savename)
# # =============================================================================

# #plot_bcsd()




















#%%
#def plot_map(img,title=None,savepath=None,savename=None,cmap='YlOrRd',
#             lonlat=[235,24.125,293.458,49.917],resolution='i',area_thresh=10000):
#    import numpy as np
#    from mpl_toolkits.basemap import Basemap
#    #import matplotlib.pyplot as plt
#    #lonlat=[235,24.125,293.458,49.917]
#    #area_thresh=10000
#    lats = np.arange(20.0,51.0,5.0)
#    lons = np.arange(235.0,300.0,10.0)
#    fig = plt.figure()
#    m = Basemap(llcrnrlon=lonlat[0],llcrnrlat=lonlat[1],urcrnrlon=lonlat[2],urcrnrlat=lonlat[3],
#                projection='cyl',resolution='i',area_thresh=area_thresh)
#    m.drawcoastlines(linewidth=1.0)
#    m.drawcountries(linewidth=1.0)
#    m.drawstates()
#
#    m.drawparallels(lats,labels=[True,False,False,False],dashes=[1,2])
#    m.drawmeridians(lons,labels=[False,False,False,True],dashes=[1,2])
#    #m.imshow(np.flipud(np.sqrt(pred)),alpha=1.0)
#    m.imshow(img,cmap=cmap,alpha=1.0)
#    plt.colorbar(fraction=0.02)
#    plt.show()
#    if title:
#        plt.title(title)
#    if savepath and savename:
#        plt.savefig(savepath+savename+'.png',dpi=1200,bbox_inches='tight')

