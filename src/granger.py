#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 14:15:59 2020

@author: wang.zife
"""

#%%
import os
import json
import numpy as np
import pandas as pd
import utils

#mode_ind = 0 # 1 #0
#station_ind = 1
def main(lag,window):
    verbose = True
    #lag = 6*12 # 4*12 # AR lag months
    #window = 3 # 1 # time window for moving average, 1 means no average
    num_min_years = 40+lag//12 # 60+4 # number of min years to be considered
    min_volumes = 100 # 9 # minimum of yearly volume (km3/yr) to be considered
    Nvalid = 10 # number of valid data
    Ntest = 20 # 10 # number of test data
    scaleback = False # True # scale back results or not
    datapath = '../data/'
    modes = ['ARonly','ARenso','ENSO']
    #mode = modes[mode_ind] # 'ARenso' # 'ARonly' # 'ENSO' #
    #columns = ['Nino12','Nino3','Nino34','Nino4','Nino12_anom','Nino3_anom','Nino34_anom','Nino4_anom','tni','soi']
    column = ['Nino12','Nino3','Nino4'] # ['Nino34_anom'] # 6 is 'Nino34_anom'
    savepath_root = '../results/analysis/Regression/riversAR/MovingAveraged/window_{}/'.format(window) # '../results/analysis/Regression/riversAR/MovingAveraged/''../results/analysis/Regression/riversAR/' #
    #savepath = '../results/analysis/Regression/riversAR/station_ind_{}/{}/'.format(station_ind,mode)
    #txtsavepath = '../results/analysis/Regression/riversAR/ARlag_{}/'.format(lag//12)
    txtsavepath = savepath_root+'ARlag_{}/'.format(lag//12)
    mainparas = {}
    mainparas['num_min_years'] = num_min_years
    mainparas['min_volumes'] = min_volumes
    mainparas['lag'] = int(lag)
    mainparas['feature'] = ' '.join(column) #str(column[0])
    mainparas['Nvalid'] = int(Nvalid)
    mainparas['Ntest'] = int(Ntest)
    mainparas['scaleback'] = scaleback
    mainparas['txtsavepath'] = str(txtsavepath)
    mainparas['verbose'] = verbose
    #%% read river flow
    riverflow_df = pd.read_csv('../data/RiverFlow/processed/riverflow.csv',index_col=0,header=0)
    info_df = pd.read_csv('../data/RiverFlow/processed/info.csv',index_col=0,header=0)
    times_df = pd.read_csv('../data/RiverFlow/processed/times.csv',index_col=0,header=0)
    #times = np.asarray(times_df,dtype=int).reshape((-1,))
    times = list(np.asarray(times_df,dtype=int).reshape((-1,)))+[201901]
    #%% read ENSO index
    indices_df, _ = utils.read_enso() # 187001 to 201912
    ##indices_df.to_csv('../data/Nino/processed/Nino_ENSO_187001-201912_df.csv')

    ##riverflow_df = pd.read_csv('../data/Denoised/denoised_riverflow.csv',index_col=0,header=0)
    ##indices_df = pd.read_csv('../data/Denoised/denoised_Nino_ENSO_187001-201912_df.csv',index_col=0)

    indices_df = indices_df[column] # select input feature

    #large_rivers = info_df.loc[(info_df['station_volume_km3/yr']>100)&(info_df['num_month']>528)]
    #mode = 'ARonly'
    #station_ind = 0
    for mode in modes[0:2]:#modes[:-1]:
        mainparas['mode'] = str(mode)
        for station_ind in [13]: #range(925):
            #if station_ind==13 or station_ind==21 or station_ind==26: continue
            ## determine start and stop yearmonth for data
            rivername = info_df['river_name'].iloc[station_ind]
            start_month_ind = info_df['start_month_ind'].iloc[station_ind] # data including start_month_ind
            stop_month_ind = info_df['stop_month_ind'].iloc[station_ind] # data not including stop_month_ind
            ## start from January and stop at December
            while times[start_month_ind]%100!=1:
                start_month_ind += 1
            while times[stop_month_ind]%100!=1:
                stop_month_ind -= 1
            assert (stop_month_ind-start_month_ind)%12==0, 'number of months must be a multiplier of 12'
            if stop_month_ind-start_month_ind<num_min_years*12:
                print('Error! too few data points to be considered!')
                print('Skipping {}-th river {}'.format(station_ind,rivername))
                continue
            #else:
                #print('Processing {}-th river {}'.format(station_ind,rivername))

            volumes = info_df['station_volume_km3/yr'].iloc[station_ind]
            if volumes<min_volumes:
                print('Error! too small volume to be considered!')
                print('Skipping {}-th river {}'.format(station_ind,rivername))
                continue
            else:
                print('Processing {}-th river {}'.format(station_ind,rivername))

            mainparas['station_ind'] = int(station_ind)
            mainparas['rivername'] = str(rivername)
            lag_start_index,start_index,end_index = times[start_month_ind-lag],times[start_month_ind],times[stop_month_ind-1] # end_index included
            mainparas['lag_start_index'],mainparas['start_index'],mainparas['end_index'] = int(lag_start_index),int(start_index),int(end_index)

            riverflow = riverflow_df[str(station_ind)].loc[start_index:end_index]
            riverflow = np.asarray(riverflow,dtype=float).reshape((-1,12))
            #riverflow = riverflow.reshape((-1,12))
            flow_year = np.sum(riverflow,axis=1) # yearly flow, unit: m^3/s
            flow_year = flow_year*30*24*3600/1e9 # unit: km^3/year

            ## moving average
            if window>1:
                flow_year = np.asarray([np.mean(flow_year[i:i+window]) for i in range(len(flow_year)-window+1)])

            flow_year_anom = flow_year-np.mean(flow_year[0:30]) # anomaly based on first 30 years, unit: km^3/year

            indice = indices_df.loc[start_index:end_index]
            #enso = indice.to_numpy().reshape((-1,12))
            #enso_annual = np.mean(enso,axis=1) # mean over a year
            enso = indice.to_numpy() # [N,D]
            enso_annual = np.asarray([np.mean(enso[i:i+12],axis=0) for i in range(0,len(enso),12)]) # mean over a year

            ## moving average
            if window>1:
                #enso_annual = np.asarray([np.mean(enso_annual[i:i+window],axis=0) for i in range(len(enso_annual)-window+1)])
                enso_annual = enso_annual[window-1:]

            #start_year, stop_year = start_index//100+4, end_index//100+1 # stop_year not included
            start_year, stop_year = start_index//100+lag//12, end_index//100+1 # stop_year not included

            ## shorter years after moving average
            start_year = start_year+window-1

            Ntotal = stop_year-start_year # number of total years available, at least 60 years
            Ntrain = Ntotal-Nvalid-Ntest
            mainparas['Ntotal'] = int(Ntotal)
            mainparas['Ntrain'] = int(Ntrain)
            years_test_start,years_test_stop = stop_year-Ntest,stop_year
            years_all_start,years_all_stop = start_year,stop_year
            year_mid = years_test_start-0.5
            mainparas['years_all_start'],mainparas['years_test_start'],mainparas['years_all_stop'] = int(years_all_start),int(years_test_start),int(years_all_stop)
            mainparas['years_test_stop'] = int(years_test_stop)
            mainparas['year_mid'] = float(year_mid)
            ## separate data
            (X_train,y_train), (X_valid,y_valid), (X_test,y_test) = utils.separatedata(mode,flow_year_anom,enso_annual,lag=lag//12,Ntrain=Ntrain,Nvalid=Nvalid,Ntest=Ntest)
            ## scale data
            (X_train,y_train), (X_valid,y_valid), (X_test,y_test), (X_scaler,y_scaler) = utils.scaledata(X_train=X_train,y_train=y_train, X_valid=X_valid,y_valid=y_valid, X_test=X_test,y_test=y_test,scalemethod='StandardScaler')
            X_scaler_mean = X_scaler.mean_
            X_scaler_std = np.sqrt(X_scaler.var_)
            y_scaler_mean = y_scaler.mean_
            y_scaler_std = np.sqrt(y_scaler.var_)
            mainparas['X_scaler_mean'] = list(X_scaler_mean)
            mainparas['X_scaler_std'] = list(X_scaler_std)
            mainparas['y_scaler_mean'] = list(y_scaler_mean)
            mainparas['y_scaler_std'] = list(y_scaler_std)

            ##
            #savepath = '../results/analysis/Regression/riversAR/ARlag_{}/stations/station_ind_{}/{}/'.format(lag//12,station_ind,mode)
            savepath = savepath_root+'ARlag_{}/stations/station_ind_{}/{}/'.format(lag//12,station_ind,mode)
            mainparas['savepath'] = savepath
            if not os.path.exists(savepath): os.makedirs(savepath)

            import regressions
            hyparas,res = regressions.regressions(X_train,y_train, X_valid,y_valid, X_test,y_test, savepath, mainparas)

            configs = {**mainparas,**hyparas,**res}
            with open(savepath+'configs.txt', 'w') as file:
                 file.write(json.dumps(configs,indent=0)) # use `json.loads` to do the reverse

            if verbose:
                import myplots
                myplots.plot_test_all_data(station_ind,savepath,mainparas)


for lag in [9]:
    lag *= 12
    for window in [1]:
        main(lag=lag,window=window)

