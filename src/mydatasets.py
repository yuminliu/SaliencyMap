#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 21:10:29 2020

@author: wang.zife
"""

import torch
import numpy as np
from skimage.transform import resize
from torch.utils.data import Dataset

class myDataset(Dataset):
    def __init__(self,fold='train',datapath='../data/',river_name='amazon',tstep=12):
        self.fold = fold
        self.tstep = tstep
        # X: [1091,11,81],192701 to 201711, 1091 months
        # y: [1080,], 192801 to 201712, 1080 months
        X,y = self.__readdata__(datapath=datapath,river_name=river_name,tstep=tstep)
        X,y = torch.from_numpy(X).float(),torch.from_numpy(y).float()  
        ## separate train, valid and test
        Ntotal = len(y)
        Ntrain = int(Ntotal*0.8)
        Nvalid = int(Ntotal*0.1)
        #Ntest = Ntotal-Ntrain-Nvalid
        y_train,y_valid,y_test = y[:Ntrain],y[Ntrain:Ntrain+Nvalid],y[Ntrain+Nvalid:]
        X_train,X_valid,X_test = X[:Ntrain+tstep-1],X[Ntrain:Ntrain+Nvalid+tstep-1],X[Ntrain+Nvalid:]
        #y_train,y_valid,y_test = torch.from_numpy(y_train).float(),torch.from_numpy(y_valid).float(),torch.from_numpy(y_test).float()
        #X_train,X_valid,X_test = torch.from_numpy(X_train).float(),torch.from_numpy(X_valid).float(),torch.from_numpy(X_test).float()
        
        self.y = {'train':y_train,'valid':y_valid,'test':y_test} # [N,]
        self.X = {'train':X_train,'valid':X_valid,'test':X_test} # [N,Nlat,Nlon]
        
    def __getitem__(self, idx):
        y = self.y[self.fold][idx] # scalar?
        #X = torch.cat([self.sst[self.fold][idx:idx+self.tstep],self.lst[self.fold][idx:idx+self.tstep]],axis=0) # [2*step,W,H]
        X = self.X[self.fold][idx:idx+self.tstep]
        #print('X.shape={},y=shape={}'.format(X.shape,y.shape))
        return X, y

    def __len__(self):
        return len(self.y[self.fold])
    
    def __readdata__(self,datapath='../data/',river_name='amazon',tstep=12):
        durations = {'amazon':[335,1425], # [335,1425], 192712 to 201810, 1091 months non stop
                    'congo':[36,1331], # [36,1331], 190301 to 201012, 1296 months non stop
                    'changjiang':[0,533], # [0,533], [536], [552,1211], [1248,1259], 190001 to 194406, 194409, 194601 to 200012,200401 to 200412, 1207 months stop
                    'parana':[60,1395], # [60,1395], 190501 to 201604, 1336 months non stop
                    'ganges':[588,1163], # [588,1163], 194901 to 199612, 576 months non stop
                    'nile':[876,1019], # [876,1019], 197301 to 198412, 144 months non stop
                    'pearl':[465,1424] # [465,1424], 193810 to 201809, 960 months non stop
                     }
        ## read in data
        y = np.load(datapath+'RiverFlow/processed/'+river_name+'.npy') # time from 190001 to 201812, totally 1428 months
        sst = np.load(datapath+'NOAA/ExtendedReconstructedSSTv4/processed/noaa_extendedresconstructed_2degree_sst.mnmean.npz')['sst'] # time from 185401 to 202002, totally 1994 months
        ind_offset_sst = 552 # ind at 190001 
        lst = np.load(datapath+'UoD/processed/uod_0.5degree_lst_global_air.mon.mean.npz')['lst'] # time from 190001 to 201712, totally 1416 months
        lst = np.transpose(resize(np.transpose(lst,axes=(1,2,0)),sst.shape[1:],order=1,preserve_range=True),axes=(2,0,1))
        #times_y = np.load(datapath+'RiverFlow/processed/times.npy') # time from 190001 to 201812, totally 1428 months
        #times_sst = np.load(datapath+'NOAA/ExtendedReconstructedSSTv4/processed/noaa_extendedresconstructed_2degree_sst.mnmean.npz')['times'] # time from 185401 to 202002, totally 1994 months
        #times_sst = np.array([int(''.join(t.split('-'))) for t in times_sst])
        ##sst_lats = np.load(datapath+'NOAA/ExtendedReconstructedSSTv4/processed/noaa_extendedresconstructed_2degree_sst.mnmean.npz')['lats'] 
        ##sst_lons = np.load(datapath+'NOAA/ExtendedReconstructedSSTv4/processed/noaa_extendedresconstructed_2degree_sst.mnmean.npz')['lons'] 
        #times_lst = np.load(datapath+'UoD/processed/uod_0.5degree_lst_global_air.mon.mean.npz')['times'] # time from 190001 to 201712, totally 1416 months
        #times_lst = np.array([int(''.join(t.split('-'))) for t in times_lst])
        
        ## get 3 month running mean
        sst_avg = np.zeros(sst.shape) # three months running mean
        for i in range(1,len(sst_avg)-1):
            sst_avg[i,:,:] = np.mean(sst[i-1:i+2,:,:],axis=0)
        lst_avg = np.zeros(lst.shape) # three months running mean
        for i in range(1,len(lst_avg)-1):
            lst_avg[i,:,:] = np.mean(lst[i-1:i+2,:,:],axis=0)
            
        ## get valid period
        start,end = durations[river_name]
        if river_name=='amazon': end = 1415+1 # end at 201712
        y = y[start+1:end] # 192801 to 201712, 1080 months
        sst_avg = sst_avg[ind_offset_sst+(start+1)-tstep:ind_offset_sst+end-1,:,:] # 192701 to 201711, 1091 months, end-start+tstep months
        lst_avg = lst_avg[(start+1)-tstep:end-1,:,:] # 192701 to 201711, 1091 months, end-start+tstep months
        
        #times_y = times_y[start+1:end] # 192801 to 201712, 1080 months
        #times_sst_avg = times_sst[ind_offset_sst+(start+1)-tstep:ind_offset_sst+end-1] # 192701 to 201711, 1091 months, end-start+tstep months
        #times_lst_avg = times_lst[(start+1)-tstep:end-1] # 192701 to 201711, 1091 months, end-start+tstep months
        
        ## get nino regions
        sst_avg = sst_avg[:,39:50,60:141] # 10N to -10N, 120E to 280E
        lst_avg = lst_avg[:,39:50,60:141] # 10N to -10N, 120E to 280E
        ## mask out ocean
        lst_avg[sst_avg!=0] = 0

        ## make the values near 0
        y = y/300000.0 # [76394.0,287332.5]-->[0,1]
        sst_avg = (sst_avg/50.0+1.0)/2 # [0.0,30.91]-->[0,1]
        lst_avg = (lst_avg/50.0+1.0)/2 # [0.0,26.89]-->[0,1]  
        
        X = sst_avg+lst_avg # [Nmon,Nlat,Nlon]==[1091,11,81]
        y = y.reshape((-1,1)) # [Nmon,]-->[Nmon,1]==[1080,1]
        
        return X,y
    
    
    
#%%
class ARDataset(Dataset):
    def __init__(self,X,y,fold='train',Ntrain=50,Nvalid=5):
        y = y.reshape((-1,1))
        X_train, y_train = X[0:Ntrain],y[0:Ntrain]
        X_valid, y_valid = X[Ntrain:Ntrain+Nvalid],y[Ntrain:Ntrain+Nvalid]
        X_test,y_test = X[Ntrain+Nvalid:],y[Ntrain+Nvalid:]
        if fold=='train':
            self.X = X_train
            self.Y = y_train
        elif fold=='valid':
            self.X = X_valid
            self.Y = y_valid
        elif fold=='test':
            self.X = X_test
            self.Y = y_test
        #self.Y /= np.max(np.abs(y_train)) # normalized to [-1,1]
        self.X, self.Y = torch.from_numpy(self.X).float(),torch.from_numpy(self.Y).float() 
        
    def __getitem__(self,idx):
        X = self.X[idx]
        y = self.Y[idx]
        return X,y
    
    def __len__(self):
        return len(self.Y)