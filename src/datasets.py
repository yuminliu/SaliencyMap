import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
#from skimage.transform import resize

#%%
# class myDataset(Dataset):
#     def __init__(self, datapath):
#         self.datapaths = sorted(glob.glob(datapath + '*.npz'))       

#     def __getitem__(self, idx):
#         data = np.load(self.datapaths[idx])
#         dataX = data['gcms'] # [Ngcm,Nlat,Nlon], ppt [0.0,1.0], tmax/tmin [-1.0,1.0]
#         datay = data['prism'] # [1,Nlat,Nlon] ppt [0.0,1.0]
#         X = torch.from_numpy(dataX).float() # [Ngcm,Nlat,Nlon], ppt [0.0,1.0], tmax/tmin [-1.0,1.0]
#         X2 = resize(np.transpose(dataX,axes=(1,2,0)),datay[0,:,:].shape,order=1,preserve_range=True) # [Nlat,Nlon,Ngcm]
#         X2 = np.transpose(X2,axes=(2,0,1))# [Ngcm,Nlat,Nlon]
#         X2 = torch.from_numpy(X2).float() # [Ngcm,Nlat,Nlon]
#         Xaux = np.concatenate((data['climatology'],data['elevation']),axis=0) # [2,Nlat,Nlon],ppt [0.0,1.0], tmax/tmin [-1.0,1.0]
#         Xaux = torch.from_numpy(Xaux).float()
#         X = [X,X2,Xaux]
        
#         y = torch.from_numpy(datay).float() #[1,Nlat,Nlon], ppt [0.0,1.0], tmax/tmin [-1.0,1.0]
        
#         return X, y

#     def __len__(self):
#         return len(self.datapaths)

#%%
# import pandas as pd
# import utils 
# def readin_riverflow_enso(window=3):   
#     #window = 3
#     column = ['Nino3'] # ['Nino12','Nino3','Nino4'] # ['Nino34_anom'] # 6 is 'Nino34_anom'
#     riverflow_df = pd.read_csv('../data/Climate/RiverFlow/processed/riverflow.csv',index_col=0,header=0)
#     #info_df = pd.read_csv('../data/Climate/RiverFlow/processed/info.csv',index_col=0,header=0)
#     #times_df = pd.read_csv('../data/Climate/RiverFlow/processed/times.csv',index_col=0,header=0)
#     #times = np.asarray(times_df,dtype=int).reshape((-1,))
#     #times = list(np.asarray(times_df,dtype=int).reshape((-1,)))+[201901]
#     #%% read ENSO index
#     indices_df, _ = utils.read_enso() # 187001 to 201912
#     indices_df = indices_df[column] # select input feature
#     #amazon = riverflow_df[['0']].loc[195001:200512].to_numpy().reshape((-1,))
#     amazon = riverflow_df[['0']].iloc[600-window+1:1272].to_numpy().reshape((-1,)) # from 1950-windown to 200512
#     amazon = np.array([np.mean(amazon[i:i+window]) for i in range(len(amazon)-window+1)]).reshape((-1,1))# moving average, result in 195001 to200512
#     #amazons = {}
#     #for window in range(1,7):
#     #    amazon = riverflow_df[['0']].iloc[600-window+1:1272].to_numpy().reshape((-1,))
#     #    amazon = [np.mean(amazon[i:i+window]) for i in range(len(amazon)-window+1)]# moving average
#     #    amazons[window] = amazon
#     nino3 = indices_df[['Nino3']].loc[195001:200512].to_numpy().reshape((-1,1))
    
#     return amazon,nino3
    


#%% CNN datasets
class myDataset_CNN(Dataset):
    def __init__(self,predictor='',predictand='',imgpath='',sst=True,fold='train',window=3,noise_std=0.0,ngcm=0):
        self.predictor = predictor
        self.predictand = predictand
        self.imgpath = imgpath
        self.sst = sst
        self.fold = fold
        #Ntotal = 672 # len(y)
        self.Ntrain = 600 # int(Ntotal*0.8)
        self.Nvalid = 36 # int(Ntotal*0.1)
        self.noise_std = noise_std
        self.ngcm = ngcm
        #Ntest = Ntotal-Ntrain-Nvalid
        # X: [672,32,80,300],195001 to 200512, 672 months
        # y: [672,], 195001 to 200512, 672 months
        X,y = self.__readdata__(imgpath=self.imgpath,window=window)
        X,y = torch.from_numpy(X).float(),torch.from_numpy(y).float()
        ## separate train, valid and test

        y_train,y_valid,y_test = y[:self.Ntrain],y[self.Ntrain:self.Ntrain+self.Nvalid],y[self.Ntrain+self.Nvalid:]
        X_train,X_valid,X_test = X[:self.Ntrain],X[self.Ntrain:self.Ntrain+self.Nvalid],X[self.Ntrain+self.Nvalid:]
        
        self.y = {'train':y_train,'valid':y_valid,'test':y_test,'all':y} # [N,]
        self.X = {'train':X_train,'valid':X_valid,'test':X_test,'all':X} # [N,Nlat,Nlon]
        
    def __getitem__(self, idx):
        y = self.y[self.fold][idx] # scalar?
        X = self.X[self.fold][idx]
        
        return X, y

    def __len__(self):
        return len(self.y[self.fold])
    
    def __readdata__(self,imgpath='../data/GCM/GCMdata/tas/processeddata/32gcms_tas_monthly_1by1_195001-200512_World.npy',
                     riverpath='../data/Climate/RiverFlow/processed/riverflow.csv',column = ['Nino3'],window=3):
        #%% tas
        if 'GCM' in self.predictor: # self.predictor=='GCM':
            #imgpath = '../data/Climate/GCM/GCMdata/tas/processeddata/32gcms_tas_monthly_1by1_195001-200512_World.npy'
            left,right = 50,350
            top,bottom = 50,130 # region 1
            #top,bottom = 67,109 #[20.5,-20.5], region 4
            #lats = np.load('../data/Climate/GCM/GCMdata/tas/processeddata/lats_gcm.npy')
            #lons = np.load('../data/Climate/GCM/GCMdata/tas/processeddata/lons_gcm.npy')
            data = np.load(imgpath) # 87.5N to -87.5N by 1, 
            if self.ngcm=='all':
                data = data[:,:,top:bottom,left:right] # 50.5E to 349.5E, 37.5N to -41.5N, # [672,32,80,300]
            elif isinstance(self.ngcm,int):
                data = data[:,self.ngcm:self.ngcm+1,top:bottom,left:right] # 50.5E to 349.5E, 37.5N to -41.5N, # one GCM [672,1,80,300]
            ## mask out land
            if self.sst:
                cobe = np.load('../data/Climate/COBE/processed/sst.mon.mean_185001-201912_1by1.npy')
                mask = np.mean(cobe,axis=0)
                mask = mask[top+2:bottom+2,left:right]
                mask = np.nan_to_num(mask,nan=1000)
                for mon in range(len(data)):
                    for gcm in range(data.shape[1]):
                        data[mon,gcm,:,:][mask==1000] = 0
        #%% combined SST region 1
        elif 'Reanalysis' in self.predictor: # self.predictor=='Reanalysis': #'sst_masked':
            #imgpath = '../data/Climate/Reanalysis/sst_cobe_hadley_noaa_uod_195001-200512_1by1_world.npy'
            left,right = 50,350
            top,bottom = 50,130 # 52,132 ##
            #top,bottom = 69,111 #[20.5,-20.5], region 4
            # lats = np.load('../data/Climate/Reanalysis/lats.npy')
            # lons = np.load('../data/Climate/Reanalysis/lons.npy')
            data = np.load(imgpath) # 89.5N to -89.5N by 1, 
            data = data[:,0:3,2:-2,:] # 87.5N to -87.5N by 1, only SST, exclude uod 
            if self.ngcm=='all':
                #data = data[:,0:3,top:bottom,left:right] # 50.5E to 349.5E, 37.5N to -41.5N, [672,3,80,300], exclude uod  
                data = data[:,:,top:bottom,left:right] # 50.5E to 349.5E, 37.5N to -41.5N, [672,4,80,300]  
            elif isinstance(self.ngcm,int):
                data = data[:,self.ngcm:self.ngcm+1,top:bottom,left:right] # 50.5E to 349.5E, 37.5N to -41.5N, [672,1,80,300], one reanalysis each
            ## mask out land
            if self.sst:
                cobe = np.load('../data/Climate/COBE/processed/sst.mon.mean_185001-201912_1by1.npy')
                mask = np.mean(cobe,axis=0)
                mask = mask[top+2:bottom+2,left:right]
                mask = np.nan_to_num(mask,nan=1000)
                for mon in range(len(data)):
                    for d in range(data.shape[1]):
                        data[mon,d,:,:][mask==1000] = 0      
            
        #%% combined SST region 2
        # imgpath = '../data/Climate/SST/sst_cobe_hadley_noaa_uod_195001-200512_1by1_world.npy'
        # left,right = 50,260
        # top,bottom = 69,111
        # lats = np.load('../data/Climate/SST/lats.npy')
        # lons = np.load('../data/Climate/SST/lons.npy')
        # data = np.load(imgpath) # 89.5N to -89.5N by 1, 
        # data = data[:,0:-1,top:bottom,left:right] # 50.5E to 259.5E, 20.5N to -20.5N, [672,3,42,210], exclude uod 
        # #data = data[:,:,top:bottom,left:right] # 50.5E to 259.5E, 20.5N to -20.5N, [672,4,42,210] 
        
        #%% combined SST region 3
        # imgpath = '../data/Climate/SST/sst_cobe_hadley_noaa_uod_195001-200512_1by1_world.npy'
        # left,right = 150,260
        # top,bottom = 60,130
        # #lats = np.load('../data/Climate/SST/lats.npy')
        # #lons = np.load('../data/Climate/SST/lons.npy')
        # data = np.load(imgpath) # 89.5N to -89.5N by 1, 
        # data = data[:,0:-1,top:bottom,left:right] # 150.5E to 259.5E, 29.5N to -39.5N, [672,3,70,110], exclude uod 
        # #data = data[:,:,top:bottom,left:right] # 150.5E to 259.5E, 29.5N to -39.5N, [672,4,70,110] 
        
        #%% amazon river flow 
        if self.predictand=='Amazon':
            # window = 3
            riverpath = '../data/Climate/RiverFlow/processed/riverflow.csv'
            riverflow_df = pd.read_csv(riverpath,index_col=0,header=0)
            ##amazon = riverflow_df[['0']].loc[195001:200512].to_numpy().reshape((-1,))
            amazon = riverflow_df[['0']].iloc[600-window+1:1272].to_numpy().reshape((-1,)) # from 1950-window to 200512
            ## moving average, result in 195001 to200512
            targets = np.array([np.mean(amazon[i:i+window]) for i in range(len(amazon)-window+1)]).reshape((-1,1))
        
        #%% Congo river flow 
        elif self.predictand=='Congo':
            # window = 3
            riverpath = '../data/Climate/RiverFlow/processed/riverflow.csv'
            riverflow_df = pd.read_csv(riverpath,index_col=0,header=0)
            congo = riverflow_df[['1']].iloc[600-window+1:1272].to_numpy().reshape((-1,)) # from 1950-window to 200512
            ## moving average, result in 195001 to200512
            targets = np.array([np.mean(congo[i:i+window]) for i in range(len(congo)-window+1)]).reshape((-1,1))        
        
        #%% Columbia
        # # window = 3
        # riverpath = '../data/Climate/RiverFlow/processed/riverflow.csv'
        # riverflow_df = pd.read_csv(riverpath,index_col=0,header=0)
        # columbia = riverflow_df[['20']].iloc[600-window+1:1272].to_numpy().reshape((-1,)) # from 1950-window to 200512
        # #print('columbia[:20]={}'.format(columbia[:20]))
        # ## moving average, result in 195001 to200512
        # targets = np.array([np.mean(columbia[i:i+window]) for i in range(len(columbia)-window+1)]).reshape((-1,1))
        # #print('targets[:20]={}'.format(targets[:20]))
        

        #%% read ENSO index
#        ##column = ['Nino34_anom'] # ['Nino34'] # ['Nino3'] # ['Nino12','Nino3','Nino4'] #  6 is 'Nino34_anom'
#        indices_df, _ = utils.read_enso() # 187001 to 201912
#        # #indices_df = indices_df[column] # select input feature
#       
#        # nino3 = indices_df[['Nino3']].iloc[960-window+1:1632].to_numpy().reshape((-1,)) # from 1950-windown to 200512
#        # # moving average, result in 195001 to200512
#        # nino3 = np.array([np.mean(nino3[i:i+window]) for i in range(len(nino3)-window+1)]).reshape((-1,1))
#       
#        nino34 = indices_df[['Nino34']].iloc[960-window+1:1632].to_numpy().reshape((-1,)) # from 1950-windown to 200512
#        # moving average, result in 195001 to200512
#        nino34 = np.array([np.mean(nino34[i:i+window]) for i in range(len(nino34)-window+1)]).reshape((-1,1))
#       
#        targets = indices_df[['Nino34_anom']].iloc[960-window+1:1632].to_numpy().reshape((-1,)) # from 1950-windown to 200512
#        # moving average, result in 195001 to200512
#        targets = np.array([np.mean(targets[i:i+window]) for i in range(len(targets)-window+1)]).reshape((-1,1))

        #%% Nino3: average between area in 5N-5S, 150W-90W
        #targets = np.mean(np.load(imgpath)[:,:,82:94,210:270],axis=(1,2,3)).reshape((-1,1))
        
        #%% GCMNino34: average between area in 5N-5S, 170W-120W
        # gcmpath = '../data/Climate/GCM/GCMdata/tas/processeddata/32gcms_tas_monthly_1by1_195001-200512_World.npy'
        # #targets = np.mean(np.load(gcmpath)[:,:,82:94,190:240],axis=(1,2,3)).reshape((-1,1))
        # #### SingleGCMNino34
        # targets = np.mean(np.load(gcmpath)[:,2:3,82:94,190:240],axis=(1,2,3)).reshape((-1,1))
        # #lons = np.load('../data/Climate/GCM/GCMdata/tas/processeddata/lons_gcm.npy')
        # #print('lons[190]={},lons[240]={}'.format(lons[190],lons[240]))
        # #lats = np.load('../data/Climate/GCM/GCMdata/tas/processeddata/lats_gcm.npy')
        # #print('lats[82]={},lats[94]={}'.format(lats[82],lats[94]))
        
        #%% India Dipole Mode Index (DMI)
        # dmi_df = pd.read_csv('../data/Climate//DMI/processed/DMI_Standard_PSL_Format_187001-202012_series.csv',index_col=0)
        # dmi = dmi_df.iloc[960-window+1:1632].to_numpy().reshape((-1,)) # from 1950-windown to 200512
        # # moving average, result in 195001 to200512
        # dmi = np.array([np.mean(dmi[i:i+window]) for i in range(len(dmi)-window+1)]).reshape((-1,1))

        # import matplotlib.pyplot as plt
        # fig = plt.figure(figsize=(12,5))
        # plt.plot(congo,label='Nino34')
        # #plt.plot(targets,label='Nino34_anom')
        # #plt.savefig('../data/Climate/Nino/processed/nino34_vs_nino34_anom.png',dpi=1200,bbox_inches='tight')
        # plt.show()





        X = data
        y = targets # dmi # nino34_anom # nino34 # amazon # congo # nino3 # 

        ## standardize
        self.data_mean = np.mean(X[:self.Ntrain],axis=0)
        self.data_std = np.std(X[:self.Ntrain],axis=0)+1e-5
        for t in range(len(X)):
            X[t,:,:,:] = (X[t,:,:,:]-self.data_mean)/self.data_std
        self.y_mean = np.mean(y[:self.Ntrain],axis=0)
        self.y_std = np.std(y[:self.Ntrain],axis=0)
        y = (y-self.y_mean)/self.y_std
        
        ## add noise to input
        #noise = np.random.normal(loc=0.0,scale=self.noise_std,size=X.shape)
        #X += noise
        
        return X, y



#%%
# class myDataset_CNN_(Dataset):
#     def __init__(self,fold='train',datapath='../data/',river_name='amazon',tstep=12):
#         self.fold = fold
#         self.tstep = tstep
#         # X: [1091,11,81],192701 to 201711, 1091 months
#         # y: [1080,], 192801 to 201712, 1080 months
#         X,y = self.__readdata__(datapath=datapath,river_name=river_name,tstep=tstep)
#         X,y = torch.from_numpy(X).float(),torch.from_numpy(y).float()  
#         ## separate train, valid and test
#         Ntotal = len(y)
#         Ntrain = int(Ntotal*0.8)
#         Nvalid = int(Ntotal*0.1)
#         #Ntest = Ntotal-Ntrain-Nvalid
#         y_train,y_valid,y_test = y[:Ntrain],y[Ntrain:Ntrain+Nvalid],y[Ntrain+Nvalid:]
#         X_train,X_valid,X_test = X[:Ntrain+tstep-1],X[Ntrain:Ntrain+Nvalid+tstep-1],X[Ntrain+Nvalid:]
#         #y_train,y_valid,y_test = torch.from_numpy(y_train).float(),torch.from_numpy(y_valid).float(),torch.from_numpy(y_test).float()
#         #X_train,X_valid,X_test = torch.from_numpy(X_train).float(),torch.from_numpy(X_valid).float(),torch.from_numpy(X_test).float()
        
#         self.y = {'train':y_train,'valid':y_valid,'test':y_test} # [N,]
#         self.X = {'train':X_train,'valid':X_valid,'test':X_test} # [N,Nlat,Nlon]
        
#     def __getitem__(self, idx):
#         y = self.y[self.fold][idx] # scalar?
#         #X = torch.cat([self.sst[self.fold][idx:idx+self.tstep],self.lst[self.fold][idx:idx+self.tstep]],axis=0) # [2*step,W,H]
#         X = self.X[self.fold][idx:idx+self.tstep]
#         #print('X.shape={},y=shape={}'.format(X.shape,y.shape))
#         return X, y

#     def __len__(self):
#         return len(self.y[self.fold])
    
#     def __readdata__(self,datapath='../data/',river_name='amazon',tstep=12):
#         durations = {'amazon':[335,1425], # [335,1425], 192712 to 201810, 1091 months non stop
#                     'congo':[36,1331], # [36,1331], 190301 to 201012, 1296 months non stop
#                     'changjiang':[0,533], # [0,533], [536], [552,1211], [1248,1259], 190001 to 194406, 194409, 194601 to 200012,200401 to 200412, 1207 months stop
#                     'parana':[60,1395], # [60,1395], 190501 to 201604, 1336 months non stop
#                     'ganges':[588,1163], # [588,1163], 194901 to 199612, 576 months non stop
#                     'nile':[876,1019], # [876,1019], 197301 to 198412, 144 months non stop
#                     'pearl':[465,1424] # [465,1424], 193810 to 201809, 960 months non stop
#                      }
#         ## read in data
#         y = np.load(datapath+'RiverFlow/processed/'+river_name+'.npy') # time from 190001 to 201812, totally 1428 months
#         sst = np.load(datapath+'NOAA/ExtendedReconstructedSSTv4/processed/noaa_extendedresconstructed_2degree_sst.mnmean.npz')['sst'] # time from 185401 to 202002, totally 1994 months
#         ind_offset_sst = 552 # ind at 190001 
#         lst = np.load(datapath+'UoD/processed/uod_0.5degree_lst_global_air.mon.mean.npz')['lst'] # time from 190001 to 201712, totally 1416 months
#         lst = np.transpose(resize(np.transpose(lst,axes=(1,2,0)),sst.shape[1:],order=1,preserve_range=True),axes=(2,0,1))
#         #times_y = np.load(datapath+'RiverFlow/processed/times.npy') # time from 190001 to 201812, totally 1428 months
#         #times_sst = np.load(datapath+'NOAA/ExtendedReconstructedSSTv4/processed/noaa_extendedresconstructed_2degree_sst.mnmean.npz')['times'] # time from 185401 to 202002, totally 1994 months
#         #times_sst = np.array([int(''.join(t.split('-'))) for t in times_sst])
#         ##sst_lats = np.load(datapath+'NOAA/ExtendedReconstructedSSTv4/processed/noaa_extendedresconstructed_2degree_sst.mnmean.npz')['lats'] 
#         ##sst_lons = np.load(datapath+'NOAA/ExtendedReconstructedSSTv4/processed/noaa_extendedresconstructed_2degree_sst.mnmean.npz')['lons'] 
#         #times_lst = np.load(datapath+'UoD/processed/uod_0.5degree_lst_global_air.mon.mean.npz')['times'] # time from 190001 to 201712, totally 1416 months
#         #times_lst = np.array([int(''.join(t.split('-'))) for t in times_lst])
        
#         ## get 3 month running mean
#         sst_avg = np.zeros(sst.shape) # three months running mean
#         for i in range(1,len(sst_avg)-1):
#             sst_avg[i,:,:] = np.mean(sst[i-1:i+2,:,:],axis=0)
#         lst_avg = np.zeros(lst.shape) # three months running mean
#         for i in range(1,len(lst_avg)-1):
#             lst_avg[i,:,:] = np.mean(lst[i-1:i+2,:,:],axis=0)
            
#         ## get valid period
#         start,end = durations[river_name]
#         if river_name=='amazon': end = 1415+1 # end at 201712
#         y = y[start+1:end] # 192801 to 201712, 1080 months
#         sst_avg = sst_avg[ind_offset_sst+(start+1)-tstep:ind_offset_sst+end-1,:,:] # 192701 to 201711, 1091 months, end-start+tstep months
#         lst_avg = lst_avg[(start+1)-tstep:end-1,:,:] # 192701 to 201711, 1091 months, end-start+tstep months
        
#         #times_y = times_y[start+1:end] # 192801 to 201712, 1080 months
#         #times_sst_avg = times_sst[ind_offset_sst+(start+1)-tstep:ind_offset_sst+end-1] # 192701 to 201711, 1091 months, end-start+tstep months
#         #times_lst_avg = times_lst[(start+1)-tstep:end-1] # 192701 to 201711, 1091 months, end-start+tstep months
        
#         ## get nino regions
#         sst_avg = sst_avg[:,39:50,60:141] # 10N to -10N, 120E to 280E
#         lst_avg = lst_avg[:,39:50,60:141] # 10N to -10N, 120E to 280E
#         ## mask out ocean
#         lst_avg[sst_avg!=0] = 0

#         ## make the values near 0
#         y = y/300000.0 # [76394.0,287332.5]-->[0,1]
#         sst_avg = (sst_avg/50.0+1.0)/2 # [0.0,30.91]-->[0,1]
#         lst_avg = (lst_avg/50.0+1.0)/2 # [0.0,26.89]-->[0,1]  
        
#         X = sst_avg+lst_avg # [Nmon,Nlat,Nlon]==[1091,11,81]
#         y = y.reshape((-1,1)) # [Nmon,]-->[Nmon,1]==[1080,1]
        
#         return X,y
    
    
    
#%%
# class ARDataset(Dataset):
#     def __init__(self,X,y,fold='train',Ntrain=50,Nvalid=5):
#         y = y.reshape((-1,1))
#         X_train, y_train = X[0:Ntrain],y[0:Ntrain]
#         X_valid, y_valid = X[Ntrain:Ntrain+Nvalid],y[Ntrain:Ntrain+Nvalid]
#         X_test,y_test = X[Ntrain+Nvalid:],y[Ntrain+Nvalid:]
#         if fold=='train':
#             self.X = X_train
#             self.Y = y_train
#         elif fold=='valid':
#             self.X = X_valid
#             self.Y = y_valid
#         elif fold=='test':
#             self.X = X_test
#             self.Y = y_test
#         #self.Y /= np.max(np.abs(y_train)) # normalized to [-1,1]
#         self.X, self.Y = torch.from_numpy(self.X).float(),torch.from_numpy(self.Y).float() 
        
#     def __getitem__(self,idx):
#         X = self.X[idx]
#         y = self.Y[idx]
#         return X,y
    
#     def __len__(self):
#         return len(self.Y)



#%%
# =============================================================================
# class myDataset_REDNet(Dataset):
#     def __init__(self, datapath,variable):
#         self.datapaths = sorted(glob.glob(datapath + '*.npz'))       
#         self.variable = variable
#         
#     def __getitem__(self, idx):
#         data = np.load(self.datapaths[idx])
# 
#         if self.variable=='ppt':
#             X = data['gcms'] # ppt: [0.0,1.0]
#             y = data['prism'] # ppt: [0.0,1.0]
#         elif self.variable=='tmax' or self.variable=='tmin':
#             X = (data['gcms']+1.0)/2.0 # tmax/tmin: [-1,1]-->[0,1]
#             y = (data['prism']+1.0)/2.0 # tmax/tmin: [-1,1]-->[0,1]
#         X = np.mean(X,axis=0) # [Ngcm,Nlat,Nlon] --> [Nlat,Nlon]
#         X = resize(X,y.shape[1:],order=1,preserve_range=True) #[Nlat,Nlon]
#         
#         X = torch.from_numpy(X[np.newaxis,...]).float() #[Nlat,Nlon] --> [1,Nlat,Nlon]
#         y = torch.from_numpy(y).float() #[1,Nlat,Nlon]
# 
#         return X, y
# 
#     def __len__(self):
#         return len(self.datapaths)
# =============================================================================

#%%
# =============================================================================
# class myDataset_ESPCN(Dataset):
#     def __init__(self, datapath, variable):
#         self.datapaths = sorted(glob.glob(datapath + '*.npz')) 
#         self.variable = variable
# 
#     def __getitem__(self, idx):
#         data = np.load(self.datapaths[idx])
#         if self.variable=='ppt':
#             X = data['gcms'] # ppt: [0.0,1.0]
#             y = data['prism'] # ppt: [0.0,1.0]
#         elif self.variable=='tmax' or self.variable=='tmin':
#             X = (data['gcms']+1.0)/2.0 # tmax/tmin: [-1,1]-->[0,1]
#             y = (data['prism']+1.0)/2.0 # tmax/tmin: [-1,1]-->[0,1]
# 
#         X = torch.from_numpy(X).float() #[Ngcm,Nlat,Nlon]
#         y = torch.from_numpy(y).float() #[1,Nlat,Nlon]
# 
#         return X, y
# 
#     def __len__(self):
#         return len(self.datapaths)
# =============================================================================



# =============================================================================
# from skimage.io import imread
# #from skimage.transform import resize
# from skimage.color import rgb2gray
# #import numpy as np
# class ImageNetDataset(Dataset):
#     def __init__(self, datapath, is_train=True, scale=4, patch_size=None, transform=None):
#         if is_train:
#             self.datapaths = sorted(glob.glob(datapath + '*/*.JPEG'))
#         else:
#             self.datapaths = sorted(glob.glob(datapath + '*/*.JPEG'))
#         self.scale = scale
#         self.patch_size = patch_size
#         self.transform = transform
# 
#     def __getitem__(self, idx):
#         img = imread(self.datapaths[idx])/255.0
#         #assert len(img.shape)==3, "error: img.shape={}!=3\n".format(img.shape)
#         #if len(img.shape)==2:
#         #    img = np.repeat(img[:,:,np.newaxis],repeats=3,axis=2)
#         #    img += 0.1*np.random.random(size=img.shape)
#         
#         if self.patch_size:
#             crop_I = np.random.randint(0,img.shape[0]-self.patch_size[0]+1)
#             crop_J = np.random.randint(0,img.shape[1]-self.patch_size[1]+1)
#             img = img[crop_I:crop_I+self.patch_size[0],crop_J:crop_J+self.patch_size[1],:]
# 
#         H,W,C = img.shape
#         #H, W = H-H%self.scale, W-W%self.scale
#         #img = img[0:H,0:W,:]
#         #print('img.shape={}'.format(img.shape))
#         X = []
#         for order in range(6):
#             #temp = resize(img,output_shape=(W//scale,H//scale,C),order=order,preserve_range=True)
#             X.append(resize(img,output_shape=(H//self.scale,W//self.scale),order=order,preserve_range=True))
#         X = np.transpose(np.concatenate(X,axis=2),axes=(2,0,1)) # [C,H,W]
#         y = rgb2gray(img) #[H,W]
#         y = y[np.newaxis,:,:] #[1,H,W]
# 
#         X = torch.from_numpy(X).float() #[C,H,W]
#         y = torch.from_numpy(y).float() #[1,H,W]
#         #print('X.size()={}'.format(X.size()))
#         #print('y.size()={}'.format(y.size()))
#         # if len(y.size())<len(X.size()):
#         #     #assert y.size()==X.size()[-2:], 'X and y should have the same height and width!'
#         #     y = torch.unsqueeze(y,dim=0) # [1,Nlat,Nlon]
#         if self.transform:
#             y = self.transform(y)
#             X = self.transform(X)
# 
#         return X, y
# 
#     def __len__(self):
#         return len(self.datapaths)
# =============================================================================




# =============================================================================
# class PRISMDataset(Dataset):
#     def __init__(self, datapath, use_gcm=True, patch_size=None, transform=None):
#         self.datapaths = sorted(glob.glob(datapath + '*.npz'))   
#         self.use_gcm = use_gcm
#         self.patch_size = patch_size
#         self.transform = transform
# 
#     def __getitem__(self, idx):
#         data = np.load(self.datapaths[idx])
#         y = data['prism'] # [1,Nlat,Nlon]        
#         elevation = data['elevation'] #[1,Nlat,Nlon]
# 
#         #%% interpolate to the same size with y
#         if self.use_gcm:
#             gcms = np.mean(data['gcms'],axis=0) #[Ngcm,Nlat,Nlon] --> [Nlat,Nlon]
#                         
#         else:
#             gcms = resize(y[0,:,:],(y.shape[1]//2,y.shape[2]//2),order=1,preserve_range=True) #
# 
#         #print('before interpolation: gcms.shape={}'.format(gcms.shape))
#         gcms = resize(gcms,y.shape[1:],order=1,preserve_range=True) #
#         #print('after interpolation: gcms.shape={}'.format(gcms.shape))  
#         
#         X = np.concatenate((gcms[np.newaxis,...],elevation),axis=0) # [C,Nlat,Nlon]
#         
#         if self.patch_size:
#             crop_I = np.random.randint(0,X.shape[1]-self.patch_size[0]+1)
#             crop_J = np.random.randint(0,X.shape[2]-self.patch_size[1]+1)
#             X = X[:,crop_I:crop_I+self.patch_size[0],crop_J:crop_J+self.patch_size[1]]
#             y = y[:,crop_I:crop_I+self.patch_size[0],crop_J:crop_J+self.patch_size[1]]
#             
#         X = torch.from_numpy(X).float() #[Nlat,Nlon,2]      
#         y = torch.from_numpy(y).float() #[1,Nlat,Nlon]
# 
#         if self.transform:
#             y = self.transform(y)
#             X = self.transform(X)
# 
#         return X, y
# 
#     def __len__(self):
#         return len(self.datapaths)
# =============================================================================

#%%
# =============================================================================
# class PRISMDataset_DeepSD2(Dataset):
#     def __init__(self, datapath, use_gcm=True, base_scale_factor=2):
#         self.datapaths = sorted(glob.glob(datapath + '*.npz'))   
#         self.use_gcm = use_gcm
#         self.bsf = base_scale_factor
# 
#     def __getitem__(self, idx):
#         data = np.load(self.datapaths[idx])
#         y = data['prism'] # [1,Nlat,Nlon]        
#         elevation = np.squeeze(data['elevation']) #[1,Nlat,Nlon] --> [Nlat,Nlon]
#         gcms = np.mean(data['gcms'],axis=0) #[Ngcm,Nlat,Nlon] --> [Nlat,Nlon]
#         #%% interpolate to the same size with y
#         elevation1 = resize(elevation,(self.bsf*gcms.shape[0],self.bsf*gcms.shape[1]),order=1,preserve_range=True) #
#         elevation2 = resize(elevation,(self.bsf*self.bsf*gcms.shape[0],self.bsf*self.bsf*gcms.shape[1]),order=1,preserve_range=True) 
#              
#         X = torch.from_numpy(gcms[np.newaxis,...]).float() #[1,Nlat,Nlon]      
#         X2 = torch.from_numpy(elevation1[np.newaxis,...]).float() #[1,Nlat,Nlon]      
#         X3 = torch.from_numpy(elevation2[np.newaxis,...]).float() #[1,Nlat,Nlon]
#         X4 = torch.from_numpy(elevation[np.newaxis,...]).float() #[1,Nlat,Nlon]
#         X = [X,X2,X3,X4]
#         
#         y = torch.from_numpy(y).float() #[1,Nlat,Nlon]
# 
#         return X, y
# 
#     def __len__(self):
#         return len(self.datapaths)
# =============================================================================
    

# =============================================================================
# '''
# class myDataset(Dataset):
#     def __init__(self, images_dir, patch_size, scale, transform=None):
#         self.image_files = sorted(glob.glob(images_dir + '*'))
#         self.patch_size = patch_size
#         self.scale = scale
#         self.transform = transform
# 
# 
#     def __getitem__(self, idx):
#         image = Image.open(self.image_files[idx]).convert('RGB')
#         
#         # crop to be divisible by scale
#         w,h = image.size[0:2]
#         w,h = (w//self.scale)*self.scale, (h//self.scale)*self.scale
#         image = image.crop((0,0,w,h))
#         
#         y = image
#         X = image.resize((w//self.scale,h//self.scale),resample=Image.LANCZOS) # interpolated to smaller size
#         X = image.resize((w,h),resample=Image.LANCZOS) # interpolated to original size, lower resolution
#         if self.transform:
#             y = self.transform(y)
#             X = self.transform(X)
#         assert(X!=y,'Error: X==y\n')
# 
#         return X, y
# 
#     def __len__(self):
#         return len(self.image_files)
# '''
# =============================================================================
