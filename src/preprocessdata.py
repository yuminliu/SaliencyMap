





#%%
def plot_preds():
    import numpy as np
    from torch.utils.data.dataloader import DataLoader
    import datasets
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    predictand = 'DMI' # 'Nino34_anom' # 'Nino34'
    folder = 'test'
    interval = 12
    save_name_prefix = predictand.lower()
    savepath = '../results/myCNN/lag_0/{}/'.format(save_name_prefix)

    #### Nino34
    # preds_gcms = np.load('../results/myCNN/lag_0/nino34/2021-03-23_17.56.37.146542_gcm_masked_nino34/pred_results_RMSE0.3993532311636982.npz')
    # preds_ssts = np.load('../results/myCNN/lag_0/nino34/2021-03-24_12.45.26.281280_sst_masked_nino34/pred_results_RMSE0.18269071993362454.npz')
    #### Nino34
    # preds_gcms = np.load('../results/myCNN/lag_0/nino34_anom/2021-03-23_20.04.36.760348_gcm_masked_nino34_anom/pred_results_RMSE0.5877410599791633.npz')
    # preds_ssts = np.load('../results/myCNN/lag_0/nino34_anom/2021-03-24_12.29.27.064729_sst_masked_nino34_anom/pred_results_RMSE0.2075148332980373.npz')
    #### DMI
    preds_gcms = np.load('../results/myCNN/lag_0/dmi/2021-03-23_20.53.38.837405_gcm_masked_dmi/pred_results_RMSE0.494626008667897.npz')
    preds_ssts = np.load('../results/myCNN/lag_0/dmi/2021-03-23_23.25.45.173063_sst_masked_dmi/pred_results_RMSE0.49948260707305614.npz')

    window,noise_std,ngcm,num_workers = 3,0.0,'all',8
    test_dataset = datasets.myDataset_CNN(fold='test',window=window,noise_std=noise_std,ngcm=ngcm)
    test_loader = DataLoader(dataset=test_dataset,batch_size=1,shuffle=False,num_workers=num_workers)
    ys = []
    for inputs,y in test_loader:
        y = np.squeeze(y.float().cpu().numpy()) # scalar?
        ys.append(y)
    ys = np.stack(ys,axis=0) #[Ntest,]
    year_months = np.load('../data/Climate/year_months_195001-200512.npy')

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
    ####
    fig = plt.figure(figsize=(12,5))
    plt.plot(ys,'--k',label='Groundtruth')
    plt.plot(preds_gcms['preds'],'-r',label='GCMs Prediction')
    plt.plot(preds_ssts['preds'],'-b',label='SSTs Prediction')
    plt.xticks(ticks=range(0,len(ys),interval), labels=year_months[::interval])
    plt.title('Groundtruth vs Prediction')
    plt.xlabel('Month')
    plt.ylabel(predictand)
    plt.legend()
    plt.savefig(savepath+'{}_pred_vs_time_{}.png'.format(save_name_prefix,folder),dpi=1200,bbox_inches='tight')
















#%%
def get_DMI():
    import numpy as np
    import pandas as pd

    filepath = "../data/Climate/DMI/"
    #filename = "DMI Standard PSL Format"
    #filename = "DMI Western Indian Ocean PSL Format"
    filename = "DMI Eastern Indian Ocean Standard PSL Format"
    month_names = np.array(['Year','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    data = []
    with open(filepath+filename+".txt",'r') as f:
        lines = f.readlines()#.split(' ')
        for line in lines:
            line = line.split(' ')
            line = [s for s in line if s!='']
            data.append(line)

    data = np.array(data)
    years = data[:,0].astype(np.int)
    data = data[:,1:].astype(np.float)
    data_df = pd.DataFrame(data=data,index=years,columns=month_names[1:],dtype=np.float)
    #data_df.to_csv(filepath+"processed/"+filename.replace(' ','_')+'.csv')

    data_series = data.reshape((-1,1))
    months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    index = []
    for year in years:
        for m in months:
            index.append(str(year)+m)
    data_series = pd.DataFrame(data=data_series,index=index,columns=['DMI'],dtype=np.float)
    data_series.to_csv(filepath+'processed/{}_{}-{}_series.csv'.format(filename.replace(' ','_'),index[0],index[-1]))
    #np.savez(filepath+'processed/{}_{}-{}_array.npz'.format(filename.replace(' ','_'),index[0],index[-1]),data=data,years=years)


    # y = data.reshape((-1,))
    # x = range(len(y))

    # import matplotlib.pyplot as plt
    # fig = plt.figure(figsize=(12,5))
    # #plt.plot(x,y,label='Western DMI')
    # plt.plot(x,y,label='Eastern DMI')
    # #plt.plot(x,y,label='DMI')
    # plt.xlabel("Month")
    # plt.ylabel("DMI")
    # #plt.title("Diploe Mode Index")
    # #plt.title("Western Diploe Mode Index")
    # plt.title("Eastern Diploe Mode Index")

    # savepath = "../data/Climate/DMI/processed/"
    # savename = "eastern_dmi" # "western_dmi" # "dmi" # 
    # plt.savefig(savepath+savename+'.png',dpi=1200,bbox_inches='tight')

    # plt.show()
#%%
def combine_sst():
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.transform import resize
    
    prefix = '/scratch/wang.zife/YuminLiu/DATA/'
    cobe = np.load(prefix+'COBE/processed/sst.mon.mean_185001-201912_1by1.npy')
    cobe_lats = np.load(prefix+'COBE/processed/lats.npy')
    cobe_lons = np.load(prefix+'COBE/processed/lons.npy')
    cobe_time = np.load(prefix+'COBE/processed/time.npy')
    hadley = np.load(prefix+'Hadley-NOAA/processed/MODEL.SST.HAD187001-198110.OI198111-202008_1by1.npy')
    noaa = np.load(prefix+'NOAA/processed/sst.mnmean_185401-202012_2by2.npy')
    uod = np.load(prefix+'UoD/processed/air.mon.mean.v501_190001-201712_0.5by0.5.npy')
    cobe = np.nan_to_num(cobe,nan=0)
    hadley = np.nan_to_num(hadley,nan=0)
    noaa = np.nan_to_num(noaa,nan=0)
    uod = np.nan_to_num(uod,nan=0)
    noaa_scaled = np.transpose(resize(np.transpose(noaa,axes=(1,2,0)),hadley.shape[1:],order=1,preserve_range=True),axes=(2,0,1))
    uod_scaled = np.transpose(resize(np.transpose(uod,axes=(1,2,0)),hadley.shape[1:],order=1,preserve_range=True),axes=(2,0,1))
    ## trim to 195001 to 200512
    # cobe = cobe[1200:1872,:,:]
    # hadley = hadley[960:1632,:,:]
    # noaa_scaled = noaa_scaled[1152:1824,:,:]
    # uod_scaled = uod_scaled[600:1272,:,:]
    
    # sst = np.stack((cobe,hadley,noaa_scaled,uod_scaled),axis=1)
    # np.save('/scratch/wang.zife/YuminLiu/myPythonFiles/SaliencyMap/data/Climate/SST/sst_cobe_hadley_noaa_uod_195001-200512_1by1_world.npy',sst)
    # np.save('/scratch/wang.zife/YuminLiu/myPythonFiles/SaliencyMap/data/Climate/SST/lats.npy',cobe_lats)
    # np.save('/scratch/wang.zife/YuminLiu/myPythonFiles/SaliencyMap/data/Climate/SST/lons.npy',cobe_lons)
    # np.save('/scratch/wang.zife/YuminLiu/myPythonFiles/SaliencyMap/data/Climate/SST/time.npy',cobe_time[1200:1872])
    
    ## trim to 190001 to 201912
    cobe = cobe[600:,:,:]
    hadley = hadley[360:1800,:,:]
    noaa_scaled = noaa_scaled[552:1992,:,:]
    #uod_scaled = uod_scaled[600:1272,:,:]
    
    sst = np.stack((cobe,hadley,noaa_scaled),axis=1)
    np.save('/scratch/wang.zife/YuminLiu/myPythonFiles/SaliencyMap/data/Climate/Reanalysis/sst_cobe_hadley_noaa_190001-201912_1by1_world.npy',sst)
combine_sst()
    





    
#%%
def GCM_enso_riverflow_xcorr():
    import numpy as np
    import pandas as pd
    import utils
    
    window = 3
    column = ['Nino3'] # ['Nino12','Nino3','Nino4'] # ['Nino34_anom'] # 6 is 'Nino34_anom'
    riverflow_df = pd.read_csv('../data/Climate/RiverFlow/processed/riverflow.csv',index_col=0,header=0)
    info_df = pd.read_csv('../data/Climate/RiverFlow/processed/info.csv',index_col=0,header=0)
    times_df = pd.read_csv('../data/Climate/RiverFlow/processed/times.csv',index_col=0,header=0)
    #times = np.asarray(times_df,dtype=int).reshape((-1,))
    times = list(np.asarray(times_df,dtype=int).reshape((-1,)))+[201901]
    #%% read ENSO index
    indices_df, _ = utils.read_enso() # 187001 to 201912
    indices_df = indices_df[column] # select input feature
    #amazon = riverflow_df[['0']].loc[195001:200512].to_numpy().reshape((-1,))
    amazons = {}
    for window in range(1,7):
        amazon = riverflow_df[['0']].iloc[600-window+1:1272].to_numpy().reshape((-1,))
        amazon = [np.mean(amazon[i:i+window]) for i in range(len(amazon)-window+1)]# moving average
        amazons[window] = amazon
      
    nino3 = indices_df[['Nino3']].loc[195001:200512].to_numpy().reshape((-1,))
    
    import matplotlib.pyplot as plt
    #fig = plt.figure()
    #plt.plot(amazon)
    #
    #fig = plt.figure()
    #plt.plot(nino3)
    #plt.show()
    
    left,right = 50,360
    top,bottom = 50,130
    data = np.load('/scratch/wang.zife/YuminLiu/DATA/GCM/GCMdata/tas/processeddata/32gcms_tas_monthly_1by1_195001-200512_World.npy')
    data = data[:,:,top:bottom,left:right]
    loc = data[0,:,90,180]
    
    #fig = plt.figure()
    #plt.plot(loc)
    #plt.show()
    
    fig = plt.figure()
    plt.imshow(a[0,0,:,:])
    plt.show()
    
    fig = plt.figure()
    plt.imshow(data[0,0,:,:])
    plt.show()
    
    fig = plt.figure()
    lags_nino3,corr_nino3,_,_ = plt.xcorr(nino3,loc,maxlags=24,usevlines=False,normed=True,label='nino3')
    #lags_amazon,corr_amazon,_,_ = plt.xcorr(amazon,loc,maxlags=24,usevlines=False,normed=True,label='amazon')
    lags_amazon,corr_amazon = {},{}
    for window in amazons:
        lags_amazon[window],corr_amazon[window],_,_ = plt.xcorr(amazons[window],loc,maxlags=24,usevlines=False,normed=True,label='amazon_{}'.format(window))
    #lags_,corr_,_,_ = plt.xcorr(amazon,nino3,maxlags=24,usevlines=False,normed=True)
    plt.legend()
    plt.show()



def readGCM_Ocean():
    '''
    generate whole USA map, saved on 12/11/2019
    read and process NASA GCM data and aglined
    '''
    import numpy as np
    from netCDF4 import Dataset
    from ncdump import ncdump
    from os import listdir
    from os.path import isfile, join
    import os
    
    prefix = '/scratch/wang.zife/YuminLiu/DATA/'
    variable = 'tas' # 'tasmin' #'tasmax' #['pr' 'tas' 'tasmax' 'tasmin'] 
    filepath = prefix+'GCM/GCMdata/'+variable+'/raw/'
    savepath = prefix+'GCM/GCMdata/'+variable+'/processeddata/'
    
    filenames = sorted([f for f in listdir(filepath) if isfile(join(filepath,f))])
    
    if savepath and not os.path.exists(savepath):
        os.makedirs(savepath)
        
    values_gcms = []
    gcm_names = []
    for kk,filename in enumerate(filenames):
        dataset = Dataset(filepath+filename,mode='r')
        if kk==0:
            nc_attrs, nc_dims, nc_vars = ncdump(dataset)
       
        # original longitude is from 0.5 to 359.5 by 1, 360 points 
        # original latitude is from -89.5 to 89.5 by 1, 180 points 
        # whole USA longitude from [235.5E,293.5E] by 1, 59 points
        # whole USA latitude from [24.5N, 49.5N] by 1, 26 points
        # original month from 195001 to 200512, 672 points
        
        time = dataset.variables['time'][:] # 195001 - 200512
        #lats = dataset.variables['latitude'][114:140] # [24.5N, 49.5N]
        #lons = dataset.variables['longitude'][235:294] # [235.5E, 293.5E]
        ## near-surface air temperature
        lats_gcm = dataset.variables['latitude'][2:-2] # [-87.5N, 87.5N]
        lats_gcm = np.flipud(lats_gcm) # lats from [87.5N,-87.5N]
        lons_gcm = dataset.variables['longitude'][:] # [0.5E, 359.5E]
        value_gcm = dataset.variables[variable][:,2:-2,:]#[month,lat,lon] 195001-200512, totally 
        value_gcm = np.ma.filled(value_gcm,1000)
        for t in range(len(value_gcm)):
            value_gcm[t,:,:] = np.flipud(value_gcm[t,:,:]) # lats from [87.5N,-87.5N]  
    
        if np.isnan(np.max(value_gcm)):
            print(filename + 'has NAN!\n')
            break
        if value_gcm.shape!=(672,176,360) or np.max(value_gcm==1000):
            print(filename)
            #break
            continue
        
        values_gcms.append(value_gcm) 
        gcm_names.append(filename)
    
    values_gcms = np.stack(values_gcms,axis=1) # [Nmon,Ngcm,Nlat,Nlon]
    print('values_gcms.shape={}'.format(values_gcms.shape))
    
    #time = np.array(time)
    time = []
    for i in range(values_gcms.shape[1]):
        year,month = str(1950+i//12), str(i%12+1)
        if int(month)<10: month = '0'+month
        time.append(int(year+month))
    time = np.array(time)  
    lats_gcm = np.array(lats_gcm)
    lons_gcm = np.array(lons_gcm)
    
    if savepath:
        np.save(savepath+'time_gcm.npy',time)
        np.save(savepath+'lats_gcm.npy',lats_gcm)
        np.save(savepath+'lons_gcm.npy',lons_gcm)
        np.save(savepath+'{}gcms_{}_monthly_1by1_195001-200512_World.npy'.format(len(filenames),variable),values_gcms)
        np.save(savepath+'gcm_names.npy',gcm_names)
    
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.imshow(values_gcms[0,0,:,:])
    plt.show()
    plt.vlines(x=[50,360],ymin=50,ymax=130)
    plt.hlines(y=[50,130],xmin=50,xmax=360)






#%%
def readGCM():
    #### generate whole USA map, saved on 10/22/2019
    #### read and process NASA GCM data and aglined
    import numpy as np
    from netCDF4 import Dataset
    from ncdump import ncdump
    from os import listdir
    from os.path import isfile, join
    import os
    
    variablename = 'pr' #['pr' 'tas' 'tasmax' 'tasmin'] 'pr_37models'
    filepath = '../data/Climate/GCMdata/rawdata/'+variablename+'/'
    savepath0 = None #'../data/Climate/GCMdata/processeddata/'
    savepath = None #savepath0+variablename+'/'
    # if not os.path.exists(savepath0):
    #     os.makedirs(savepath0)
    if savepath and not os.path.exists(savepath):
        os.makedirs(savepath)
    
    filenames = [f for f in listdir(filepath) if isfile(join(filepath,f))]
    
    kk = 0
    values_gcms = []
    for filename in filenames:
        #filename = 'regridded_1deg_pr_amon_inmcm4_historical_r1i1p1_195001-200512.nc'
        dataset = Dataset(filepath+filename,mode='r')
        #dataset = Dataset(filepath+filename,mode='r',format="NETCDF3")
        #dataset = Dataset(filename,mode='r')
        if kk==0:
            nc_attrs, nc_dims, nc_vars = ncdump(dataset)
            kk += 1
       
        # original longitude is from 0.5 to 359.5 by 1, 360 points 
        # original latitude is from -89.5 to 89.5 by 1, 180 points 
        # whole USA longitude from [230.5E,304.5E] by 1, 75 points
        # whole USA latitude from [20.5N, 49.5N] by 1, 30 points
        # whole USA longitude from [234.5E,295.5E] by 1, 62 points
        # whole USA latitude from [24.5N, 49.5N] by 1, 26 points
        
        # retangular USA longitude from [245.5 to 277.5] by 1, 33 points
        # retangular USA latitude from 33.5 to 49.5 by 1, 17 points
        # original month from 195001 to 200512, 672 points
        # month from 200001 to 200412, 60 months
        
        time = dataset.variables['time'][:] # 195001 - 200512
        lats = dataset.variables['latitude'][110:140] # [20.5N, 49.5N]
        lons = dataset.variables['longitude'][230:305] # [230.5E, 304.5E]
        #lats = dataset.variables['latitude'][114:140] # [24.5N, 49.5N]
        #lons = dataset.variables['longitude'][234:296] # [234.5E, 295.5E]
        #### whole USA
        ## monthly mean precipitation, unit: mm/day
        value1_gcm = dataset.variables[variablename][:,110:140,230:305]#[month,lat,lon] 195001-200512, totally 
        #value1_gcm = dataset.variables[variablename][:,114:140,234:296]#[month,lat,lon] 195001-200512, totally 
        #value2_gcm = np.ma.filled(value1_gcm,-1.0e-8)
        value2_gcm = np.ma.filled(value1_gcm,0)
        (Nmon,Nlat,Nlon) = value2_gcm.shape # [672,30,75]
        value_gcm = np.zeros((Nmon,Nlat,Nlon)) # [672,30,75]
        for t in range(Nmon):
            value_gcm[t,:,:] = np.flipud(value2_gcm[t,:,:]) # lats from [49.5N,25.5N]  
        
        #### retangular USA
        ## monthly mean precipitation, unit: mm/day
        #precipitation = dataset.variables['pr'][600:660,123:140,245:278]#[month,lat,lon]    
        #prmean_month_gcm = np.ma.filled(precipitation,np.nan)   
    
        if np.isnan(np.sum(value_gcm)):
            print(filename + '\n')
            break
        savename = filename.replace('.nc','_USA.npy') 
        np.save(savepath+savename,value_gcm)
        values_gcms.append(value_gcm) 

    values_gcms = np.stack(values_gcms,axis=0) # [Ngcm,Nmon,Nlat,Nlon]
    print('values_gcms.shape={}'.format(values_gcms.shape))
    
    time = np.array(time)
    #### whole USA
    # latitude from [20.5N, 49.5N] by 1, 30 points
    # latitude from [24.5N, 49.5N] by 1, 26 points
    lats_gcm1 = dataset.variables['latitude'][110:140]
    #lats_gcm1 = dataset.variables['latitude'][114:140] 
    lats_gcm = np.flipud(lats_gcm1) # lats from [49.5N,24.5N]
    # longitude from [230.5E, 304.5E] by 1, 75 points
    # longitude from [234.5E, 295.5E] by 1, 62 points
    lons_gcm = dataset.variables['longitude'][230:305]
    #lons_gcm = dataset.variables['longitude'][234:296] 
    lats_gcm = np.array(lats_gcm)
    lons_gcm = np.array(lons_gcm)
    #np.save(savepath0+'time_gcm.npy',time)
    #np.save(savepath0+'lats_gcm.npy',lats_gcm)
    #np.save(savepath0+'lons_gcm.npy',lons_gcm)
    #np.save(savepath0+'18gcms_prmean_monthly_1by1_195001-200512_USA.npy',values_gcms)
    
    #### retangular USA
    ### latitude from 33.5 to 49.5 by 1, 17 points
    #lats_gcm = dataset.variables['latitude'][123:140] 
    ## longitude from 245.5 to 277.5 by 1, 33 points
    #lons_gcm = dataset.variables['longitude'][245:278] 
    #np.save(savepath+'lats_gcm.npy',lats_gcm)
    #np.save(savepath+'lons_gcm.npy',lons_gcm)
    
    
    # test = dataset.variables['pr'][:,:,:]
    # test = test[:,110:140,230:305] # usa area
    # for t in range(test.shape[0]):
    #     test[t,:,:] = np.flipud(test[t,:,:])
    
    # img = np.sum(test,axis=0)
    # img = img/abs(img).max()
    # import matplotlib.pyplot as plt
    # plt.figure()
    # #plt.imshow(img)
    # #plt.imshow(value_gcm[0,:,:])
    # plt.show()


    ## save every month data
    for mon in range(values_gcms.shape[0]):
        values_gcms_per_month = np.transpose(values_gcms[:,mon,:,:],axes=(1,2,0)) # [Nlat,Nlon,Ngcm]
        if mon==0:
            print('values_gcms_per_month.shape={}'.format(values_gcms_per_month.shape))
        



#readGCM()


#%%
def readCPC():
    #### read the CPC data and aglined
    import numpy as np
    from netCDF4 import Dataset
    import os
    from ncdump import ncdump

    filepath = '../data/Climate/CPCdata/rawdata/precip.V1.0.mon.mean.nc'
    savepath = '../data/Climate/CPCdata/processeddata/cpc_prmean_monthly_0.25by0.25/'
    
    
    dataset = Dataset(filepath,mode='r')
    nc_attrs, nc_dims, nc_vars = ncdump(dataset)
    
    
    # original latitude from 20.125 to 49.875 by 0.25, 120 points
    # latitude from 20.375N to 49.625N by 0.25, 118 points
    ## latitude from 24.375N to 49.625N by 0.25, 102 points
    lats_2 = dataset.variables['lat'][:]
    #lats_2 = dataset.variables['lat'][1:-1]
    #lats_2 = dataset.variables['lat'][17:-1]  
    # original longitude from 230.125E to 304.875E by 0.25, 300 points
    # longitude from 230.375E to 304.625E by 0.25, 298 points
    # longitude from 234.375E to 295.625E by 0.25, 246 points
    lons_2 = dataset.variables['lon'][:] 
    #lons_2 = dataset.variables['lon'][1:-1]
    #lons_2 = dataset.variables['lon'][17:263] 
    # time from 195001 to 200512 by 1, 672 points
    time = dataset.variables['time'][24:696]
    
    precipitation_2 = dataset.variables['precip'][24:696,:,:]#[month,lat,lon]
    #precipitation_2 = dataset.variables['precip'][24:696,1:-1,1:-1]#[month,lat,lon]    
    #precipitation_2 = dataset.variables['precip'][24:696,17:-1,17:263]#[month,lat,lon]
    #prmean_month = np.ma.filled(precipitation_2, -1.0e-8) # daily mean precipitation, unite: mm/day
    prmean_month = np.ma.filled(precipitation_2,0) # daily mean precipitation, unite: mm/day
    
    # flip precipitation upside down regarding to latitude
    for i in range(prmean_month.shape[0]):
        prmean_month[i,:,:] = np.flipud(prmean_month[i,:,:]) 
        
    #prmean_month = prmean_month/abs(prmean_month[:]).max()
    #import matplotlib.pyplot as plt
    #for i in range(1):
    #    a = prmean_month[i,:,:]
    #    #a = np.flipud(a)
    #    plt.imshow(a)
    
    # aligned with GCM longtitude and latitude
    # average longitude from [230.375E,304.625E], 297 points
    # average latitude from [20.375N,49.625N], 117 points
    ## average longitude from [234.5E,295.5E], 245 points
    ## average latitude from [24.5N,49.5N], 101 points
    lats_2 = np.array(lats_2[::-1]) # flip latitude upside down
    lons_2 = np.array(lons_2)
    '''
    lats = np.zeros(len(lats_2)-1,)
    lons = np.zeros(len(lons_2)-1,)
    for i in range(len(lats_2)-1):
        lats[i] = 0.5*(lats_2[i]+lats_2[i+1])
    for i in range(len(lons_2)-1):
        lons[i] = 0.5*(lons_2[i]+lons_2[i+1])
    (Nmon,Nlat,Nlon) = prmean_month.shape
    prmean_month_cpc = np.zeros((Nmon,Nlat-1,Nlon-1)) # [672, 117, 297],[672, 101, 245]
    for t in range(Nmon):
        for i in range(Nlat-1):
            for j in range(Nlon-1):
                prmean_month_cpc[t,i,j] = 0.25*(prmean_month[t,i,j]+prmean_month[t,i+1,j]
                                          +prmean_month[t,i,j+1]+prmean_month[t,i+1,j+1])
    #prmean_month_cpc[prmean_month_cpc<0] = -1.0e-8
    prmean_month_cpc[prmean_month_cpc<0] = 0
    '''

    lats = lats_2
    lons = lons_2
    prmean_month_cpc = np.array(prmean_month)

    if np.isnan(np.sum(prmean_month_cpc)):
        print("Error! nan found!\n")            
        
    #    img = np.sum(prmean_month_cpc,axis=0)
    #    img = img/abs(img).max()    
    #    import matplotlib.pyplot as plt
    #    plt.figure()
    #    plt.imshow(img)
    
    time = np.array(time)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    #print('lons={}'.format(lons))
    #print('lats={}'.format(lats))
    #print('prmena_month_cpc.shape={},type(prmean_month_cpc)={}'.format(prmean_month_cpc.shape,type(prmean_month_cpc)))
    #np.save(savepath+'lons_cpc.npy',lons)
    #np.save(savepath+'lats_cpc.npy',lats) 
    #np.save(savepath+'time_cpc.npy',time)    
    #np.save(savepath+'cpc_prmean_monthly_195001-200512_USA.npy',prmean_month_cpc)
    ##import matplotlib.pyplot as plt
    ##plt.imshow(prmean_month_cpc[0,:,:])
    ##plt.show()

#readCPC()

#%%
def mergexy():
    import os
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from skimage.transform import resize


    savepath = '../data/Climate/CPC_GCMdata/'
    gcmpath = '../data/Climate/GCMdata/processeddata/18gcms_prmean_monthly_1by1_195001-200512_USA.npy'
    cpcpath = '../data/Climate/CPCdata/processeddata/cpc_prmean_monthly_0.25by0.25/cpc_prmean_monthly_0.25by0.25_195001-200512_USA.npy'
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    gcmdata = np.load(gcmpath)
    cpcdata = np.load(cpcpath)

    print('gcmdata.shape={}'.format(gcmdata.shape))
    print('cpcdata.shape={}'.format(cpcdata.shape))

    MaskMapUSA = np.sum(cpcdata,axis=0) #[Nlat,Nlon]
    MaskMapUSAsmall = resize(image=MaskMapUSA,output_shape=gcmdata.shape[2:],preserve_range=True)
    print('MaskMapUSA.shape={}'.format(MaskMapUSA.shape))
    print('MaskMapUSAsmall.shape={}'.format(MaskMapUSAsmall.shape))

    for mon in range(cpcdata.shape[0]):
        X = gcmdata[:,mon,:,:] #[Ngcm,Nlat,Nlon], mm/day
        #y = cpcdata[[mon],:,:] #[1,Nlat,Nlon], mm/day
        y = cpcdata[mon,:,:] #[Nlat,Nlon], mm/day
        
        for ngcm in range(X.shape[0]):
            X[ngcm,:,:][MaskMapUSAsmall<=0] = 0
        y[MaskMapUSA<=0] = 0
        y = y[np.newaxis,...] #[1,Nlat,Nlon], mm/day
        #X = gcmdata
        #y = cpcdata

        X = np.log(1.0+X) ##[Ngcm,Nlat,Nlon], [0,3.737655630962239]
        y = np.log(1.0+y) ##[1,Nlat,Nlon],[0,4.159132957458496]        
        #print('X.max={}'.format(max(X.flatten())))
        #print('y.max={}'.format(max(y.flatten())))
        savename = 'cpc_gcm_log1p_prmean_monthly_0.25to1.0_195001-200512_USA_month'+str(mon+1)
        np.savez(savepath+savename+'.npz',gcms=X,cpc=y)
        # if mon==0:
        #     fig = plt.figure()
        #     sns.distplot(X.flatten())
        #     plt.title('GCM')
        #     plt.show()
        #     fig = plt.figure()
        #     sns.distplot(y.flatten())
        #     plt.title('CPC')
        #     plt.show()

#mergexy()





#%%
def normalization():
    import os
    import numpy as np
    
    def normalized(resolution,folder):
        #resolution = 0.125 # 0.5
        variable = 'tmin' # 'tmax' # 'ppt' # 
        #folder = 'train' # 'test' #'val'#
        datapath = '../data/Climate/PRISM_GCMdata/{}/{}by{}/{}/'.format(variable,resolution,resolution,folder)
        savepath = '../data/Climate_new/PRISM_GCMdata/{}/{}by{}/{}/'.format(variable,resolution,resolution,folder)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        filenames = [f for f in os.listdir(datapath) if f.endswith('.npz')]
        filenames = sorted(filenames)
        #filenames = [filenames[0]]
        gcmsmin,gcmsmax = 50,-50
        prismmin,prismmax = 50,-50
        climatologymin,climatologymax = 50,-50
        elevationmin,elevationmax = 50,-50
        for filename in filenames:
            data = np.load(datapath+filename)
            gcms = data['gcms']#/5.0 # tmin, tmax: [-1,1], ppt [0,3.6]
            prism = data['prism']#/5.0 # tmin, tmax: [-1,1], ppt [0,4.2]
            elevation = data['elevation']#/10.0 # tmin, tmax [0,0.8], ppt [0,8.2]
            climatology = data['climatology']#/5.0 # tmin,tmax: [-1,1], ppt [0,3.2]
            
            np.savez(savepath+filename,gcms=gcms,prism=prism,elevation=elevation,climatology=climatology)
        
            gcmsmin,gcmsmax = min(gcmsmin,gcms.min()),max(gcmsmax,gcms.max())
            prismmin,prismmax = min(prismmin,prism.min()),max(prismmax,prism.max())
            elevationmin,elevationmax = min(elevationmin,elevation.min()),max(elevationmax,elevation.max())
            climatologymin,climatologymax = min(climatologymin,climatology.min()),max(climatologymax,climatology.max())
        
        print('resolution={},folder={}'.format(resolution,folder))
        print('gcmsmin={},max={}'.format(gcmsmin,gcmsmax))
        print('prismmin={},max={}'.format(prismmin,prismmax))
        print('elevationmin={},max={}'.format(elevationmin,elevationmax))
        print('climatologymin={},max={}\n'.format(climatologymin,climatologymax))
    
    for resolution in [0.5,0.25,0.125]:
        for folder in ['train','val','test']:
            normalized(resolution,folder)