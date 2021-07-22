






import numpy as np; np.random.seed(0)
import matplotlib.pyplot as plt
import matplotlib.ticker

class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
             self.format = r'$\mathdefault{%s}$' % self.format


z = (np.random.random((10,10)) - 0.5) * 0.2

fig, ax = plt.subplots()
plot = ax.contourf(z)
cbar = fig.colorbar(plot, format=OOMFormatter(-3,fformat="%1.1f",offset=True,mathText=True))

plt.show()
























import numpy as np
sst = np.load('../data/Climate/Reanalysis/AlignedData/sst_cobe_hadley_noaa_uod_195001-200512_1by1_world.npy')[:,0:3,:,:]
mask = np.mean(sst,axis=(0,1))
mask = mask[50:130,50:350]
print(mask)














def align_reanalysis_with_gcm():
    import numpy as np

    imgpath = '/scratch/wang.zife/YuminLiu/DATA/COBE/processed/sst.mon.mean_185001-201912_1by1.npy'
    lats = np.load('/scratch/wang.zife/YuminLiu/DATA/COBE/processed/lats.npy')
    lons = np.load('/scratch/wang.zife/YuminLiu/DATA/COBE/processed/lons.npy')
    time = np.load('/scratch/wang.zife/YuminLiu/DATA/COBE/processed/time.npy')
    data = np.load(imgpath) # 89.5N to -89.5N by 1, 
    data = data[:,2:-2,:] # 87.5N to -87.5N by 1, 
    lats = lats[2:-2]
    print('data.shape={}'.format(data.shape))
    print('lats={}'.format(lats))
    print('lons[0][-1]={},{}'.format(lons[0],lons[-1]))
    np.save('/scratch/wang.zife/YuminLiu/DATA/COBE/processed/AlignedData/sst.mon.mean_185001-201912_1by1.npy',data)
    np.save('/scratch/wang.zife/YuminLiu/DATA/COBE/processed/AlignedData/lats.npy',lats)
    np.save('/scratch/wang.zife/YuminLiu/DATA/COBE/processed/AlignedData/lons.npy',lons)
    np.save('/scratch/wang.zife/YuminLiu/DATA/COBE/processed/AlignedData/time.npy',time)


    imgpath = '../data/Climate/Reanalysis/sst_cobe_hadley_noaa_uod_195001-200512_1by1_world.npy'
    lats = np.load('../data/Climate/Reanalysis/lats.npy')
    lons = np.load('../data/Climate/Reanalysis/lons.npy')
    time = np.load('../data/Climate/Reanalysis/time.npy')
    data = np.load(imgpath) # 89.5N to -89.5N by 1, 
    data = data[:,:,2:-2,:] # 87.5N to -87.5N by 1, 
    lats = lats[2:-2]
    np.save('../data/Climate/Reanalysis/AlignedData/sst_cobe_hadley_noaa_uod_195001-200512_1by1_world.npy',data)
    np.save('../data/Climate/Reanalysis/AlignedData/lats.npy',lats)
    np.save('../data/Climate/Reanalysis/AlignedData/lons.npy',lons)
    np.save('../data/Climate/Reanalysis/AlignedData/time.npy',time)

    print('data.shape={}'.format(data.shape))
    print('lats={}'.format(lats))
    print('lons[0][-1]={},{}'.format(lons[0],lons[-1]))

    imgpath = '/scratch/wang.zife/YuminLiu/DATA/GCM/GCMdata/tas/processeddata/32gcms_tas_monthly_1by1_195001-200512_World.npy'
    lats = np.load('/scratch/wang.zife/YuminLiu/DATA/GCM/GCMdata/tas/processeddata/lats_gcm.npy')
    lons = np.load('/scratch/wang.zife/YuminLiu/DATA/GCM/GCMdata/tas/processeddata/lons_gcm.npy')
    data = np.load(imgpath) # 87.5N to -87.5N by 1, 
    print('data.shape={}'.format(data.shape))
    print('lats={}'.format(lats))
    print('lons[0][-1]={},{}'.format(lons[0],lons[-1]))


def test_plot_worldmap():
    import os
    os.environ['PROJ_LIB'] = 'C:\\WIN10ProgramFiles\\anaconda3\\pkgs\\basemap-1.3.0-py37ha7665c8_0\\Library\\share\\basemap\\'
    from mpl_toolkits.basemap import Basemap
    from matplotlib import pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection 
    import numpy as np


    #lonlat = [235.5,24.5,293.5,49.5]
    lonlat = [-20,-80,330,80]
    watercolor = 'white' # '#46bcec'
    cmap = 'YlOrRd' # 'rainbow' # 'Accent' #'YlGn' #'hsv' #'seismic' # 
    alpha = 0.7
    projection = 'cyl' # 'merc' # 
    resolution = 'i' # 'h' # 'l' #
    area_thresh = 10000
    clim = None
    parallels = np.arange(-80.0,80.0,10.0)
    meridians = np.arange(-20.0,330.0,30.0) # label lons, 60E to -9.5W, or 60E to 350E
        
    #fig = plt.figure(figsize=(18,8))
    fig = plt.figure()
    fig.set_size_inches([18,8])
    ax = fig.add_subplot(111) 
    m = Basemap(llcrnrlon=lonlat[0],llcrnrlat=lonlat[1],urcrnrlon=lonlat[2],urcrnrlat=lonlat[3],
                projection=projection,resolution=resolution,area_thresh=area_thresh)
    m.drawcoastlines(linewidth=1.0,color='k')
    #m.drawcountries(linewidth=1.0,color='k')
    #m.drawstates(linewidth=0.2,color='k')
    #m.drawrivers(color='dodgerblue',linewidth=1.0,zorder=1)
    #m.fillcontinents(color='w',alpha=0.1)
    #m.drawmapboundary(fill_color=watercolor)
    m.fillcontinents(color = 'gray',alpha=1.0,lake_color=watercolor)
    m.drawparallels(parallels,labels=[True,False,False,False],dashes=[1,2])
    m.drawmeridians(meridians,labels=[False,False,False,True],dashes=[1,2])
    # img = np.flipud(img)
    # m.imshow(img,cmap=cmap,alpha=alpha,zorder=1)
    # m.colorbar(fraction=0.02)
    # plot Amazon River basin
    #m.readshapefile('..\\data\\Shapefiles\\AmazonBasin\\amapoly_ivb', 'AmazonBasin', drawbounds=False)
    m.readshapefile('..\\data\\Shapefiles\\AmazonBasinLimits-master\\amazon_sensulatissimo_gmm_v1', 'AmazonBasin', drawbounds=True)
    patches = []
    for info, shape in zip(m.AmazonBasin_info, m.AmazonBasin):
        shape = np.array(shape)
        shape[:,0] += 360 # transform negative longitude to positive
        patches.append(Polygon(xy=shape, closed=True))
        # if info['area']>10:
        #     x,y = zip(*shape)
        #     x = tuple(map(lambda e:e+360,x))
        #     m.plot(x,y,marker=None,color='m')
    ax.add_collection(PatchCollection(patches, facecolor='r', edgecolor='r', alpha=0.5))


    m.readshapefile('..\\data\\Shapefiles\\congo_basin_polyline\\congo_basin_polyline', 'CongoBasin', drawbounds=False)
    patches,shapes = [],[]
    for info, shape in zip(m.CongoBasin_info, m.CongoBasin):
        print('info={},len(shape)={}'.format(info,len(shape)))
        shapes.append(np.array(shape))
    shapes = [shapes[2],shapes[0],shapes[4],shapes[3],shapes[1]]
    shapes = np.concatenate(shapes,axis=0)
    patches.append(Polygon(xy=shapes, closed=True))
    ax.add_collection(PatchCollection(patches, facecolor='m', edgecolor='m', alpha=0.5))

    def draw_rectangle(lats, lons, m, facecolor='red', alpha=0.5, edgecolor='k',fill=False,linewidth=1):
        x, y = m(lons, lats)
        xy = zip(x,y)
        rect = Polygon(list(xy),facecolor=facecolor,alpha=alpha,edgecolor=edgecolor,fill=fill)
        plt.gca().add_patch(rect)
    ## plot enso regions
    nino12_lats,nino12_lons = [-10,0,0,-10],[270,270,280,280]
    nino3_lats,nino3_lons = [-5,5,5,-5],[210,210,270,270]
    nino34_lats,nino34_lons = [-5,5,5,-5],[190,190,240,240]
    #oni_lats,oni_lons = nino34_lats,nino34_lons
    nino4_lats,nino4_lons = [-5,5,5,-5],[160,160,210,210]

    draw_rectangle(nino4_lats,nino4_lons,m,facecolor='g',alpha=0.8,edgecolor='g',fill=True)
    draw_rectangle(nino3_lats,nino3_lons,m,facecolor='b',alpha=0.8,edgecolor='b',fill=True)
    draw_rectangle(nino34_lats,nino34_lons,m,edgecolor='k',fill=False,linewidth=4)
    draw_rectangle(nino12_lats,nino12_lons,m,facecolor='r',alpha=0.8,edgecolor='r',fill=True)

    ## add text
    def add_text(lat, lon, m, text,fontsize=12,**kwargs):
        x, y = m(lon, lat)
        plt.text(x,y,text,**kwargs)
    add_text(lat=0,lon=165,m=m,text='Nino4')
    add_text(lat=0,lon=245,m=m,text='Nino3')
    add_text(lat=10,lon=215,m=m,text='Nino3.4')
    add_text(lat=-15,lon=270,m=m,text='Nino1+2')
    add_text(lat=5,lon=40,m=m,text='Congo Basin')
    add_text(lat=5,lon=310,m=m,text='Amazon Basin')

    savepath = '../data/'
    savename = 'worldmap'
    #plt.savefig(savepath+savename+'.png',dpi=1200,bbox='tight')
    plt.show()

    # fig = plt.figure()
    # fig.set_size_inches([18,8])
    # ax = fig.add_subplot(111)
    
    # ## plot basemap, rivers and countries
    # #m = Basemap(llcrnrlat=19.5, urcrnrlat=26.0, llcrnrlon=99.6, urcrnrlon=107.5, resolution='h')
    # m = Basemap(llcrnrlat=-80, urcrnrlat=80.0, llcrnrlon=-20, urcrnrlon=330, resolution='h')
    # m.arcgisimage(service='World_Shaded_Relief')
    # m.drawrivers(color='dodgerblue',linewidth=1.0,zorder=1)
    # m.drawcountries(color='k',linewidth=1.25)











# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import utils


# indices_df, years = utils.read_enso() # 187001 to 201912
# ## trim to 195001 to 201012
# nino34_df = indices_df[['Nino34']].iloc[960:1692]
# nino3_df = indices_df[['Nino3']].iloc[960:1692]
# nino4_df = indices_df[['Nino4']].iloc[960:1692]
# riverpath = '../data/Climate/RiverFlow/processed/riverflow.csv'
# riverflow_df = pd.read_csv(riverpath,index_col=0,header=0) #
# ## Amazon: 192801 to 201712, Congo: 190301 to 201012
# ## trim to 195001 to 201012
# amazon_df = riverflow_df[['0']].iloc[600:1332]#.to_numpy().reshape((-1,)) # from 195001 to 200512
# congo_df = riverflow_df[['1']].iloc[600:1332]

# fig,axes = plt.subplots(5,1,sharex=True)
# axes[0].plot(nino34_df)
# axes[1].plot(nino3_df)
# axes[2].plot(nino4_df)
# axes[3].plot(amazon_df) 
# axes[4].plot(congo_df)
# plt.show()


# nino34 = nino34_df.to_numpy().reshape((-1,))
# nino3 = nino3_df.to_numpy().reshape((-1,))
# nino4 = nino4_df.to_numpy().reshape((-1,))
# amazon = amazon_df.to_numpy().reshape((-1,))
# congo = congo_df.to_numpy().reshape((-1,))


# nino34_std1,nino34_std2 = np.std(nino34[-240:-120]),np.std(nino34[-120:])
# nino3_std1,nino3_std2 = np.std(nino3[-240:-120]),np.std(nino3[-120:])
# nino4_std1,nino4_std2 = np.std(nino4[-240:-120]),np.std(nino4[-120:])
# amazon_std1,amazon_std2 = np.std(amazon[-240:-120]),np.std(amazon[-120:])
# congo_std1,congo_std2 = np.std(congo[-240:-120]),np.std(congo[-120:])

# nino34_std_change = (nino34_std2-nino34_std1)/nino34_std1
# nino3_std_change = (nino3_std2-nino3_std1)/nino3_std1
# nino4_std_change = (nino4_std2-nino4_std1)/nino4_std1
# amazon_std_change = (amazon_std2-amazon_std1)/amazon_std1
# congo_std_change = (congo_std2-congo_std1)/congo_std1
# print('nino34_std_change={},\nnino3_std_change={},\nnino4_std_change={}'.format(nino34_std_change,nino3_std_change,nino4_std_change))
# print('amazon_std_change={},\ncongo_std_change={}'.format(amazon_std_change,congo_std_change))


# nino34_year = np.sum(nino34.reshape((-1,12)),axis=1)
# nino3_year = np.sum(nino3.reshape((-1,12)),axis=1)
# nino4_year = np.sum(nino4.reshape((-1,12)),axis=1)
# amazon_year = np.sum(amazon.reshape((-1,12)),axis=1)
# congo_year = np.sum(congo.reshape((-1,12)),axis=1)
# year = range(1950,2011)
# fig,axes = plt.subplots(5,1,sharex=True)
# axes[0].plot(year,nino34_year)
# axes[1].plot(year,nino3_year)
# axes[2].plot(year,nino4_year)
# axes[3].plot(year,amazon_year) 
# axes[4].plot(year,congo_year)
# plt.show()

# data = np.load('../data/Climate/Reanalysis/sst_cobe_hadley_noaa_190001-201912_1by1_world.npy')
# print('data.shape={}'.format(data.shape))

# cobe,hadley,noaa = data[:,0,:,:],data[:,1,:,:],data[:,2,:,:]

# cobe_mean = np.mean(cobe[1032:1392,:,:],axis=0)
# cobe_anomaly = np.mean(cobe[1260:1272,:,:],axis=0)-cobe_mean
# fig = plt.figure()
# plt.imshow(cobe_anomaly)
# plt.colorbar(fraction=0.02)
# plt.show()

# hadley_mean = np.mean(hadley[1032:1392,:,:],axis=0)
# hadley_anomaly = np.mean(hadley[1260:1272,:,:],axis=0)-hadley_mean
# fig = plt.figure()
# plt.imshow(hadley_anomaly)
# plt.colorbar(fraction=0.02)
# plt.show()

# noaa_mean = np.mean(noaa[1032:1392,:,:],axis=0)
# noaa_anomaly = np.mean(noaa[1260:1272,:,:],axis=0)-noaa_mean
# fig = plt.figure()
# plt.imshow(noaa_anomaly)
# plt.colorbar(fraction=0.02)
# plt.show()


# from matplotlib.patches import Polygon
# from matplotlib.collections import PatchCollection 
# import numpy as np
# from mpl_toolkits.basemap import Basemap
# from matplotlib import pyplot as plt
 


# lonlat = [235.5,24.5,293.5,49.5]
# lonlat = [0,-80,350,80]
# watercolor = 'white' # '#46bcec'
# cmap = 'YlOrRd' # 'rainbow' # 'Accent' #'YlGn' #'hsv' #'seismic' # 
# alpha = 0.7
# projection = 'merc' # 'cyl' # 
# resolution = 'l' #'i' # 'h'
# area_thresh = 10000
# clim = None

# fig = plt.figure()
# #fig.set_size_inches([17.05,8.15])
# ax = fig.add_subplot(111)
 
# ###plot basemap, rivers and countries
# ###m = Basemap(llcrnrlat=19.5, urcrnrlat=26.0, llcrnrlon=99.6, urcrnrlon=107.5, resolution='h')
# m = Basemap(llcrnrlat=-80, urcrnrlat=80.0, llcrnrlon=-22, urcrnrlon=300, resolution='h')
# m.arcgisimage(service='World_Shaded_Relief')
# m.drawrivers(color='dodgerblue',linewidth=1.0,zorder=1)
# m.drawcountries(color='k',linewidth=1.25)

# m.readshapefile('../data/Shapefiles/AmazonBasinLimits-master/amazon_sensulatissimo_gmm_v1', 'Basin', drawbounds=False)
# patches = []
# for info, shape in zip(m.Basin_info, m.Basin):
#     shape = np.array(shape)
#     shape[:,0] += 360 # transform negative longitude to positive
#     patches.append(Polygon(xy=shape, closed=True))
#     # if info['area']>10:
#     #     x,y = zip(*shape)
#     #     x = tuple(map(lambda e:e+360,x))
#     #     m.plot(x,y,marker=None,color='m')
# ax.add_collection(PatchCollection(patches, facecolor='r', edgecolor='r', alpha=0.5))

# plt.show()





# fig = plt.figure() 
# m = Basemap(llcrnrlon=lonlat[0],llcrnrlat=lonlat[1],urcrnrlon=lonlat[2],urcrnrlat=lonlat[3],
#             projection=projection,resolution=resolution,area_thresh=area_thresh)
# m.drawcoastlines(linewidth=1.0,color='k')
# m.drawcountries(linewidth=1.0,color='k')
# m.drawstates(linewidth=0.2,color='k')
# m.drawrivers(color='dodgerblue',linewidth=1.0,zorder=1)
# #m.fillcontinents(color='w',alpha=0.1)
# m.drawmapboundary(fill_color=watercolor)
# m.fillcontinents(color = 'white',alpha=1.0,lake_color=watercolor)
# # m.drawparallels(parallels,labels=[True,False,False,False],dashes=[1,2])
# # m.drawmeridians(meridians,labels=[False,False,False,True],dashes=[1,2])
# # img = np.flipud(img)
# # m.imshow(img,cmap=cmap,alpha=alpha,zorder=1)
# # m.colorbar(fraction=0.02)














# import pandas as pd
# import numpy as np
# import utils

# column = ['Nino12','Nino3','Nino4'] # ['Nino34_anom'] # 6 is 'Nino34_anom'
# riverflow_df = pd.read_csv('../data/RiverFlow/processed/riverflow.csv',index_col=0,header=0)
# indices_df, _ = utils.read_enso() # 187001 to 201912
# indices_df = indices_df[column] # select input feature
# info_df = pd.read_csv('../data/RiverFlow/processed/info.csv',index_col=0,header=0)
# times_df = pd.read_csv('../data/RiverFlow/processed/times.csv',index_col=0,header=0)
# #times = np.asarray(times_df,dtype=int).reshape((-1,))
# times = list(np.asarray(times_df,dtype=int).reshape((-1,)))+[201901]

# #print('{}'.format(abs(0.324-0.919)/0.919))



# import numpy as np
# import plots

# lonlat = [50.5,-41.5,349.5,37.5] # [50.5,-42.5,349.5,37.5] # [-124.5,24.5,-66.5,49.5] # map area, [left,bottom,right,top]
# lonlat = [50.5,-41.5,-10.5,37.5]
# parallels = np.arange(-40.0,40.0,10.0)
# meridians = np.arange(60.0,350.0,30.0) # label lons, 60E to -9.5W, or 60E to 350E

# masks = np.load('../data/Climate/Reanalysis/masks_cobe_hadley_noaa_uod_1by_world.npy')
# img = masks[0,52:132,50:350] # cobe region 1

# watercolor = 'white' # '#46bcec'
# cmap = 'YlOrRd' # 'rainbow' # 'Accent' #'YlGn' #'hsv' #'seismic' # 
# alpha = 0.7
# projection = 'merc' # 'cyl' # 
# resolution = 'i' # 'h'
# area_thresh = 10000
# clim = None
# pos_lons, pos_lats = [], [] # None, None # to plot specific locations
# verbose = False # True


# title = 'test'
# savepath = None
# savename = None

# plots.plot_map(img,title=title,savepath=savepath,savename=savename,cmap=cmap,alpha=alpha,
#                 lonlat=lonlat,projection=projection,resolution=resolution,area_thresh=area_thresh,
#                 parallels=parallels,meridians=meridians,pos_lons=pos_lons, pos_lats=pos_lats,clim=clim,
#                 watercolor=watercolor,verbose=verbose)









# import numpy as np

# imgpath = '../data/Climate/SST/sst_cobe_hadley_noaa_uod_195001-200512_1by1_world.npy'
# left,right = 50,350
# top,bottom = 52,132 ##50,130
# #lats = np.load('../data/Climate/SST/lats.npy')
# #lons = np.load('../data/Climate/SST/lons.npy')
# data = np.load(imgpath) # 89.5N to -89.5N by 1, 
# ###data = data[:,0:-1,2:-2,:] # 87.5N to -87.5N by 1, 
# #data = data[:,1:2,top:bottom,left:right] # 50.5E to 349.5E, 37.5N to -41.5N, [672,1,80,300], one reanalysis each
# data = data[:,0:3,top:bottom,left:right] # 50.5E to 349.5E, 37.5N to -41.5N, [672,3,80,300], exclude uod  
# #data = data[:,:,top:bottom,left:right] # 50.5E to 349.5E, 37.5N to -41.5N, [672,4,80,300]   
    
# cobe = np.load('/scratch/wang.zife/YuminLiu/DATA/COBE/processed/sst.mon.mean_185001-201912_1by1.npy')
# mask = np.mean(cobe,axis=0)
# mask = mask[top:bottom,left:right]
# mask = np.nan_to_num(mask,nan=1000)
# for mon in range(len(data)):
#     for d in range(data.shape[1]):
#         data[mon,d,:,:][mask==1000] = 0  
# print('data.max={},data.min={}'.format(np.max(data),np.min(data)))


def get_X_y(predictor,predictand):
    #%% tas
    import numpy as np
    import utils
    Ntrain = 600 # int(Ntotal*0.8)
    Nvalid = 36 # int(Ntotal*0.1)
    if predictor=='GCM':
        imgpath = '/scratch/wang.zife/YuminLiu/DATA/GCM/GCMdata/tas/processeddata/32gcms_tas_monthly_1by1_195001-200512_World.npy'
        left,right = 50,350
        top,bottom = 50,130
        lats = np.load('/scratch/wang.zife/YuminLiu/DATA/GCM/GCMdata/tas/processeddata/lats_gcm.npy')
        lons = np.load('/scratch/wang.zife/YuminLiu/DATA/GCM/GCMdata/tas/processeddata/lons_gcm.npy')
        data = np.load(imgpath) # 87.5N to -87.5N by 1, 
        #data = data[:,self.ngcm:self.ngcm+1,top:bottom,left:right] # 50.5E to 349.5E, 37.5N to -41.5N, # one GCM [672,1,80,300]
        data = data[:,:,top:bottom,left:right] # 50.5E to 349.5E, 37.5N to -41.5N, # [672,32,80,300]
        lats = lats[top:bottom]
        lons = lons[left:right]

        cobe = np.load('/scratch/wang.zife/YuminLiu/DATA/COBE/processed/sst.mon.mean_185001-201912_1by1.npy')
        mask = np.mean(cobe,axis=0)
        mask = mask[top+2:bottom+2,left:right]
        mask = np.nan_to_num(mask,nan=1000)
        for mon in range(len(data)):
            for gcm in range(data.shape[1]):
                data[mon,gcm,:,:][mask==1000] = 0

        ## detrend
        mean_m,std_m = {},{}
        for gcm in range(data.shape[1]):
            for i in range(data.shape[2]):
                for j in range(data.shape[3]):
                    data_n = data[:,gcm,i,j]
                    for m in range(12):
                        mean_m[m] = np.mean(data_n[m::12])
                        std_m[m] = np.std(data_n[m::12],ddof=1)
                        data[m::12,gcm,i,j] = (data_n[m::12]-mean_m[m])/(std_m[m]+1e-10)

    elif predictor=='Reanalysis':
        pass
        # #%% combined SST region 1
        # imgpath = '../data/Climate/Reanalysis/sst_cobe_hadley_noaa_uod_195001-200512_1by1_world.npy'
        # left,right = 50,350
        # top,bottom = 52,132 ##50,130
        # lats = np.load('../data/Climate/Reanalysis/lats.npy')
        # lons = np.load('../data/Climate/Reanalysis/lons.npy')
        # data = np.load(imgpath) # 89.5N to -89.5N by 1, 
        # ###data = data[:,0:-1,2:-2,:] # 87.5N to -87.5N by 1, 
        # #data = data[:,1:2,top:bottom,left:right] # 50.5E to 349.5E, 37.5N to -41.5N, [672,1,80,300], one reanalysis each
        # data = data[:,0:3,top:bottom,left:right] # 50.5E to 349.5E, 37.5N to -41.5N, [672,3,80,300], exclude uod  
        # #data = data[:,:,top:bottom,left:right] # 50.5E to 349.5E, 37.5N to -41.5N, [672,4,80,300] 
        # lats = lats[top:bottom]
        # lons = lons[left:right]  
            
        # cobe = np.load('/scratch/wang.zife/YuminLiu/DATA/COBE/processed/sst.mon.mean_185001-201912_1by1.npy')
        # mask = np.mean(cobe,axis=0)
        # mask = mask[top:bottom,left:right]
        # mask = np.nan_to_num(mask,nan=1000)
        # for mon in range(len(data)):
        #     for d in range(data.shape[1]):
        #         data[mon,d,:,:][mask==1000] = 0      
       
        # ### detrend
        # mean_m,std_m = {},{}
        # for d in range(data.shape[1]):
        #     for i in range(data.shape[2]):
        #         for j in range(data.shape[3]):
        #             data_n = data[:,d,i,j]
        #             for m in range(12):
        #                 mean_m[m] = np.mean(data_n[m::12])
        #                 std_m[m] = np.std(data_n[m::12],ddof=1)
        #                 data[m::12,d,i,j] = (data_n[m::12]-mean_m[m])/(std_m[m]+1e-15)
        
    window = 3
    #%% Congo river flow 
    # # window = 3
    # riverpath = '../data/Climate/RiverFlow/processed/riverflow.csv'
    # riverflow_df = pd.read_csv(riverpath,index_col=0,header=0)
    # congo = riverflow_df[['1']].iloc[600-window+1:1272].to_numpy().reshape((-1,)) # from 1950-window to 200512
    # ## moving average, result in 195001 to200512
    # congo = np.array([np.mean(congo[i:i+window]) for i in range(len(congo)-window+1)]).reshape((-1,1))        

    #%% read ENSO index
    ##column = ['Nino34_anom'] # ['Nino34'] # ['Nino3'] # ['Nino12','Nino3','Nino4'] #  6 is 'Nino34_anom'
    indices_df, _ = utils.read_enso() # 187001 to 201912
    # #indices_df = indices_df[column] # select input feature

    # nino3 = indices_df[['Nino3']].iloc[960-window+1:1632].to_numpy().reshape((-1,)) # from 1950-windown to 200512
    # # moving average, result in 195001 to200512
    # nino3 = np.array([np.mean(nino3[i:i+window]) for i in range(len(nino3)-window+1)]).reshape((-1,1))
    if predictand=='Nino34':
        pass
        # nino34 = indices_df[['Nino34']].iloc[960-window+1:1632].to_numpy().reshape((-1,)) # from 1950-windown to 200512
        # # moving average, result in 195001 to200512
        # targets = np.array([np.mean(nino34[i:i+window]) for i in range(len(nino34)-window+1)]).reshape((-1,1))
    # elif predictand=='Nino34_anom':
    #     targets = indices_df[['Nino34_anom']].iloc[960-window+1:1632].to_numpy().reshape((-1,)) # from 1950-windown to 200512
    #     # moving average, result in 195001 to200512
    #     targets = np.array([np.mean(targets[i:i+window]) for i in range(len(targets)-window+1)]).reshape((-1,1))
    # elif predictand=='GCMNino34':
    #     gcmpath = '/scratch/wang.zife/YuminLiu/DATA/GCM/GCMdata/tas/processeddata/32gcms_tas_monthly_1by1_195001-200512_World.npy'
    #     #%% Nino3: average between area in 5N-5S, 150W-90W
    #     #nino3 = np.mean(np.load(gcmpath)[:,0:1,82:94,210:270],axis=(1,2,3)).reshape((-1,1))
    #     #%% Nino34: average between area in 5N-5S, 170W-120W
    #     #targets = np.mean(np.load(gcmpath)[:,:,82:94,190:240],axis=(1,2,3)).reshape((-1,1))

    elif predictand=='SingleGCMNino34':
        gcmpath = '/scratch/wang.zife/YuminLiu/DATA/GCM/GCMdata/tas/processeddata/32gcms_tas_monthly_1by1_195001-200512_World.npy'
        #%% Nino3: average between area in 5N-5S, 150W-90W
        #nino3 = np.mean(np.load(gcmpath)[:,0:1,82:94,210:270],axis=(1,2,3)).reshape((-1,1))
        #%% Nino34: average between area in 5N-5S, 170W-120W
        targets = np.mean(np.load(gcmpath)[:,0:1,82:94,190:240],axis=(1,2,3)).reshape((-1,1))

    # elif predictand=='DMI':
    #     #%% India Dipole Mode Index (DMI)
    #     dmi_df = pd.read_csv('../data/Climate//DMI/processed/DMI_Standard_PSL_Format_187001-202012_series.csv',index_col=0)
    #     dmi = dmi_df.iloc[960-window+1:1632].to_numpy().reshape((-1,)) # from 1950-windown to 200512
    #     # moving average, result in 195001 to200512
    #     targets = np.array([np.mean(dmi[i:i+window]) for i in range(len(dmi)-window+1)]).reshape((-1,1))


    #### detrend y
    mean_m,std_m = {},{}
    for m in range(12):
        mean_m[m] = np.mean(targets[m::12,0])
        std_m[m] = np.std(targets[m::12,0],ddof=1)
        targets[m::12,0] = (targets[m::12,0]-mean_m[m])/(std_m[m]+1e-15)


    # import matplotlib.pyplot as plt
    # fig = plt.figure(figsize=(12,5))
    # plt.plot(nino34,label='Nino34')
    # plt.plot(targets,label='Nino34_anom')
    # plt.savefig('../data/Climate/Nino/processed/nino34_vs_nino34_anom.png',dpi=1200,bbox_inches='tight')
    # plt.show()


    X = data
    y = targets # dmi # nino34_anom # nino34 # amazon # congo # nino3 # 

    ## normalize
    data_mean = np.mean(X[:Ntrain],axis=0)
    data_std = np.std(X[:Ntrain],axis=0)+1e-5
    for t in range(len(X)):
        X[t,:,:,:] = (X[t,:,:,:]-data_mean)/data_std
    y_mean = np.mean(y[:Ntrain],axis=0)
    y_std = np.std(y[:Ntrain],axis=0)
    y = (y-y_mean)/y_std

    ## add noise to input
    #noise = np.random.normal(loc=0.0,scale=self.noise_std,size=X.shape)
    #X += noise

    return X, y, lats,lons








# import numpy as np

# year_months = []
# for year in range(1950,2006):
#     for month in range(1,13):
#         if month<10:
#             year_months.append(int(str(year)+'0'+str(month)))
#         else:
#             year_months.append(int(str(year)+str(month)))
# year_months = np.array(year_months)

# ##np.save('../data/Climate/year_months_195001-200512.npy',year_months)

# a = range(50)


# import pandas as pd

# data = pd.read_csv('../data/Climate/Nino/processed/Nino34_anom_187001-201912_series.csv',index_col=0,header=0)


# import matplotlib.pyplot as plt
# #plt.switch_backend('Qt5Agg')
# #%matplotlib qt
# fig = plt.figure()
# plt.plot(data)
# plt.show()


def get_DMI():
    import numpy as np
    import pandas as pd

    filepath = "../data/Climate/DMI/"
    #filename = "DMI Standard PSL Format"
    filename = "DMI Western Indian Ocean PSL Format"
    #filename = "DMI Eastern Indian Ocean Standard PSL Format"
    month_names = np.array(['Year','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    data = []
    with open(filepath+filename+".txt",'r') as f:
        lines = f.readlines()#.split(' ')
        for line in lines:
            line = line.split(' ')
            line = [s for s in line if s!='']
            data.append(line)

    data = np.array(data)
    years = data[:,0]
    data = data[:,1:]
    data_df = pd.DataFrame(data=data,index=years,columns=month_names[1:],dtype=np.float)
    #data_df.to_csv(filepath+"processed/"+filename.replace(' ','_')+'.csv')

# =============================================================================
# import numpy as np
# import plots
# 
# path = '../results/myCNN/lag_0/SingleGCM/'
# filename = 'myCNN_amazon_saliency_maps'
# 
# salencymap_mgcm = np.load('../results/myCNN/lag_0/2021-02-17_23.06.48.428983_gcm_masked_amazon_region1/2021-02-17_23.06.48.428983_Saliency/myCNN_amazon_saliency_maps.npy')
# 
# salencymap_mgcm = np.mean(salencymap_mgcm,axis=(0,1))
# 
# saliencymaps_sgcm = []
# for ngcm in range(32): # [13,15]:# 
#     filepath = path+'{}/{}_Saliency/'.format(ngcm,ngcm)
#     saliencymap = np.load(filepath+filename+'.npy')
#     saliencymap = plots.get_saliency(saliencymap,fillnan=0.0,threshold=0.15) # [Nmonth,Nlat,Nlon]
#     #saliencymap = np.mean(saliencymap,axis=0)
#     saliencymap = saliencymap[0]
#     saliencymaps_sgcm.append(saliencymap)
# 
# ngcm = 13
# saliencymap_sgcm = saliencymaps_sgcm[ngcm]
# 
# 
# import matplotlib
# matplotlib.use('Qt5Agg')
# ## gcm region 1
# lonlat = [50.5,-41.5,349.5,37.5] # [50.5,-42.5,349.5,37.5] # [-124.5,24.5,-66.5,49.5] # map area, [left,bottom,right,top]
# parallels = np.arange(-40.0,40.0,10.0)
# meridians = np.arange(60.0,350.0,30.0) # label lons, 60E to -9.5W, or 60E to 350E
# 
# watercolor = 'white' # '#46bcec'
# cmap = 'rainbow' # 'YlOrRd' # 'Accent' #'YlGn' #'hsv' #'seismic' # 
# alpha = 0.7
# projection = 'merc' # 'cyl' # 
# resolution = 'i' # 'h'
# area_thresh = 10000
# clim = None
# pos_lons, pos_lats = [], [] # None, None # to plot specific locations
# verbose = True # False # 
# 
# 
# img = (saliencymap_sgcm-salencymap_mgcm)
# #title = 'Saliency Map Difference GCM {} to MultiGCM'.format(ngcm)
# title = 'Single Month {} Saliency Map Difference GCM {} to MultiGCM'.format(0,ngcm)
# savepath = '../results/myCNN/lag_0/SingleGCM/postprocessed/' # None # 
# savename = title.lower().replace(' ','_') # None
# plots.plot_map(img,title=title,savepath=savepath,savename=savename,cmap=cmap,alpha=alpha,
#                  lonlat=lonlat,projection=projection,resolution=resolution,area_thresh=area_thresh,
#                  parallels=parallels,meridians=meridians,pos_lons=pos_lons, pos_lats=pos_lats,clim=clim,
#                  watercolor=watercolor,verbose=verbose)
# =============================================================================
        
        
        
# =============================================================================
# import os
# import numpy as np
# import datasets
# 
# path = '../results/myCNN/lag_0/SingleGCM/'
# test_dataset = datasets.myDataset_CNN(fold='test',window=3,noise_std=0.0)
# ys = []
# for y in test_dataset:
#     ys.append(y[1].numpy()[0])
# ys = np.array(ys)
# 
# preds_mgcm = np.load('../results/myCNN/lag_0/2021-02-17_23.06.48.428983_gcm_masked_amazon_region1/pred_results_RMSE0.2839257769608812.npz')['preds']
# 
# preds_sgcm = []
# rmses_sgcm = []
# for ngcm in range(32):
#     filepath = path+str(ngcm)+'/'
#     for file in os.listdir(filepath):
#         if file.startswith('pred_results_RMSE'):
#             res = np.load(filepath+file)
#             preds_sgcm.append(res['preds'])
#             rmses_sgcm.append(res['rmse'])
# 
# preds_sgcm = np.stack(preds_sgcm,axis=1)
# preds_sgcm_mean = np.mean(preds_sgcm,axis=1)
# preds_sgcm_std = np.std(preds_sgcm,axis=1)
# 
# rmses_sgcm = np.stack(rmses_sgcm,axis=0)
# rmses_sgcm_mean = np.mean(rmses_sgcm)
# 
# import matplotlib.pyplot as plt
# 
# fig = plt.figure(figsize=(12,5))
# plt.plot(ys,'--b',label='Groundtruth')
# plt.plot(preds_mgcm,'-r',label='MultiGCM Prediction')
# plt.plot(preds_sgcm_mean,'-g',label='SingleGCM Prediction')
# plt.fill_between(range(len(ys)),y1=preds_sgcm_mean+preds_sgcm_std,y2= preds_sgcm_mean-preds_sgcm_std,alpha=0.3,color='g',label='SingleGCM 1 std')
# plt.legend()
# plt.xlabel('Month')
# plt.ylabel('River flow')
# plt.title('Groundtruth vs Predictions')
# plt.show()
# savepath = '../results/myCNN/lag_0/SingleGCM/'
# savename = 'pred_vs_time'
# plt.savefig(savepath+savename+'.png',dpi=1200,bbox_inches='tight')
# 
# fig = plt.figure(figsize=(12,5))
# plt.plot(ys,'--b',label='Groundtruth')
# plt.plot(preds_mgcm,'-r',label='MultiGCM Prediction')
# plt.plot(preds_sgcm[:,13],'-g',label='GCM_13 Prediction')
# plt.plot(preds_sgcm[:,15],'-y',label='GCM_15 Prediction')
# plt.legend()
# plt.xlabel('Month')
# plt.ylabel('River flow')
# plt.title('Groundtruth vs Predictions')
# plt.show()
# savepath = '../results/myCNN/lag_0/SingleGCM/'
# savename = 'pred_vs_time_sample_gcms'
# plt.savefig(savepath+savename+'.png',dpi=1200,bbox_inches='tight')
# 
# 
# 
# fig = plt.figure(figsize=(12,5))
# plt.axhline(y=0.2839257769608812,xmin=0,xmax=32,color='r',linestyle='-',label='MultiGCM')
# plt.axhline(y=rmses_sgcm_mean,xmin=0,xmax=32,color='g',linestyle='-',label='SingleGCM Mean')
# plt.scatter(x=range(len(rmses_sgcm)),y=rmses_sgcm,c='g',label='SingleGCM')
# plt.title('RMSE When Using Single GCM or Multiple GCMs')
# plt.xlabel('GCM')
# plt.ylabel('RMSE')
# plt.legend()
# plt.show()
# savepath = '../results/myCNN/lag_0/SingleGCM/'
# savename = 'rmses'
# plt.savefig(savepath+savename+'.png',dpi=1200,bbox_inches='tight')
# =============================================================================


# =============================================================================
# #%%  cal std
# import numpy as np
# import matplotlib
# matplotlib.use('Qt5Agg')
# import matplotlib.pyplot as plt
# import datasets
# #preds1 = np.load('../results/myCNN/lag_0/gcm_masked_amazon_region1_std/2021-02-24_12.15.31.890235/pred_results_RMSE0.29375510890548745.npz')['preds']
# #preds2 = np.load('../results/myCNN/lag_0/gcm_masked_amazon_region1_std/2021-02-24_12.29.05.564894/pred_results_RMSE0.31581716097004175.npz')['preds']
# #preds3 = np.load('../results/myCNN/lag_0/gcm_masked_amazon_region1_std/2021-02-24_12.48.56.271192/pred_results_RMSE0.2987614734765478.npz')['preds']
# #preds4 = np.load('../results/myCNN/lag_0/gcm_masked_amazon_region1_std/2021-02-24_13.08.22.900733/pred_results_RMSE0.37477424761496386.npz')['preds']
# #preds5 = np.load('../results/myCNN/lag_0/gcm_masked_amazon_region1_std/2021-02-24_13.22.28.931243/pred_results_RMSE0.3778239265291365.npz')['preds']
# preds1 = np.load('../results/myCNN/lag_0/sst_masked_amazon_region1_std/2021-02-24_18.08.37.842371/pred_results_RMSE0.3196171865754908.npz')['preds']
# preds2 = np.load('../results/myCNN/lag_0/sst_masked_amazon_region1_std/2021-02-24_18.15.34.429151/pred_results_RMSE0.29547176805139297.npz')['preds']
# preds3 = np.load('../results/myCNN/lag_0/sst_masked_amazon_region1_std/2021-02-24_18.22.38.995020/pred_results_RMSE0.2857826713285732.npz')['preds']
# preds4 = np.load('../results/myCNN/lag_0/sst_masked_amazon_region1_std/2021-02-24_18.37.45.805113/pred_results_RMSE0.29694662090486296.npz')['preds']
# preds5 = np.load('../results/myCNN/lag_0/sst_masked_amazon_region1_std/2021-02-24_18.49.04.326581/pred_results_RMSE0.31434949838558285.npz')['preds']
# 
# test_dataset = datasets.myDataset_CNN(fold='test',window=3,noise_std=0.0)
# ys = []
# for y in test_dataset:
#     ys.append(y[1].numpy()[0])
# ys = np.array(ys)
# 
# preds = np.stack((preds1,preds2,preds3,preds4,preds5),axis=1)
# preds_mean = np.mean(preds,axis=1)
# preds_std = np.std(preds,axis=1)
# 
# fig = plt.figure()
# plt.plot(ys,'--b',label='Groundtruth')
# plt.plot(preds_mean,'-r',label='Prediction')
# plt.fill_between(range(len(ys)),y1=preds_mean+preds_std,y2= preds_mean-preds_std,alpha=0.3,color='r',label='1 std')
# plt.legend()
# plt.xlabel('Month')
# plt.ylabel('River flow')
# plt.title('Groundtruth vs Prediction')
# plt.show()
# savepath = '../results/myCNN/lag_0/sst_masked_amazon_region1_std/'
# savename = 'pred_vs_time_std'
# plt.savefig(savepath+savename+'.png',dpi=1200,bbox_inches='tight')
# =============================================================================

# =============================================================================
# import numpy as np
# import plots
# import matplotlib.pyplot as plt
# 
# savepath = None
# pos_lons,pos_lats = [],[]
# 
# path = '../results/myCNN/lag_0/2021-01-31_17.09.43.637895_sst_amazon_region1/2021-01-31_17.09.43.637895_Saliency/'
# saliencymap = np.load(path+'myCNN_amazon_saliency_maps.npy')
# 
# ## region 1
# left,right = 50,350
# top,bottom = 52,132 # lat from 37.5N to -41.5S?, 50.5E to 349.5E
# lonlat = [50.5,-41.5,349.5,37.5] # [-124.5,24.5,-66.5,49.5] # map area, [left,bottom,right,top]
# parallels = np.arange(-40.0,40.0,10.0)
# meridians = np.arange(60.0,350.0,30.0) # label lons, 60E to -9.5W, or 60E to 350E
# ## region 3
# #left,right = 150,260
# #top,bottom = 60,130
# #lonlat = [150.5,-39.5,259.5,29.5] # # map area, [left,bottom,right,top]
# #parallels = np.arange(-30.0,30.0,10.0)
# #meridians = np.arange(150.0,300.0,30.0) # label lons, 60E to -9.5W, or 60E to 350E
# 
# imgpath = '../data/Climate/SST/sst_cobe_hadley_noaa_uod_195001-200512_1by1_world.npy'
# lats = np.load('../data/Climate/SST/lats.npy')
# lons = np.load('../data/Climate/SST/lons.npy')
# data = np.load(imgpath) # 89.5N to -89.5N by 1, 
# #data = data[:,0:-1,top:bottom,left:right] # 150.5E to 259.5E, 29.5N to -39.5N, [672,3,70,110], exclude uod 
# #data = data[:,:,top:bottom,left:right] # 150.5E to 259.5E, 29.5N to -39.5N, [672,3,70,110], 
# cobe = data[:,0,:,:] # SST
# hadley = data[:,1,:,:] # SST+LST
# noaa = data[:,2,:,:] # SST
# uod = data[:,3,:,:] # LST
# 
# masks = np.mean(data,axis=0)
# cobe_mask = np.mean(cobe,axis=0)
# hadley_mask = np.mean(hadley,axis=0)
# noaa_mask = np.mean(noaa,axis=0)
# uod_mask = np.mean(uod,axis=0)
# np.save('../data/Climate/SST/masks_cobe_hadley_noaa_uod_1by_world.npy',masks)
# 
# for mon in range(len(saliencymap)):
#     for dn in range(saliencymap.shape[1]):
#         saliencymap[mon,dn,:,:][cobe_mask!=0] = 0
#         
# 
# img = np.sum(np.sum(saliencymap,axis=0),axis=0) # cobe_mask # saliencymap[0,1,:,:]
# #img = img/np.max(img)
# fig = plt.figure()
# plt.imshow(img)
# plt.colorbar(fraction=0.02)
# 
# 
# cmap = 'YlOrRd' # 'viridis' # 'rainbow' # 'Accent' #'YlGn' #'hsv' #'seismic' # 
# alpha = 0.7
# projection = 'merc' # 'cyl' # 
# #lonlat=[235.5,24.5,293.5,49.5]
# #lonlat = [-124.5,24.5,-66.5,49.5] # map area, [left,bottom,right,top]
# resolution = 'i' # 'h'
# area_thresh = 10000
# clim = None
# #parallels = np.arange(20.0,51.0,10.0)
# #meridians = np.arange(-125.0,-60.0,10.0) # label lons, 125W to 60W, or 235E to 300E
# 
# watercolor = '#46bcec' # 'white' # 
# verbose = True
# 
# 
# img = data[0,3,:,:]
# title = 'Input Reanalysis SST'
# savename = title.lower().replace(' ','_') # None
# plots.plot_map(img,title=title,savepath=savepath,savename=savename,cmap=cmap,alpha=alpha,
#                  lonlat=lonlat,projection=projection,resolution=resolution,area_thresh=area_thresh,
#                  parallels=parallels,meridians=meridians,pos_lons=pos_lons, pos_lats=pos_lats,clim=clim,
#                  watercolor=watercolor,verbose=verbose)
# =============================================================================


# =============================================================================
# import numpy as np
# import matplotlib.pyplot as plt
# import plots
# 
# #savepath = '/scratch/wang.zife/YuminLiu/myPythonFiles/SaliencyMap/results/Climate/PRISM_GCM/YNet/ppt/scale8/2020-12-25_21.18.55.261503/' # None # 
# #results = np.load('/scratch/wang.zife/YuminLiu/myPythonFiles/SaliencyMap/results/Climate/PRISM_GCM/YNet/ppt/scale8/2020-12-25_21.18.55.261503/pred_results_MSE1.4519149793605919.npz')
# #data = np.load('/scratch/wang.zife/YuminLiu/myPythonFiles/SaliencyMap/data/Climate/PRISM_GCMdata/ppt/0.125by0.125/test/prism_gcm_log1p_prmean_monthly_0.125to1.0_195001-200512_USA_month672.npz')
# #mask = np.load('/scratch/wang.zife/YuminLiu/myPythonFiles/SaliencyMap/data/Climate/PRISM_GCMdata/ppt/0.125by0.125/prism_USAmask_0.125by0.125.npz')
# #savepath = '/scratch/wang.zife/YuminLiu/myPythonFiles/SaliencyMap/results/Climate/PRISM_GCM/YNet/tmax/scale8/2021-01-11_17.42.26.526113/' # None # 
# #results = np.load('/scratch/wang.zife/YuminLiu/myPythonFiles/SaliencyMap/results/Climate/PRISM_GCM/YNet/tmax/scale8/2021-01-11_17.42.26.526113/pred_results_MSE2.6174423025033082.npz')
# #data = np.load('/scratch/wang.zife/YuminLiu/myPythonFiles/SaliencyMap/data/Climate/PRISM_GCMdata/tmax/0.125by0.125/test/prism_gcm_divide50_tmax_monthly_0.125to1.0_195001-200512_USA_month672.npz')
# #mask = np.load('/scratch/wang.zife/YuminLiu/myPythonFiles/SaliencyMap/data/Climate/PRISM_GCMdata/tmax/0.125by0.125/prism_USAmask_0.125by0.125.npz')
# savepath = '/scratch/wang.zife/YuminLiu/myPythonFiles/SaliencyMap/results/Climate/PRISM_GCM/YNet/tmin/scale8/2021-01-11_17.33.42.045321/' # None # 
# results = np.load('/scratch/wang.zife/YuminLiu/myPythonFiles/SaliencyMap/results/Climate/PRISM_GCM/YNet/tmin/scale8/2021-01-11_17.33.42.045321/pred_results_MSE2.0315981468925437.npz')
# data = np.load('/scratch/wang.zife/YuminLiu/myPythonFiles/SaliencyMap/data/Climate/PRISM_GCMdata/tmin/0.125by0.125/test/prism_gcm_divide50_tmin_monthly_0.125to1.0_195001-200512_USA_month672.npz')
# mask = np.load('/scratch/wang.zife/YuminLiu/myPythonFiles/SaliencyMap/data/Climate/PRISM_GCMdata/tmin/0.125by0.125/prism_USAmask_0.125by0.125.npz')
# 
# gcms = data['gcms']
# prisms = data['prism']
# y_preds = results['y_preds']
# mask_HR = mask['USAmask_HR']
# mask_LR = mask['USAmask_LR']
# 
# loc_lats, loc_lons = [30,120],[20,240] 
# latlonpath = '../data/Climate/PRISM_GCMdata/ppt/0.125by0.125/'#.format(variable)
# lats_prism = np.load(latlonpath+'lats_prism.npy')
# lons_prism = np.load(latlonpath+'lons_prism.npy')-360
# pos_lats = lats_prism[loc_lats]
# pos_lons = lons_prism[loc_lons]
# 
# #gcms_mean = np.mean(np.expm1(gcms*5.0),axis=0) # [Nlat,Nlon]
# #prism = np.expm1(prisms*5.0)[0,:,:] # [0.0,1.0]-->[0.0,R], unit: mm/day
# #y_pred = y_preds[-1,:,:] # [0.0,1.0]-->[0.0,R], unit: mm/day
# gcms_mean = np.mean(gcms*50,axis=0) # [Nlat,Nlon]
# prism = 50*prisms[0,:,:] # [0.0,1.0]-->[0.0,R], unit: mm/day
# y_pred = y_preds[-1,:,:] # [0.0,1.0]-->[0.0,R], unit: mm/day
# 
# rmse = np.sqrt(np.sum((prism-y_pred)**2))
# 
# gcms_mean[mask_LR==0] = np.nan
# prism[mask_HR==0] = np.nan
# y_pred[mask_HR==0] = np.nan
# 
# cmap = 'YlOrRd' # 'viridis' # 'rainbow' # 'Accent' #'YlGn' #'hsv' #'seismic' # 
# alpha = 0.7
# projection = 'merc' # 'cyl' # 
# #lonlat=[235.5,24.5,293.5,49.5]
# lonlat = [-124.5,24.5,-66.5,49.5] # map area, [left,bottom,right,top]
# resolution = 'i' # 'h'
# area_thresh = 10000
# clim = None
# parallels = np.arange(20.0,51.0,10.0)
# meridians = np.arange(-125.0,-60.0,10.0) # label lons, 125W to 60W, or 235E to 300E
# watercolor = '#46bcec' # 'white' # 
# verbose = True
# 
# 
# img = gcms_mean
# title = 'Input GCM Mean'
# savename = title.lower().replace(' ','_') # None
# plots.plot_map(img,title=title,savepath=savepath,savename=savename,cmap=cmap,alpha=alpha,
#                  lonlat=lonlat,projection=projection,resolution=resolution,area_thresh=area_thresh,
#                  parallels=parallels,meridians=meridians,pos_lons=pos_lons, pos_lats=pos_lats,clim=clim,
#                  watercolor=watercolor,verbose=verbose)
# 
# img = y_pred
# title = 'Prediction'
# savename = title.lower().replace(' ','_') # None
# plots.plot_map(img,title=title,savepath=savepath,savename=savename,cmap=cmap,alpha=alpha,
#                  lonlat=lonlat,projection=projection,resolution=resolution,area_thresh=area_thresh,
#                  parallels=parallels,meridians=meridians,pos_lons=pos_lons, pos_lats=pos_lats,clim=clim,
#                  watercolor=watercolor,verbose=verbose)
# 
# img = prism
# title = 'Ground Truth'
# savename = title.lower().replace(' ','_') # None
# plots.plot_map(img,title=title,savepath=savepath,savename=savename,cmap=cmap,alpha=alpha,
#                  lonlat=lonlat,projection=projection,resolution=resolution,area_thresh=area_thresh,
#                  parallels=parallels,meridians=meridians,pos_lons=pos_lons, pos_lats=pos_lats,clim=clim,
#                  watercolor=watercolor,verbose=verbose)
# 
# img = abs(y_pred-prism)
# title = 'Absolute Difference between Prediction and Ground Truth'
# savename = title.lower().replace(' ','_') # None
# plots.plot_map(img,title=title,savepath=savepath,savename=savename,cmap=cmap,alpha=alpha,
#                  lonlat=lonlat,projection=projection,resolution=resolution,area_thresh=area_thresh,
#                  parallels=parallels,meridians=meridians,pos_lons=pos_lons, pos_lats=pos_lats,clim=clim,
#                  watercolor=watercolor,verbose=verbose)
# 
# =============================================================================

# =============================================================================
# import os
# import numpy as np
# 
# datapath = '/scratch/wang.zife/YuminLiu/myPythonFiles/Downscaling/data/Climate/PRISM_GCMdata/ppt/0.125by0.125/test/'
# filenames = [f for f in os.listdir(datapath) if f.endswith('.npz')]
# filenames = sorted(filenames)
# 
# gcms = []
# prism = []
# for filename in filenames:
#     data = np.load(datapath+filename)
#     gcms.append(data['gcms'])
#     prism.append(data['prism'])
# elevation = data['elevation']
# climatology = data['climatology']
# gcms = np.stack(gcms,axis=0)
# prism = np.stack(prism,axis=0)
# 
# import matplotlib.pyplot as plt
# fig = plt.figure()
# plt.imshow(np.sum(np.std(gcms,axis=1),axis=0))
# plt.title('GCMs ppt')
# plt.show()
# 
# fig = plt.figure()
# plt.imshow(np.sum(prism,axis=(0,1)))
# plt.title('PRISM ppt')
# plt.show()
# 
# 
# datapath = '/scratch/wang.zife/YuminLiu/myPythonFiles/Downscaling/data/Climate/PRISM_GCMdata/tmin/0.125by0.125/test/'
# filenames = [f for f in os.listdir(datapath) if f.endswith('.npz')]
# filenames = sorted(filenames)
# 
# gcms = []
# prism = []
# for filename in filenames:
#     data = np.load(datapath+filename)
#     gcms.append(data['gcms'])
#     prism.append(data['prism'])
# elevation = data['elevation']
# climatology = data['climatology']
# gcms = np.stack(gcms,axis=0)
# prism = np.stack(prism,axis=0)
# 
# import matplotlib.pyplot as plt
# fig = plt.figure()
# plt.imshow(np.sum(np.std(gcms,axis=1),axis=0))
# plt.title('GCMs tmin')
# plt.show()
# 
# fig = plt.figure()
# plt.imshow(np.sum(prism,axis=(0,1)))
# plt.title('PRISM tmin')
# plt.show()
# 
# 
# #import matplotlib.pyplot as plt
# #fig = plt.figure()
# #plt.imshow(data[0,0,:,:])
# #plt.title('0')
# #plt.show()
# #import matplotlib.pyplot as plt
# #fig = plt.figure()
# #plt.imshow(data[0,3,:,:])
# #plt.title('3')
# #plt.show()
# 
# =============================================================================

# =============================================================================
# def visualize_sst():
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from skimage.transform import resize
#     
#     prefix = '/scratch/wang.zife/YuminLiu/DATA/'
#     
#     #%% COBE, ocean only, 1 by 1, 89.5N to -89.5N, 0.5E to 359.5E
#     cobe = np.load(prefix+'COBE/processed/sst.mon.mean_185001-201912_1by1.npy')
#     cobe_lats = np.load(prefix+'COBE/processed/lats.npy')
#     cobe_lons = np.load(prefix+'COBE/processed/lons.npy')
#     cobe_time = np.load(prefix+'COBE/processed/time.npy')
#     cobe[cobe==np.nan] = 0
#     
#     fig = plt.figure()
#     plt.imshow(cobe[0,:,:])
#     plt.title('COBE')
#     plt.show()
#     
#     #%% Hadley, both ocean and land, 1 by 1, 89.5N - -89.5N, 0.5E-359.5E
#     hadley = np.load(prefix+'Hadley-NOAA/processed/MODEL.SST.HAD187001-198110.OI198111-202008_1by1.npy')
#     hadley_lats = np.load(prefix+'Hadley-NOAA/processed/lats.npy')
#     hadley_lons = np.load(prefix+'Hadley-NOAA/processed/lons.npy')
#     hadley_time = np.load(prefix+'Hadley-NOAA/processed/time.npy')
#     hadley[hadley==np.nan] = 0
#     
#     fig = plt.figure()
#     plt.imshow(hadley[0,:,:])
#     plt.title('Hadley')
#     plt.show()
#     
#     #%% NOAA, ocean only, 2 by 2, 88N to -88N, 0E to 358E
#     noaa = np.load(prefix+'NOAA/processed/sst.mnmean_185401-202012_2by2.npy')
#     noaa_lats = np.load(prefix+'NOAA/processed/lats.npy')
#     noaa_lons = np.load(prefix+'NOAA/processed/lons.npy')
#     noaa_time = np.load(prefix+'NOAA/processed/time.npy')
#     noaa[noaa==np.nan] = 0
#     
#     fig = plt.figure()
#     plt.imshow(noaa[0,:,:])
#     plt.title('NOAA')
#     plt.show()
#     
#     noaa_scaled = np.transpose(resize(np.transpose(noaa,axes=(1,2,0)),hadley.shape[1:],order=1,preserve_range=True),axes=(2,0,1))
#     
#     #%% UoD, land only, 0.5 by 0.5, 89.75N to -89.75N, 0.25E to 359.75E
#     uod = np.load(prefix+'UoD/processed/air.mon.mean.v501_190001-201712_0.5by0.5.npy')
#     uod_lats = np.load(prefix+'UoD/processed/lats.npy')
#     uod_lons = np.load(prefix+'UoD/processed/lons.npy')
#     uod_time = np.load(prefix+'UoD/processed/time.npy')
#     uod[uod==np.nan] = 0
#     
#     fig = plt.figure()
#     plt.imshow(uod[0,:,:])
#     plt.title('UoD')
#     plt.show()
#     
#     uod_scaled = np.transpose(resize(np.transpose(uod,axes=(1,2,0)),hadley.shape[1:],order=1,preserve_range=True),axes=(2,0,1))
# =============================================================================

#%%
# =============================================================================
# def combine_sst():
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from skimage.transform import resize
#     
#     prefix = '/scratch/wang.zife/YuminLiu/DATA/'
#     cobe = np.load(prefix+'COBE/processed/sst.mon.mean_185001-201912_1by1.npy') # Ocean only
#     cobe_lats = np.load(prefix+'COBE/processed/lats.npy')
#     cobe_lons = np.load(prefix+'COBE/processed/lons.npy')
#     cobe_time = np.load(prefix+'COBE/processed/time.npy')
#     hadley = np.load(prefix+'Hadley-NOAA/processed/MODEL.SST.HAD187001-198110.OI198111-202008_1by1.npy') # land and Ocean
#     noaa = np.load(prefix+'NOAA/processed/sst.mnmean_185401-202012_2by2.npy') # Ocean only
#     uod = np.load(prefix+'UoD/processed/air.mon.mean.v501_190001-201712_0.5by0.5.npy') # land only
#     cobe = np.nan_to_num(cobe,nan=0)
#     hadley = np.nan_to_num(hadley,nan=0)
#     noaa = np.nan_to_num(noaa,nan=0)
#     uod = np.nan_to_num(uod,nan=0)
#     noaa_scaled = np.transpose(resize(np.transpose(noaa,axes=(1,2,0)),hadley.shape[1:],order=1,preserve_range=True),axes=(2,0,1))
#     uod_scaled = np.transpose(resize(np.transpose(uod,axes=(1,2,0)),hadley.shape[1:],order=1,preserve_range=True),axes=(2,0,1))
#     ## trim to 195001 to 200512
#     cobe = cobe[1200:1872,:,:]
#     hadley = hadley[960:1632,:,:]
#     noaa_scaled = noaa_scaled[1152:1824,:,:]
#     uod_scaled = uod_scaled[600:1272,:,:]
#     
#     sst = np.stack((cobe,hadley,noaa_scaled,uod_scaled),axis=1)
#     np.save('/scratch/wang.zife/YuminLiu/myPythonFiles/SaliencyMap/data/Climate/SST/sst_cobe_hadley_noaa_uod_195001-200512_1by1_world.npy',sst)
#     np.save('/scratch/wang.zife/YuminLiu/myPythonFiles/SaliencyMap/data/Climate/SST/lats.npy',cobe_lats)
#     np.save('/scratch/wang.zife/YuminLiu/myPythonFiles/SaliencyMap/data/Climate/SST/lons.npy',cobe_lons)
#     np.save('/scratch/wang.zife/YuminLiu/myPythonFiles/SaliencyMap/data/Climate/SST/time.npy',cobe_time[1200:1872])
# =============================================================================

#hadley_lats = np.load(prefix+'Hadley-NOAA/processed/lats.npy') 
#noaa_lats = np.load(prefix+'NOAA/processed/lats.npy') 
#uod_lats = np.load(prefix+'UoD/processed/lats.npy') 
#
#fig = plt.figure()
#plt.imshow(cobe[0,:,:])
#plt.title('cobe')
#
#fig = plt.figure()
#plt.imshow(hadley[0,:,:])
#plt.title('hadley')
#
#fig = plt.figure()
#plt.imshow(noaa[0,:,:])
#plt.title('noaa')

# =============================================================================
# #%% ICOADS 2 Degree Enhanced dataset
# ## https://psl.noaa.gov/data/gridded/data.coads.2deg.html
# from netCDF4 import Dataset,num2date
# from ncdump import ncdump
# import numpy as np
# import os
# 
# prefix = '/scratch/wang.zife/YuminLiu/DATA/' # '/scratch/wang.zife/YuminLiu/myPythonFiles/RiverFlow/data/' # 
# folder = 'PRISM/monthly/'
# dataname = 'PRISM_tdmean_stable_4kmM3_1950-2005_monthly_total' # 'sst.mnmean'
# varname = 'tdmean' # dataname.split('.')[0]
# #savename = dataname+'_190001-201712_0.5by0.5'
# datapath = prefix+folder+'tdmean/'
# savepath = prefix+folder+'processed/'
# dataset = Dataset(datapath+dataname+'.nc',mode='r')
# nc_attrs, nc_dims, nc_vars = ncdump(dataset)
# #### 'Air Temperature Monthly Mean at Surface', unites: degree Celsius
# sst = dataset.variables[varname][:] # air
# ##mask = np.ma.getmask(np.sum(np.abs(sst),axis=0))
# sst = np.ma.filled(sst,np.nan)
# #### time from 180001 to 202005, totally 2645 months
# time = dataset.variables['time'][:] # 'days since 1800-1-1'
# #time = np.ma.filled(time,1e+20)
# time_unit = dataset.variables['time'].units # get unit 
# tvalue = num2date(time,units=time_unit)
# times= np.array([int(i.strftime("%Y%m")) for i in tvalue]) # to display dates as string
# 
# #### lon, from 1E to 359E by 2, 180 points
# lons = dataset.variables['lon'][:]
# lons = np.ma.filled(lons,1e+20)
# #### lat, from 89N to -89N by 2, 90 points
# lats = dataset.variables['lat'][:]
# lats = np.ma.filled(lats,1e+20)
# 
# #sst = np.flipud(sst)
# #lats = np.flipud(lats)
# 
# ####
# import matplotlib.pyplot as plt
# 
# fig = plt.figure()
# plt.imshow(sst[0,:,:])
# plt.show()
# 
# #fig = plt.figure()
# #plt.imshow(np.sum(sst[6::12,:,:]-sst[0::12,:,:],axis=0))
# #plt.show()
# #
# #fig = plt.figure()
# #plt.imshow(sst[6,:,:]-sst[0,:,:])
# #plt.show()
# 
# #if savepath:
# #    if not os.path.exists(savepath):
# #        os.makedirs(savepath)
# #    np.save(savepath+savename+'.npy',sst)
# #    np.save(savepath+'lats.npy',lats)
# #    np.save(savepath+'lons.npy',lons)
# #    np.save(savepath+'time.npy',times)
# =============================================================================
    










#%%
#import numpy as np
#datapath = '/scratch/wang.zife/YuminLiu/myPythonFiles/RiverFlow/data/'
#data = np.load(datapath+'NOAA/ExtendedReconstructedSSTv4/processed/noaa_extendedresconstructed_2degree_sst.mnmean.npz')
#sst = data['sst'] # time from 185401 to 202002, totally 1994 months
#lons = data['lons']
#lats = data['lats']
#time = data['times']
#mask = data['mask']
#
#import matplotlib
#matplotlib.use('Qt5Agg')
#import matplotlib.pyplot as plt
#fig = plt.figure()
#plt.imshow(sst[0,:,:])
#plt.show()




# =============================================================================
# def GCM_enso_riverflow_xcorr():
#     import numpy as np
#     import pandas as pd
#     import utils
#     
#     window = 3
#     column = ['Nino3'] # ['Nino12','Nino3','Nino4'] # ['Nino34_anom'] # 6 is 'Nino34_anom'
#     riverflow_df = pd.read_csv('../data/Climate/RiverFlow/processed/riverflow.csv',index_col=0,header=0)
#     info_df = pd.read_csv('../data/Climate/RiverFlow/processed/info.csv',index_col=0,header=0)
#     times_df = pd.read_csv('../data/Climate/RiverFlow/processed/times.csv',index_col=0,header=0)
#     #times = np.asarray(times_df,dtype=int).reshape((-1,))
#     times = list(np.asarray(times_df,dtype=int).reshape((-1,)))+[201901]
#     #%% read ENSO index
#     indices_df, _ = utils.read_enso() # 187001 to 201912
#     indices_df = indices_df[column] # select input feature
#     #amazon = riverflow_df[['0']].loc[195001:200512].to_numpy().reshape((-1,))
#     amazons = {}
#     for window in range(1,7):
#         amazon = riverflow_df[['0']].iloc[600-window+1:1272].to_numpy().reshape((-1,))
#         amazon = [np.mean(amazon[i:i+window]) for i in range(len(amazon)-window+1)]# moving average
#         amazons[window] = amazon
#       
#     nino3 = indices_df[['Nino3']].loc[195001:200512].to_numpy().reshape((-1,))
#     
#     import matplotlib.pyplot as plt
#     #fig = plt.figure()
#     #plt.plot(amazon)
#     #
#     #fig = plt.figure()
#     #plt.plot(nino3)
#     #plt.show()
#     
#     left,right = 50,360
#     top,bottom = 50,130
#     data = np.load('/scratch/wang.zife/YuminLiu/DATA/GCM/GCMdata/tas/processeddata/32gcms_tas_monthly_1by1_195001-200512_World.npy')
#     data = data[:,:,top:bottom,left:right]
#     loc = data[0,:,90,180]
#     
#     #fig = plt.figure()
#     #plt.plot(loc)
#     #plt.show()
#     
#     fig = plt.figure()
#     plt.imshow(a[0,0,:,:])
#     plt.show()
#     
#     fig = plt.figure()
#     plt.imshow(data[0,0,:,:])
#     plt.show()
#     
#     fig = plt.figure()
#     lags_nino3,corr_nino3,_,_ = plt.xcorr(nino3,loc,maxlags=24,usevlines=False,normed=True,label='nino3')
#     #lags_amazon,corr_amazon,_,_ = plt.xcorr(amazon,loc,maxlags=24,usevlines=False,normed=True,label='amazon')
#     lags_amazon,corr_amazon = {},{}
#     for window in amazons:
#         lags_amazon[window],corr_amazon[window],_,_ = plt.xcorr(amazons[window],loc,maxlags=24,usevlines=False,normed=True,label='amazon_{}'.format(window))
#     #lags_,corr_,_,_ = plt.xcorr(amazon,nino3,maxlags=24,usevlines=False,normed=True)
#     plt.legend()
#     plt.show()
# 
# 
# 
# def readGCM_Ocean():
#     '''
#     generate whole USA map, saved on 12/11/2019
#     read and process NASA GCM data and aglined
#     '''
#     import numpy as np
#     from netCDF4 import Dataset
#     from ncdump import ncdump
#     from os import listdir
#     from os.path import isfile, join
#     import os
#     
#     prefix = '/scratch/wang.zife/YuminLiu/DATA/'
#     variable = 'tas' # 'tasmin' #'tasmax' #['pr' 'tas' 'tasmax' 'tasmin'] 
#     filepath = prefix+'GCM/GCMdata/'+variable+'/raw/'
#     savepath = prefix+'GCM/GCMdata/'+variable+'/processeddata/'
#     
#     filenames = sorted([f for f in listdir(filepath) if isfile(join(filepath,f))])
#     
#     if savepath and not os.path.exists(savepath):
#         os.makedirs(savepath)
#         
#     values_gcms = []
#     gcm_names = []
#     for kk,filename in enumerate(filenames):
#         dataset = Dataset(filepath+filename,mode='r')
#         if kk==0:
#             nc_attrs, nc_dims, nc_vars = ncdump(dataset)
#        
#         # original longitude is from 0.5 to 359.5 by 1, 360 points 
#         # original latitude is from -89.5 to 89.5 by 1, 180 points 
#         # whole USA longitude from [235.5E,293.5E] by 1, 59 points
#         # whole USA latitude from [24.5N, 49.5N] by 1, 26 points
#         # original month from 195001 to 200512, 672 points
#         
#         time = dataset.variables['time'][:] # 195001 - 200512
#         #lats = dataset.variables['latitude'][114:140] # [24.5N, 49.5N]
#         #lons = dataset.variables['longitude'][235:294] # [235.5E, 293.5E]
#         ## near-surface air temperature
#         lats_gcm = dataset.variables['latitude'][2:-2] # [-87.5N, 87.5N]
#         lats_gcm = np.flipud(lats_gcm) # lats from [87.5N,-87.5N]
#         lons_gcm = dataset.variables['longitude'][:] # [0.5E, 359.5E]
#         value_gcm = dataset.variables[variable][:,2:-2,:]#[month,lat,lon] 195001-200512, totally 
#         value_gcm = np.ma.filled(value_gcm,1000)
#         for t in range(len(value_gcm)):
#             value_gcm[t,:,:] = np.flipud(value_gcm[t,:,:]) # lats from [49.5N,24.5N]  
#     
#         if np.isnan(np.max(value_gcm)):
#             print(filename + 'has NAN!\n')
#             break
#         if value_gcm.shape!=(672,176,360) or np.max(value_gcm==1000):
#             print(filename)
#             #break
#             continue
#         
#         values_gcms.append(value_gcm) 
#         gcm_names.append(filename)
#     
#     values_gcms = np.stack(values_gcms,axis=1) # [Nmon,Ngcm,Nlat,Nlon]
#     print('values_gcms.shape={}'.format(values_gcms.shape))
#     
#     #time = np.array(time)
#     time = []
#     for i in range(values_gcms.shape[1]):
#         year,month = str(1950+i//12), str(i%12+1)
#         if int(month)<10: month = '0'+month
#         time.append(int(year+month))
#     time = np.array(time)  
#     lats_gcm = np.array(lats_gcm)
#     lons_gcm = np.array(lons_gcm)
#     
#     if savepath:
#         np.save(savepath+'time_gcm.npy',time)
#         np.save(savepath+'lats_gcm.npy',lats_gcm)
#         np.save(savepath+'lons_gcm.npy',lons_gcm)
#         np.save(savepath+'{}gcms_{}_monthly_1by1_195001-200512_World.npy'.format(len(filenames),variable),values_gcms)
#         np.save(savepath+'gcm_names.npy',gcm_names)
#     
#     import matplotlib.pyplot as plt
#     fig = plt.figure()
#     plt.imshow(values_gcms[0,0,:,:])
#     plt.show()
#     plt.vlines(x=[50,360],ymin=50,ymax=130)
#     plt.hlines(y=[50,130],xmin=50,xmax=360)
# =============================================================================



# =============================================================================
# import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
# import pandas as pd
# import io
# 
# u = u"""latitude,longitude
# 42.357778,-71.059444
# 39.952222,-75.163889
# 25.787778,-80.224167
# 30.267222, -97.763889"""
# 
# # read in data to use for plotted points
# buildingdf = pd.read_csv(io.StringIO(u), delimiter=",")
# lat = buildingdf['latitude'].values
# lon = buildingdf['longitude'].values
# 
# # determine range to print based on min, max lat and lon of the data
# margin = 2 # buffer to add to the range
# lat_min = min(lat) - margin
# lat_max = max(lat) + margin
# lon_min = min(lon) - margin
# lon_max = max(lon) + margin
# 
# # create map using BASEMAP
# m = Basemap(llcrnrlon=lon_min,
#             llcrnrlat=lat_min,
#             urcrnrlon=lon_max,
#             urcrnrlat=lat_max,
#             #lat_0=(lat_max - lat_min)/2,
#             #lon_0=(lon_max-lon_min)/2,
#             projection='merc',
#             resolution = 'h',
#             area_thresh=10000.,
#             )
# m.drawcoastlines()
# m.drawcountries()
# m.drawstates()
# m.drawmapboundary(fill_color='#46bcec')
# m.fillcontinents(color = 'white',lake_color='#46bcec')
# # convert lat and lon to map projection coordinates
# lons, lats = m(lon, lat)
# # plot points as red dots
# m.scatter(lons, lats, marker = 'o', color='r', zorder=5)
# plt.show()
# =============================================================================




# =============================================================================
# import time
# import json
# #import numpy as np
# import torch
# from torch.utils.data.dataloader import DataLoader
# import datasets
# import models
# import utils
# from inference import inference
# import numpy as np
# from skimage.transform import resize
# 
# start_time = time.time()
# 
# is_debug = False # True # 
# variable = 'ppt' #'tmax' #  'tmin' # 
# resolution = 0.125 # 0.5 # 
# num_epochs = 300 # 100 # 
# 
# #### model parameter setting
# if variable=='ppt':
#     input_channels = 35 #
# elif variable=='tmax' or variable=='tmin':
#     input_channels = 33 # 35 #
# output_channels = 1
# hidden_channels = 64 # number of feature maps for hidden layers
# num_layers = 15 # number of Conv/Deconv layer pairs
# scale = int(1/resolution) # 8 # 2 # downscaling factor
# use_climatology = True
# 
# #model_path = '../results/Climate/PRISM_GCM/YNet/tmax/scale2/2020-12-23_20.19.46.827931/'
# model_path = '../results/Climate/PRISM_GCM/YNet/ppt/scale8/2020-12-25_21.18.55.261503/'
# model_name = 'YNet_epoch_best.pth'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 
# checkpoint_pathname = model_path+model_name
# checkpoint = torch.load(checkpoint_pathname, map_location=lambda storage, loc: storage)
# 
# model = models.YNet(input_channels=input_channels,output_channels=output_channels,
#                     hidden_channels=hidden_channels,num_layers=num_layers,
#                     scale=scale,use_climatology=use_climatology)
# model.load_state_dict(checkpoint['model_state_dict'])
# model = model.to(device)
# model.eval()
# 
# #test_datapath = '../data/Climate/PRISM_GCMdata/tmax/0.5by0.5/test/'
# #filename = 'prism_gcm_divide50_tmax_monthly_0.5to1.0_195001-200512_USA_month637.npz'
# test_datapath = '../data/Climate/PRISM_GCMdata/ppt/0.125by0.125/test/'
# filename = 'prism_gcm_log1p_prmean_monthly_0.125to1.0_195001-200512_USA_month637.npz'
# data = np.load(test_datapath+filename)
# X = data['gcms'] # tmax tmin:[-1,1], ppt: [0.0,1.0], [Ngcm,Nlat,Nlon]
# y = data['prism'] # [1,Nlat,Nlon], tmax/tmin:[-1.0,1.0], ppt:[0.0,1.0]
# 
# input1 = torch.from_numpy(X[np.newaxis,...]).float() #[Ngcm,Nlat,Nlon]-->[1,Ngcm,Nlat,Nlon]
# X2 = resize(np.transpose(X,axes=(1,2,0)),y.shape[1:],order=1,preserve_range=True) # [Nlat,Nlon,Ngcm]
# X2 = np.transpose(X2,axes=(2,0,1))# [Ngcm,Nlat,Nlon]
# input2 = torch.from_numpy(X2[np.newaxis,...]).float() # [Ngcm,Nlat,Nlon]
# inputs = [input1,input2]
# if use_climatology:
#     Xaux = np.concatenate((data['climatology'],data['elevation']),axis=0)  # [2,Nlat,Nlon]
#     input3 = torch.from_numpy(Xaux[np.newaxis,...]).float() #[1,2,Nlat,Nlon] --> [1,2,Nlat,Nlon]
#     inputs += [input3]
# inputs = [e.to(device) for e in inputs]  
# inputs = [e.requires_grad_() for e in inputs]  
# y_pred = model(*inputs) # [1,1,Nlat,Nlon]
# 
# import matplotlib.pyplot as plt
# fig = plt.figure()
# plt.imshow(y_pred.cpu().detach().squeeze().numpy())
# 
# y_pred_single_locations = y_pred[0,0,30,20]
# 
# y_pred_single_locations.backward()
# 
# gradient_maps = [e.grad.data.abs() for e in inputs]
# 
# print('saliency_map.size={}'.format(gradient_maps[0].size()))
# 
# saliency_gcm = gradient_maps[0].squeeze() # GCM
# saliency_gcm,_ = torch.max(saliency_gcm,dim=0)
# print('saliency_gcm.size={}'.format(saliency_gcm.size()))
# 
# saliency_aux = gradient_maps[2].squeeze() # Climatology
# print('saliency_aux.size={}'.format(saliency_aux.size()))
# saliency_climatology,saliency_elevation = saliency_aux[0,:,:],saliency_aux[1,:,:]
# 
# #import matplotlib.pyplot as plt
# fig = plt.figure()
# plt.imshow(saliency_gcm.cpu().detach().squeeze().numpy())
# 
# #import matplotlib.pyplot as plt
# fig = plt.figure()
# plt.imshow(saliency_climatology.cpu().detach().squeeze().numpy())
# 
# #import matplotlib.pyplot as plt
# fig = plt.figure()
# plt.imshow(saliency_elevation.cpu().detach().squeeze().numpy())
# 
# 
# =============================================================================



# =============================================================================
# #%% https://medium.com/datadriveninvestor/visualizing-neural-networks-using-saliency-maps-in-pytorch-289d8e244ab4
# #IMPORTS
# 
# import torch
# import torchvision
# import torchvision.transforms as T
# import numpy as np
# import matplotlib.pyplot as plt
# from torchsummary import summary
# import requests
# from PIL import Image
# 
# #Using VGG-19 pretrained model for image classification
# 
# model = torchvision.models.vgg19(pretrained=False)
# model.load_state_dict(torch.load('./vgg19.pth'))
# for param in model.parameters():
#     param.requires_grad = False
#     
# def download(url,fname):
#     response = requests.get(url)
#     with open(fname,"wb") as f:
#         f.write(response.content)
#     
# # Downloading the image    
# #download("https://specials-images.forbesimg.com/imageserve/5db4c7b464b49a0007e9dfac/960x0.jpg?fit=scale","input.jpg")
# 
# # Opening the image
# img = Image.open('../data/input.jpg') 
# 
# # Preprocess the image
# def preprocess(image, size=224):
#     transform = T.Compose([
#         T.Resize((size,size)),
#         T.ToTensor(),
#         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         T.Lambda(lambda x: x[None]),
#     ])
#     return transform(image)
# 
# '''
#     Y = (X - μ)/(σ) => Y ~ Distribution(0,1) if X ~ Distribution(μ,σ)
#     => Y/(1/σ) follows Distribution(0,σ)
#     => (Y/(1/σ) - (-μ))/1 is actually X and hence follows Distribution(μ,σ)
# '''
# def deprocess(image):
#     transform = T.Compose([
#         T.Lambda(lambda x: x[0]),
#         T.Normalize(mean=[0, 0, 0], std=[4.3668, 4.4643, 4.4444]),
#         T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
#         T.ToPILImage(),
#     ])
#     return transform(image)
# 
# def show_img(PIL_IMG):
#     plt.imshow(np.asarray(PIL_IMG))
#     
# # preprocess the image
# X = preprocess(img)
# 
# # we would run the model in evaluation mode
# model.eval()
# 
# print(model)
# 
# # we need to find the gradient with respect to the input image, so we need to call requires_grad_ on it
# X.requires_grad_()
# 
# '''
# forward pass through the model to get the scores, note that VGG-19 model doesn't perform softmax at the end
# and we also don't need softmax, we need scores, so that's perfect for us.
# '''
# 
# scores = model(X)
# 
# print('scores.size={}'.format(scores.size()))
# 
# # Get the index corresponding to the maximum score and the maximum score itself.
# score_max_index = scores.argmax()
# score_max = scores[0,score_max_index]
# 
# print('score_max={}'.format(score_max))
# 
# '''
# backward function on score_max performs the backward pass in the computation graph and calculates the gradient of 
# score_max with respect to nodes in the computation graph
# '''
# score_max.backward()
# 
# a = X.grad.data.abs()
# 
# '''
# Saliency would be the gradient with respect to the input image now. But note that the input image has 3 channels,
# R, G and B. To derive a single class saliency value for each pixel (i, j),  we take the maximum magnitude
# across all colour channels.
# '''
# saliency, _ = torch.max(X.grad.data.abs(),dim=1)
# 
# # code to plot the saliency map as a heatmap
# plt.imshow(saliency[0], cmap=plt.cm.hot)
# plt.axis('off')
# plt.show()
# =============================================================================



# =============================================================================
# def normalization():
#     import os
#     import numpy as np
#     
#     def normalized(resolution,folder):
#         #resolution = 0.125 # 0.5
#         variable = 'tmin' # 'tmax' # 'ppt' # 
#         #folder = 'train' # 'test' #'val'#
#         datapath = '../data/Climate/PRISM_GCMdata/{}/{}by{}/{}/'.format(variable,resolution,resolution,folder)
#         savepath = '../data/Climate_new/PRISM_GCMdata/{}/{}by{}/{}/'.format(variable,resolution,resolution,folder)
#         if not os.path.exists(savepath):
#             os.makedirs(savepath)
#         filenames = [f for f in os.listdir(datapath) if f.endswith('.npz')]
#         filenames = sorted(filenames)
#         #filenames = [filenames[0]]
#         gcmsmin,gcmsmax = 50,-50
#         prismmin,prismmax = 50,-50
#         climatologymin,climatologymax = 50,-50
#         elevationmin,elevationmax = 50,-50
#         for filename in filenames:
#             data = np.load(datapath+filename)
#             gcms = data['gcms']#/5.0 # tmin, tmax: [-1,1], ppt [0,3.6]
#             prism = data['prism']#/5.0 # tmin, tmax: [-1,1], ppt [0,4.2]
#             elevation = data['elevation']#/10.0 # tmin, tmax [0,0.8], ppt [0,8.2]
#             climatology = data['climatology']#/5.0 # tmin,tmax: [-1,1], ppt [0,3.2]
#             
#             np.savez(savepath+filename,gcms=gcms,prism=prism,elevation=elevation,climatology=climatology)
#         
#             gcmsmin,gcmsmax = min(gcmsmin,gcms.min()),max(gcmsmax,gcms.max())
#             prismmin,prismmax = min(prismmin,prism.min()),max(prismmax,prism.max())
#             elevationmin,elevationmax = min(elevationmin,elevation.min()),max(elevationmax,elevation.max())
#             climatologymin,climatologymax = min(climatologymin,climatology.min()),max(climatologymax,climatology.max())
#         
#         print('resolution={},folder={}'.format(resolution,folder))
#         print('gcmsmin={},max={}'.format(gcmsmin,gcmsmax))
#         print('prismmin={},max={}'.format(prismmin,prismmax))
#         print('elevationmin={},max={}'.format(elevationmin,elevationmax))
#         print('climatologymin={},max={}\n'.format(climatologymin,climatologymax))
#     
#     for resolution in [0.5,0.25,0.125]:
#         for folder in ['train','val','test']:
#             normalized(resolution,folder)
# =============================================================================












# =============================================================================
# import glob
# import numpy as np
# 
# variable = 'ppt' #'tmin' #'ppt' #'tmax'
# resolution = 0.5
# datapath = '../data/Climate/PRISM_GCMdata/{}/{}by{}/train/'.format(variable,resolution,resolution)
# #valid_datapath = '../data/Climate/PRISM_GCMdata/{}/{}by{}/val/'.format(variable,resolution,resolution)
# 
# datapaths = sorted(glob.glob(datapath + '*.npz')) 
# filepath = datapaths[0]
# data = np.load(filepath)
# gcms = data['gcms']
# gcms.shape
# 
# =============================================================================
# =============================================================================
# import numpy as np
# datapath = '../data/Climate/PRISM_GCMdata/ppt/0.5by0.5/test/prism_gcm_log1p_prmean_monthly_0.5to1.0_195001-200512_USA_month637.npz'
# datapath = '/home/yumin/DS//DATA/GCM/GCMdata/ppt/35channels/processeddata_26by59_points/lons_gcm.npy'
# data = np.load(datapath)
# #gcm = data['gcms']
# 
# =============================================================================


# =============================================================================
# import os
# import numpy as np
# datapath = '../data/Climate/PRISM_GCMdata/tmin/0.5by0.5/train/'
# train_filenames = [f for f in os.listdir(datapath) if f.endswith('.npz')]
# train_filenames = sorted(train_filenames)
# 
# gcms_train = []
# prism_train = []
# for filename in train_filenames:
#     data = np.load(datapath+filename)
#     gcms_train.append(data['gcms']) # [Ngcm,Nlat,Nlon]
#     prism_train.append(np.squeeze(data['prism'])) # [1,Nlat,Nlon] --> [Nlat,Nlon]
# gcms_train = np.stack(gcms_train,axis=1) #[Ngcm,Nlat,Nlon] --> [Ngcm,Nmon,Nlat,Nlon]
# prism_train = np.stack(prism_train,axis=0) # [Nlat,Nlon] --> [Nmon,Nlat,Nlon]    
# print('gcms_train.shape={}\nprism_train.shape={}'.format(gcms_train.shape,prism_train.shape))
# mask = np.sum(prism_train,axis=0)
# import matplotlib.pyplot as plt
# fig = plt.figure()
# plt.imshow(mask)
# 
# =============================================================================












# =============================================================================
# '''
# obs: observed values, 1d array, [Ntrain,]
# model: model values, 1d array, [Ntrain,]
# output: cdfs of observed and model variables, and bins, 1d arrays
# '''
# import numpy as np
# obs = np.arange(-50,100)
# model = np.arange(0,150)
# nbin = 20
# if len(model)>len(obs):
#     model = model[:len(obs)] # should just use training data
# max_value = max(np.amax(obs),np.amax(model))
# width = max_value/nbin
# xbins = np.arange(0.0,max_value+width,width)
# # create PDF
# pdfobs, _ = np.histogram(obs,bins=xbins)
# pdfmodel, _ = np.histogram(model,bins=xbins)
# # create CDF with zero in first entry.
# cdfobs = np.insert(np.cumsum(pdfobs),0,0.0)
# cdfmodel = np.insert(np.cumsum(pdfmodel),0,0.0)
#     
#     
#     
#     
#     
# import numpy as np
# obs2 = np.arange(-50,100)
# model2 = np.arange(0,150)
# nbin2 = 20
# if len(model2)>len(obs2):
#     model2 = model2[:len(obs2)] # should just use training data
# max_value2 = max(np.amax(obs2),np.amax(model2))
# width2 = max_value2/nbin2
# xbins2 = np.arange(-50,max_value2+width2,width2)
# # create PDF
# pdfobs2, _ = np.histogram(obs2,bins=xbins2)
# pdfmodel2, _ = np.histogram(model2,bins=xbins2)
# # create CDF with zero in first entry.
# cdfobs2 = np.insert(np.cumsum(pdfobs2),0,0.0)
# cdfmodel2 = np.insert(np.cumsum(pdfmodel2),0,0.0)    
#     
#     
#     
#     
#     
# import matplotlib.pyplot as plt
# #rng = np.random.RandomState(10)  # deterministic random data
# #a = np.hstack((rng.normal(size=1000),rng.normal(loc=5, scale=2, size=1000)))
# #_ = plt.hist(a, bins='auto')  # arguments are passed to np.histogram
# #plt.title("Histogram with 'auto' bins")
# #Text(0.5, 1.0, "Histogram with 'auto' bins")
# fig = plt.figure()
# plt.hist(obs,bins=xbins)
# plt.hist(model,bins=xbins)
# fig = plt.figure()
# plt.hist(obs2,bins=xbins2)
# plt.hist(model2,bins=xbins2)
# plt.show()    
# =============================================================================
    
    
    
    
    
# =============================================================================
# import numpy as np
# 
# path = '/home/yumin/DS/DATA/GCM/GCMdata/ppt/35channels/processeddata_26by59_points/'
# latsname = 'lats_gcm.npy'
# lonsname = 'lons_gcm.npy'
# 
# lats = np.load(path+latsname)
# lons = np.load(path+lonsname)
# 
# =============================================================================


# =============================================================================
# # Draw the locations of cities on a map of the US
# import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
# from geopy.geocoders import Nominatim
# import math
# 
# cities = [["Chicago",10],
#           ["Boston",10],
#           ["New York",5],
#           ["San Francisco",25]]
# scale = 5
# 
# map = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
#         projection='lcc',lat_1=32,lat_2=45,lon_0=-95)
# 
# # load the shapefile, use the name 'states'
# map.readshapefile('../data/Climate/map/st99_d00', name='states', drawbounds=True)
# 
# # Get the location of each city and plot it
# #geolocator = Nominatim()
# #for (city,count) in cities:
# #    loc = geolocator.geocode(city)
# #    x, y = map(loc.longitude, loc.latitude)
# #    map.plot(x,y,marker='o',color='Red',markersize=int(math.sqrt(count))*scale)
# #plt.show()
# =============================================================================


#%%
# =============================================================================
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap as Basemap
# from matplotlib.colors import rgb2hex
# from matplotlib.patches import Polygon
# # Lambert Conformal map of lower 48 states.
# m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
#         projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
# # draw state boundaries.
# # data from U.S Census Bureau
# # http://www.census.gov/geo/www/cob/st2000.html
# shp_info = m.readshapefile('../data/Climate/map/st99_d00','states',drawbounds=True)
# # population density by state from
# # http://en.wikipedia.org/wiki/List_of_U.S._states_by_population_density
# popdensity = np.load('../data/Climate/map/state_popdensity.npy',allow_pickle=True).item()
# 
# # choose a color for each state based on population density.
# colors={}
# statenames=[]
# cmap = plt.cm.hot # use 'hot' colormap
# vmin = 0; vmax = 450 # set range.
# for shapedict in m.states_info:
#     statename = shapedict['NAME']
#     # skip DC and Puerto Rico.
#     if statename not in ['District of Columbia','Puerto Rico']:
#         pop = popdensity[statename]
#         # calling colormap with value between 0 and 1 returns
#         # rgba value.  Invert color range (hot colors are high
#         # population), take sqrt root to spread out colors more.
#         colors[statename] = cmap(1.-np.sqrt((pop-vmin)/(vmax-vmin)))[:3]
#     statenames.append(statename)
# # cycle through state names, color each one.
# ax = plt.gca() # get current axes instance
# for nshape,seg in enumerate(m.states):
#     # skip DC and Puerto Rico.
#     if statenames[nshape] not in ['District of Columbia','Puerto Rico']:
#         #color = rgb2hex(colors[statenames[nshape]]) 
#         #poly = Polygon(seg,facecolor=color,edgecolor=color)
#         poly = Polygon(seg,facecolor='w',edgecolor='k')
#         ax.add_patch(poly)
# plt.title('Filling State Polygons by Population Density')
# plt.show()
# =============================================================================
