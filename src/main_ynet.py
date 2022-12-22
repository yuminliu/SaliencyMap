#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 13:38:44 2021

@author: wang.zife
"""
import os
#import json
#import torch
import numpy as np
#from torch.utils.data.dataloader import DataLoader
#import models
#import datasets
#import plots

from mpl_toolkits.basemap import Basemap
#import matplotlib
#matplotlib.use('Agg')
##matplotlib.use('Qt5Agg')
#import matplotlib.pyplot as plt

#%%
#def get_saliency(datapathname,varname,fillnan=None,threshold=0.05):
def get_saliency(saliency_maps,fillnan=None,threshold=0.05):
    #data = np.load(datapathname)
    #saliency_maps = data[varname]
    #saliency_maps = abs(saliency_maps[:,gcm,:,:])
    #saliency_maps = np.amax(abs(saliency_maps),axis=1)
    saliency_maps = np.sum(abs(saliency_maps),axis=1)
    #saliency_maps = np.std(abs(saliency_maps),axis=1)
    max_per_map = np.amax(saliency_maps,axis=(1,2))
    if fillnan!=None:
        for mon in range(len(saliency_maps)):
            saliency_maps[mon,:,:][saliency_maps[mon,:,:]<threshold*max_per_map[mon]] = fillnan
    return saliency_maps

#%%
def plot_map(img,title=None,savepath=None,savename=None,cmap='YlOrRd',alpha=0.8,
             lonlat=[235.5,24.5,293.5,49.5],projection='merc',resolution='i',area_thresh=10000,
             parallels=[30,40],meridians=[-115,-105,-95,-85,-75],pos_lons=None, pos_lats=None,
             clim=None,watercolor='#46bcec',verbose=True):
    
    import matplotlib
    if verbose:
        matplotlib.use('Qt5Agg')
    else:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure() 
    m = Basemap(llcrnrlon=lonlat[0],llcrnrlat=lonlat[1],urcrnrlon=lonlat[2],urcrnrlat=lonlat[3],
                projection=projection,resolution=resolution,area_thresh=area_thresh)
    m.drawcoastlines(linewidth=1.0,color='k')
    m.drawcountries(linewidth=1.0,color='k')
    m.drawstates(linewidth=0.2,color='k')
    #m.fillcontinents(color='w',alpha=0.1)
    m.drawmapboundary(fill_color=watercolor)
    m.fillcontinents(color = 'white',alpha=1.0,lake_color=watercolor)
    m.drawparallels(parallels,labels=[True,False,False,False],dashes=[1,2])
    m.drawmeridians(meridians,labels=[False,False,False,True],dashes=[1,2])
    img = np.flipud(img)
    m.imshow(img,cmap=cmap,alpha=alpha,zorder=1)
    m.colorbar(fraction=0.02)
    ## plot scatter points
    if len(pos_lons)==len(pos_lats)>0:
        # convert lat and lon to map projection coordinates
        lonss, latss = m(pos_lons, pos_lats)
        # plot points as red dots
        m.scatter(lonss, latss, marker = 'o', color='k', zorder=2)
        # left,right,bottom,top = 234.5,295.5,49.5,24.5
        ##im = plt.imshow(img, extent=(left, right, bottom, top))
        text_lons,text_lats = m(pos_lons+5,pos_lats)
        plt.text(text_lons[0],text_lats[0],'A',fontsize=15,ha='left',va='center',color='k')
        plt.text(text_lons[1],text_lats[1],'B',fontsize=15,ha='left',va='center',color='k')
    if clim:# set color limits
        plt.clim(clim)
    
    if title:
        plt.title(title)
    if savepath and savename:
        plt.savefig(savepath+savename+'.png',dpi=1200,bbox_inches='tight')
    plt.show()

#def downscaling_ppt_tmax_tmin():
variable = 'ppt' # 'tmax' # 'tmin' # 
method = 'Years' # 'Seasons' # 'NaturalMonths' # 'Months' # 
var2modelpath = {'ppt':'2020-12-25_21.18.55.261503','tmax':'2021-01-11_17.42.26.526113','tmin':'2021-01-11_17.33.42.045321'}
model_path = var2modelpath[variable] # '2021-01-11_17.42.26.526113''2021-01-11_17.33.42.045321' # '2021-01-11_17.42.26.526113' # '2020-12-25_21.18.55.261503'
loc_lats, loc_lons = [30,120],[20,240] # 120,240 # latitude and longitude of location we want to get saliency map
datapath = '../results/Climate/PRISM_GCM/YNet/{}/scale8/{}_Saliency/'.format(variable,model_path)
latlonpath = '../data/Climate/PRISM_GCMdata/ppt/0.125by0.125/'#.format(variable)
datapathname1 = datapath+'lat_30_lon_20/lat_30_lon_20_y_preds_saliency_gcms_saliency_gcms_large_saliency_aux.npz'
datapathname2 = datapath+'lat_120_lon_240/lat_120_lon_240_y_preds_saliency_gcms_saliency_gcms_large_saliency_aux.npz'

varname = 'saliency_aux' # 'saliency_gcms' # 
cmap = 'YlOrRd' # 'rainbow' # 'Accent' #'YlGn' #'hsv' #'seismic' # 
alpha = 0.7
projection = 'merc' # 'cyl' # 
#lonlat=[235.5,24.5,293.5,49.5]
lonlat = [-124.5,24.5,-66.5,49.5] # map area, [left,bottom,right,top]
resolution = 'i' # 'h'
area_thresh = 10000
clim = None
parallels = np.arange(20.0,51.0,10.0)
meridians = np.arange(-125.0,-60.0,10.0) # label lons, 125W to 60W, or 235E to 300E
watercolor = '#46bcec'
verbose = True

lats_prism = np.load(latlonpath+'lats_prism.npy')
lons_prism = np.load(latlonpath+'lons_prism.npy')-360
pos_lats = lats_prism[loc_lats]
pos_lons = lons_prism[loc_lons]
   
#saliency_maps_loc1 = get_saliency(datapathname1,varname,fillnan=0)
#saliency_maps_loc2 = get_saliency(datapathname2,varname,fillnan=0)
saliency_maps_loc1 = get_saliency(np.load(datapathname1)[varname],fillnan=0,threshold=0.15)
saliency_maps_loc2 = get_saliency(np.load(datapathname2)[varname],fillnan=0,threshold=0.15)

saliency_maps = saliency_maps_loc1+saliency_maps_loc2


var2title = {'ppt':'Precipitation','tmax':'Max Temperature','tmin':'Min Temperature'}
ind2month = {1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',7:'July',8:'August',9:'September',10:'October',11:'November',12:'December'}
ind2season = {1:'Spring(MAM)',2:'Summer(JJA)',3:'Autumn(SON)',4:'Winter(DJF)'}  
ind2year = {1:2003,2:2004,3:2005}
if method=='Months':
    months = [[i] for i in range(36)]
elif method=='NaturalMonths':
    months = [[i,i+12,i+24] for i in range(12)]
elif method=='Seasons':
    months = [[-1,0,1,11,12,13,23,24,25],[2,3,4,14,15,16,26,27,28],[5,6,7,17,18,19,29,30,31],[8,9,10,20,21,22,32,33,34]]
elif method=='Years':
    months = [list(range(i,i+12)) for i in range(0,36,12)]
    
savepath = None # datapath+'SaliencyMaps/{}/'.format(method) # 
if savepath and not os.path.exists(savepath):
    os.makedirs(savepath)
    
for i,month in enumerate(months):
    #if i>0: break
    #month = list(range(6,36,12)) # [0]
    
    if method=='Months':
        YEAR, MONTH = str(2003+i//12), ind2month[i%12+1] 
        savename = '{}_{}_month_{}'.format(varname,variable,i)
        title = 'Saliency Map for {} at Two Locations in {} {}'.format(var2title[variable],MONTH,YEAR)
    elif method=='NaturalMonths':
        savename = '{}_{}_month_{}'.format(varname,variable,ind2month[i+1])
        title = 'Saliency Map for {} at Two Locations in {}'.format(var2title[variable],ind2month[i+1])
    elif method=='Seasons':
        savename = '{}_{}_season_{}'.format(varname,variable,ind2season[i+1])
        title = 'Saliency Map for {} at Two Locations in {}'.format(var2title[variable],ind2season[i+1])
    elif method=='Years':
        savename = '{}_{}_year_{}'.format(varname,variable,ind2year[i+1])
        title = 'Saliency Map for {} at Two Locations in {}'.format(var2title[variable],ind2year[i+1])

    img = np.mean(saliency_maps[month,:,:],axis=0)
    img[img==0] = np.nan
    
    plot_map(img,title=title,savepath=savepath,savename=savename,cmap=cmap,alpha=alpha,
                 lonlat=lonlat,projection=projection,resolution=resolution,area_thresh=area_thresh,
                 parallels=parallels,meridians=meridians,pos_lons=pos_lons, pos_lats=pos_lats,clim=clim,
                 watercolor=watercolor,verbose=verbose)