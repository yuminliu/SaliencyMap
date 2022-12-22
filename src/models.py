#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 17:34:32 2019

@author: yumin
"""
import math
import torch
from torch import nn

#%% Version 3
torch.autograd.set_detect_anomaly(True)
class YNet(nn.Module):
    def __init__(self,input_channels=1,output_channels=1,hidden_channels=64,num_layers=15,scale=4,use_climatology=True):
        super(YNet, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.scale = scale
        self.use_climatology = use_climatology
        self.fusion_layer = None

        conv_layers = []
        deconv_layers = []
        conv_layers.append(nn.Sequential(nn.Conv2d(self.input_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
                                         nn.ReLU(inplace=True)))
        for i in range(self.num_layers - 1):
            conv_layers.append(nn.Sequential(nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True)))
        for i in range(self.num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
                                               nn.ReLU(inplace=True),
                                               nn.Conv2d(self.hidden_channels,self.hidden_channels,kernel_size=3,padding=1),
                                               nn.ReLU(inplace=True)))
        deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1, output_padding=0),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(self.hidden_channels,self.input_channels,kernel_size=3,stride=1,padding=1)))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)
        self.upsample_layers = nn.Sequential(nn.Conv2d(self.input_channels,self.input_channels,kernel_size=3,stride=1,padding=1),
                                                 nn.ReLU(inplace=True),
                                                 nn.Upsample(scale_factor=self.scale,mode='bilinear',align_corners=False),
                                                 nn.Conv2d(self.input_channels,self.input_channels,kernel_size=3,stride=1,padding=1),
                                                 nn.ReLU(inplace=True))
        if self.use_climatology:
            self.fusion_layer = nn.Sequential(nn.Conv2d(2*self.input_channels+2,self.hidden_channels,kernel_size=3,stride=1,padding=1),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(self.hidden_channels,self.output_channels,kernel_size=1,stride=1,padding=0))#,nn.ReLU(inplace=True))

    def forward(self, X, X2=None, Xaux=None):
        residual = X
        conv_feats = []
        for i in range(self.num_layers):
            X = self.conv_layers[i](X)
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
                conv_feats.append(X)
        conv_feats_idx = 0
        for i in range(self.num_layers):
            X = self.deconv_layers[i](X)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
                X = X + conv_feat
                X = self.relu(X)
        X = X+residual
        X = self.relu(X)
        X = self.upsample_layers(X)
        X = torch.cat([X,X2],dim=1)
        if self.use_climatology and (Xaux is not None):
            X = self.fusion_layer(torch.cat([X,Xaux],dim=1)) # [Nbatch,Nchannel,Nlat,Nlon]
        
        return X

#%%
class CNN_nature(nn.Module):
    def __init__(self,input_channels=6,hidden_channels=50,fc_channels=50):
        '''
        imput image is height*width*channel=24*72*6
        '''
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        super(CNN_nature,self).__init__()
        self.cnn1 = nn.Sequential(nn.Conv2d(self.input_channels, self.hidden_channels,
                           kernel_size=(8,4),stride=1,padding=(4,2), bias=True),nn.Tanh())
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn2 = nn.Sequential(nn.Conv2d(self.hidden_channels, self.hidden_channels,
                           kernel_size=(4,2),stride=1,padding=(2,1), bias=True),nn.Tanh())
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn3 = nn.Sequential(nn.Conv2d(self.hidden_channels, self.hidden_channels,
                           kernel_size=(4,2),stride=1,padding=(2,1), bias=True),nn.Tanh())
        self.fc = nn.Linear(6*18*hidden_channels,fc_channels)
        self.activation = nn.Tanh()
        #self.layers = nn.Sequential(cnn1,pool1,cnn2,pool2,cnn3,fc)
        
    def forward(self,X):
        #print('before cnn1: X.shape={}'.format(X.shape))
        X = self.cnn1(X)
        #print('after cnn1: X.shape={}'.format(X.shape))
        X = self.pool1(X)
        #print('after pool1: X.shape={}'.format(X.shape))
        X = self.cnn2(X)
        #print('after cnn2: X.shape={}'.format(X.shape))
        X = self.pool2(X)
        #print('after pool2: X.shape={}'.format(X.shape))
        X = self.cnn3(X)
        X = X[:,:,0:-1,0:-1]
        #print('after cnn3: X.shape={}'.format(X.shape))
        X = X.reshape(X.shape[0],-1)
        #print('after view: X.shape={}'.format(X.shape))
        X = self.fc(X)
        #print('after fc: X.shape={}'.format(X.shape))
        return X
        #return self.layers(X)
        
class CNN(nn.Module):
    def __init__(self,input_channels=32,output_channels=1,hidden_channels=[16,32,64],
                 fc_channels=[128,64],drop_rate=0.5,activation='ReLU'):
        '''
        imput image is N*channel*height*width=None*32*80*300
        '''
        super(CNN,self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.channels = [input_channels]+hidden_channels
        #self.num_hidden_layers = len(hidden_channels)
        self.fc_channels = [5*18*self.channels[-1]]+fc_channels+[output_channels] # sst region 1, input size [80,300]
        #self.fc_channels = [2*13*self.channels[-1]]+fc_channels+[output_channels] # sst region 2 , input size [42,210]
        #self.fc_channels = [4*6*self.channels[-1]]+fc_channels+[output_channels] # sst region 3 , input size [70,110]
        #self.fc_channels = [2*18*self.channels[-1]]+fc_channels+[output_channels] # sst region 4 , input size [42,300]
        self.drop_rate = drop_rate
        if activation=='ReLU':
            self.activation = nn.ReLU()
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(negative_slope=0.1)
        elif activation=='RReLU':
            self.activation = nn.RReLU()
        elif activation=='Tanh':
            self.activation = nn.Tanh()

        cnn_layers = [] # cnn-->activation-->dropout
        for i in range(len(self.channels)-1):
            cnn_layers.append(nn.Conv2d(self.channels[i], self.channels[i+1],kernel_size=(3,3),stride=1,padding=(1,1), bias=True))
            cnn_layers.append(self.activation)
            if self.drop_rate:
                cnn_layers.append(nn.Dropout(p=self.drop_rate))
            if True:#i:# and i%2==0:
                cnn_layers.append(nn.MaxPool2d(kernel_size=(2,2)))
        self.cnn_layers = nn.Sequential(*cnn_layers)

        last_cnn_layers = []
        last_cnn_layers.append(nn.Conv2d(self.channels[-1], self.channels[-1],kernel_size=(1,1),stride=1,padding=(0,0), bias=True))
        last_cnn_layers.append(self.activation)
        if self.drop_rate:
            last_cnn_layers.append(nn.Dropout(p=self.drop_rate))  
        last_cnn_layers.append(nn.MaxPool2d(kernel_size=(2,2)))
        self.last_cnn_layers = nn.Sequential(*last_cnn_layers)    

        fc_layers = [] # fc-->activation-->dropout
        for i in range(len(self.fc_channels)-1):
            fc_layers.append(nn.Linear(self.fc_channels[i],self.fc_channels[i+1]))
            #fc_layers.append(self.activation)
            if self.drop_rate:
                fc_layers.append(nn.Dropout(p=self.drop_rate))
        self.fc_layers = nn.Sequential(*fc_layers)
        
    def forward(self,X):
        #print('before cnn1: X.shape={}'.format(X.shape))
        X = self.cnn_layers(X)
        #print('after cnn_layers: X.shape={}'.format(X.shape))
        X = self.last_cnn_layers(X)
        #print('after last_cnn_layers: X.shape={}'.format(X.shape))
        X = X.reshape(X.shape[0],-1)
        #print('after view: X.shape={}'.format(X.shape))
        X = self.fc_layers(X)
        #print('after fc: X.shape={}'.format(X.shape))
        return X
        #return self.layers(X)

#%% CAM
class CAM(nn.Module):
    def __init__(self,input_channels=1,output_channels=1,hidden_channels=[16,32,64],
                 fc_channels=[128],drop_rate=0.5,activation='ReLU'):
        '''
        imput image is N*channel*height*width=None*32*80*300
        '''
        super(CAM,self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.channels = [input_channels]+hidden_channels
        #self.num_hidden_layers = len(hidden_channels)
        self.fc_channels = [self.channels[-1]]+[output_channels] # sst region 1, input size [80,300]
        #self.fc_channels = [2*13*self.channels[-1]]+fc_channels+[output_channels] # sst region 2 , input size [42,210]
        #self.fc_channels = [4*6*self.channels[-1]]+fc_channels+[output_channels] # sst region 3 , input size [70,110]
        self.drop_rate = drop_rate
        if activation=='ReLU':
            self.activation = nn.ReLU()
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(negative_slope=0.1)
        elif activation=='RReLU':
            self.activation = nn.RReLU()
        elif activation=='Tanh':
            self.activation = nn.Tanh()

        cnn_layers = [] # cnn-->activation-->dropout
        for i in range(len(self.channels)-1):
            cnn_layers.append(nn.Conv2d(self.channels[i], self.channels[i+1],kernel_size=(3,3),stride=1,padding=(1,1), bias=True))
            cnn_layers.append(self.activation)
            if self.drop_rate:
                cnn_layers.append(nn.Dropout(p=self.drop_rate))
            if True:#i:# and i%2==0:
                cnn_layers.append(nn.MaxPool2d(kernel_size=(2,2)))
        self.cnn_layers = nn.Sequential(*cnn_layers)

        last_cnn_layers = []
        last_cnn_layers.append(nn.Conv2d(self.channels[-1], self.channels[-1],kernel_size=(1,1),stride=1,padding=(0,0), bias=True))
        last_cnn_layers.append(self.activation)
        if self.drop_rate:
            last_cnn_layers.append(nn.Dropout(p=self.drop_rate))  
        #last_cnn_layers.append(nn.MaxPool2d(kernel_size=(5,18)))
        self.last_maxpool_layer = nn.MaxPool2d(kernel_size=(5,18))
        self.last_cnn_layers = nn.Sequential(*last_cnn_layers)    

        fc_layers = [] # fc-->activation-->dropout
        for i in range(len(self.fc_channels)-1):
            fc_layers.append(nn.Linear(self.fc_channels[i],self.fc_channels[i+1]))
            #fc_layers.append(self.activation)
            if self.drop_rate:
                fc_layers.append(nn.Dropout(p=self.drop_rate))
        self.fc_layers = nn.Sequential(*fc_layers)
        
    def forward(self,X):
        #print('before cnn1: X.shape={}'.format(X.shape))
        X = self.cnn_layers(X)
        #print('after cnn_layers: X.shape={}'.format(X.shape))
        Xcnn = self.last_cnn_layers(X)
        #print('after last_cnn_layers: Xcnn.shape={}'.format(X.shape))
        X = self.last_maxpool_layer(Xcnn)
        X = X.reshape(X.shape[0],-1)
        #X = torch.reshape(Xcnn,(Xcnn.shape[0],-1))
        #print('after view: X.shape={}'.format(X.shape))
        X = self.fc_layers(X)
        #print('after fc: X.shape={},Xcnn.shape={}'.format(X.shape,Xcnn.shape))
        return X,Xcnn
        #return self.layers(X)

#%% DNN
class DNN(nn.Module):
    def __init__(self,input_channels=36,hidden_channels=[20,10,5],output_channels=1,drop_rate=0.0):
        super(DNN,self).__init__()
        self.drop_rate = drop_rate
        layer_sizes = [input_channels]+hidden_channels+[output_channels]
        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i],layer_sizes[i+1]))
            if i<len(layer_sizes)-2:
                layers.append(nn.Tanh())
            if self.drop_rate:
                layers.append(nn.Dropout(p=self.drop_rate))
        self.layers = nn.Sequential(*layers)
        
    def forward(self,X):
        X = self.layers(X)
        return X
        


#%% ESPCN 
# =============================================================================
# # from https://github.com/leftthomas/ESPCN
# import torch
# class ESPCNNet(nn.Module):
#     def __init__(self, scale,input_channels=1,output_channels=1):
#         super(ESPCNNet, self).__init__()
#         self.input_channels = input_channels
#         self.output_channels = output_channels
#         self.conv1 = nn.Conv2d(self.input_channels, 64, (5, 5), (1, 1), (2, 2))
#         self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
#         self.conv3 = nn.Conv2d(32, self.output_channels * (scale ** 2), (3, 3), (1, 1), (1, 1))
#         self.pixel_shuffle = nn.PixelShuffle(scale)
# 
#     def forward(self, x):
#         x = torch.tanh(self.conv1(x))
#         x = torch.tanh(self.conv2(x))
#         x = torch.sigmoid(self.pixel_shuffle(self.conv3(x)))
#         return x
# =============================================================================
    
   
#%% original REDNet30
# =============================================================================
# class REDNet30(nn.Module):
#     def __init__(self, input_channels=1, output_channels=1, num_layers=15, hidden_channels=64):
#         super(REDNet30, self).__init__()
#         self.num_layers = num_layers
#         self.input_channels = input_channels
#         self.output_channels = output_channels
# 
#         conv_layers = []
#         deconv_layers = []
#         conv_layers.append(nn.Sequential(nn.Conv2d(self.input_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
#                                          nn.ReLU(inplace=True)))
#         for i in range(num_layers - 1):
#             conv_layers.append(nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
#                                              nn.ReLU(inplace=True)))
#         for i in range(num_layers - 1):
#             deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
#                                                nn.ReLU(inplace=True)))
#         deconv_layers.append(nn.ConvTranspose2d(hidden_channels, self.output_channels, kernel_size=3, stride=2, padding=1, output_padding=1))
#         self.conv_layers = nn.Sequential(*conv_layers)
#         self.deconv_layers = nn.Sequential(*deconv_layers)
#         self.relu = nn.ReLU(inplace=True)
# 
#     def forward(self, x):
#         residual = x
#         conv_feats = []
#         for i in range(self.num_layers):
#             x = self.conv_layers[i](x)
#             if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
#                 conv_feats.append(x)
#         conv_feats_idx = 0
#         for i in range(self.num_layers):
#             x = self.deconv_layers[i](x)
#             if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
#                 conv_feat = conv_feats[-(conv_feats_idx + 1)]
#                 conv_feats_idx += 1
#                 x = x + conv_feat
#                 x = self.relu(x)
#         x += residual
#         x = self.relu(x)
# 
#         return x    
# =============================================================================
    