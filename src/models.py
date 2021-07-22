#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 17:34:32 2019

@author: yumin
"""
#import math
#import torch
from torch import nn

#%% CNN model      
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
        #self.intermediate_outputs = {}
        #self.gradients = []
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
        #self.last_cnn_layers = last_cnn_layers

        fc_layers = [] # fc-->activation-->dropout
        for i in range(len(self.fc_channels)-1):
            fc_layers.append(nn.Linear(self.fc_channels[i],self.fc_channels[i+1]))
            #fc_layers.append(self.activation)
            if self.drop_rate:
                fc_layers.append(nn.Dropout(p=self.drop_rate))
        self.fc_layers = nn.Sequential(*fc_layers)
        #self.fc_layers = fc_layers
        
    def forward(self,X):
        #print('before cnn1: X.shape={}'.format(X.shape))
        X = self.cnn_layers(X)
        # self.intermediate_outputs['input'] = X.detach().numpy()[0]
        # for i in range(len(self.cnn_layers)):
        #     X = self.cnn_layers[i](X)
        #     self.intermediate_outputs['cnn_{}'.format(i)] = X.detach().numpy()[0]
        #print('after cnn_layers: X.shape={}'.format(X.shape))
        X = self.last_cnn_layers(X)
        # for i in range(len(self.last_cnn_layers)):
        #     X = self.last_cnn_layers[i](X)
        #     self.intermediate_outputs['last_cnn_{}'.format(i)] = X.detach().numpy()[0]
        #print('after last_cnn_layers: X.shape={}'.format(X.shape))
        X = X.reshape(X.shape[0],-1)
        #print('after view: X.shape={}'.format(X.shape))
        X = self.fc_layers(X)
        # for i in range(len(self.fc_layers)):
        #     X = self.fc_layers[i](X)
        #     self.intermediate_outputs['fc_{}'.format(i)] = X.detach().numpy()[0]
        #print('after fc: X.shape={}'.format(X.shape))
        return X
        #return self.layers(X)

    # def save_gradient(self,module,grad_input,grad_output):
    #     self.gradients.append(grad_output[0].item())


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
        

