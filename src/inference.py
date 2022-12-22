#%%
def inference(hyparas,paras):
    import os
    import torch
    import numpy as np
    from skimage.transform import resize
    import models
    
    use_climatology = hyparas['use_climatology']
    savepath = paras['savepath']
    variable = paras['variable']
    test_datapath = paras['test_datapath']
    device = paras['device']
    checkpoint_name = paras['checkpoint_name']
    verbose = paras['verbose']
    torch.backends.cudnn.benchmark = True
    checkpoint_pathname = savepath+checkpoint_name
    filenames = [f for f in os.listdir(test_datapath) if f.endswith('.npz')]
    filenames = sorted(filenames)
    
    #%%
    #model = models.YNet(input_channels=input_channels,output_channels=output_channels,
    #                    hidden_channels=hidden_channels,num_layers=num_layers,
    #                    scale=scale,use_climatology=use_climatology)
    model = models.YNet(**hyparas)
    checkpoint = torch.load(checkpoint_pathname, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    ys,y_preds = [],[]
    for filename in filenames:
        data = np.load(test_datapath+filename)
        X = data['gcms'] # tmax tmin:[-1,1], ppt: [0.0,1.0], [Ngcm,Nlat,Nlon]
        y = data['prism'] # [1,Nlat,Nlon], tmax/tmin:[-1.0,1.0], ppt:[0.0,1.0]
        
        input1 = torch.from_numpy(X[np.newaxis,...]).float() #[Ngcm,Nlat,Nlon]-->[1,Ngcm,Nlat,Nlon]
        X2 = resize(np.transpose(X,axes=(1,2,0)),y.shape[1:],order=1,preserve_range=True) # [Nlat,Nlon,Ngcm]
        X2 = np.transpose(X2,axes=(2,0,1))# [Ngcm,Nlat,Nlon]
        input2 = torch.from_numpy(X2[np.newaxis,...]).float() # [Ngcm,Nlat,Nlon]
        inputs = [input1,input2]
        if use_climatology:
            Xaux = np.concatenate((data['climatology'],data['elevation']),axis=0)  # [2,Nlat,Nlon]
            input3 = torch.from_numpy(Xaux[np.newaxis,...]).float() #[1,2,Nlat,Nlon] --> [1,2,Nlat,Nlon]
            inputs += [input3]
        inputs = [e.to(device) for e in inputs]    
        with torch.no_grad():
            y_pred = model(*inputs) # [1,1,Nlat,Nlon]

        y_pred = np.squeeze(y_pred.cpu().detach().numpy()) # [1,1,Nlat,Nlon]-->[Nlat,Nlon]] 
        y = np.squeeze(y) # [1,Nlat,Nlon]-->[Nlat,Nlon]] 
        y_preds.append(y_pred)
        ys.append(y)
    
    y_preds = np.stack(y_preds,axis=0) #[Ntest,Nlat,Nlon], unit: mm/day
    ys = np.stack(ys,axis=0) #[Ntest,Nlat,Nlon], unit: mm/day
    if variable=='ppt':
        y_preds = np.expm1(y_preds*5.0) # [0.0,1.0]-->[0.0,R], unit: mm/day
        ys = np.expm1(ys*5.0) # [0.0,1.0]-->[0.0,R], unit: mm/day
        X_mean = np.mean(np.expm1(X*5.0),axis=0) # [Nlat,Nlon]
    elif variable=='tmax' or variable=='tmin':
        y_preds = y_preds*50.0 # [Nlat,Nlon] unit: Celsius
        ys = ys*50.0 # [Nlat,Nlon] unit: Celsius
        X_mean = np.mean(X*50.0,axis=0) # [Nlat,Nlon]
    else:
        print('Error! variable not recognized!')
    mse = np.mean((y_preds-ys)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_preds-ys))
    print('test data MSE={},\nRMSE={}\nMAE={}'.format(mse,rmse,mae))
    if savepath:
        np.savez(savepath+'pred_results_MSE{}.npz'.format(mse),y_preds=y_preds,mse=mse,rmse=rmse,mae=mae)
    
    #%% plot figures
    if verbose:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt 
        fig,axs = plt.subplots(2,2)
        axs[0,0].imshow(y)
        axs[0,0].set_title('y')
        axs[1,0].imshow(y_pred)
        axs[1,0].set_title('Prediction')
        axs[0,1].imshow(X_mean)
        axs[0,1].set_title('Input GCM mean')
        diff_y_pred = np.abs(y-y_pred)
        diff1 = axs[1,1].imshow(diff_y_pred)
        axs[1,1].set_title('Abs(y-pred)')
        fig.colorbar(diff1,ax=axs[1,1],fraction=0.05)
        ## hide x labels and  tick labels for top plots and y ticks for right plots
        for ax in axs.flat:
            ax.label_outer()
        if savepath:
            plt.savefig(savepath+'pred_vs_groundtruth.png',dpi=1200,bbox_inches='tight')
        #plt.show()
        
    
    return mae,mse,rmse

