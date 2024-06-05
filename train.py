# Predict satellite data based on L3 data (including missing data)
# Alexander Barth

import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
import numpy as np
import os
from model import UNet
import time

class NetCDFLoader():
    def __init__(self,filename,varname,device,npast,indices,meanx = None):
        ds = xr.open_dataset(filename)
        x = np.float32(ds[varname][indices,:,:].data)

        if meanx == None:
            self.meanx = np.nanmean(x)
        else:
            self.meanx = meanx

        self.x = x - self.meanx
        self.npast = npast
        print("variable:      ",varname)
        print("shape:         ",x.shape)
        print("mean:          ",self.meanx)
        self.device = device

    def __len__(self):
        return self.x.shape[0] - self.npast

    def __getitem__(self,i):
        xi = self.x[i:i+self.npast,:,:].copy()
        mask = np.isnan(xi)
        xi[mask] = 0

        xin = torch.tensor(np.concatenate((xi,mask)), device=self.device)
        xtrue = torch.tensor(self.x[np.newaxis,i+self.npast,:,:], device=self.device)

        return (xin,xtrue)

def loss_function_MSE(xout,xtrue):
    m = torch.isfinite(xtrue)
    return torch.mean((xout[m] - xtrue[m])**2)

def loss_function_DINCAE(xout,xtrue):
    m_rec = xout[:,0:1,:,:]
    log_σ2_rec = xout[:,1:2,:,:]
    σ2_rec = torch.exp(log_σ2_rec)

    m = torch.isfinite(xtrue)
    difference2 = (m_rec[m] - xtrue[m])**2
    cost = torch.mean(difference2/σ2_rec[m] + log_σ2_rec[m])
    return cost


def train(model,dataset_train,nepochs,
          train_loss_function = loss_function_MSE,
          npast = 7,
          device = torch.device('cpu'),
          batchsize = 4,
          learning_rate = 0.001):


    training_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=batchsize, shuffle=True)
    model = model.to(device=device)

    # Test data loader
    xin,xtrue = next(iter(training_loader))

    # Test model
    xout = model(xin)
    print("shape of input x: ",xin.shape)
    print("shape of true x:  ",xtrue.shape)
    print("shape of output:  ",xout.shape)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    start = time.time()
    losses = []

    for epoch in range(nepochs):
        running_loss = 0.

        for (i,(xin,xtrue)) in enumerate(training_loader):
            optimizer.zero_grad()
            xout = model(xin)
            loss = train_loss_function(xout,xtrue)
            loss.backward()

            # Adjust learning weights
            optimizer.step()
            running_loss += loss.item()

            #if i % 10 == 0:
            #    print("    loss ",i,loss.item())

        losses.append(running_loss / len(training_loader))
        print("loss ",epoch,losses[-1])

    print("training time (seconds)",time.time() - start)
    return (model,losses)
