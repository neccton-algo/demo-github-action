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
import urllib

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
        print("variable ",varname)
        print("shape: ",x.shape)
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

def get(url,filename):
    "download from `url` and save as `filename` unless the file is already present"
    if not os.path.isfile(filename):
        urllib.request.urlretrieve("https://dox.ulg.ac.be/index.php/s/b3DWpYysuw6itOz/download", filename)


os.environ["CI"] = "true"

if os.environ.get("CI","false") == "true":
    filename = "cmems_obs-sst_glo_phy_my_l3s_P1D-m_multi-vars_9.15W-41.95E_30.05N-55.55N_2022-01-01-2022-01-31.nc"

    get("https://dox.ulg.ac.be/index.php/s/wKuyuGvX3bujc40/download",filename)

    train_indices = range(0,20)
    test_indices = range(20,31)
    nepochs = 10
else:
    filename = "cmems_obs-sst_glo_phy_my_l3s_P1D-m_multi-vars_9.15W-41.95E_30.05N-55.55N_1982-01-01-2022-12-31.nc"
    train_indices = range(0,14975-365)
    test_indices = range((14975-365),14975)
    nepochs = 50


def loss_function(xout,xtrue):
    m = torch.isfinite(xtrue)
    return torch.mean((xout[m] - xtrue[m])**2)


varname = "adjusted_sea_surface_temperature"
npast = 7
learning_rate = 0.001
device = torch.device('cuda')

dataset_train = NetCDFLoader(filename,varname,device,npast,train_indices)
dataset_test = NetCDFLoader(filename,varname,device,npast,test_indices, meanx = dataset_train.meanx)

training_loader = torch.utils.data.DataLoader(dataset_train, batch_size=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False)

# Instantiate the model
model = UNet(2*npast,1)
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

for epoch in range(nepochs):
    running_loss = 0.

    for (i,(xin,xtrue)) in enumerate(training_loader):
        optimizer.zero_grad()
        xout = model(xin)
        loss = loss_function(xout,xtrue)
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        running_loss += loss.item()

        #if i % 10 == 0:
        #    print("    loss ",i,loss.item())


    print("loss ",epoch,running_loss / len(training_loader))

print("training time (seconds)",time.time() - start)
