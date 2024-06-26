import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),".."))
import model
import train
import urllib

def get(url,filename):
    "download from `url` and save as `filename` unless the file is already present"
    if not os.path.isfile(filename):
        print(f"getting {url}")
        urllib.request.urlretrieve(url, filename)

def test_nn():
    filename = "cmems_obs-sst_glo_phy_my_l3s_P1D-m_multi-vars_9.15W-41.95E_30.05N-55.55N_2022-01-01-2022-01-31.nc"

    get("https://dox.ulg.ac.be/index.php/s/wKuyuGvX3bujc40/download",filename)

    train_indices = range(0,20)
    nepochs = 3
    #device = torch.device('cuda')
    device = torch.device('cpu')
    varname = "adjusted_sea_surface_temperature"
    npast = 7

    dataset_train = train.NetCDFLoader(filename,varname,device,npast,train_indices)

    # Instantiate the model
    unet_model = model.UNet(2*npast,1)

    unet_model,losses = train.train(
        unet_model,dataset_train,nepochs,
        npast = 7,
        batchsize = 4,
        device = device,
        learning_rate = 0.001)

    assert losses[-1] < losses[0]

# cpu 32 sec
# gpu 1 sec

#test_nn()
