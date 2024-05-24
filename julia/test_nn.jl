using Test
using Downloads: download

include("train.jl")

function get(url,filename)
    if !isfile(filename)
        println("getting $url")
        download(url, filename)
    end
end


filename = "cmems_obs-sst_glo_phy_my_l3s_P1D-m_multi-vars_9.15W-41.95E_30.05N-55.55N_2022-01-01-2022-01-31.nc"

get("https://dox.ulg.ac.be/index.php/s/wKuyuGvX3bujc40/download",filename)

train_indices = 1:20
nepochs = 3
device = cpu
varname = "adjusted_sea_surface_temperature"
npast = 7

# Load training data
dataset_train = NetCDFLoader3(filename,varname,npast,train_indices)

# Instantiate the model
model = UNet(2*npast,1)


#=
in_channels = 14
out_channels = 2


xin = zeros(Float32,256,256,in_channels,3);
model = UNet(in_channels,out_channels)
xout = model(xin)
size(xout) == (256,256,out_channels,3)

sum(length.(Flux.params(model))) == 34533442
=#

#train_loss_function = loss_function_DINCAE
train_loss_function = loss_function_MSE

model,losses = train(model,dataset_train,nepochs;
                     npast = 7,
                     device = device,
                     batchsize = 4,
                     learning_rate = 0.001)

@test losses[end] < losses[1]
