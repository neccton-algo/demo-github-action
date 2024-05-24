using PyPlot
using Statistics
using NCDatasets
using DINCAE: getobs
using DINCAE
using BSON


include("train.jl")

model_name = "/home/abarth/src/demo-github-action/old/model-12785:14609-loss_function_DINCAE.bson"

model_name = "/home/abarth/src/demo-github-action/model-12786:14609-loss_function_DINCAE-100.bson"
model_name = "/home/abarth/src/demo-github-action/model-12786:14609-loss_function_DINCAE-100.bson"


BSON.@load model_name model_cpu meanx

filename = "cmems_obs-sst_glo_phy_my_l3s_P1D-m_multi-vars_9.15W-41.95E_30.05N-55.55N_1982-01-01-2022-12-31.nc"

train_indices = 1:(14975-365-1)
train_indices = 14975-6*365:(14975-365-1)
test_indices = (14975-365):14975

varname = "adjusted_sea_surface_temperature"
npast = 7

#dataset_train = NetCDFLoader3(filename,varname,device,npast,train_indices)
dataset_test = NetCDFLoader3(filename,varname,npast,test_indices, meanx = meanx)

loader_test = DINCAE.DataBatches(Array{Float32,4},dataset_test,1; shuffle=false)

#dataset_train.meanx


#mask = mean(isfinite.(dataset_train.x),dims=3) .> 0.05
mask = mean(isfinite.(dataset_test.x),dims=3)[:,:,1] .> 0.05

xin,xxtrue = first(loader_test)

ds = NCDataset(filename)
lon = ds["longitude"][:]
lat = ds["latitude"][:]
time = ds["time"][:]
close(ds)

device = gpu
model = model_cpu |> device;

xout = model(device(xin));
m_rec = cpu(xout)[:,:,1,1] .+ meanx .- 273.15
xxtrue2 = cpu(xxtrue)[:,:,1,1]  .+ meanx .- 273.15

m_rec[.!mask] .= NaN
xxtrue2[.!mask] .= NaN

sigma_rec = sqrt.(exp.(cpu(xout)[:,:,2,1]))
sigma_rec[.!mask] .= NaN


pl = Vector{Any}(undef,3)
plt.ion()
plt.gcf().set_size_inches(8, 8, forward=true)
plt.clf()

function myplot(data)
    item = plt.pcolor(lon,lat,data')
    plt.colorbar(orientation="horizontal")
    plt.gca().set_aspect(1/cos(pi * mean(lat) / 180))
    return item
end

plt.subplot(2,2,1)
pl[1] = myplot(xxtrue2)
plt.title("target observation")

plt.subplot(2,2,2)
pl[2] = myplot(m_rec)
plt.title("1-day forecast")
pl[2].set_clim(pl[1].get_clim())

plt.subplot(2,2,3)
pl[3] = myplot(sigma_rec)
plt.title("1-day forecast (exp. err. std.)")
pl[3].set_clim(0,1.5)

plt.savefig("example_unet_test_jl_dincae_loss.png")
