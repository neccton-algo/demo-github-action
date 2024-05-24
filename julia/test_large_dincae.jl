using CUDA
using BSON

include("train.jl")

filename = "cmems_obs-sst_glo_phy_my_l3s_P1D-m_multi-vars_9.15W-41.95E_30.05N-55.55N_1982-01-01-2022-12-31.nc"
#train_indices = 1:(14975-365-1)

#test
#train_indices = 14975-2*365:(14975-365-1)
train_indices = 14975-6*365+1:(14975-365-1)

#test_indices = (14975-365):14975

nepochs = 1
nepochs = 100
#device = cpu
device = gpu
varname = "adjusted_sea_surface_temperature"
npast = 7
learning_rate = 5f-4

# Load training data
dataset_train = NetCDFLoader3(filename,varname,npast,train_indices)

# Instantiate the model
model = UNet2(2*npast,2);

train_loss_function = loss_function_DINCAE
#train_loss_function = loss_function_MSE

model,losses = train(model,dataset_train,nepochs;
                     train_loss_function = train_loss_function,
                     npast = 7,
                     device = device,
                     batchsize = 8,
                     learning_rate = learning_rate)

GC.gc()
model_cpu = model |> cpu
model_name = "model-$(train_indices)-$(train_loss_function)-$nepochs.bson"
@show model_name
meanx = dataset_train.meanx
BSON.@save model_name model_cpu losses meanx

# 1y: 110 s
# 1y: 25 s
# 1y: 27 s prepare batch on cpu then move to gpu
# 1y: 12.6 s with DINCAE loader

# all data
# 472.227602 s: use views
# 361.471912 s: DINCAE loader

# 1824 data: 14975-6*365+1:(14975-365-1)
# 11443.748211 s: julia (3 threads)

# pytorch
# training time (seconds) 382.6549620628357
