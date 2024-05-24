# Predict satellite data based on L3 data (including missing data)
# Alexander Barth

using NCDatasets
using Flux
import Base: length, getindex, size
using Statistics
using Random
import DINCAE
import DINCAE: getobs!, getobs, nobs, sizex, sizey
using Base.Threads
using ChainRulesCore

include("model.jl")


struct NetCDFLoader3{T,Tdevice} <: DINCAE.AbstractDataset{T,3}
    x::Array{T,3}
    meanx::T
    npast::Int
    device::Tdevice
end

function NetCDFLoader3(filename::AbstractString,varname,npast,indices; meanx = nothing)
    ds = NCDataset(filename,maskingvalue = NaN)
    x = Float32.(ds[varname][:,:,indices])

    if meanx == nothing
        #meanx = nanmean(x)
        # more precise
        meanx = Float32(mean(map(Float64,filter(isfinite,x))))
    end

    x .= x .- meanx
    println("variable:      ",varname)
    println("shape:         ",size(x))
    println("mean:          ",meanx)

    NetCDFLoader3(x,meanx,npast,cpu)
end

sizex(nl::NetCDFLoader3) = (size(nl.x,1),size(nl.x,2),nl.npast*2)
sizey(nl::NetCDFLoader3) = (size(nl.x,1),size(nl.x,2),1)

function getobs!(nl::NetCDFLoader3,(xin,xtrue),ind::Int)
    @inbounds for j = 1:size(nl.x,2)
        for i = 1:size(nl.x,1)
            for n = 0:(nl.npast-1)
                if isnan(nl.x[i,j,ind+n])
                    xin[i,j,2*n+1] = 0
                    xin[i,j,2*n+2] = 1
                else
                    xin[i,j,2*n+1] = nl.x[i,j,ind+n]
                    xin[i,j,2*n+2] = 0
                end

            end

            xtrue[i,j,1] = nl.x[i,j,ind+nl.npast]
        end
    end
    return (xin,xtrue)
end

nobs(nl::NetCDFLoader3) = size(nl.x)[end] - nl.npast

function loss_function_MSE(xout,xtrue)
    m = isfinite.(xtrue)
    return mean((xout[m] - xtrue[m]).^2)
end

function loss_function_DINCAE(xout,xtrue; eps = 1f-5)
    m_rec = @view xout[:,:,1:1,:]
    log_σ2_rec = @view xout[:,:,2:2,:]
    σ2_rec = @fastmath exp.(log_σ2_rec)

    m = isfinite.(xtrue)
    difference2 = (m_rec[m] - xtrue[m]).^2
    cost = mean(difference2./σ2_rec[m] + log_σ2_rec[m])
    return cost
end

function train(model,dataset_train,nepochs;
               train_loss_function = loss_function_MSE,
               npast = 7,
               device = cpu,
               batchsize = 4,
               learning_rate = 0.001)


    training_loader = DINCAE.DataBatches(Array{Float32,4},dataset_train,batchsize; shuffle=true)

    model = model |> device;

    # Test data loader
    xin_cpu,xtrue_cpu = first(training_loader)
    (xin,xtrue) = device.((xin_cpu,xtrue_cpu))

    # Test model
    xout = model(xin)
    println("threads:         ",nthreads())
    println("size of input x: ",size(xin))
    println("size of true x:  ",size(xtrue))
    println("size of output:  ",size(xout))
    println("nepochs:         ",nepochs)

    # Test loss
    loss = train_loss_function(xout,xtrue)
    println("initial loss:    ",loss)

    opt_state = Flux.setup(Adam(learning_rate), model)

    losses = zeros(Float32,nepochs)

    @time for epoch = 1:nepochs
        running_loss = 0.

            for (i,(xin_cpu,xtrue_cpu)) in enumerate(DINCAE.PrefetchDataIter(training_loader))
            #for (i,(xin_cpu,xtrue_cpu)) in enumerate(training_loader)
                #(i,(xin,xtrue)) = first(enumerate(training_loader))

                (xin,xtrue) = device.((xin_cpu,xtrue_cpu))

                loss, grads = Flux.withgradient(model) do m
                    xout = m(xin)
                    train_loss_function(xout,xtrue)
                end

                if !isfinite(loss)
                    @warn("stopping optimization loss: $loss")
                    break
                end

                # Adjust learning weights
                Flux.update!(opt_state, model, grads[1])
                running_loss += loss

                if (i-1) % 100 == 0
                    #println("    loss ",i," ",loss)
                end
            end
        losses[epoch] = running_loss / length(training_loader)
        println("loss ",epoch," ",losses[epoch])

        GC.gc()
        shuffle!(training_loader)
    end
    return (model,losses)
end
