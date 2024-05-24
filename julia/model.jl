# Flexible UNet implement with arbitary number of channels and depth
# Alexander Barth

using Flux
using NCDatasets


function conv2d2((in_channels,out_channels))
    return [
        Conv((3,3),in_channels => out_channels,pad=1),
        BatchNorm(out_channels),
        relu,
        Conv((3,3),out_channels => out_channels,pad=1),
        BatchNorm(out_channels),
        relu,
    ]
end

function upsampling((in_channels,out_channels))
    return Chain(
        Upsample(2),
        Conv((3,3),in_channels => out_channels,pad=1),
        BatchNorm(out_channels),
        relu,
    )
end


# concatenate channels
cat_channels(mx,x) = cat(mx, x, dims=3)

# helper function to show the size of a layer
function showsize(tag)
    return function s(x)
        @show tag,size(x)
        return x
    end
end

# individual block of a UNet
function unet_block(l,nchannels,in_channels,out_channels,activation)
    nchan = nchannels[l]
    if l == 1
        nin,nout = in_channels,out_channels
    else
        nin = nchannels[l-1]
        nout = nchannels[l]
    end
    @show l,nchannels,nin,nout,nchan

    if l == length(nchannels)
        return Chain(conv2d2(nchannels[l-1] => nchan)...)
    end

    inner = unet_block(l+1,nchannels,in_channels,out_channels,activation)


    return Chain(
        #showsize("A $nin"),
        conv2d2(nin => nchan)...,
        SkipConnection(
            Chain(
                MaxPool((2,2)),
                inner,
                #showsize("C $(nchannels[l+1]) => $nchan"),
                Upsample(2),
                Conv((3,3),nchannels[l+1] => nchan,pad = SamePad()),
                BatchNorm(nchan),
                activation,

                #ConvTranspose((2,2),nchannels[l+1] => nchan,#=activation,=#stride=2),
                #showsize("D"),
            ),
            cat_channels
        ),
        #showsize("G $(2*nchan)"),
        conv2d2(2*nchan => nchan)...,
        #showsize("out level $nout"),
    )
end

function UNet(in_channels,out_channels,n_channels = [64,128,256,512,1024])
    Chain(
        unet_block(1,n_channels,in_channels,n_channels[1],relu),
        Conv((1,1),n_channels[1] => out_channels)
    )
end
