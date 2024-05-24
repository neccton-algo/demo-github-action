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

# class AttentionBlock(nn.Module):
#     function __init__(self, Fg, Fl, Fint=255):
#         super(AttentionBlock,self).__init__()
#         self.Wg = Conv(Fg,Fint,kernel_size=1)
#         self.Wx = Conv(Fl,Fint,kernel_size=1,bias=False)
#         self.Wx = Conv(Fl,Fint,kernel_size=1)
#         self.phi = Conv(Fint,1,kernel_size=1)

#     # x high res information to be scaled
#     # g low res information upsampled as gating signal
#     function forward(self,x,g):
#         q_att = self.phi(F.relu(self.Wg(g) + self.Wx(x)))
#         alpha = F.sigmoid(q_att)

#         x_hat = alpha * x_l
#         return x_hat


# class SkipConnection(nn.Module):
#     function __init__(self, module):
#         super().__init__()
#         self.module = module

#     function forward(self, inputs):

#         return torch.cat((self.module(inputs),inputs),dim=1)

struct UNet_old
    EncConv
    DecConv
    UpConv
    ConvOut
    ConvInner
end

Flux.@functor UNet_old

function UNet_old(in_channels,out_channels,n_channels = [64,128,256,512,1024])
    enc_channels = [in_channels,n_channels...]
    dec_channels = [out_channels,n_channels...]

    nlevels = length(n_channels)

    EncConv = [conv2d2(enc_channels[i-1] => enc_channels[i]) for i in 2:(nlevels)]

    DecConv = [conv2d2(2*dec_channels[i] => dec_channels[i]) for i in 2:(nlevels)]

    UpConv = [upsampling(dec_channels[i+1] => dec_channels[i]) for i in 2:(nlevels)]
    ConvOut = Conv((1,1),dec_channels[2] => dec_channels[1])
    ConvInner = conv2d2(enc_channels[end-1] => enc_channels[end])

    UNet_old(
        EncConv,
        DecConv,
        UpConv,
        ConvOut,
        ConvInner)
end

function (self::UNet_old)(xin)
    nlevels = length(self.EncConv)+1
#    x = Vector{Any}(undef,nlevels-1)
    xi = xin

    x = [begin
        xi = self.EncConv[l](xi)
        xl = xi
        println(" ll ",l,size(xi))
        xi = maxpool(xi,(2,2))
        println(" mp ",l,size(xi));
             xl
         end
         for l in 1:(nlevels-1)
    ]
    # for l in 1:(nlevels-1)
    #     xi = self.EncConv[l](xi)
    #     x[l] = xi
    #     println(" ll ",l,size(xi))
    #     xi = maxpool(xi,(2,2))
    #     println(" mp ",l,size(xi))
    # end

    xi = self.ConvInner(xi)
    println(" inner ",size(xi))

    for l in reverse(1:(nlevels-1))
        println(l)
        xi = self.UpConv[l](xi)
        #println(" up ",l,size(xi),size(x[l]))
        xi = cat(x[l],xi,dims=3)
        #xi = cat(ones(Float32,size(xi)),xi,dims=3)
        println(" cat ",size(xi))
        xi = self.DecConv[l](xi)
        println(" c2 ",size(xi))
    end

    xi = self.ConvOut(xi)
    return xi
end


#=
in_channels = 14
out_channels = 2


xin = zeros(Float32,256,256,in_channels,3);
model = UNet_old(in_channels,out_channels)
xout = model(xin)
model.nlevels
length(model.EncConv)
size(xout) == (256,256,out_channels,3)
=#




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

function UNet2(in_channels,out_channels,n_channels = [64,128,256,512,1024])
    Chain(
        unet_block(1,n_channels,in_channels,n_channels[1],relu),
        Conv((1,1),n_channels[1] => out_channels)
    )
end

#=
in_channels = 14
out_channels = 2


xin = zeros(Float32,256,256,in_channels,3);
model = UNet2(in_channels,out_channels)
xout = model(xin)
size(xout) == (256,256,out_channels,3)

sum(length.(Flux.params(model))) == 34533442
=#
