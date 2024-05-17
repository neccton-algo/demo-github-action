# Flexible UNet implement with arbitary number of channels and depth
# Alexander Barth

import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
import numpy as np
import os

def conv2d2(in_channels,out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


def upsampling(in_channels,out_channels):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2),
        nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


class AttentionBlock(nn.Module):
    def __init__(self, Fg, Fl, Fint=255):
        super(AttentionBlock,self).__init__()
        self.Wg = nn.Conv2d(Fg,Fint,kernel_size=1)
        self.Wx = nn.Conv2d(Fl,Fint,kernel_size=1,bias=False)
        self.Wx = nn.Conv2d(Fl,Fint,kernel_size=1)
        self.phi = nn.Conv2d(Fint,1,kernel_size=1)

    # x high res information to be scaled
    # g low res information upsampled as gating signal
    def forward(self,x,g):
        q_att = self.phi(F.relu(self.Wg(g) + self.Wx(x)))
        alpha = F.sigmoid(q_att)

        x_hat = alpha * x_l
        return x_hat


class SkipConnection(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):

        return torch.cat((self.module(inputs),inputs),dim=1)

class UNet(nn.Module):
    def __init__(self, in_channels,out_channels,n_channels = [64,128,256,512,1024]):
        super().__init__()
        enc_channels = [in_channels,*n_channels]
        dec_channels = [out_channels,*n_channels]

        nlevels = len(n_channels)
        self.nlevels = nlevels
        self.EncConv = nn.ModuleList([conv2d2(enc_channels[i-1],enc_channels[i]) for i in range(1,nlevels)])
        self.DecConv = nn.ModuleList([conv2d2(2*dec_channels[i],dec_channels[i]) for i in range(1,nlevels)])
        self.UpConv = nn.ModuleList([upsampling(dec_channels[i+1],dec_channels[i]) for i in range(1,nlevels)])
        self.ConvOut = nn.Conv2d(dec_channels[1],dec_channels[0],kernel_size=1)
        self.ConvInner = conv2d2(enc_channels[-2],enc_channels[-1])

    def forward(self, xin):
        nlevels = self.nlevels
        x = [None] * (nlevels-1)
        xi = xin

        for l in range(nlevels-1):
            xi = self.EncConv[l](xi)
            x[l] = xi
            #print(" ll ",l,xi.shape)
            xi = F.max_pool2d(xi,2)
            #print(" mp ",l,xi.shape)

        xi = self.ConvInner(xi)
        #print(" inner ",xi.shape)

        for l in reversed(range(nlevels-1)):
            #print(l)
            xi = self.UpConv[l](xi)
            #print(" up ",l,xi.shape)
            xi = torch.cat((x[l],xi),dim=1)
            #print(" cat ",xi.shape)
            xi = self.DecConv[l](xi)
            #print(" c2 ",xi.shape)

        xi = self.ConvOut(xi)
        return xi
