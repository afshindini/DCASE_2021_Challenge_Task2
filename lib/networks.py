""" Network architectures.
"""
# pylint: disable=W0221,W0622,C0103,R0913

##
import torch
import torch.nn as nn
import torch.nn.parallel

##
def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)

class Encoder(nn.Module):
    def __init__(self,nz,ndf):
        super().__init__()
        self.block1 = nn.Sequential(
          nn.Conv2d(1,128,5,2,2,bias=False),
          nn.LeakyReLU(inplace=True),
          nn.BatchNorm2d(128),
          nn.Conv2d(128,128,5,1,2,bias=False),
          nn.LeakyReLU(inplace=True),
          nn.BatchNorm2d(128),
          nn.Conv2d(128,256,5,2,2,bias=False),
          nn.LeakyReLU(inplace=True),
          nn.BatchNorm2d(256),
          nn.Conv2d(256,256,5,2,2,bias=False),
          nn.LeakyReLU(inplace=True),
        )

    def forward(self,x):
        x = self.block1(x)
        return x


class Decoder(nn.Module):
    def __init__(self,nz,ngf):
        super().__init__()
        self.block5 = nn.Sequential(
          nn.ConvTranspose2d(256,512,5,2,2,output_padding=1,bias=False),
          nn.BatchNorm2d(512),
          nn.ReLU(inplace=True),
          nn.ConvTranspose2d(512,512,5,2,2,output_padding=1,bias=False),
          nn.BatchNorm2d(512),
          nn.ReLU(inplace=True),
          nn.ConvTranspose2d(512,1,5,2,2,output_padding=1,bias=False),
          nn.ReLU(inplace=True),
          
        )
    def forward(self,x):
        x = self.block5(x)
        return x

class Discrim(nn.Module):
    def __init__(self,nz,ndf):
        super().__init__()
        self.block1 = nn.Sequential(
          nn.Conv2d(1,128,5,2,2,bias=False),
          nn.LeakyReLU(inplace=True),
          nn.BatchNorm2d(128),
          nn.Conv2d(128,128,5,1,2,bias=False),
          nn.LeakyReLU(inplace=True),
          nn.BatchNorm2d(128),
          nn.Conv2d(128,256,5,2,2,bias=False),
          nn.LeakyReLU(inplace=True),
          nn.BatchNorm2d(256),
          nn.Conv2d(256,256,5,2,2,bias=False),
          nn.LeakyReLU(inplace=True),
          nn.BatchNorm2d(256),
          nn.Conv2d(256,256,5,2,2,bias=False),
          nn.LeakyReLU(inplace=True),
        )
    def forward(self,x):
        x = self.block1(x)
        return x


class NetD(nn.Module):
    """
    DISCRIMINATOR NETWORK
    """

    def __init__(self, opt):
        super(NetD, self).__init__()
        model = Discrim( opt.nz, opt.ndf)
        self.features = nn.Sequential(model.block1)
        self.classifier = nn.Sequential(nn.Linear(8*20*256,1))
        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        features = self.features(x)
        o = features.view(x.shape[0],-1)
        
        classifier = self.classifier(o)
        classifier = classifier.view(-1, 1).squeeze(1)

        return classifier, features

##
class NetG(nn.Module):
    """
    GENERATOR NETWORK
    """

    def __init__(self, opt):
        super(NetG, self).__init__()
        self.encoder1 = Encoder( opt.nz, opt.ngf)
        self.decoder = Decoder(opt.nz, opt.ngf)
        self.encoder2 = Encoder( opt.nz, opt.ngf)

    def forward(self, x):
        latent_i = self.encoder1(x)
        gen_imag = self.decoder(latent_i)
        latent_o = self.encoder2(gen_imag)
        return gen_imag, latent_i, latent_o
