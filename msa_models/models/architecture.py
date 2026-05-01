# Design differnent basic network architecture aginst MSA
import typing as T
import functools
from functools import partial, reduce
import functools as ft
from typing import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import torch.nn as nn
from torch.optim import lr_scheduler

from torch.nn.modules.flatten import Flatten
from torch.nn import Identity
import torch.nn.parallel as nn_parallel

Tensor = torch.Tensor
NormLayer = T.Union[T.Type[nn.BatchNorm2d], 
                    T.Type[nn.InstanceNorm2d]]
ParialNorm = T.TypeVar("ParialNorm", 
                      ft.partial[nn.BatchNorm2d],
                      ft.partial[nn.InstanceNorm2d])
# PartialNorm = T.Union[T.Type[ft.partial[nn.BatchNorm2d]],
#                       T.Type[ft.partial[nn.InstanceNorm2d]]]


import os
import sys

prj_dir = os.path.dirname(os.path.dirname(__file__))
if prj_dir not in sys.path:
    sys.path.append(prj_dir)

from math import ceil

def get_filter(filt_size:int = 3):
    a: Tensor
    if(filt_size == 1):
        a = torch.tensor([1.])
    elif(filt_size == 2):
        a = torch.tensor([1., 1.])
    elif(filt_size == 3):
        a = torch.tensor([1., 2., 1.])
    elif(filt_size == 4):
        a = torch.tensor([1., 3., 3., 1.])
    elif(filt_size == 5):
        a = torch.tensor([1., 4., 6., 4., 1.])
    elif(filt_size == 6):
        a = torch.tensor([1., 5., 10., 10., 5., 1.])
    elif(filt_size == 7):
        a = torch.tensor([1., 6., 15., 20., 15., 6., 1.])
    else:
        raise NotImplementedError(f"filt_size = {filt_size} is not implemented")

    # filt = torch.Tensor(a[:, None] * a[None, :])
    filt = a[:, None] * a[None, :]

    return filt / filt.sum()


class Downsample(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = (int(1. * (filt_size - 1) / 2), int(ceil(1. * (filt_size - 1) / 2)), int(1. * (filt_size - 1) / 2), int(ceil(1. * (filt_size - 1) / 2)))
        self.pad_sizes = tuple([pad_size + pad_off for pad_size in self.pad_sizes])
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        # r, c = filt.shape
        # self.register_buffer('filt', filt.reshape((1, 1, r, c)).repeat(self.channels, 1, 1, 1))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    # def forward(self, inp:Tensor):
    #     if(self.filt_size == 1):
    #         if(self.pad_off == 0):
    #             return inp[:, :, ::self.stride, ::self.stride]
    #         else:
    #             return self.pad(inp)[:, :, ::self.stride, ::self.stride]
    #     else:
    #         assert isinstance(self.filt, Tensor)
    #         return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

    # improve the pad for large batch size
    def forward(self, inp: Tensor):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            assert isinstance(self.filt, Tensor)
            # Avoid the 32-bit index limitation of ReflectionPad2d
            # Leave some margin for the threshold, e.g., 2**30
            THRESHOLD = 2**30
            if inp.numel() < THRESHOLD:
                return F.conv2d(self.pad(inp), self.filt,
                                stride=self.stride, groups=inp.shape[1])
            # control
            per_sample = inp[0].numel()
            chunk = max(1, THRESHOLD // per_sample)
            outs = []
            for sub in inp.split(chunk, dim=0):
                outs.append(F.conv2d(self.pad(sub), self.filt,
                                     stride=self.stride, groups=inp.shape[1]))
            return torch.cat(outs, dim=0)
class Upsample(nn.Module):
    def __init__(self, channels, pad_type='repl', filt_size=4, stride=2):
        super(Upsample, self).__init__()
        self.filt_size = filt_size
        # self.filt_odd = np.mod(filt_size, 2) == 1
        self.filt_odd = (filt_size % 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size) * (stride**2)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        # r, c = filt.shape
        # self.register_buffer('filt', filt.reshape((1, 1, r, c)).repeat(self.channels, 1, 1, 1))

        self.pad = get_pad_layer(pad_type)((1, 1, 1, 1))

    def forward(self, inp : Tensor):
        assert isinstance(self.filt, Tensor)
        ret_val = F.conv_transpose2d(self.pad(inp), self.filt, stride=self.stride, padding=1 + self.pad_size, groups=inp.shape[1])[:, :, 1:, 1:]
        if(self.filt_odd):
            return ret_val
        else:
            return ret_val[:, :, :-1, :-1]


def get_pad_layer(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        raise KeyError('Pad type [%s] not recognized' % pad_type)
    return PadLayer


def get_norm_layer(norm_type="instance"):
    """
    """
    layer_dict = dict(
        batch=partial(nn.BatchNorm2d, affine=True, track_running_stats=True),
        instance=partial(nn.InstanceNorm2d, affine=False, track_running_stats=False),
        none=nn.Identity()
    )
    assert norm_type in layer_dict, \
        f"normalization layer {norm_type} is not found"
    return layer_dict[norm_type]

def init_weights(net : nn.Module, init_type : str='normal', init_gain: float=0.02, debug:bool=False):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m : nn.Module):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            assert isinstance(m.weight.data, Tensor)
            if debug:
                print(classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=int(init_gain))
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
              
            if hasattr(m, 'bias') and m.bias is not None:
                assert isinstance(m.bias.data, Tensor)
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            assert isinstance(m.weight.data, Tensor)
            assert isinstance(m.bias.data, Tensor)
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type : str='normal', init_gain : float=0.02, 
             gpu_ids=[], debug=False, initialize_weights=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    # if len(gpu_ids) > 0:
    #     assert(torch.cuda.is_available())
        # net.to(gpu_ids[0])
        # net = nn_parallel.DataParallel(net, gpu_ids)  # multi-GPUs for non-AMP training
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net

class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, 
                 norm_layer: T.Union[NormLayer,
                                     nn.Identity,
                                     ft.partial[nn.BatchNorm2d],
                                     ft.partial[nn.InstanceNorm2d]]=nn.BatchNorm2d, 
                 use_dropout=False, n_blocks=6, padding_type='reflect', 
                 no_antialias=False, no_antialias_up=False, opt=None):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            assert isinstance(norm_layer, ft.partial)
            use_bias = norm_layer.func == nn.InstanceNorm2d
            nl = norm_layer.func(ngf)
        else:
            assert isinstance(norm_layer, nn.BatchNorm2d) or \
                isinstance(norm_layer, nn.InstanceNorm2d) or \
                isinstance(norm_layer, nn.Identity)
            use_bias = norm_layer == nn.InstanceNorm2d
            nl = norm_layer(ngf)
            

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 nl,
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if(no_antialias):
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            if no_antialias_up:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(ngf * mult // 2),
                          nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,  # output_padding=1,
                                    bias=use_bias),
                          norm_layer(ngf * mult // 2),
                          nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input: Tensor):
        """Standard forward"""
        fake : Tensor = self.model(input)
        return fake


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), 
                       norm_layer(dim), 
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), 
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out
