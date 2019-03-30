#
# For licensing see accompanying LICENSE.txt file.
# Copyright (C) 2018-2019 Apple Inc. All Rights Reserved.
#
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as M
from collections import OrderedDict


AFFINE_BN = False


def init_weights(module):
    ''' helper to do xavier initialization '''
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            print("initializing ", m, " with xavier init")
            nn.init.xavier_uniform(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                print("initial bias from ", m, " with zeros")
                nn.init.constant(m.bias, 0.0)
        elif isinstance(m, nn.Sequential):
            for mod in m:
                init_weights(mod)

    return module


class View(nn.Module):
    def __init__(self, shape):
        """ use this to reshape into any shape """
        super(View, self).__init__()
        self.shape = shape

    def forward(self, inputs):
        return inputs.view(*self.shape)


class Identity(nn.Module):
    """ use this for linear activation """
    def __init__(self, inplace=True):
        super(Identity, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x if self.inplace else x.clone()


class Zero(nn.Module):
    """ use this for linear activation """
    def __init__(self, inplace=True):
        super(Zero, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * 0


class ConvNet(nn.Module):
    def __init__(self, in_num_channels,
                 filter_sizes=(5, 4, 4, 4, 4, 1, 1),
                 num_channels=(32, 64, 128, 256, 512, 512, 512),
                 strides=(1, 2, 1, 2, 1, 1, 1),
                 padding=(0, 0, 0, 0, 0, 0, 0),
                 use_bn=True,
                 use_bias=False,
                 activation_fn=nn.ELU):
        """ helper function to build convolutional network """
        super(ConvNet, self).__init__()  # call superclass
        print("CREATING ConvNet: ", len(filter_sizes))

        assert len(filter_sizes) == len(num_channels) == len(strides) == len(padding)

        network = []
        current_channel = in_num_channels

        for i, (filt, stride, out_channel, pad) in enumerate(zip(filter_sizes, strides, num_channels, padding)):
            # - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
            network.append(('conv_%d' % i,
                            nn.Conv2d(in_channels=current_channel,
                                      out_channels=out_channel,
                                      padding=pad,
                                      kernel_size=filt,
                                      stride=stride,
                                      bias=use_bias)))
            if activation_fn == nn.Sigmoid:
                network.append(('activ_%d' % i, nn.Sigmoid()))
            else:
                network.append(('activ_%d' % i, activation_fn(inplace=True)))

            if use_bn:
                # add batch norm after activation
                network.append(('bn_%d' % i, nn.BatchNorm2d(out_channel, eps=0.001, momentum=0.01, affine=AFFINE_BN)))

            current_channel = out_channel

        # ordered-dict for a sequential nn-module
        self.net = nn.Sequential(OrderedDict(network))

        # initialize weights to xavier
        self.net = init_weights(self.net)

    def forward(self, x):
        return self.net(x)


class Dense(nn.Module):
    def __init__(self, input_size,
                 layers=(512, 512, 512),
                 use_bn=True,
                 activation_fn=nn.ELU,
                 bias=True):
        """ helper function to build dense network """
        super(Dense, self).__init__()  # call superclass

        network = []
        current_size = input_size

        for i, dim in enumerate(layers):
            # - Input: :math:`(N, feat )`
            network.append(('dense_%d' % i,
                            nn.Linear(in_features=int(current_size),
                                      out_features=int(dim),
                                      bias=bias)))
            network.append(('activ_%d' % i, activation_fn(inplace=True)))
            if use_bn:
                network.append(('bn_%d' % i, nn.BatchNorm1d(dim, eps=0.001, momentum=0.01, affine=AFFINE_BN)))

            current_size = dim

        # use ordered-dict to house in a sequential nn-module
        self.net = nn.Sequential(OrderedDict(network))
        self.net = init_weights(self.net)

    def forward(self, x):
        return self.net(x)


def upsample(img, output_img_shape, mode='bilinear'):
    """ simple helper for upsampling """
    return F.upsample(img, size=output_img_shape, mode=mode)
