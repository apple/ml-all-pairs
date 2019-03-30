#
# For licensing see accompanying LICENSE.txt file.
# Copyright (C) 2018-2019 Apple Inc. All Rights Reserved.
#
from torch import mean, cat, squeeze
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np
import copy
from code_pytorch.layers import Identity, Dense, ConvNet


def chan_smax(x):
    x = x.permute(0, 2, 3, 1)
    x = nnf.softmax(x)
    x = x.permute(0, 3, 1, 2)
    return x


def get_activation(activ_str):
    activations = {
        'sigmoid': nnf.sigmoid,
        'softmax': chan_smax,
        'relu': nnf.relu,
        'tanh': nnf.tanh,
        'softplus': nnf.softplus,
        'softsign': nnf.softsign,
        'identity': lambda x: x,
        'selu': nnf.selu,
        'elu': nnf.elu,
        'zero': lambda x: 0.0 * x,
    }
    return activations[activ_str]


class TypeNet(nn.Module):
    def __init__(self, input_img_shape, output_size, num_top_features,
                 debug=False, activations=('softmax', 'softmax')):
        super(TypeNet, self).__init__()
        self.input_shape = input_img_shape  # [chan, H, W]
        self.output_size = output_size  # scalar
        self.num_top_features = num_top_features
        self.debug = debug
        self.activations = copy.deepcopy(activations)
        self.last_conv_size = 128

        # build the actual model
        self.feature_net = ConvNet(input_img_shape[0],
                                   filter_sizes=[3, 5, 5, 3],
                                   num_channels=[128, 128, 128, self.last_conv_size],
                                   strides=[1, 2, 1, 1],
                                   padding=[1, 2, 2, 1],  # just filter_size[i] // 2
                                   activation_fn=nn.ELU, use_bias=False,
                                   use_bn=True)

        self.type_branches = []
        for i in range(len(self.activations)):
            name = 'typenet{}'.format(i)
            tn = ConvNet(self.last_conv_size, filter_sizes=[1],
                         num_channels=[self.num_top_features], strides=[1], padding=[0],
                         activation_fn=Identity, use_bias=False,
                         use_bn=False)
            setattr(self, name, tn)
            self.type_branches.append(getattr(self, name))

        # put into a ModuleList, so pytorch manages submodules
        self.type_branches = nn.ModuleList(self.type_branches)

        in_size = 3 * self.num_top_features
        self.prednet = nn.Sequential(
            Dense(in_size, layers=[in_size, in_size // 2, in_size // 4], activation_fn=nn.ELU, use_bn=True),
            Dense(in_size // 4, layers=[self.output_size], use_bn=False, activation_fn=Identity)
        )

    @staticmethod
    def combine_types(type_list, activations, features):
        result = get_activation(activations[0])(type_list[0](features))
        for i in range(1, len(activations)):
            t = type_list[i]
            act = get_activation(activations[i])
            result = result + act(t(features))
        return result

    def forward(self, x):
        # extract the features
        features = self.feature_net(x)
        if self.debug:
            print("feature size = ", features.size())

        # get the combined types
        types_value = TypeNet.combine_types(self.type_branches, self.activations, features).contiguous()

        max_pooled = nnf.max_pool2d(types_value, kernel_size=5, stride=1, padding=2)
        other_pooled = nnf.max_pool2d(types_value, kernel_size=3, stride=1, padding=1)

        # some debug traces
        if self.debug:
            print("types_value = ", types_value.size())

        # build each_hist
        pooled_prod = int(np.prod(list(max_pooled.size())[2:]))
        types_value_prod = int(np.prod(list(types_value.size())[2:]))

        each_hist = cat([
            mean(types_value.view(-1, self.num_top_features, types_value_prod), -1),
            mean(max_pooled.view(-1, self.num_top_features, pooled_prod), -1),
            mean(other_pooled.view(-1, self.num_top_features, pooled_prod), -1),
        ], 1)

        hist = squeeze(each_hist)
        if self.debug:
            print("hist = ", hist.size())

        # finally run through densenet
        return self.prednet(hist)
