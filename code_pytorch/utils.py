#
# For licensing see accompanying LICENSE.txt file.
# Copyright (C) 2018-2019 Apple Inc. All Rights Reserved.
#
import torch
import numpy as np
from torch.autograd import Variable


def softmax_accuracy(preds, targets, size_average=True):
    pred = to_data(preds).max(1)[1]  # get the index of the max log-probability
    reduction_fn = torch.mean if size_average is True else torch.sum
    return reduction_fn(pred.eq(to_data(targets)).cpu().type(torch.FloatTensor))


def squeeze_expand_dim(tensor, axis):
    """ helper to squeeze a multi-dim tensor and then
        unsqueeze the axis dimension if dims < 4"""
    tensor = torch.squeeze(tensor)
    if len(list(tensor.size())) < 4:
        return tensor.unsqueeze(axis)
    else:
        return tensor


def one_hot_np(num_cols, indices):
    num_rows = len(indices)
    mat = np.zeros((num_rows, num_cols))
    mat[np.arange(num_rows), indices] = 1
    return mat


def one_hot(size, index, use_cuda=False):
    """ Creates a matrix of one hot vectors."""
    mask = long_type(use_cuda)(*size).fill_(0)
    ones = 1
    if isinstance(index, Variable):
        ones = Variable(long_type(use_cuda)(index.size()).fill_(1))
        mask = Variable(mask, volatile=index.volatile)

    ret = mask.scatter_(1, index, ones)
    return ret


def to_data(tensor_or_var):
    """simply returns the data"""
    if type(tensor_or_var) is Variable:
        return tensor_or_var.data
    else:
        return tensor_or_var


def float_type(use_cuda):
    return torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


def int_type(use_cuda):
    return torch.cuda.IntTensor if use_cuda else torch.IntTensor


def long_type(use_cuda):
    return torch.cuda.LongTensor if use_cuda else torch.LongTensor

