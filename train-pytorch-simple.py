#
# For licensing see accompanying LICENSE.txt file.
# Copyright (C) 2018-2019 Apple Inc. All Rights Reserved.
#
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as nnf
from torch.autograd import Variable
# local imports for graphing / dataloading / basic layers
from code_pytorch.utils import softmax_accuracy
from code_pytorch.grid_loader import GridDataLoader, set_sample_spec
from code_pytorch.typenet import TypeNet
from allpairs.grid_generator import SampleSpec

parser = argparse.ArgumentParser(description='Typenet')
# Task parameters
parser.add_argument('--max-train-samples', type=int, default=100000000)
parser.add_argument('--num-top-features', type=int, default=64)
# grid parameters
parser.add_argument('--num-pairs', type=str, default=4)
parser.add_argument('--num-classes', type=str, default=4)
# Model parameters
parser.add_argument('--batch-size', type=int, default=400)
parser.add_argument('--activations', nargs='+', default=['softmax', 'softmax'], type=str)
parser.add_argument('--lr', type=float, default=1e-3)
# Device parameters
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--no-cuda', action='store_true', default=False)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


def train(num_trained_on, model, optimizer, data_loader, loss_function=nnf.cross_entropy):
    """ Note: nnf.cross_entropy is like softmax_cross_entropy_with_logits """
    model.train()

    for batch_idx, (data, target) in enumerate(data_loader.train_loader):
        def op(index_of_batch, batch_data, batch_target):
            if args.cuda:
                batch_data, batch_target = batch_data.cuda(), batch_target.cuda().squeeze()

            batch_data, batch_target = Variable(batch_data), Variable(batch_target)

            optimizer.zero_grad()

            # compute loss
            predictions = model(batch_data)
            loss = loss_function(predictions, batch_target)
            loss.backward()
            optimizer.step()

            # log every nth interval
            if index_of_batch % 20 == 0:
                correct = softmax_accuracy(predictions, batch_target)
                num_samples = num_trained_on + index_of_batch * len(batch_data)
                percent_complete = float(num_samples) / args.max_train_samples
                batch_progress = 100.0 * index_of_batch / len(data_loader.train_loader)
                print('{}: {}% {}%, Loss: {:.6f} Acc: {:.4f}'.format(
                    num_samples, percent_complete, batch_progress, loss.data[0], correct))

        op(batch_idx, data, target)


def test(model, data_loader, loss_function=nnf.cross_entropy):
    """ Note: nnf.cross_entropy is like softmax_cross_entropy_with_logits """
    model.eval()
    test_loss = 0.0
    correct = 0

    loader = data_loader.test_loader
    for data, target in loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda().squeeze()

        data, target = Variable(data, volatile=True), Variable(target)

        predictions = model(data)
        loss_t = loss_function(predictions, target)
        test_loss += loss_t.data[0]

        # compute accuracy
        correct += softmax_accuracy(predictions, target, size_average=False)

    test_loss /= len(loader)  # loss function already averages over batch size
    correct_percent = 100.0 * correct / len(loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(test_loss, correct_percent))


def get_data_loader():
    """ helper to return the data loader """
    im_dim = [76, 96][int(args.num_pairs) > 5]
    set_sample_spec(int(args.num_pairs), int(args.num_classes), im_dim=im_dim)
    batches_per_epoch = 50000 // args.batch_size
    print("{} batches of {} in effective epoch of {} size".format(batches_per_epoch, args.batch_size, 50000))
    loader = GridDataLoader(batch_size=args.batch_size, batches_per_epoch=batches_per_epoch)
    return loader


def run():
    data = get_data_loader()
    typenet = TypeNet(data.img_shp, data.output_size, args.num_top_features, activations=args.activations)

    if args.ngpu > 1: # parallelize across multiple GPU's
        typenet = nn.DataParallel(typenet)
    if args.cuda: # push to cuda
        typenet = typenet.cuda()

    # main training loop
    optimizer = optim.Adam(typenet.parameters(), lr=args.lr)
    num_trained_on = 0
    while num_trained_on < args.max_train_samples:
        train(num_trained_on, typenet, optimizer, data)
        num_trained_on += args.batch_size
        test(typenet, data)

    # close all the generators, needed to deal with multi-process generators
    SampleSpec.close_generators()


if __name__ == "__main__":
    run()
