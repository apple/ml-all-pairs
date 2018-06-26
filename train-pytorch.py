from __future__ import print_function
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as nnf
import numpy as np
from torch.autograd import Variable

# local imports for graphing / dataloading / basic layers
from code_pytorch.utils import softmax_accuracy
from code_pytorch.grid_loader import GridDataLoader, set_sample_spec
from code_pytorch.typenet import TypeNet
from allpairs.grid_generator import SampleSpec
from utils.file_logger import FileLogger

parser = argparse.ArgumentParser(description='Typenet', fromfile_prefix_chars='@')

# Global parameters
parser.add_argument('--debug', action='store_true',
                    help='print debug messages to stderr')

parser.add_argument('--note', type=str, default="", help='a note to log')
parser.add_argument('--id', type=str, default="", help='an id to append to the run name')
parser.add_argument('--filelog', type=str, default="results", help='log to files in directory')

# Task parameters
parser.add_argument('--epochs', type=int, default=2000, metavar='N',
                    help='number of epochs to train (default: 2000)')
parser.add_argument('--num-top-features', type=int, default=64,
                    help='number of top conv features (default: 64)')

# grid parameters
parser.add_argument('--num-pairs', type=str, default="4", help='how many pairs')
parser.add_argument('--num-classes', type=str, default="4", help='num of different symbols')
parser.add_argument('--reset-every', type=str, default="", help='reset generator after this number of samples')

# Model parameters
parser.add_argument('--batch-size', type=int, default=400, metavar='N',
                    help='input batch size for training (default: 400)')
parser.add_argument('--activations', nargs='+', 
                    help='list of activations', default=['softmax', 'softmax'], type=str)
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate (default: 1e-3)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')

# Device parameters
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of gpus available (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


def get_short_name():
    result = str(args.num_pairs) + '-' + str(args.num_classes)
    result += '_Cnap'
    result += str(args.num_top_features)
    for act in args.activations:
        result += '_' + short_activation_name(act)
    result += '_' + str(args.ngpu) + '_' + str(args.batch_size) + '_' + str(args.lr)
    return result + '_' + str(args.id)


def short_activation_name(long_name):
    lookup = {
        'softmax':  'Sm',
        'sigmoid':  'S',
        'relu':     'R',
        'tanh':     'T',
        'softplus': 'Sp',
        'softsign': 'Ss',
        'selu':     'Se',
        'elu':      'E',
        'identity': 'I',
        'zero':     'Z',
    }
    assert long_name in lookup, "short_activation_name missing for {}".format(long_name)
    return lookup[long_name]


def train(epoch, model, optimizer, data_loader, logger, loss_function=nnf.cross_entropy):
    """ Note: nnf.cross_entropy is like softmax_cross_entropy_with_logits """
    model.train()

    for batch_idx, (data, target) in enumerate(data_loader.train_loader):
        # @timing
        def op(index_of_batch, batch_data, batch_target):
            if args.cuda:
                batch_data, batch_target = batch_data.cuda(), batch_target.cuda().squeeze()

            batch_data, batch_target = Variable(batch_data), Variable(batch_target)

            optimizer.zero_grad()

            # run the model
            predictions = model(batch_data)

            # compute loss
            loss = loss_function(predictions, batch_target)
            loss.backward()
            optimizer.step()

            # compute accuracy
            correct = softmax_accuracy(predictions, batch_target)

            # log every nth interval
            if index_of_batch % args.log_interval == 0:
                run_id = ''
                if args.filelog != '':
                    run_id = logger.uuid

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.4f}  {}'.format(
                    epoch, index_of_batch * len(batch_data), len(data_loader.train_loader.dataset),
                    100.0 * index_of_batch / len(data_loader.train_loader), loss.data[0], correct, run_id))

                if logger is not None:
                    logger.train_log(epoch, loss.data[0], correct)
                    logger.record('train', 'epoch', epoch)
                    logger.record('train', 'loss', loss.data[0])
                    logger.record('train', 'acc', correct)
                    logger.new_row('train')

        op(batch_idx, data, target)

        if args.debug:
            params = list(model.parameters())
            num_params = 0
            for p in params:
                print("paramSize = {}".format(p.size()))
                num_params += int(np.prod(list(p.size())))
            print("{} is the total num of parameters".format(num_params))
            sys.exit()


def test(epoch, model, data_loader, logger, loss_function=nnf.cross_entropy):
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

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(loader.dataset),
        correct_percent))

    # plot the test accuracy and loss
    if logger is not None:
        logger.test_log(epoch, test_loss, correct_percent / 100.0)
        logger.copy_sources()
        logger.record('test', 'epoch', epoch)
        logger.record('test', 'loss', test_loss)
        logger.record('test', 'acc', correct_percent / 100.0)
        logger.new_row('test')


def get_data_loader():
    """ helper to return the data loader """
    reset_every = None
    if len(args.reset_every) > 0:
        reset_every = int(args.reset_every)

    im_dim = 76
    if int(args.num_pairs) > 5:
        # increase the dimensionality of the image
        im_dim = 96

    set_sample_spec(int(args.num_pairs), int(args.num_classes), reset_every=reset_every, im_dim=im_dim)
    batches_per_epoch = 50000 // args.batch_size
    print("{} batches of {} in effective epoch of {} size".format(batches_per_epoch, args.batch_size, 50000))
    loader = GridDataLoader(batch_size=args.batch_size,
                            batches_per_epoch=batches_per_epoch)

    return loader


def run():
    # get the dataloader we want to work with
    data_loader = get_data_loader()

    # build main typenet object
    typenet = TypeNet(data_loader.img_shp,
                      data_loader.output_size,
                      args.num_top_features,
                      debug=args.debug,
                      activations=args.activations)

    # parallelize across multiple GPU's
    if args.ngpu > 1:
        typenet = nn.DataParallel(typenet)
        print('Devices:', typenet.device_ids)

    # push to cuda
    if args.cuda:
        typenet = typenet.cuda()

    # build optimizer
    print("training...")
    optimizer = optim.Adam(typenet.parameters(), lr=args.lr)

    # build the logger object
    logger = None
    if args.filelog != '':
        short_name = get_short_name()
        logger = FileLogger(
                        {
                            'train': ['epoch', 'loss', 'acc'],
                            'test':  ['epoch', 'loss', 'acc'],
                        },
                        [
                            __file__,
                            'allpairs/grid_generator.py',
                            'allpairs/symbol_drawing.py',
                        ],
                        path=args.filelog,
                        file_prefix=short_name,
                        args=args
                    )
        logger.set_info('note', args.note)
        logger.set_info('uuid', logger.uuid)
        logger.set_info('args', str(args))

    # main training loop
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train(epoch, typenet, optimizer, data_loader, logger)
        acc = test(epoch, typenet, data_loader, logger)
        best_acc = max(best_acc, acc)
        print('best acc: {}'.format(best_acc))

    # close all the generators, needed to deal with multi-process generators
    SampleSpec.close_generators()


if __name__ == "__main__":
    run()
