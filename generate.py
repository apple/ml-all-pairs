from __future__ import print_function
import sys
import argparse
from allpairs.grid_generator import SampleSpec
import scipy.misc
import numpy as np
from allpairs.grid_generator import SampleSpec
import os
import time


parser = argparse.ArgumentParser(description='Generate All-Pairs')
parser.add_argument('--num-pairs', type=int, default=4, help='how many pairs')
parser.add_argument('--num-classes', type=int, default=4, help='num of different symbols')
parser.add_argument('--pixels', type=int, default=72, help='size in pixels of image side (square)')
parser.add_argument('--num', type=int, default=10, help='number of images to generate (png format)')
parser.add_argument('--dest', type=str, help='directory to save results into', required=True)
parser.add_argument('--csv', type=str, help='path to csv file to save groundtruth', required=True)
args = parser.parse_args()


def progress(i, max_plus_one):
    bar_len = 100
    fraction = float(i+1) / float(max_plus_one)
    fill_len = int(bar_len * fraction)
    bar = '#' * fill_len + ' ' * (bar_len - fill_len)
    print('\r |{}| {}%%'.format(bar, 100.0*fraction), end = '\r')
    if i + 1 == max_plus_one:
        print()


def run():
    f = open(args.csv, 'w')
    spec = SampleSpec(args.num_pairs, args.num_classes, im_dim=args.pixels, min_cell=15, max_cell=18)
    ground_truth = []
    for i in range(args.num):
        progress(i, args.num)
        images, labels, stats = spec.blocking_generate_with_stats(1)
        outpath = os.path.join(args.dest, '{}.png'.format(str(i).zfill(6)))
	scipy.misc.imsave(outpath, images[0, 0, :, :])
        f.write(str(labels[0]))
        f.write('\n')


if __name__ == "__main__":
    run()
