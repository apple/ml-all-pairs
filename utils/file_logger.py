#
# For licensing see accompanying LICENSE.txt file.
# Copyright (C) 2018-2019 Apple Inc. All Rights Reserved.
#
from shutil import copyfile
import copy
import os
import yaml
import random
from time import time


# a random semi-pronoucable uuid
def rand_id(num_syllables=2, num_parts=3):
    part1 = ['s', 't', 'r', 'ch', 'b', 'c', 'w', 'z', 'h', 'k', 'p', 'ph', 'sh', 'f', 'fr']
    part2 = ['a', 'oo', 'ee', 'e', 'u', 'er', ]
    seps = ['_', ]  # [ '-', '_', '.', ]
    result = ""
    for i in range(num_parts):
        if i > 0:
            result += seps[random.randrange(len(seps))]
        indices1 = [random.randrange(len(part1)) for _ in range(num_syllables)]
        indices2 = [random.randrange(len(part2)) for _ in range(num_syllables)]
        for i1, i2 in zip(indices1, indices2):
            result += part1[i1] + part2[i2]
    return result


# log to a directory
# source_list: a list of files to save to the results directory
# log_def: definition of the logging { 'log_name1': [ keys ], 'log_name2': [ keys ], ...}
class FileLogger:
    def __init__(self, log_def, description='', source_list=(), path="results/", include_wall_time=True, file_prefix='', args=None):
        self.uuid = rand_id()
        self.dest_path = path
        self.description = description
        if not os.path.isdir(path):
            os.makedirs(path)

        self.fp = {}
        self.num_rows = 0
        self.log_def = copy.deepcopy(log_def)
        self.values = {}
        self.info = {}
        self.start_time = None
        self.file_prefix = file_prefix
        if include_wall_time:
            self.start_time = time()

        for log_name, items in log_def.items():
            csv_path = os.path.join(self.dest_path, file_prefix + self.uuid + "-" + log_name + '.csv')
            print('FILE LOG = ' + csv_path)
            self.fp[log_name] = open(csv_path, 'w')
            self.values[log_name] = {}
            for item in items:
                self.values[log_name][item] = 0.0
        self.source_list = source_list
        self.sourcesCopied = False

    def copy_sources(self):
        if self.sourcesCopied:
            return
        self.sourcesCopied = True
        for path in self.source_list:
            print("copying {}".format(path))
            _, filename = os.path.split(path)
            copyfile(path, os.path.join(self.dest_path, self.uuid + "-" + filename))

    def set_info(self, key, value):  # rewrites yaml file each call
        self.info[key] = value
        with open(os.path.join(self.dest_path, self.uuid + '-info.yml'), 'w') as outfile:
            yaml.dump(self.info, outfile, default_flow_style=False)

    def record(self, log_name, key, value):
        self.values[log_name][key] = value

    def new_row(self, log_name):
        row = str(self.num_rows)
        self.num_rows += 1
        if self.start_time is not None:
            row += ',' + str(time() - self.start_time)
        for key in self.log_def[log_name]:
            row += ',' + str(self.values[log_name][key])
        self.fp[log_name].write(row + "\n")
        self.fp[log_name].flush()

    def test_log(self, epoch, test_loss, test_acc):
        pass
    
    def train_log(self, epoch, loss, acc):
        pass
