from __future__ import print_function
import numpy as np
import time
import scipy.misc
from scipy.ndimage.morphology import grey_dilation
from multiprocessing import Process, Queue
from .symbol_drawing import draw_symbol
from random import shuffle as non_np_shuffle

def save_batch_as_png(batch, labels, dir_path_with_slash, add_label=True):
    """save a whole batch as separate png files with the label in the filename"""
    for i in range(len(batch)):
        dest = dir_path_with_slash + str(i)
        if add_label:
            dest += '-' + str(labels[i])

        print(batch.shape)
        scipy.misc.imsave(dest + ".png", batch[i, 0, :, :])


def make_image_parts(symbol_list, min_cell_size, max_cell_size):
    """get the sub-images for each symbol to place in the main image"""
    result = []
    for s in symbol_list:
        sym_size = np.random.randint(min_cell_size, high=max_cell_size + 1)
        im = np.zeros((sym_size, sym_size))
        draw_symbol(im, s)
        result.append(im)
    return result


def collision(part, part_size, here, result):  # assumes black background
    """was there a pixel collision"""
    sum_of_product = np.sum(np.multiply(part, result[here[0]:here[0] + part_size,
                                                     here[1]:here[1] + part_size]))
    return sum_of_product > 0.0


def find_location(result, part, image_dim, part_dim, margin, max_collisions):
    """find a collision free location"""
    size_with_margin = part_dim + 2*margin
    mask = np.zeros((size_with_margin, size_with_margin))
    mask[margin:part_dim + margin, margin:part_dim + margin] = part
    mask = grey_dilation(mask, size=2*margin + 1, mode='constant', cval=0.0)
    im_end = image_dim - size_with_margin + 1
    here = (np.random.randint(0, im_end), np.random.randint(0, im_end))
    collision_count = 0
    while collision(mask, size_with_margin, here, result):
        collision_count += 1
        assert max_collisions is None or collision_count < max_collisions, "too many collisions"
        here = (np.random.randint(0, im_end), np.random.randint(0, im_end))

    return (here[0] + margin, here[1] + margin), collision_count


def make_image_rand_placement(image_dim, symbol_list, min_cell, max_cell, max_collisions=None, initial_image=None):
    """use random placement trials to make an image"""
    parts = make_image_parts(symbol_list, min_cell, max_cell)
    if initial_image is None:
        result = np.zeros((image_dim, image_dim))
    else:
        result = initial_image
    collision_count = 0.0
    for part in parts:
        try:
            part_dim = part.shape[0]
            loc, num_collisions = find_location(result, part, image_dim, part_dim, 2, max_collisions)
            collision_count += num_collisions
            assert max_collisions is None or collision_count < max_collisions, "too many collisions"
        except AssertionError:
            return None

        result[loc[0]:loc[0] + part_dim, loc[1]:loc[1] + part_dim] += part
    result = np.maximum(np.minimum(result, 1.0), 0.0)
    return result


def to_images_rand_placement(image_dim, symbol_data, min_cell, max_cell, max_collisions=40):
    """keep trying to make an image"""
    retry_count = 0
    data = []
    for row in symbol_data:
        image = None
        while image is None:
            image = make_image_rand_placement(image_dim, row,
                                              min_cell, max_cell,
                                              max_collisions=max_collisions)
            if image is None:
                retry_count += 1

        data.append(image)

    data = np.array(data)
    return data, retry_count


class ProcessHolder:
    def __init__(self):
        self._processes = []

    def __del__(self):
        if len(self._processes) > 0:
            self.terminate_all()

    def terminate_all(self):
        print("removing generator processes...")
        while len(self._processes) > 0:
            p = self._processes.pop()
            p.terminate()

    def append(self, p):
        self._processes.append(p)

    def count(self):
        return len(self._processes)


all_gen_processes = ProcessHolder()


class GenProccesses:
    def __init__(self, sample_spec, num_processes=1):
        self.gen_qs = []
        self.gen_processes = []
        self.num_processes = num_processes
        self.spec = sample_spec
        self.time_seed = int(time.time())
        print("starting data generators: {}".format(num_processes))
        for i in range(num_processes):
            self.gen_qs.append(Queue())
            seed = self.time_seed + i
            p = Process(target=GenProccesses.generator_repeat_process,
                        args=(self.gen_qs[-1], self.spec, seed))
            p.start()
            self.gen_processes.append(p)
            all_gen_processes.append(p)
            print("created generator process {}".format(all_gen_processes.count()))

    @staticmethod
    def generator_repeat_process(q, spec, seed):
        np.random.seed(seed)
        if spec.reset_every is None:
            while True:
                if q.qsize() < 1000:
                    q.put(spec.blocking_generate(1))
        else:
            reseed_every = 1000
            assert (spec.reset_every % reseed_every) == 0, 'reset_every needs to be a multiple of 1000'
            num_seeds = spec.reset_every // reseed_every
            seeds = list(set(np.random.randint(1, high=100000000, size=4*num_seeds)))[0:num_seeds]

            # small chance that seeds is too small since duplicates might have been produced
            while len(seeds) < num_seeds:
                seeds.append(np.random.randint(1, high=100000000))

            used_seeds = []

            pre_generated = []
            while True:
                if len(pre_generated) == 0:
                    for i in range(reseed_every):
                        pre_generated.append(spec.blocking_generate(1))
                    non_np_shuffle(pre_generated)
                    used_seeds.append(seeds.pop())
                    np.random.seed(used_seeds[-1])

                if len(seeds) == 0:
                    seeds, used_seeds = used_seeds, seeds
                    # we don't use numpy here, so the shuffles are not dependent on the local seeds
                    non_np_shuffle(seeds)

                if q.qsize() < 1000:
                    q.put(pre_generated.pop())

    @staticmethod
    def generator_process(q, spec, seed):
        np.random.seed(seed)
        while True:
            if q.qsize() < 1000:
                q.put(spec.blocking_generate(1))

    def __call__(self, batch_size):
        if batch_size == 1:
            # randomize to spread the load (might not be needed, but simple to do)
            start = np.random.randint(0, self.num_processes)
            for i in range(self.num_processes):
                q = self.gen_qs[(start + i) % self.num_processes]
                if q.qsize() > 100:  # don't use near empty queues
                    return q.get()

            return self.spec.blocking_generate(batch_size)

        print("non optimized batch size (blocking)")
        return self.spec.blocking_generate(batch_size)  # TODO: this is blocking


class SampleSpec:
    def __init__(self, num_pairs, num_classes, im_dim, min_cell, max_cell, symbol_reordering=None, reset_every=None):
        self.num_pairs = num_pairs
        self.num_classes = num_classes
        self.im_dim = im_dim
        self.min_cell = min_cell
        self.max_cell = max_cell
        self.symbol_reordering = symbol_reordering
        self.retry_rate = 0.0
        self.generators = None
        self.reset_every = reset_every

    @staticmethod
    def close_generators():
        all_gen_processes.terminate_all()

    def make_batch(self, num_samples):
        data = []
        labels = np.random.randint(0, 2, (num_samples, 1))
        for which in labels:
            which = which[0]
            row = np.random.randint(0, self.num_classes, self.num_pairs)
            row = np.append(row, row)
            if which == 0:
                row[0] = (row[0] + np.random.randint(1, self.num_classes)) % self.num_classes

            np.random.shuffle(row)
            data.append(row)
        return data, labels

    def blocking_generate_with_stats(self, batch_size):
        vis_data, vis_labels = self.make_batch(batch_size)
        vis_input, num_retries = to_images_rand_placement(self.im_dim, vis_data,
                                                          self.min_cell, self.max_cell)
        vis_input = np.expand_dims(vis_input, 1).astype(np.float32)
        vis_labels = vis_labels.flatten().astype(np.int64)
        return vis_input, vis_labels, {'num_retries': num_retries,
                                       'num_generated': batch_size}

    def blocking_generate(self, batch_size):
        images, labels, stats = self.blocking_generate_with_stats(batch_size)
        return images, labels

    def generate(self, batch_size=1):
        # return self.blocking_generate(batch_size) # this line will short circuit the processes
        num_processes = 8
        if self.reset_every is not None:
            num_processes = 1

        if self.generators is None:
            print("creating generator processes")
            self.generators = GenProccesses(self, num_processes=num_processes)

        return self.generators(batch_size)

def run():
    import argparse
    parser = argparse.ArgumentParser(description='All Pairs Generator')
    parser.add_argument('--num-examples', type=int, default=10,
                        help='number of examples to generate (default: 10)')
    parser.add_argument('--num-pairs', type=int, default=4,
                        help='number of pairs to generate (default: 4)')
    parser.add_argument('--num-classes', type=int, default=4,
                        help='number of classes to generate (default: 4)')
    args = parser.parse_args()

    spec = SampleSpec(args.num_pairs, args.num_classes,
                      im_dim=76, min_cell=15, max_cell=18)
    vis_input, vis_labels, stats \
        = spec.blocking_generate_with_stats(args.num_examples)
    print(stats)
    save_batch_as_png(vis_input, vis_labels, "", add_label=True)


if __name__ == '__main__':
    run()
