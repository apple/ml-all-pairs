import sys
sys.path.append('.')
from allpairs.grid_generator import SampleSpec
import scipy.misc
import numpy as np


def find_index(labels, target):
    for i in range(0, len(labels)):
        if labels[i] == target:
            return i
    return "target not found"


def run():
    start_num = 2
    end_num = 8
    dest = 'all_pairs_survey.png'

    next_x = 0
    for i, n in enumerate(range(start_num, end_num + 1)):
        im_dim = [76, 96][n > 5]
        next_x += im_dim + 1

    result_size = (2 * 96 + 1, next_x - 1)
    print(result_size)
    result = np.ones((2 * 96 + 1, next_x - 1))

    next_y = 0
    for target_label in [0, 1]:
        next_x = 0
        print('target_label', target_label)
        for i, n in enumerate(range(start_num, end_num + 1)):
            im_dim = [76, 96][n > 5]
            dy = [10, 0][n > 5]
            spec = SampleSpec(n, n, im_dim=im_dim, min_cell=15, max_cell=18)
            vis_input, vis_labels, stats = spec.blocking_generate_with_stats(200)
            sample_index = find_index(vis_labels, target_label)
            y = next_y + dy
            result[y:y + im_dim, next_x:next_x + im_dim] = vis_input[sample_index, 0, :, :]
            next_x += im_dim + 1
            print(stats)
            # vis_input = 1.0 - vis_input
        next_y += 96 + 1

    scipy.misc.imsave(dest, result)


if __name__ == '__main__':
    run()
