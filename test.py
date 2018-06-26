import unittest
from allpairs.grid_generator import SampleSpec
from utils.image_hash import image_hash
import numpy as np
from hashlib import sha256


class TestAllPairs(unittest.TestCase):

    def test_images_same(self):
        correct = ((
                       ('e11b8020b9f8fabe', '18a3ae95b53eb2a3'),
                       ('b1cbdc248365556e', '3a45220b17d7bf49'),
                       ('7fbfff719fe71393', 'b436c33047a91227'),
                       ('07d1403088b6c624', '6b0a099525d4b709'),
                       ('deb6823a3285233a', 'c6c1a74630ba13ca'),
                       ('e6433a7d3bfb7c5e', 'e7884540c7d4364b'),
                       ('90228d0abfa1c73b', '7ce1c8786cc40475'),
                       ('c4be817429e61d25', '6dcd05d7168be05e'),
                       ('be5a2d302cae9f18', '89659712a3f2c7c7'),
                       ('0a42707c9f1c3c32', '35b3d810a33533c8'),
            ),(
                        ('d1a12878c42725a5', '1759c3b44205f0fb'),
                        ('041d9ff827beba3e', 'bf9aa5fbcf176355'),
                        ('0162327e25617782', '8ee558aac58bb032'),
                        ('de3571b7bd364931', 'f439cfcbd075b6bf'),
                        ('98ffaea06e268eda', 'ba5bad1a9b1676eb'),
                        ('3c042e60e1ca15b6', '14d7b3a418f5510a'),
                        ('367ca722d496bf33', '052a7ce6679df288'),
                        ('f7321cb15555ce55', '231060e55e4e987d'),
                        ('361d0c54a66fe7e2', '2dc6219f8ee96f1d'),
                        ('f6e04ac9005b6d58', '64ec9871f8d58228'),
            ))
        np.random.seed(1234)
        im_dim = 96
        num = 10
        for which, n in enumerate([4, 8]):
            spec = SampleSpec(n, n, im_dim=im_dim, min_cell=15, max_cell=18)
            vis_input, vis_labels, stats = spec.blocking_generate_with_stats(num)
            for i in range(num):
                im = vis_input[i, 0, :, :]
                (h0, h1) = image_hash(im)
                h0 = h0[0:16]
                h1 = h1[0:16]
                # print(h0, h1)
                self.assertEqual(h0, correct[which][i][0])
                self.assertEqual(h1, correct[which][i][1])
            # print()
        self.assertTrue(True)


# if __name__ == '__main__':
#     unittest.main()


suite = unittest.TestLoader().loadTestsFromTestCase(TestAllPairs)
unittest.TextTestRunner(verbosity=2).run(suite)
