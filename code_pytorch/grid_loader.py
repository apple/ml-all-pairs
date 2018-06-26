import torch
from torch.utils.data import Dataset
import numpy as np
from allpairs.grid_generator import SampleSpec

DEFAULT_PAIRS = 4
DEFAULT_CLASSES = 4
sample_spec = SampleSpec(num_pairs=DEFAULT_PAIRS, num_classes=DEFAULT_CLASSES, im_dim=76, min_cell=15, max_cell=18)


def set_sample_spec(num_pairs, num_classes, reset_every=None, im_dim=76):
    global sample_spec
    assert sample_spec.generators is None, 'attempting to redefine spec after it has been used'
    sample_spec = SampleSpec(num_pairs=num_pairs, num_classes=num_classes, im_dim=im_dim,
                             min_cell=15, max_cell=18, reset_every=reset_every)


class ToTensor(object):
    """simple override to add context to ToTensor"""
    def __init__(self, numpy_base_type=np.float32):
        self.numpy_base_type = numpy_base_type

    def __call__(self, index, img, context):
        result = img.astype(self.numpy_base_type) - 0.5  # TODO: this is not the right place to subtract 0.5
        result = np.expand_dims(result, axis=0)
        result = torch.from_numpy(result)
        return result


class GridGenerator(Dataset):
    def __init__(self, batch_size, batches_per_epoch,
                 transform=None,
                 target_transform=None):
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.current_batch = 0
        self.transform = transform
        self.target_transform = target_transform

        # XXX: lots of other code does len(dataset) to get size,
        #      so accommodate that with a simple 0 list
        self.dataset = [0] * batches_per_epoch * batch_size

    def __len__(self):
        return self.batches_per_epoch * self.batch_size

    def __iter__(self):
        return self

    def __next__(self):
        # Python 3 compatibility
        return self.next()

    def __getitem__(self, index):
        return self.next()

    def next(self):
        self.current_batch += 1
        img, target = sample_spec.generate(1)
        img = np.squeeze(img)

        # apply our transform
        if self.transform is not None:
            for transform in self.transform:
                if transform is not None:
                    img = transform(0, img, None)

        return img, target


class GridDataLoader(object):
    def __init__(self, batch_size, batches_per_epoch, test_frac=0.2, use_cuda=False):
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.output_size = 2

        # build the torch train dataloader
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        train_dataset = GridGenerator(batch_size, batches_per_epoch, transform=[ToTensor()])
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
            **kwargs)

        # tabulate samples counts
        total_samples = batches_per_epoch * batch_size
        num_test_batches = int(test_frac * total_samples) // batch_size
        num_test_samples = num_test_batches * batch_size

        # generate all test samples independently and store away
        print("starting full test set [%d samples] generation [this might take a while]..." % num_test_samples)
        old_state = np.random.get_state()
        np.random.seed(123456)  # always make the same test set
        test_imgs, test_labels, stats = sample_spec.blocking_generate_with_stats(num_test_samples)
        print("generator retry rate = {}".format(stats['num_retries']/float(stats['num_generated'])))
        np.random.set_state(old_state)
        test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_imgs - 0.5), torch.from_numpy(test_labels))
        print("test samples successfully generated...")

        # generate test dataloader using above samples
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            drop_last=True,
            shuffle=False,  # don't shuffle here to keep consistency
            **kwargs)

        # grab a test sample to set the size in the loader
        test_img, _ = train_dataset.__next__()
        self.img_shp = list(test_img.size())
        print("derived image shape = ", self.img_shp)
