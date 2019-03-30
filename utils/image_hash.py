#
# For licensing see accompanying LICENSE.txt file.
# Copyright (C) 2018-2019 Apple Inc. All Rights Reserved.
#
import numpy as np
from hashlib import sha256

def hash(val):
    val_type = val.__class__.__name__
    if type(val).__name__ == 'memoryview':
        return sha256(val.tobytes())
    else:
        return sha256(''.join(tuple(val)))


def image_hash(im):
    im = np.asarray(im)
    h1 = hash(im.data)
    im = im > 0
    h2 = hash(im.data)
    return h1.hexdigest(), h2.hexdigest()
