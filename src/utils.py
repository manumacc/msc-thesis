from time import perf_counter

import numpy as np
from PIL import Image


class Profiling(object):
    def __init__(self, phase):
        self.phase = phase

    def __enter__(self):
        self.start = perf_counter()
        print(f"- {self.phase}", flush=True)
        return self

    def __exit__(self, *args):
        t = perf_counter() - self.start
        print(f"  {t:.3f} seconds", flush=True)


def pil_to_ndarray(image):
    """Convert a PIL Image instance to a numpy ndarray"""

    x = np.asarray(image, dtype='float32')

    if len(x.shape) == 2:
        x = x.reshape((x.shape[0], x.shape[1], 1))
    elif len(x.shape) != 3:
        raise ValueError(f"Unsupported image shape: {x.shape}")

    return x


def ndarray_to_pil(x):
    """Convert a numpy ndarray to a PIL Image instance"""

    x = np.asarray(x, dtype='uint8')

    if x.shape[2] == 1:
        return Image.fromarray(x[:, :, 0], 'L')
    elif x.shape[2] == 3:
        return Image.fromarray(x, 'RGB')
    else:
        raise ValueError(f"Unsupported image shape: {x.shape}")


def load_image_as_pil(path, grayscale=False, target_size=None):
    """Load an image into a PIL Image instance."""

    image = Image.open(path)

    if grayscale:
        if image.mode != 'L':
            image = image.convert('L')
    else:
        if image.mode != 'RGB':
            image = image.convert('RGB')

    if target_size:
        width, height = target_size[0], target_size[1]
        if image.size != (width, height):
            image = image.resize((width, height), Image.NEAREST)

    return image
