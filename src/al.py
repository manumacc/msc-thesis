import os
import random

import numpy as np

from PIL import Image


class ActiveLearning:
    def __init__(self, path, classes, target_size=None, sample=None, init_size=None, seed=None):
        self.X, self.y = self.load_dataset(path, classes, target_size=target_size, sample=sample, seed=seed)
        self.idx_train = self.create_initial(init_size=init_size, seed=seed)

    def load_dataset(self, path, classes, target_size=None, sample=None, seed=None):
        """
        Load an image dataset into a numpy array X along with target classes in y

        Assume images to be divided by class into different folders

        path/
            class1/
                image1.jpg
                image2.jpg
                ...
            class2/
            ...
            classN/

        sample: how many elements to sample from each class
        seed: if set, seeds the sample

        X: (n samples, weight, height, channels)
        y: (n samples, )
        [to implement] class_id (dict): {0: classname0, 1: classname1, ...}
        """

        X = []
        y = []

        for i, cls in enumerate(classes):
            cpath = os.path.join(path, cls)
            (_, _, filenames) = next(os.walk(cpath))

            if sample:
                random.seed(seed)
                filenames = random.sample(filenames, k=sample)

            arrs = []

            for f in filenames:
                fpath = os.path.join(cpath, f)

                img = self._load_img(fpath, target_size=target_size)
                arr = self._pil_to_ndarray(img)
                arrs.append(arr)

            X_c = np.stack(arrs)
            y_c = np.full(len(arrs), i)

            X.append(X_c)
            y.append(y_c)

        X = np.concatenate(X)
        y = np.concatenate(y)

        return X, y

    def create_initial(self, init_size=None, seed=None):
        """Create initial training dataset and unlabeled pool"""

        if init_size is None:
            init_size = 0.1

        rng = np.random.default_rng(seed)

        idx = np.arange(0, self.X.shape[0])
        idx_init = rng.choice(idx, size=int(len(idx) * init_size), replace=False)

        return idx_init

    def get_training(self):
        """Get current training dataset"""

        return self.idx_train

    def get_pool(self, batch_size=None):
        """Get current unlabeled pool. Accepts batching"""

        idx = np.arange(0, self.X.shape[0])
        idx_pool = np.delete(idx, self.idx_train)

        return idx_pool

    def add_to_training(self, idx):
        """Move a batch of labeled data from the unlabeled pool to the training
        dataset
        """

        self.idx_train = np.concatenate(self.idx_train, idx)

    @staticmethod
    def _load_img(path, grayscale=False, target_size=None):
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

    @staticmethod
    def _pil_to_ndarray(image):
        """Convert a PIL Image instance to a numpy ndarray"""

        x = np.asarray(image, dtype='float32')

        if len(x.shape) == 2:
            x = x.reshape((x.shape[0], x.shape[1], 1))
        elif len(x.shape) != 3:
            raise ValueError(f"Unsupported image shape: {(x.shape,)}")

        return x

    @staticmethod
    def _ndarray_to_pil(x):
        """Convert a numpy ndarray to a PIL Image instance"""

        x = np.asarray(x, dtype='uint8')

        if x.shape[2] == 1:
            return Image.fromarray(x[:, :, 0], 'L')
        elif x.shape[2] == 3:
            return Image.fromarray(x, 'RGB')
        else:
            raise ValueError(f"Unsupported image shape: {(x.shape,)}")