import os
import random

import numpy as np

from PIL import Image


class ActiveLearning:
    def __init__(self,
                 path_train,
                 path_test,
                 query_strategy,
                 model,
                 preprocess_input_fn,
                 target_size,
                 class_sample_size_train,
                 class_sample_size_test,
                 init_size,
                 val_size,
                 seed=None,
                 model_callbacks=None):
        self.query_strategy = query_strategy

        self.model = model
        self.model_callbacks = model_callbacks
        self.preprocess_input_fn = preprocess_input_fn
        self.target_size = target_size

        self.init_size = init_size
        self.val_size = val_size

        self.X, self.y, classes_train = self.load_dataset(path_train, class_sample_size_train, seed=seed)
        self.X_test, self.y_test, classes_test = self.load_dataset(path_test, class_sample_size_test, seed=seed)

        if classes_train != classes_test:
            raise ValueError("Train and test classes differ")

        self.classes = classes_train

        self.idx_train = self.initialize_dataset(init_size, seed=seed)

        self.logs = {"train": [], "test": []}

    def load_dataset(self, path, sample_size, seed=None):
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

        sample_size: how many elements to sample from each class
        seed: if set, seeds the sample

        X: (n samples, width, height, channels) -- non-preprocessed PIL images in np format
        y: (n samples, )
        [to implement] class_id (dict): {0: classname0, 1: classname1, ...}
        """

        X, y = [], []

        (_, dirs, _) = next(os.walk(path))
        classes = sorted(dirs)  # ensure order

        for i, cls in enumerate(classes):
            print(f"Loading class {i}: {cls}")

            cpath = os.path.join(path, cls)
            (_, _, filenames) = next(os.walk(cpath))
            filenames = sorted(filenames)

            if sample_size:
                random.seed(seed)
                filenames = random.sample(filenames, k=sample_size)

            arrs = []

            for f in filenames:
                fpath = os.path.join(cpath, f)

                img = self._load_img(fpath, target_size=self.target_size)
                arr = self._pil_to_ndarray(img)
                arrs.append(arr)

            X_c = np.stack(arrs)
            y_c = np.full(len(arrs), i)

            X.append(X_c)
            y.append(y_c)

        X, y = np.concatenate(X), np.concatenate(y)
        y = self._one_hot_encode(y)

        return X, y, classes

    def initialize_dataset(self, init_size, seed=None):
        """Create initial training set and unlabeled pool.

        The initial training set is used to train a baseline model.

        init_size: initial training set size
        val_size: validation set size
        """

        rng = np.random.default_rng(seed)

        idx = np.arange(0, self.X.shape[0])
        idx_init = rng.choice(idx, size=int(len(idx) * init_size), replace=False)

        return idx_init

    def get_train(self, preprocess=True):
        """Get current training dataset"""

        X_train = self.X[self.idx_train]
        if preprocess:
            X_train = self.preprocess_input_fn(X_train)

        return X_train, self.y[self.idx_train]

    def get_test(self, preprocess=True):
        X_test = self.X_test
        if preprocess:
            X_test = self.preprocess_input_fn(np.copy(X_test))

        return X_test, self.y_test

    def get_pool(self, preprocess=True, get_indices=False):
        """Get current unlabeled pool"""

        idx = np.arange(0, self.X.shape[0])
        idx_pool = np.delete(idx, self.idx_train)

        X_pool = self.X[idx_pool]
        if preprocess:
            X_pool = self.preprocess_input_fn(X_pool)

        if get_indices:
            return X_pool, idx_pool
        else:
            return X_pool

    def add_to_training(self, idx):
        """Move a batch of labeled data from the unlabeled pool to the training
        dataset
        """

        self.idx_train = np.concatenate([self.idx_train, idx])

    def query(self, X_pool, n_query_instances, seed=None, **query_kwargs):
        """
        X_pool: pool data
        n_query_instances: how many instances to query from the pool
        """

        return self.query_strategy(X_pool, n_query_instances, seed=seed, **query_kwargs)

    def model_fit(self, X_train, y_train, batch_size, n_epochs):
        history = self.model.fit(X_train, y_train,
                                 validation_split=self.val_size,
                                 batch_size=batch_size,
                                 epochs=n_epochs,
                                 callbacks=self.model_callbacks if self.model_callbacks else [],
                                 shuffle=True)
        return history

    def model_evaluate(self, X_test, y_test, batch_size):
        test_metrics = self.model.evaluate(X_test, y_test,
                                           batch_size=batch_size)
        return test_metrics

    def learn(self,
              n_loops,
              n_query_instances,
              batch_size,
              n_epochs,
              seed=None,
              require_raw_pool=False,
              **query_kwargs):
        """Runs active learning loops.

        Args:
            n_loops: Number of active learning loops
            n_query_instances: Number of instances to query at each iteration
            batch_size: Batch size for model fit and evaluation
            n_epochs: Number of epochs for model fit
            seed: Reproducibility for query strategy
            require_raw_pool: If True, also extract unprocessed unlabeled pool
                and pass it to query strategy
            **query_kwargs: Query strategy kwargs
        """

        print("Total dataset size:", len(self.X))

        X_train, y_train = self.get_train()
        history = self.model_fit(X_train, y_train, batch_size, n_epochs)
        self.logs["train"].append(history)
        del X_train, y_train

        X_test, y_test = self.get_test()  # Reusable at every iteration
        test_metrics = self.model_evaluate(X_test, y_test, batch_size)
        self.logs["test"].append(test_metrics)

        # Active learning loop
        for i in range(n_loops):
            print(f"Iteration #{i+1}")

            X_pool, idx_pool = self.get_pool(preprocess=False if require_raw_pool else True,
                                             get_indices=True)
            idx_query = self.query(X_pool, n_query_instances, seed=seed, **query_kwargs)
            self.add_to_training(idx_pool[idx_query])

            X_train, y_train = self.get_train()
            history = self.model_fit(X_train, y_train, batch_size, n_epochs)
            self.logs["train"].append(history)
            del X_train, y_train

            test_metrics = self.model_evaluate(X_test, y_test, batch_size)
            self.logs["test"].append(test_metrics)

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
            raise ValueError(f"Unsupported image shape: {x.shape}")

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
            raise ValueError(f"Unsupported image shape: {x.shape}")

    @staticmethod
    def _one_hot_encode(y):
        if y.ndim != 1:
            raise ValueError(f"Unsupported shape: {y.shape}")

        y_one_hot = np.zeros((y.size, y.max()+1))
        y_one_hot[np.arange(y.size), y] = 1

        return y_one_hot
