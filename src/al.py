import os
import random

import numpy as np
import tensorflow as tf

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
        self.model_initial_weights = model.get_weights()
        self.model_callbacks = model_callbacks
        self.preprocess_input_fn = preprocess_input_fn
        self.target_size = target_size

        self.init_size = init_size
        self.val_size = val_size

        self.X_train_init, self.y_train_init, self.X_pool, self.y_pool, self.X_test, self.y_test, self.classes = (
            self.initialize_dataset(path_train, path_test, class_sample_size_train, class_sample_size_test, seed=seed)
        )

        self.idx_queried = np.array([], dtype=int)

        self.logs = {"train": [], "test": []}

    def load_data_from_directory(self, path, sample_size, seed=None):
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

    def initialize_dataset(self,
                           path_train,
                           path_test,
                           class_sample_size_train,
                           class_sample_size_test,
                           seed=None):
        """Initialize initial training set, unlabeled pool and test set."""

        # Training set and unlabeled pool
        X, y, cls_train = self.load_data_from_directory(path_train, class_sample_size_train, seed=seed)

        self._joint_shuffle(X, y, seed=seed)

        init_bound = int(len(X) * self.init_size)
        X_train_init, y_train_init = X[:init_bound], y[:init_bound]
        X_pool, y_pool = X[init_bound:], y[init_bound:]

        print(f"Length of initial training set: {len(X_train_init)}")
        print(f"Length of pool dataset: {len(X_pool)}")

        # Test set
        X_test, y_test, cls_test = self.load_data_from_directory(path_test, class_sample_size_test, seed=seed)

        if cls_train != cls_test:
            raise ValueError("Train and test classes differ")

        print(f"Length of test dataset: {len(X_test)}")

        return X_train_init, y_train_init, X_pool, y_pool, X_test, y_test, cls_train

    def get_train(self, preprocess=True):
        """Get current training dataset along with the validation set."""

        print("Getting current training")
        print("Concatenating X_train_init and labeled pool (may cause OOM)")
        X_train = np.concatenate([self.X_train_init, self.X_pool[self.idx_queried]])
        y_train = np.concatenate([self.y_train_init, self.y_pool[self.idx_queried]])
        print(f"Shape of current train: {X_train.shape} labels {y_train.shape}")

        print("Shuffling training dataset")
        # This is required due to model.fit validation_split behaviour
        self._joint_shuffle(X_train, y_train)

        if preprocess:
            print("Preprocessing X_train")
            X_train = self._prepare_dataset(X_train)

        return X_train, y_train

    def get_pool(self, preprocess=True, get_metadata=False):
        """Get current unlabeled pool"""

        print("Getting current pool (may cause OOM)")
        X_pool = np.delete(self.X_pool, self.idx_queried, axis=0)
        idx_pool = np.delete(np.arange(0, len(self.X_pool)), self.idx_queried)
        print(f"Size of current pool: {X_pool.shape}")

        if preprocess:
            print("Preprocessing X_pool")
            X_pool = self._prepare_dataset(X_pool)

        if get_metadata:
            metadata = {
                "len": len(X_pool),
                "is_raw": False if preprocess else True,
            }

            return X_pool, idx_pool, metadata
        else:
            return X_pool, idx_pool

    def get_test(self, preprocess=True):
        if preprocess:
            print("Preprocessing test dataset")
            self.X_test = self._prepare_dataset(self.X_test)

        return self.X_test, self.y_test

    def _prepare_dataset(self, X, preprocess=True):
        if preprocess:
            X = self.preprocess_input_fn(X)  # Note: this overwrites X

        return X

    def add_to_train(self, idx):
        """Move a batch of labeled data from the unlabeled pool to the training
        dataset
        """

        self.idx_queried = np.concatenate([self.idx_queried, idx])
        print(f"Total amount of queried samples post-query: {len(self.idx_queried)}")

    def query(self, X_pool, metadata, n_query_instances, seed=None, **query_kwargs):
        """

        Args:
            X_pool:
            metadata:
            n_query_instances: number of instances to select from the pool
            seed:
            **query_kwargs:

        Returns:

        """

        return self.query_strategy(X_pool, metadata, n_query_instances, seed=seed, **query_kwargs)

    def reset_model_weights(self):
        self.model.set_weights(self.model_initial_weights)

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

        print("Total dataset size:", len(self.X_train_init) + len(self.X_pool))

        X_test, y_test = self.get_test()

        # Active learning loop
        for i in range(n_loops+1):
            print(f"* Iteration #{i}")

            if i > 0:
                X_pool, idx_pool, metadata = self.get_pool(preprocess=False if require_raw_pool else True,
                                                           get_metadata=True)
                print("Querying")
                idx_query = self.query(X_pool, metadata, n_query_instances, seed=seed, **query_kwargs)
                print(f"Queried {len(idx_query)} samples.")
                self.add_to_train(idx_pool[idx_query])
                print("Deleting pool")
                del X_pool, idx_query

                print("Resetting model initial weights")
                self.reset_model_weights()

            X_train, y_train = self.get_train(preprocess=True)
            print("(debug) Composition of current training set:")
            print(np.unique(self._one_hot_decode(y_train), return_counts=True))
            print("Fitting model")
            history = self.model.fit(X_train, y_train,
                                     validation_split=self.val_size,
                                     batch_size=batch_size,
                                     epochs=n_epochs,
                                     shuffle=True,
                                     callbacks=self.model_callbacks if self.model_callbacks else [])
            self.logs["train"].append(history.history)
            print("Deleting training set")
            del X_train, y_train

            print("Evaluating model")
            test_metrics = self.model.evaluate(X_test, y_test,
                                               batch_size=batch_size)
            self.logs["test"].append(test_metrics)

    @staticmethod
    def _joint_shuffle(a, b, seed=42):
        """Jointly shuffles two ndarrays in-place."""
        rng = np.random.default_rng(seed)
        rng.shuffle(a)
        rng = np.random.default_rng(seed)
        rng.shuffle(b)

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

    @staticmethod
    def _one_hot_decode(y_one_hot):
        if y_one_hot.ndim != 2:
            raise ValueError(f"Unsupported shape: {y_one_hot.shape}")

        y = np.argmax(y_one_hot, axis=1)

        return y
