import os
import pathlib
import random
import pickle

import numpy as np
from tensorflow.keras.datasets import cifar10

from PIL import Image

from utils import pil_to_ndarray


class ActiveLearning:
    def __init__(self,
                 path_train,
                 path_test,
                 query_strategy,
                 model_initialization_fn,
                 preprocess_input_fn,
                 target_size,
                 class_sample_size_train,
                 class_sample_size_test,
                 init_size,
                 val_size,
                 dataset=None,
                 save_models=False,
                 dataset_seed=None):
        self.query_strategy = query_strategy

        self.model = None
        self.model_weights_checkpoint = None
        self.model_initialization_fn = model_initialization_fn

        self.preprocess_input_fn = preprocess_input_fn
        self.target_size = target_size

        self.init_size = init_size
        self.val_size = val_size

        if dataset:
            self.X_train_init, self.y_train_init, self.X_val, self.y_val, self.X_pool, self.y_pool, self.X_test, self.y_test = (
                self.initialize_dataset(dataset, seed=dataset_seed)
            )
        else:
            self.X_train_init, self.y_train_init, self.X_val, self.y_val, self.X_pool, self.y_pool, self.X_test, self.y_test = (
                self.initialize_dataset_from_directory(path_train, path_test, class_sample_size_train, class_sample_size_test, seed=dataset_seed)
            )

        self.idx_queried_last = None
        self.idx_queried = np.array([], dtype=int)

        self.save_models = save_models

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
                arr = pil_to_ndarray(img)
                arrs.append(arr)

            X_c = np.stack(arrs)
            y_c = np.full(len(arrs), i)

            X.append(X_c)
            y.append(y_c)

        X, y = np.concatenate(X), np.concatenate(y)
        y = self._one_hot_encode(y)

        return X, y, classes

    def initialize_dataset(self, dataset_name, seed=None):
        if dataset_name == "cifar-10":
            (X, y), (X_test, y_test) = cifar10.load_data()
            y = self._one_hot_encode(y)
            y_test = self._one_hot_encode(y_test)

            self._joint_shuffle(X, y, seed=seed)

            init_bound = int(len(X) * self.init_size)
            X_train, y_train = X[:init_bound], y[:init_bound]
            X_pool, y_pool = X[init_bound:], y[init_bound:]

            val_bound = int(len(X_train) * self.val_size)
            X_val, y_val = X_train[:val_bound], y_train[:val_bound]
            X_train_init, y_train_init = X_train[val_bound:], y_train[val_bound:]
        elif dataset_name == "imagenet-25":
            with open("data/imagenet_25/imagenet_25_train.pkl", "rb") as f:
                (X, y) = pickle.load(f)
            y = self._one_hot_encode(y)

            with open("data/imagenet_25/imagenet_25_val.pkl", "rb") as f:
                (X_val, y_val) = pickle.load(f)
            y_val = self._one_hot_encode(y_val)

            with open("data/imagenet_25/imagenet_25_test.pkl", "rb") as f:
                (X_test, y_test) = pickle.load(f)
            y_test = self._one_hot_encode(y_test)

            self._joint_shuffle(X, y, seed=seed)

            if self.init_size < 1:
                init_bound = int(len(X) * self.init_size)
            else:
                init_bound = self.init_size
            X_train_init, y_train_init = X[:init_bound], y[:init_bound]
            X_pool, y_pool = X[init_bound:], y[init_bound:]
        else:
            raise ValueError(f"No dataset with name {dataset_name} found")

        print(f"Length of initial training set: {len(X_train_init)}")
        print(f"Length of validation set: {len(X_val)}")
        print(f"Length of pool set: {len(X_pool)}")
        print(f"Length of test dataset: {len(X_test)}")

        return X_train_init, y_train_init, X_val, y_val, X_pool, y_pool, X_test, y_test

    def initialize_dataset_from_directory(self,
                                          path_train,
                                          path_test,
                                          class_sample_size_train,
                                          class_sample_size_test,
                                          seed=None):
        """Initialize initial training set, validation set, unlabeled pool and
        test set.
        """

        # Training set, validation set and unlabeled pool
        X, y, cls_train = self.load_data_from_directory(path_train, class_sample_size_train, seed=seed)

        self._joint_shuffle(X, y, seed=seed)

        init_bound = int(len(X) * self.init_size)
        X_train, y_train = X[:init_bound], y[:init_bound]
        X_pool, y_pool = X[init_bound:], y[init_bound:]

        val_bound = int(len(X_train) * self.val_size)
        X_val, y_val = X_train[:val_bound], y_train[:val_bound]
        X_train_init, y_train_init = X_train[val_bound:], y_train[val_bound:]

        print(f"Length of initial training set: {len(X_train_init)}")
        print(f"Length of validation set: {len(X_val)}")
        print(f"Length of pool set: {len(X_pool)}")

        # Test set
        X_test, y_test, cls_test = self.load_data_from_directory(path_test, class_sample_size_test, seed=seed)

        if cls_train != cls_test:
            raise ValueError("Train and test classes differ")

        print(f"Length of test dataset: {len(X_test)}")

        return X_train_init, y_train_init, X_val, y_val, X_pool, y_pool, X_test, y_test

    def initialize_base_model(self, model_name):
        self.model, loss_fn, optimizer = self.model_initialization_fn()

        print("Optimizer configuration")
        print(optimizer.get_config())

        print("Loading base model weights")
        path_model = pathlib.Path("models", model_name, model_name)
        self.model.load_weights(path_model)

        print("Compiling model")
        self.model.compile(optimizer=optimizer,
                           loss=loss_fn,
                           metrics=["accuracy"])

    def get_train(self, preprocess=True, seed=None):
        """Get current training dataset along with the validation set."""

        print("Getting current training set")
        print("Concatenating X_train_init and labeled pool")
        X_train = np.concatenate([self.X_train_init, self.X_pool[self.idx_queried]])
        y_train = np.concatenate([self.y_train_init, self.y_pool[self.idx_queried]])
        print(f"Shape of current train: {X_train.shape} labels {y_train.shape}")

        print("Shuffling training dataset")
        # This is required due to model.fit validation_split behaviour
        self._joint_shuffle(X_train, y_train, seed=seed)

        if preprocess:
            print("Preprocessing X_train")
            X_train = self._prepare_dataset(X_train)

        return X_train, y_train

    def get_val(self, preprocess=True):
        X_val = np.copy(self.X_val)
        y_val = np.copy(self.y_val)

        if preprocess:
            print("Preprocessing val dataset")
            X_val = self._prepare_dataset(X_val)

        return X_val, y_val

    def get_pool(self, preprocess=True, get_metadata=False):
        """Get current unlabeled pool"""

        print("Getting current pool")
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
        X_test = np.copy(self.X_test)
        y_test = np.copy(self.y_test)

        if preprocess:
            print("Preprocessing test dataset")
            X_test = self._prepare_dataset(X_test)

        return X_test, y_test

    def _prepare_dataset(self, X, preprocess=True):
        if preprocess:
            X = self.preprocess_input_fn(X)  # This overwrites X

        return X

    def add_to_train(self, idx):
        """Move a batch of labeled data from the unlabeled pool to the training
        dataset
        """

        self.idx_queried = np.concatenate([self.idx_queried, idx])
        self.idx_queried_last = idx
        print(f"Total amount of queried samples post-query: {len(self.idx_queried)}")

    def query(self, X_pool, metadata, n_query_instances, seed=None, **query_kwargs):
        """Call to query strategy function"""

        self.query_strategy.set_model(self.model, self.preprocess_input_fn)
        return self.query_strategy(X_pool, metadata, n_query_instances, seed=seed, **query_kwargs)

    def learn(self,
              n_loops,
              n_query_instances,
              batch_size,
              n_epochs,
              callbacks,
              base_model_name,
              dir_logs=None,
              seed=None,
              **query_kwargs):
        """Runs active learning loops.

        Args:
            n_loops: Number of active learning loops
            n_query_instances: Number of instances to query at each iteration
            batch_size: Batch size for model fit and evaluation
            n_epochs: Number of epochs for model fit
            callbacks:
            base_model_name:
            dir_logs:
            seed: Reproducibility for query strategy
            **query_kwargs: Query strategy kwargs
        """

        logs = {"train": [], "test": []}
        pathlib.Path("logs", dir_logs).mkdir(parents=True, exist_ok=False)

        print("Total dataset size:", len(self.X_train_init) + len(self.X_pool))

        X_test, y_test = self.get_test()
        X_val, y_val = self.get_val()
        print("Composition of validation set:")
        print(np.unique(self._one_hot_decode(y_val), return_counts=True))

        print("Initializing base model")
        self.initialize_base_model(base_model_name)

        print("Evaluating base model")
        test_metrics = self.model.evaluate(X_test, y_test, batch_size=batch_size)
        logs["base_test"] = test_metrics

        # Active learning loop
        for i in range(n_loops):
            print(f"* Iteration #{i+1}")

            if i > 0:
                self.model, loss_fn, optimizer = self.model_initialization_fn()

                print("Optimizer configuration")
                print(optimizer.get_config())

                self.model.compile(optimizer=optimizer,
                                   loss=loss_fn,
                                   metrics=["accuracy"])

                print("Setting weights to last iteration weights")
                self.model.set_weights(self.model_weights_checkpoint)

            X_pool, idx_pool, metadata = self.get_pool(preprocess=False, get_metadata=True)
            print("Querying")
            idx_query = self.query(X_pool, metadata, n_query_instances, seed=seed, **query_kwargs)
            print(f"Queried {len(idx_query)} samples.")
            self.add_to_train(idx_pool[idx_query])
            del X_pool, idx_query

            X_train, y_train = self.get_train(preprocess=True, seed=seed)
            print("Composition of current training set:")
            print(np.unique(self._one_hot_decode(y_train), return_counts=True))
            print("Fitting model")
            history = self.model.fit(X_train, y_train,
                                     validation_data=(X_val, y_val),
                                     batch_size=batch_size,
                                     epochs=n_epochs,
                                     shuffle=True,
                                     callbacks=callbacks)
            logs["train"].append(history.history)
            del X_train, y_train

            if self.save_models:
                print("Saving model")
                self.model.save(pathlib.Path("logs", dir_logs, f"model_iter_{i+1}"))

            print("Evaluating model")
            test_metrics = self.model.evaluate(X_test, y_test, batch_size=batch_size)
            logs["test"].append(test_metrics)

            self.model_weights_checkpoint = self.model.get_weights()

        return logs

    def train_base(self,
                   model_name,
                   batch_size,
                   n_epochs,
                   callbacks,
                   seed=None):
        """Train and save base model"""

        logs = {"train": [], "test": None}

        model, loss_fn, optimizer = self.model_initialization_fn(base=True)

        print("Optimizer configuration")
        print(optimizer.get_config())

        print("Compiling model")
        model.compile(optimizer=optimizer,
                      loss=loss_fn,
                      metrics=["accuracy"])

        X_train, y_train = self.get_train(preprocess=True, seed=seed)
        X_val, y_val = self.get_val()
        print("Composition of initial training set:")
        print(np.unique(self._one_hot_decode(y_train), return_counts=True))

        print("Fitting base model")
        history = model.fit(X_train, y_train,
                            validation_data=(X_val, y_val),
                            batch_size=batch_size,
                            epochs=n_epochs,
                            shuffle=True,
                            callbacks=callbacks)
        logs["train"].append(history.history)
        del X_train, y_train

        X_test, y_test = self.get_test()
        print("Evaluating model")
        test_metrics = model.evaluate(X_test, y_test, batch_size=batch_size)
        del X_test, y_test

        logs["test"] = test_metrics

        print("Saving base model weights")
        path_model = pathlib.Path("models", model_name)
        path_model.mkdir(parents=True, exist_ok=False)
        model.save_weights(pathlib.Path(path_model, model_name))

        return logs

    @staticmethod
    def _joint_shuffle(a, b, seed=0):
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
    def _one_hot_encode(y):
        y = y.squeeze()

        if y.ndim != 1:
            raise ValueError(f"Unsupported shape: {y.shape}")

        y_one_hot = np.zeros((y.size, y.max() + 1))
        y_one_hot[np.arange(y.size), y] = 1

        return y_one_hot

    @staticmethod
    def _one_hot_decode(y_one_hot):
        if y_one_hot.ndim != 2:
            raise ValueError(f"Unsupported shape: {y_one_hot.shape}")

        y = np.argmax(y_one_hot, axis=1)

        return y
