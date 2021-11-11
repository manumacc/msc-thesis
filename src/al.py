import pathlib

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend

from dataset.labels import get_labels_by_name
from dataset.record import load_dataset_from_tfrecord


NUM_PARALLEL_READS = 4
CYCLE_LENGTH = 4
BLOCK_LENGTH = 16


def debug(message):
    print(f"[DEBUG]: {message}")


class ActiveLearning:
    def __init__(self,
                 dataset_name,
                 dataset_path,
                 query_strategy,
                 model_initialization_fn,
                 preprocess_input_fn,
                 target_size,
                 save_models=False):
        self.ds_init, self.ds_pool, self.ds_val, self.ds_test = (
            self.initialize_dataset(dataset_name, dataset_path, num_parallel_reads=NUM_PARALLEL_READS)
        )

        self.n_classes = len(get_labels_by_name(dataset_name))

        self.query_strategy = query_strategy

        self.idx_labeled_pool = []
        self.ds_augment = None

        self.model = None
        self.model_weights_checkpoint = None
        self.model_initialization_fn = model_initialization_fn
        self.save_models = save_models

        self.preprocess_input_fn = preprocess_input_fn
        self.target_size = target_size

    def initialize_dataset(self,
                           dataset_name,
                           dataset_path,
                           num_parallel_reads=None):
        return self._load_splits_from_tfrecord(dataset_name, dataset_path, num_parallel_reads=num_parallel_reads)

    @staticmethod
    def _load_splits_from_tfrecord(dataset_name, dataset_path, num_parallel_reads=None):
        ds_init = load_dataset_from_tfrecord(f"{dataset_name}-init", path=dataset_path, load_id=False, num_parallel_reads=num_parallel_reads)
        ds_pool = load_dataset_from_tfrecord(f"{dataset_name}-pool", path=dataset_path, load_id=True, num_parallel_reads=num_parallel_reads, deterministic=True)
        ds_val = load_dataset_from_tfrecord(f"{dataset_name}-val", path=dataset_path, load_id=False, num_parallel_reads=num_parallel_reads)
        ds_test = load_dataset_from_tfrecord(f"{dataset_name}-test", path=dataset_path, load_id=False, num_parallel_reads=num_parallel_reads)

        return ds_init, ds_pool, ds_val, ds_test

    def initialize_base_model(self,
                              model_name):
        self.model, loss_fn, optimizer, lr_init = self.model_initialization_fn()

        print("Optimizer configuration")
        print(optimizer.get_config())

        print("Loading base model weights")
        path_model = pathlib.Path("models", model_name, model_name)
        self.model.load_weights(path_model)

        print("Compiling model")
        self.model.compile(optimizer=optimizer,
                           loss=loss_fn,
                           metrics=["accuracy"])

        backend.set_value(self.model.optimizer.lr, lr_init)

    def get_train(self,
                  batch_size,
                  shuffle_buffer_size=1000,
                  cache_train=False,
                  cache_labeled_pool=True,
                  data_augmentation=True):
        if cache_train and cache_labeled_pool:
            print("WARNING: no need to cache labeled pool when caching training set already")

        ds_train_list = [self.ds_init]

        if len(self.idx_labeled_pool) > 0:
            print("Labeled pool is present")
            ds_labeled_pool = self._filter_dataset_by_index(self.ds_pool, self.idx_labeled_pool)
            ds_labeled_pool = self._remove_index_from_pool(ds_labeled_pool)

            # # DEBUG
            # print("Iterating labeled pool")
            # c = 0
            # labpoolidx = []
            # for x, y in ds_labeled_pool:
            #     c += 1
            #     labpoolidx.append(y)
            # debug(f"Total number of elements inside labeled pool: {c}")
            # debug(f"Class distribution: {np.unique(np.array(labpoolidx), return_counts=True)}")
            # # /DEBUG

            if cache_labeled_pool:
                ds_labeled_pool = ds_labeled_pool.cache()

            ds_train_list.append(ds_labeled_pool)
        else:
            print("Labeled pool is not present")

        if self.ds_augment is not None:
            print("Augmented set is present")
            ds_train_list.append(self.ds_augment)
        else:
            print("Augmented set is not present")

        if len(ds_train_list) == 1:  # only ds_init
            print("Only ds_init")
            ds_train = self.ds_init
        else:
            print("Interleaving datasets")
            # Interleave training datasets
            ds_train = tf.data.Dataset.from_tensor_slices(ds_train_list)
            ds_train = ds_train.interleave(lambda x: x, cycle_length=CYCLE_LENGTH, block_length=BLOCK_LENGTH, num_parallel_calls=tf.data.AUTOTUNE)

        if data_augmentation:
            ds_train = self._apply_data_augmentation(ds_train)

        ds_train = self._preprocess_dataset(ds_train)

        if cache_train:
            # note: it probably doesn't make sense to cache this since it's
            #  mostly initial training set and may be read directly from files
            #  produced by `generate_dataset.py`. Instead, cache the labeled pool.
            ds_train = ds_train.cache()
        ds_train = (
            ds_train
            .shuffle(shuffle_buffer_size)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        return ds_train

    def get_unlabeled_pool(self):
        ds_pool = self.ds_pool
        if len(self.idx_labeled_pool) > 0:
            ds_pool = self._filter_dataset_by_index(ds_pool, self.idx_labeled_pool, keep_indices=False)

        metadata = {
            "n_classes": self.n_classes,
        }

        # we don't preprocess the unlabeled pool at this time as there may be
        # query strategies that require raw data to operate.
        # ds_pool examples are structured as (index, (image, label))
        return ds_pool, metadata

    @staticmethod
    def _filter_dataset_by_index(dataset, index, keep_indices=True):
        # Declare a tf hashtable
        key_tensor = tf.constant(index)
        val_tensor = tf.ones_like(key_tensor)

        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(key_tensor, val_tensor),
            default_value=0)

        if keep_indices:  # keep specified indices
            def hash_table_filter(i, v):
                table_value = table.lookup(tf.cast(i, tf.int32))  # 1 if index in arr, else 0
                keep_record = tf.cast(table_value, tf.bool)
                return keep_record
        else:  # filter out specified indices
            def hash_table_filter(i, v):
                table_value = table.lookup(tf.cast(i, tf.int32))
                keep_record = tf.cast(table_value, tf.bool)
                return tf.math.logical_not(keep_record)

        dataset_filtered = dataset.filter(hash_table_filter)

        return dataset_filtered

    def get_val(self, batch_size):
        ds_val = self.ds_val
        ds_val = self._preprocess_dataset(ds_val)
        ds_val = (
            ds_val
            .batch(batch_size)
            .cache()  # in-memory cache (we can afford a low memory cost amortized by fetching val at each epoch)
            .prefetch(tf.data.AUTOTUNE)
        )
        # no need to shuffle val.
        return ds_val

    def get_test(self, batch_size):
        ds_test = self.ds_test
        ds_test = self._preprocess_dataset(ds_test)
        ds_test = (
            ds_test
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        # no need to cache test as it is read rarely.
        # no need to shuffle it either.
        return ds_test

    def _preprocess_dataset(self, dataset, only_one_hot=False):
        if only_one_hot:
            def tf_map_preprocess(x, y):
                return x, tf.one_hot(y, depth=self.n_classes)
        else:
            def tf_map_preprocess(x, y):
                return self.preprocess_input_fn(x), tf.one_hot(y, depth=self.n_classes)

        ds_preprocess = (
            dataset
            .map(tf_map_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        )

        return ds_preprocess

    @staticmethod
    def _apply_data_augmentation(dataset):
        def tf_map_horizontal_flip(x, y):
            return tf.image.random_flip_left_right(x), y

        ds_augmented = (
            dataset
            .map(tf_map_horizontal_flip, num_parallel_calls=tf.data.AUTOTUNE)
        )

        return ds_augmented

    @staticmethod
    def _remove_index_from_pool(ds_labeled_pool):
        def tf_map_remove_index(i, v):
            return v[0], v[1]

        ds_noindex = (
            ds_labeled_pool
            .map(tf_map_remove_index, num_parallel_calls=tf.data.AUTOTUNE)
        )

        return ds_noindex

    def query(self, ds_pool, metadata, n_query_instances, current_iter, seed=None, **query_kwargs):
        self.query_strategy.set_model(self.model, self.preprocess_input_fn)

        indices = self.query_strategy(ds_pool, metadata, n_query_instances, current_iter, seed=seed, **query_kwargs)
        print(f"Queried {len(indices)} samples from unlabeled pool")

        self.idx_labeled_pool.extend(indices)
        print(f"Total number of queried samples post-query: {len(self.idx_labeled_pool)}")

        ds_augment = self.query_strategy.get_ds_augment()
        if ds_augment is not None:
            self.ds_augment = ds_augment

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
        """
        """

        logs = {"train": [], "test": [], "test_preds": []}
        pathlib.Path("logs", dir_logs).mkdir(parents=True, exist_ok=False)

        print("Initializing base model")
        self.initialize_base_model(base_model_name)

        ds_val = self.get_val(batch_size)
        ds_test = self.get_test(batch_size)

        print("Evaluating base model")
        test_metrics = self.model.evaluate(ds_test)
        logs["base_test"] = test_metrics

        # Active learning loop
        for i in range(n_loops):
            print(f"* Iteration #{i+1}")

            # Reset augmented set
            self.ds_augment = None

            if i > 0:
                self.model, loss_fn, optimizer, lr_init = self.model_initialization_fn()

                print("Optimizer configuration")
                print(optimizer.get_config())

                self.model.compile(optimizer=optimizer,
                                   loss=loss_fn,
                                   metrics=["accuracy"])

                print("Setting weights to last iteration weights")
                self.model.set_weights(self.model_weights_checkpoint)
                backend.set_value(self.model.optimizer.lr, lr_init)

            ds_pool, metadata = self.get_unlabeled_pool()

            print("Querying unlabeled pool")
            self.query(ds_pool, metadata, n_query_instances, current_iter=i, seed=seed, **query_kwargs)

            ds_train = self.get_train(batch_size)
            print("Fitting model")
            history = self.model.fit(ds_train,
                                     validation_data=ds_val,
                                     epochs=n_epochs,
                                     callbacks=callbacks)
            logs["train"].append(history.history)

            if self.save_models:
                print(f"Saving model at iteration {i+1}")
                self.model.save(pathlib.Path("logs", dir_logs, f"model_iter_{i+1}"))

            print("Evaluating model")
            test_metrics = self.model.evaluate(ds_test)
            logs["test"].append(test_metrics)

            print("Fetching predictions")
            preds = self.model.predict(ds_test)
            logs["test_preds"].append(preds)

            self.model_weights_checkpoint = self.model.get_weights()

        return logs

    def train_base(self,
                   model_name,
                   batch_size,
                   n_epochs,
                   callbacks,
                   seed=None):
        logs = {"train": [], "test": None, "test_preds": None}

        model, loss_fn, optimizer, _ = self.model_initialization_fn(base=True)

        print("Optimizer configuration")
        print(optimizer.get_config())

        print("Compiling model")
        model.compile(optimizer=optimizer,
                      loss=loss_fn,
                      metrics=["accuracy"])

        ds_train = self.get_train(batch_size)
        ds_val = self.get_val(batch_size)

        print("Fitting base model")
        history = model.fit(ds_train,
                            validation_data=ds_val,
                            epochs=n_epochs,
                            callbacks=callbacks)
        logs["train"].append(history.history)

        print("Evaluating model")
        ds_test = self.get_test(batch_size)
        test_metrics = model.evaluate(ds_test)
        logs["test"] = test_metrics

        print("Fetching predictions")
        preds = model.predict(ds_test)
        logs["test_preds"] = preds

        print("Saving base model weights")
        path_model = pathlib.Path("models", model_name)
        path_model.mkdir(parents=True, exist_ok=False)
        model.save_weights(pathlib.Path(path_model, model_name))

        return logs
