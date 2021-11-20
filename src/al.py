import pathlib
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend

from dataset.metadata import get_labels_by_name, get_size_by_name
from dataset.record import load_dataset_from_tfrecord


NUM_PARALLEL_READS = 4


class ActiveLearning:
    def __init__(self,
                 dataset_name,
                 dataset_path,
                 query_strategy,
                 model_initialization_fn,
                 preprocess_input_fn,
                 target_size,
                 save_models=False):
        self.ds_init = None
        self.ds_pool = None
        self.ds_val = None
        self.ds_test = None
        self.n_classes = None
        self.ds_init_len = None

        self.initialize_dataset(dataset_name, dataset_path, num_parallel_reads=NUM_PARALLEL_READS)

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
        self.ds_init, self.ds_pool, self.ds_val, self.ds_test = (
            self._load_splits_from_tfrecord(dataset_name, dataset_path, num_parallel_reads=num_parallel_reads)
        )

        self.n_classes = len(get_labels_by_name(dataset_name))
        self.ds_init_len = get_size_by_name(dataset_name, "init")

    @staticmethod
    def _load_splits_from_tfrecord(dataset_name, dataset_path, num_parallel_reads=None):
        ds_init = load_dataset_from_tfrecord(f"{dataset_name}-init", path=dataset_path, load_id=False, num_parallel_reads=num_parallel_reads, deterministic=False)
        ds_pool = load_dataset_from_tfrecord(f"{dataset_name}-pool", path=dataset_path, load_id=True, num_parallel_reads=num_parallel_reads, deterministic=True)
        ds_val = load_dataset_from_tfrecord(f"{dataset_name}-val", path=dataset_path, load_id=False, num_parallel_reads=num_parallel_reads, deterministic=True)
        ds_test = load_dataset_from_tfrecord(f"{dataset_name}-test", path=dataset_path, load_id=False, num_parallel_reads=num_parallel_reads, deterministic=True)

        return ds_init, ds_pool, ds_val, ds_test

    def initialize_base_model(self, model_name):
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
                  shuffle_buffer_size=2000,
                  labeled_pool_cache=True,
                  data_augmentation=True):
        ds_train_list = [self.ds_init]
        ds_train_list_len = [self.ds_init_len]  # needed for uniform sampling

        # Labeled pool
        if len(self.idx_labeled_pool) > 0:
            print("Labeled pool is present")
            ds_labeled_pool = self._filter_dataset_by_index(self.ds_pool, self.idx_labeled_pool)
            ds_labeled_pool = self._remove_index_from_pool(ds_labeled_pool)

            if labeled_pool_cache:
                ds_labeled_pool = ds_labeled_pool.cache()

            ds_labeled_pool = ds_labeled_pool.shuffle(shuffle_buffer_size)

            ds_train_list.append(ds_labeled_pool)
            ds_train_list_len.append(len(self.idx_labeled_pool))
        else:
            print("Labeled pool is not present")

        # Augmented set
        if self.ds_augment is not None:
            print("Augmented set is present")
            ds_augment_len = 0
            for _ in self.ds_augment:
                ds_augment_len += 1
            print(f"Length of augmented set: {ds_augment_len}")

            ds_augment = self.ds_augment.shuffle(shuffle_buffer_size)
            ds_train_list.append(ds_augment)
            ds_train_list_len.append(ds_augment_len)
        else:
            print("Augmented set is not present")

        if len(ds_train_list) == 1:  # only ds_init
            print("Only ds_init")
            ds_train = self.ds_init
        else:
            print("Interleaving datasets via weighted sampling")
            # Interleaving is done in a weighted fashion, as the sizes of the
            # datasets are all different. We want to take elements from the
            # datasets such that they are all well shuffled together.
            weights = np.array(ds_train_list_len) / np.sum(ds_train_list_len)
            ds_train = tf.data.experimental.sample_from_datasets(ds_train_list, weights=weights, seed=None)  # non-deterministic shuffle

        if data_augmentation:
            ds_train = self._apply_data_augmentation(ds_train)

        ds_train = self._preprocess_dataset(ds_train)
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
            .map(tf_map_remove_index, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
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
              logs_path,
              logs_partial_path,
              resume_job_dir=None,
              seed=None,
              **query_kwargs):
        """
        """

        LAST_ITER_FILE = "last_iter.txt"

        logs = {"train": [], "test": [], "test_preds": []}

        with open(pathlib.Path(logs_partial_path, LAST_ITER_FILE), "w") as f:
            f.write("-1")

        ds_val = self.get_val(batch_size)
        ds_test = self.get_test(batch_size)

        if resume_job_dir is not None:
            resume_job_path = pathlib.Path("logs", resume_job_dir)
            print(f"Resuming job from path {str(resume_job_path)}")

            # Skip current AL loop to correct iteration
            with open(pathlib.Path(resume_job_path, "partial", LAST_ITER_FILE)) as f:
                last_iter = int(f.read())

            skip_to_iter = last_iter + 1

            # Reload last iteration's weights
            self.model, loss_fn, optimizer, lr_init = self.model_initialization_fn()
            self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

            print(f"Reloading weights from iteration #{last_iter+1} (i={last_iter})")
            self.model.load_weights(pathlib.Path(resume_job_path, "partial", "model"))
            backend.set_value(self.model.optimizer.lr, lr_init)

            print("Evaluating reloaded model")
            self.model.evaluate(ds_test)

            # Reload logs up to last iteration
            with open(pathlib.Path(resume_job_path, "partial", "logs_partial.pkl"), "rb") as f:
                logs = pickle.load(f)

            # Reload labeled pool indices
            with open(pathlib.Path(resume_job_path, "partial", "idx_labeled_pool.pkl"), "rb") as f:
                self.idx_labeled_pool = pickle.load(f)
            print(f"Loaded labeled pool indices of length {len(self.idx_labeled_pool)}")
        else:
            print("Initializing base model")
            self.initialize_base_model(base_model_name)

            print("Evaluating base model")
            test_metrics = self.model.evaluate(ds_test)
            logs["base_test"] = test_metrics

            skip_to_iter = None

        # Active learning loop
        for i in range(n_loops):
            print(f"* Iteration #{i+1} (i={i}) begins")

            if skip_to_iter is not None and i < skip_to_iter:
                print(f"Iteration #{i+1} (i={i}) skipped")
                continue

            # Reset augmented set
            self.ds_augment = None

            if skip_to_iter is None and i > 0:
                self.model, loss_fn, optimizer, lr_init = self.model_initialization_fn()

                print("Optimizer configuration")
                print(optimizer.get_config())

                self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

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
                self.model.save(pathlib.Path(logs_path, f"model_iter_{i+1}"))

            print("Evaluating model")
            test_metrics = self.model.evaluate(ds_test)
            logs["test"].append(test_metrics)

            print("Fetching predictions")
            preds = self.model.predict(ds_test)
            logs["test_preds"].append(preds)

            self.model_weights_checkpoint = self.model.get_weights()

            # Parachute: save partial model
            skip_to_iter = None  # deactivate skipping

            self.model.save_weights(
                pathlib.Path(logs_partial_path, "model"),
                overwrite=True
            )
            with open(pathlib.Path(logs_partial_path, LAST_ITER_FILE), "w") as f:
                f.write(f"{i}")
            with open(pathlib.Path(logs_partial_path, "logs_partial.pkl"), "wb") as f:
                pickle.dump(logs, f)
            with open(pathlib.Path(logs_partial_path, "idx_labeled_pool.pkl"), "wb") as f:
                pickle.dump(self.idx_labeled_pool, f)

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
