import tensorflow as tf
import numpy as np


class QueryStrategy:
    def __init__(self):
        self.model = None
        self.preprocess_input_fn = None
        self.ds_augment = None

    def set_model(self, model, preprocess_input_fn=None):
        self.model = model
        self.preprocess_input_fn = preprocess_input_fn

    def __call__(self, ds_pool, metadata, n_query_instances, current_iter, seed=None):
        raise NotImplementedError("Can't call a base class")

    def _preprocess_pool_dataset(self, ds_pool, metadata, query_batch_size):
        def tf_map_preprocess(i, v):
            return self.preprocess_input_fn(v[0]), tf.one_hot(v[1], depth=metadata["n_classes"])

        ds_pool_preprocess = (
            ds_pool
            .map(tf_map_preprocess, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)  # we lose the index here
            .batch(query_batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        return ds_pool_preprocess

    @staticmethod
    def _get_pool_indices(ds_pool):
        idx_pool = []
        print("Iterating pool to fetch indices")
        for idx, _ in ds_pool:
            idx_pool.append(idx)

        return np.array(idx_pool)

    def get_ds_augment(self):
        return self.ds_augment
