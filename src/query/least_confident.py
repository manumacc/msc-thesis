import numpy as np
import tensorflow as tf

from qs import QueryStrategy

class LeastConfidentQueryStrategy(QueryStrategy):
    def __call__(self,
                 ds_pool,
                 metadata,
                 n_query_instances,
                 current_iter,
                 seed=None,
                 query_batch_size=None):
        """Selects the instances whose best labeling is the least confident.

        Confidence is measured as the prediction output after the softmax layer.
        """

        # todo: remove this and add metadata["indices"]
        idx_pool = []
        print("Iterating pool")
        for idx, _ in ds_pool:
            idx_pool.append(idx)
        idx_pool = np.array(idx_pool)

        def tf_map_preprocess(i, v):
            return self.preprocess_input_fn(v[0]), tf.one_hot(v[1], depth=metadata["n_classes"])

        ds_pool_preprocess = (
            ds_pool
            .map(tf_map_preprocess, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)  # we lose the index here
            .batch(query_batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        preds = self.model.predict(ds_pool_preprocess,
                                   verbose=1)

        max_preds = preds.max(axis=1)
        idx_query = np.argsort(max_preds)[:n_query_instances]

        return idx_pool[idx_query]
