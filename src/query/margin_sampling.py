import numpy as np
import tensorflow as tf

from qs import QueryStrategy

class MarginSamplingQueryStrategy(QueryStrategy):
    def __call__(self,
                 ds_pool,
                 metadata,
                 n_query_instances,
                 current_iter,
                 seed=None,
                 query_batch_size=None,
                 shuffle_buffer_size=1000):
        """Selects the instances whose margin is smallest.

        The margin for a given instance is defined as the difference between the
        first and second most likely predictions. A small margin signifies a
        more ambiguous sample.
        """

        # todo: remove this and add metadata["indices"] array that contains all
        #  current indices inside the training pool (we can derive this by
        #  using self.idx_labeled_pool and the length of the whole pool (lab+unl)
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
                                   verbose=1)  # (len(X_pool), n_classes)

        preds_sorted = np.sort(preds, axis=-1)[..., ::-1]  # Sort each prediction vector (descending)
        margins = preds_sorted[..., 0] - preds_sorted[..., 1]  # Subtract the largest predictions from the second-largest
        idx_query = np.argsort(margins)[:n_query_instances]  # Select smallest margins

        return idx_pool[idx_query]
