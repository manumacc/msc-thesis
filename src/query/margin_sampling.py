import numpy as np

from qs import QueryStrategy

class MarginSamplingQueryStrategy(QueryStrategy):
    def __call__(self,
                 ds_pool,
                 ds_train,
                 metadata,
                 n_query_instances,
                 current_iter,
                 seed=None,
                 query_batch_size=None):
        """Selects the instances whose margin is smallest.

        The margin for a given instance is defined as the difference between the
        first and second most likely predictions. A small margin signifies a
        more ambiguous sample.
        """

        idx_pool = self._get_pool_indices(ds_pool)
        ds_pool_preprocess = self._preprocess_pool_dataset(ds_pool, metadata, query_batch_size)

        preds = self.model.predict(ds_pool_preprocess,
                                   verbose=1)  # (len(X_pool), n_classes)

        preds_sorted = np.sort(preds, axis=-1)[..., ::-1]  # Sort each prediction vector (descending)
        margins = preds_sorted[..., 0] - preds_sorted[..., 1]  # Subtract the largest predictions from the second-largest
        idx_query = np.argsort(margins)[:n_query_instances]  # Select smallest margins

        return idx_pool[idx_query]
