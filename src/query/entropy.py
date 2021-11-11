import numpy as np
import tensorflow as tf

from qs import QueryStrategy

class EntropyQueryStrategy(QueryStrategy):
    def __call__(self,
                 ds_pool,
                 metadata,
                 n_query_instances,
                 current_iter,
                 seed=None,
                 query_batch_size=None):
        """Selects the instances whose entropy is highest.

        Entropy takes into account all predictions for each sample. Higher
        entropy corresponds to higher uncertainty.
        """

        idx_pool = self._get_pool_indices(ds_pool)
        ds_pool_preprocess = self._preprocess_dataset(ds_pool, metadata, query_batch_size)

        preds = self.model.predict(ds_pool_preprocess,
                                   verbose=1)

        entropy = -np.sum(preds * np.log(preds), axis=-1)
        idx_sorted = np.argsort(entropy)[::-1]  # Descending order
        idx_query = idx_sorted[:n_query_instances]

        return idx_pool[idx_query]
