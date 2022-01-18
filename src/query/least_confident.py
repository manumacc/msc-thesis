import numpy as np

from qs import QueryStrategy

class LeastConfidentQueryStrategy(QueryStrategy):
    def __call__(self,
                 ds_pool,
                 ds_train,
                 metadata,
                 n_query_instances,
                 current_iter,
                 seed=None,
                 query_batch_size=None):
        """Selects the instances whose best labeling is the least confident.

        Confidence is measured as the prediction output after the softmax layer.
        """

        idx_pool = self._get_pool_indices(ds_pool)
        ds_pool_preprocess = self._preprocess_pool_dataset(ds_pool, metadata, query_batch_size)

        preds = self.model.predict(ds_pool_preprocess,
                                   verbose=1)

        max_preds = preds.max(axis=1)
        idx_query = np.argsort(max_preds)[:n_query_instances]

        return idx_pool[idx_query]
