import numpy as np

from qs import QueryStrategy

class RandomQueryStrategy(QueryStrategy):
    def __call__(self,
                 ds_pool,
                 ds_train,
                 metadata,
                 n_query_instances,
                 current_iter,
                 seed=None):
        idx_pool = self._get_pool_indices(ds_pool)

        rng = np.random.default_rng(seed)
        idx_query = rng.choice(idx_pool, size=n_query_instances, replace=False)

        return idx_query
