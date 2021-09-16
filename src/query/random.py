import numpy as np

from qs import QueryStrategy

class RandomQueryStrategy(QueryStrategy):
    def __call__(self, X_pool, n_query_instances, seed=None):
        rng = np.random.default_rng(seed)

        idx = np.arange(0, X_pool.shape[0])
        idx_query = rng.choice(idx, size=n_query_instances, replace=False)

        return idx_query
