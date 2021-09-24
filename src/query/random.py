import numpy as np

from qs import QueryStrategy

class RandomQueryStrategy(QueryStrategy):
    def __call__(self, X_pool, metadata, n_query_instances, model, seed=None):
        rng = np.random.default_rng(seed)

        idx = np.arange(0, metadata["len"])
        idx_query = rng.choice(idx, size=n_query_instances, replace=False)

        return idx_query
