import numpy as np

from qs import QueryStrategy

class RandomQueryStrategy(QueryStrategy):
    def __call__(self,
                 ds_pool,
                 metadata,
                 n_query_instances,
                 current_iter,
                 seed=None):
        rng = np.random.default_rng(seed)

        # todo: remove this and substitute with metadata["indices"]
        idx_pool = []
        print("Iterating pool")
        for idx, _ in ds_pool:
            idx_pool.append(idx)
        idx_pool = np.array(idx_pool)

        idx_query = rng.choice(idx_pool, size=n_query_instances, replace=False)
        del idx_pool

        return idx_query
