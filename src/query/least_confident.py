import numpy as np

from qs import QueryStrategy

class LeastConfidentQueryStrategy(QueryStrategy):
    def __call__(self, X_pool, n_query_instances, seed=None, batch_size=None):
        """
        Query the instances whose best labeling is the least confident,
        according to the model outputs (softmax).
        """

        preds = self.model.predict(X_pool,
                                   batch_size=batch_size,
                                   verbose=1)

        max_preds = preds.max(axis=1)
        idx_query = np.argsort(max_preds)[:n_query_instances]

        return idx_query
