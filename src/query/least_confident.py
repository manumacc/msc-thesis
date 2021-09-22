import numpy as np

from qs import QueryStrategy

class LeastConfidentQueryStrategy(QueryStrategy):
    def __call__(self, X_pool, n_query_instances, seed=None, query_batch_size=None):
        """Selects the instances whose best labeling is the least confident.

        Confidence is measured as the prediction output after the softmax layer.

        Args:
            X_pool: Unlabeled pool from which to query instances
            n_query_instances: Number of instances to query
            seed: No effect

        Returns:
            idx_query: Indices of selected samples
        """

        preds = self.model.predict(X_pool,
                                   batch_size=query_batch_size,
                                   verbose=1)

        max_preds = preds.max(axis=1)
        idx_query = np.argsort(max_preds)[:n_query_instances]

        return idx_query
