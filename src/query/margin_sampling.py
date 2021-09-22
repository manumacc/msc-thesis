import numpy as np

from qs import QueryStrategy

class MarginSamplingQueryStrategy(QueryStrategy):
    def __call__(self, X_pool, metadata, n_query_instances, seed=None, query_batch_size=None):
        """Selects the instances whose margin is smallest.

        The margin for a given instance is defined as the difference between the
        first and second most likely predictions. A small margin signifies a
        more ambiguous sample.

        Args:
            X_pool:
            metadata:
            n_query_instances:
            seed:
            query_batch_size:

        Returns:

        """

        preds = self.model.predict(X_pool,
                                   batch_size=query_batch_size,
                                   verbose=1)

        preds_sorted = np.sort(preds, axis=-1)[..., ::-1]  # Sort each prediction vector (descending)
        margins = preds_sorted[..., 0] - preds_sorted[..., 1]  # Subtract the largest predictions from the second-largest
        idx_query = np.argsort(margins)[:n_query_instances]  # Select smallest margins

        return idx_query
