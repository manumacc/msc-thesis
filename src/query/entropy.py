import numpy as np

from qs import QueryStrategy

class EntropyQueryStrategy(QueryStrategy):
    def __call__(self, X_pool, y_pool, n_query_instances, current_iter, seed=None, query_batch_size=None):
        """Selects the instances whose entropy is highest.

        Entropy takes into account all predictions for each sample. Higher
        entropy corresponds to higher uncertainty.

        Args:
            X_pool:
            n_query_instances:
            current_iter:
            seed:
            query_batch_size:

        Returns:

        """

        X_pool = self.preprocess_input_fn(X_pool)

        preds = self.model.predict(X_pool,
                                   batch_size=query_batch_size,
                                   verbose=1)  # (len(X_pool), n_classes)

        entropy = -np.sum(preds * np.log(preds), axis=-1)
        idx_sorted = np.argsort(entropy)[::-1]  # Descending order
        idx_query = idx_sorted[:n_query_instances]

        return idx_query
