import numpy as np

from qs import QueryStrategy
from query.entropy import EntropyQueryStrategy

class HybridQueryStrategy(QueryStrategy):
    def __call__(self,
                 ds_pool,
                 ds_train,
                 metadata,
                 n_query_instances,
                 current_iter,
                 n_query_entropy=None,
                 seed=None,
                 query_batch_size=None):
        """Selects the most uncertain and diverse instances, based on Smailagic et al."""

        idx_pool = self._get_pool_indices(ds_pool)

        ds_pool_preprocess = self._preprocess_pool_dataset(ds_pool, metadata, query_batch_size)

        preds = self.model.predict(ds_pool_preprocess, verbose=1)
        features = self.extractor(ds_pool_preprocess)

        print("features shape:", features.shape)

        raise ValueError

        # Compute entropy of all samples
        print("DEBUG: Computing entropy...")
        entropy = -np.sum(preds * np.log(preds), axis=-1)
        idx_sorted = np.argsort(entropy)[::-1]  # Descending order
        idx_query_entropy = idx_sorted[:n_query_entropy]

        print(f"DEBUG: Selected {len(idx_query_entropy)} samples via entropy")

        # Extract features of training samples
        # train_features = self.extractor(ds_train)




        #return idx_pool[idx_query]
        return idx_pool[idx_query_entropy]

    def dist(self, ...):
        pass