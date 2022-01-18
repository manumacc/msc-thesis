import numpy as np

from qs import QueryStrategy


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

        n_query_entropy = 5000  # todo: REMOVE THIS AND MAKE IT A HYPERPARAM

        idx_pool = self._get_pool_indices(ds_pool)

        ds_pool_preprocess = self._preprocess_pool_dataset(ds_pool, metadata, query_batch_size)

        preds = self.model.predict(ds_pool_preprocess, verbose=1)

        # Compute entropy of all samples
        print("Computing entropy")
        entropy = -np.sum(preds * np.log(preds), axis=-1)
        idx_sorted = np.argsort(entropy)[::-1]  # Descending order
        idx_query_entropy = idx_sorted[:n_query_entropy]

        print(f"Selected {len(idx_query_entropy)} samples via entropy")

        # Extract features from pool
        # todo: memory consumption can be reduced if we fetch the feature vectors of only
        #  ds_pool_preprocess[idx_query_entropy].
        print("Extracting features from pool (only selected samples)")
        features = []
        for pool_batch, _ in ds_pool_preprocess:
            features.append(self.extractor(pool_batch).numpy())
        features = np.concatenate(features)  # shape (num elements pool, num features)
        features = features[idx_query_entropy]  # subset of selected features
        print(features.shape)

        # Extract features of training samples
        print("Extracting features from training samples")
        features_train = []
        for train_batch, _ in ds_train:
            features_train.append(self.extractor(train_batch).numpy())
        features_train = np.concatenate(features_train)  # shape (num elements train, num features)

        # Iteratively select samples that are most diverse w.r.t. training samples
        idx_query = []
        while len(idx_query) < n_query_instances:
            # Compute centroid of training samples
            centroid_train = features_train.mean(axis=0)

            # Calculate Euclidean distance between all samples in pool and the centroid
            scores_pool = np.array([np.linalg.norm(features_i - centroid_train) for features_i in features])  # todo: this may be vectorized, check axis=?

            if len(idx_query) > 0:
                m = np.zeros(len(features), dtype=bool)
                m[idx_query] = True
                scores_pool = np.ma.array(scores_pool, mask=m)  # mask already selected elements

            # Select index of highest scoring element (the one "most distant" from training elements)
            i = np.argmax(scores_pool)

            # Add the element to the training set features
            features_train = np.concatenate([features_train, features[[i]]])

            idx_query.append(i)

        return idx_pool[idx_query]