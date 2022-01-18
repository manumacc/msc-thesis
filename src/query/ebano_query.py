import numpy as np
import tensorflow as tf
from PIL import Image

from ebano.batchebano import Explainer
from qs import QueryStrategy

from utils import Profiling, ndarray_to_pil, pil_to_ndarray

class EBAnOQueryStrategy(QueryStrategy):
    def __call__(self,
                 ds_pool,
                 ds_train,
                 metadata,
                 n_query_instances,
                 current_iter,
                 seed=None,
                 query_batch_size=None,
                 **query_kwargs):
        augment = query_kwargs["augment"]
        base_strategy = query_kwargs["base_strategy"]
        query_limit = query_kwargs["query_limit"]
        augment_limit = query_kwargs["augment_limit"]
        min_diff = query_kwargs["min_diff"]

        idx_pool = self._get_pool_indices(ds_pool)
        ds_pool_preprocess = self._preprocess_pool_dataset(ds_pool, metadata, query_batch_size)

        # Predict
        preds = self.model.predict(ds_pool_preprocess, verbose=1)  # (len(X_pool), n_classes)
        cois = np.argmax(preds, axis=1)  # class-of-interest is the prediction

        # Select subset of indices (if specified)
        if query_kwargs["subset"] is not None:
            if query_kwargs["subset"] > len(idx_pool):
                print(f"Subset size {query_kwargs['subset']} is higher than pool size {len(idx_pool)}.")
                print(f"Setting subset size equal to pool size.")
                idx_to_process = idx_pool
            else:
                # Subset is taken randomly from the unlabeled pool
                rng = np.random.default_rng(seed)
                idx_to_process = rng.choice(idx_pool, size=query_kwargs["subset"], replace=False)
                print(f"Fetched a subset of size {len(idx_to_process)} over {len(idx_pool)}")
        else:
            idx_to_process = idx_pool
            print(f"Fetched all elements")

        # Explain via BatchEBAnO
        self.explainer = Explainer(
            model=self.model,
            n_classes=query_kwargs["n_classes"],
            input_shape=query_kwargs["input_shape"],
            layers_to_analyze=query_kwargs["layers_to_analyze"],
        )

        nPIR_best = []
        nPIRP_best = []
        X_masks = []

        X_buffer = []
        cois_buffer = []
        batch_count = 0

        idx_pool_subset = []
        preds_subset = []
        for (tf_i, (tf_x, tf_y)), coi, pred in zip(ds_pool, cois, preds):  # Non-preprocessed pool
            if tf_i.numpy() not in idx_to_process:
                continue

            X_buffer.append(tf_x.numpy())
            cois_buffer.append(coi)

            idx_pool_subset.append(tf_i.numpy())
            preds_subset.append(pred)

            if len(X_buffer) == query_batch_size:  # buffer filled
                with Profiling(f"Processing batch {batch_count+1}"):
                    results = self.ebano_process(X_buffer, cois_buffer, seed=seed, **query_kwargs)
                    for r in results:
                        nPIR_best.append(r["nPIR_best"])
                        nPIRP_best.append(r["nPIRP_best"])
                        if augment:
                            X_masks.append(r["X_masks"])

                X_buffer.clear()
                cois_buffer.clear()

                batch_count += 1

        # Process final batch, if it exists
        if len(X_buffer) > 0:
            print(f"Processing last batch of size {len(X_buffer)}")
            results = self.ebano_process(X_buffer, cois_buffer, seed=seed, **query_kwargs)
            for r in results:
                nPIR_best.append(r["nPIR_best"])
                nPIRP_best.append(r["nPIRP_best"])
                if augment:
                    X_masks.append(r["X_masks"])
        else:
            print("Last batch is empty")

        assert len(nPIR_best) == len(nPIRP_best)
        idx_pool_subset = np.array(idx_pool_subset)
        preds_subset = np.array(preds_subset)

        # Note that idx_pool_subset contains REAL indices from the pool dataset,
        # i.e., the ones you get when iterating through ds_pool.
        # Note that nPIR_best, nPIRP_best, X_masks are all ordered in the same
        # way as idx_pool_subset, as they are extracted concurrently.
        # Also, idx_pool_subset and preds_subset correspond to each other.

        # EBAnO query
        idx_candidates, nPIR_max_f_i = self.query_most_influential_has_low_precision_difference_rank(nPIR_best, nPIRP_best, min_diff=min_diff)
        idx_candidates = idx_pool_subset[idx_candidates]
        print(f"Candidates queried by EBAnO: {len(idx_candidates)} with diff={min_diff}")
        # idx_candidates contains REAL indices from the pool dataset. These
        # are ORDERED in a ranked fashion, per the ebano specifications.
        # This means that any element in idx_candidates satisfies:
        #   min(idx_pool_subset) < x < max(idx_pool_subset)
        # NOTE: nPIR_max_f_i ordering is the same as idx_pool_subset and NOT
        # the same as idx_candidates. This makes nPIR_max_f_i consistent with
        # the ordering of nPIR_best, nPIRP_best, X_masks.

        # Limit number of candidates queried by EBAnO
        if query_limit > n_query_instances:
            query_limit = n_query_instances
            print(f"WARNING: query_limit set to {n_query_instances}")
        if augment and augment_limit > n_query_instances:
            augment_limit = n_query_instances
            print(f"WARNING: augment_limit set to {n_query_instances}")

        idx_query_ebano = idx_candidates[:query_limit]
        # idx_query_ebano contains again REAL indices.
        print(f"EBAnO selected {len(idx_query_ebano)} samples.")

        # Mix EBAnO query with chosen base strategy
        if len(idx_query_ebano) < n_query_instances:  # If too few queried by EBAnO, add more by selecting among non-candidates using base_strategy
            # Since we already queried the samples at pool indices idx_query_ebano,
            # we remove them from the pool from which we take the "other" samples
            # using the chosen base strategy.
            idx_others = []
            preds_others = []
            for i, pred in zip(idx_pool_subset, preds_subset):
                if i not in idx_query_ebano:
                    idx_others.append(i)
                    preds_others.append(pred)
            idx_others = np.array(idx_others)
            preds_others = np.array(preds_others)
            # idx_others contains REAL indices

            n_missing = n_query_instances - len(idx_query_ebano)

            if base_strategy == "random":
                rng = np.random.default_rng(seed)
                idx_additional = rng.choice(idx_others, size=n_missing, replace=False)
            elif base_strategy == "least-confident":
                max_preds_others = preds_others.max(axis=1)
                idx_of_idx_others_sorted = np.argsort(max_preds_others)
                idx_additional = idx_others[idx_of_idx_others_sorted][:n_missing]
            elif base_strategy == "entropy":
                entropy_others = -np.sum(preds_others * np.log(preds_others), axis=-1)
                idx_of_idx_others_sorted = np.argsort(entropy_others)[::-1]
                idx_additional = idx_others[idx_of_idx_others_sorted][:n_missing]
            else:
                raise ValueError(f"Base strategy {base_strategy} not implemented.")

            idx_query = np.concatenate([idx_query_ebano, idx_additional])
        else:
            idx_query = idx_query_ebano

        # Augmented dataset
        self.ds_augment = None  # reset augmented set
        if augment and len(idx_candidates) > 0:
            idx_augment = idx_candidates[:augment_limit]  # Limit number of candidates queried by EBAnO
            # idx_augment contains REAL indices
            print(f"Create ds_augment with {len(idx_augment)} samples")

            X_augment = []
            y_augment = []
            for (tf_i, (tf_x, tf_y)) in ds_pool:
                if tf_i.numpy() not in idx_augment:
                    continue

                x_original = tf_x.numpy()

                i_of_idx_pool_subset = np.argwhere(idx_pool_subset == tf_i.numpy())[0,0]

                x_masks = X_masks[i_of_idx_pool_subset]
                f_i = nPIR_max_f_i[i_of_idx_pool_subset]

                x_perturbed = self.get_perturbed_image(x_original, x_masks, f_i, perturb_filter=self.explainer.perturb_filter)
                X_augment.append(x_perturbed)
                y_augment.append(tf_y.numpy())

            X_augment = np.array(X_augment)
            y_augment = np.array(y_augment)

            self.ds_augment = tf.data.Dataset.from_tensor_slices((X_augment, y_augment))

        return idx_query

    def ebano_process(self, X, cois, seed=None, **kwargs):
        results = self.explainer.fit_batch(
            np.array(X),
            cois=cois,
            preprocess_input_fn=self.preprocess_input_fn,
            hypercolumn_features=kwargs["hypercolumn_features"],
            hypercolumn_reduction=kwargs["hypercolumn_reduction"],
            clustering=kwargs["clustering"],
            min_features=kwargs["min_features"],
            max_features=kwargs["max_features"],
            display_plots=False,
            return_results=True,
            use_gpu=False,
            seed=seed,
            niter=kwargs["niter"],
        )

        return results

    @staticmethod
    def query_most_influential_has_low_precision_difference_rank(nPIR, nPIRP, min_diff=0.):
        """
        Selects samples with the highest difference between their maximum nPIR
        i.e., the nPIR corresponding to the most influential feature, and the
        associated nPIRP. Samples are ranked from highest to lowest difference.
        The goal is to select samples whose most influential interpretable
        feature has highest nPIR and lowest nPIRP.

        Returns a sorted (n,) array of indices corresponding to queried images,
        from best-to-query to worst-to-query.
        Returns an array of shape (n,) containing the indices of the
        interpretable feature corresponding to the highest nPIR for each image.
        """

        n = len(nPIR)

        margin = np.empty(n)
        nPIR_max_f_i = np.empty(n, dtype=int)
        for i in range(n):
            idx = np.argmax(nPIR[i])
            nPIR_max_i = nPIR[i][idx]
            nPIRP_of_nPIR_max_i = nPIRP[i][idx]
            margin[i] = nPIR_max_i - nPIRP_of_nPIR_max_i
            nPIR_max_f_i[i] = idx

        idx = np.arange(n)
        mask = margin > min_diff  # create mask without samples under minimum diff
        margin_masked = margin[mask]  # remove samples under min diff
        idx_masked = idx[mask]  # remove idx of samples under min diff

        idx_margin_mask_sorted = np.argsort(margin_masked)[::-1]  # in order of best-to-query to worst-to-query
        idx_candidates = idx_masked[idx_margin_mask_sorted]  # retrieve real indices from masked idx array

        return idx_candidates, nPIR_max_f_i

    @staticmethod
    def get_perturbed_image(x_original, x_masks, f_i, perturb_filter=None, flip=False):
        x_original_pil = ndarray_to_pil(x_original)
        x_perturbed = x_original_pil.filter(perturb_filter)  # Fully perturbed image

        x_perturbed_f_i = x_original_pil.copy()

        # Invert mask
        x_mask = x_masks[f_i]
        x_mask_inverse = np.copy(x_mask)
        x_mask_inverse[x_mask == 255] = 0
        x_mask_inverse[x_mask == 0] = 255
        x_mask = x_mask_inverse

        # Apply mask
        x_mask_img = Image.fromarray(x_mask, mode="L")
        x_perturbed_f_i.paste(x_perturbed, mask=x_mask_img)
        if flip:
            x_perturbed_f_i = x_perturbed_f_i.transpose(Image.FLIP_LEFT_RIGHT)
        x_perturbed_f_i = pil_to_ndarray(x_perturbed_f_i)

        return x_perturbed_f_i
