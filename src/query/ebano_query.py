import numpy as np
from PIL import Image

from ebano.batchebano import Explainer
from qs import QueryStrategy

from utils import Profiling, ndarray_to_pil, pil_to_ndarray

class EBAnOQueryStrategy(QueryStrategy):
    def __call__(self,
                 X_pool,
                 y_pool,
                 n_query_instances,
                 current_iter,
                 seed=None,
                 query_batch_size=32,
                 n_classes=None,
                 input_shape=None,
                 layers_to_analyze=None,
                 hypercolumn_features=None,
                 hypercolumn_reduction=None,
                 clustering=None,
                 min_features=2,
                 max_features=5,
                 use_gpu=False,
                 augment=None,
                 strategy=None,
                 base_strategy=None,
                 query_limit=None,
                 augment_limit=None,
                 min_diff=None,
                 eps=None,
                 **ebano_kwargs):

        self.augmented_set = None  # reset augmented set

        # Predict
        X_pool_preprocessed = self.preprocess_input_fn(np.copy(X_pool))
        preds = self.model.predict(X_pool_preprocessed,
                                   batch_size=query_batch_size,
                                   verbose=1)  # (len(X_pool), n_classes)
        del X_pool_preprocessed
        cois = np.argmax(preds, axis=1)

        # Explain via BatchEBAnO
        nPIR_best = []
        nPIRP_best = []
        X_masks = []

        explainer = Explainer(
            model=self.model,
            n_classes=n_classes,
            input_shape=input_shape,
            layers_to_analyze=layers_to_analyze,
        )

        n_batches = int(np.ceil(len(X_pool) / query_batch_size))
        for i in range(n_batches):
            batch_len = len(X_pool[i*query_batch_size:(i+1)*query_batch_size])
            if batch_len == 0:  # skip batch if empty
                print("Empty batch")
                pass

            with Profiling(f"Processing batch {i+1}/{n_batches} of size {batch_len}"):
                results = explainer.fit_batch(
                    X_pool[i*query_batch_size:(i+1)*query_batch_size],
                    cois=cois[i*query_batch_size:(i+1)*query_batch_size],
                    preprocess_input_fn=self.preprocess_input_fn,  # data is already preprocessed
                    hypercolumn_features=hypercolumn_features,
                    hypercolumn_reduction=hypercolumn_reduction,
                    clustering=clustering,
                    min_features=min_features,
                    max_features=max_features,
                    display_plots=False,
                    return_results=True,
                    use_gpu=False,
                    seed=seed,
                    niter=ebano_kwargs["niter"],
                )

            for r in results:
                nPIR_best.append(r["nPIR_best"])
                nPIRP_best.append(r["nPIRP_best"])
                if augment:
                    X_masks.append(r["X_masks"])

        assert len(nPIR_best) == len(nPIRP_best)

        # EBAnO query
        if strategy == "rank":
            idx_candidates, nPIR_max_f_i = self.query_most_influential_has_low_precision_difference_rank(nPIR_best, nPIRP_best, min_diff=min_diff)
            print(f"Candidates queried by EBAnO: {len(idx_candidates)} with diff={min_diff}")
        else:
            idx_candidates, nPIR_max_f_i = self.query_most_influential_has_low_precision(nPIR_best, nPIRP_best, eps=eps)
            np.random.default_rng(seed).shuffle(idx_candidates)  # shuffle to eventually randomly choose a subset
            print(f"Candidates queried by EBAnO: {len(idx_candidates)} with epsilon={eps}")

        # Limit number of candidates queried by EBAnO
        if query_limit > n_query_instances:
            query_limit = n_query_instances
            print(f"WARNING: query_limit set to {n_query_instances}")
        if augment and augment_limit > n_query_instances:
            augment_limit = n_query_instances
            print(f"WARNING: augment_limit set to {n_query_instances}")

        idx_query_ebano = idx_candidates[:query_limit]

        # Mix EBAnO query with chosen base strategy
        if len(idx_query_ebano) < n_query_instances:  # If too few queried by EBAnO, add more by selecting among non-candidates using base_strategy
            idx_others = np.delete(np.arange(len(X_pool)), idx_query_ebano)
            n_missing = n_query_instances - len(idx_query_ebano)

            if base_strategy == "random":
                rng = np.random.default_rng(seed)
                idx_additional = rng.choice(idx_others, size=n_missing, replace=False)
            elif base_strategy == "least-confident":
                max_preds = preds.max(axis=1)
                idx_idx_others_sorted = np.argsort(max_preds[idx_others])
                idx_additional = idx_others[idx_idx_others_sorted][:n_missing]
            else:
                raise ValueError(f"Base strategy {base_strategy} not implemented.")

            idx_query = np.concatenate([idx_query_ebano, idx_additional])
        else:
            idx_query = idx_query_ebano

        if augment:  # Create augmented dataset
            print("Create augmented set")

            idx_augment = idx_candidates[:augment_limit]  # Limit number of candidates queried by EBAnO

            X_augmented_set = []
            y_augmented_set = []
            for i in idx_augment:
                x_original = X_pool[i]

                x_masks = X_masks[i]
                f_i = nPIR_max_f_i[i]
                x_perturbed = self.get_perturbed_image(x_original, x_masks, f_i, perturb_filter=explainer.perturb_filter, flip=True)

                X_augmented_set.append(x_perturbed)
                y_augmented_set.append(y_pool[i])

            X_augmented_set = np.array(X_augmented_set)
            y_augmented_set = np.array(y_augmented_set)

            self.augmented_set = (X_augmented_set, y_augmented_set)

        return idx_query

    @staticmethod
    def query_most_influential_has_low_precision(nPIR, nPIRP, eps=0.):
        """
        Select samples whose most influential interpretable feature on the class
        of interest, i.e., the feature that has highest nPIR, is not focused on
        the class of interest, i.e., it has low associated nPIRP. Only choose
        samples whose most influential interpretable feature has nPIR above 0
        (most samples satisfy this requirement).

        Returns a list of indices corresponding to queried images.
        Returns an array of shape (n,) containing the indices of the
        interpretable feature corresponding to the highest nPIR for each image.
        """

        n = len(nPIR)

        nPIR_max = np.empty(n)
        nPIRP_of_nPIR_max = np.empty(n)
        nPIR_max_f_i = np.empty(n, dtype=int)
        for i in range(n):
            idx = np.argmax(nPIR[i])
            nPIR_max[i] = nPIR[i][idx]
            nPIRP_of_nPIR_max[i] = nPIRP[i][idx]
            nPIR_max_f_i[i] = idx

        idx_candidates = []
        for i in range(n):
            if nPIR_max[i] > 0. and nPIRP_of_nPIR_max[i] < (0. + eps):
                idx_candidates.append(i)

        return idx_candidates, nPIR_max_f_i

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
