import numpy as np

from ebano.batchebano import Explainer
from qs import QueryStrategy

class EBAnOMixQueryStrategy(QueryStrategy):
    def __call__(self,
                 X_pool,
                 n_query_instances,
                 current_iter,
                 seed=None,
                 current_iteration=None,
                 switch_iteration=4,
                 switch_first_method="margin-sampling",
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
                 **ebano_kwargs):

        if current_iter < switch_iteration:
            if switch_first_method == "margin-sampling":
                from query.margin_sampling import MarginSamplingQueryStrategy
                qs = MarginSamplingQueryStrategy()
                qs.set_model(self.model, self.preprocess_input_fn)
                return qs(X_pool, n_query_instances, current_iter, seed=seed, query_batch_size=query_batch_size)
            elif switch_first_method == "random":
                from query.random import RandomQueryStrategy
                qs = RandomQueryStrategy()
                return qs(X_pool, n_query_instances, current_iter, seed=seed)
            else:
                raise ValueError(f"Unknown query method {switch_first_method}")

        # Predict
        X_pool_preprocessed = self.preprocess_input_fn(np.copy(X_pool))
        preds = self.model.predict(X_pool,
                                   batch_size=query_batch_size,
                                   verbose=1)  # (len(X_pool), n_classes)
        del X_pool_preprocessed
        cois = np.argmax(preds, axis=1)

        # Explain via BatchEBAnO
        nPIR_best = []
        nPIRP_best = []

        explainer = Explainer(
            model=self.model,
            n_classes=n_classes,
            input_shape=input_shape,
            layers_to_analyze=layers_to_analyze,
        )

        n_batches = len(X_pool) // query_batch_size + 1

        for i in range(n_batches):
            batch_len = len(X_pool[i*query_batch_size:(i+1)*query_batch_size])
            if batch_len == 0:  # skip last batch if empty
                print("Empty batch")
                pass

            print(f"Processing batch {i+1}/{n_batches} of size {batch_len}")

            nPIR_best_batch, nPIRP_best_batch = explainer.fit_batch(
                X_pool[i*query_batch_size:(i+1)*query_batch_size],
                cois=cois[i*query_batch_size:(i+1)*query_batch_size],
                preprocess_input_fn=self.preprocess_input_fn,  # data is already preprocessed
                hypercolumn_features=hypercolumn_features,
                hypercolumn_reduction=hypercolumn_reduction,
                clustering=clustering,
                min_features=min_features,
                max_features=max_features,
                display_plots=False,
                return_indices=True,
                use_gpu=False,
                seed=seed,
                niter=ebano_kwargs["niter"],
            )

            nPIR_best.extend(nPIR_best_batch)
            nPIRP_best.extend(nPIRP_best_batch)

        # Process indices for query
        # ...

        # TODO: actually implement this
        idx = np.arange(0, len(X_pool))
        idx_query = np.random.choice(idx, size=n_query_instances, replace=False)
        return idx_query
