import numpy as np

from qs import QueryStrategy

class MixQueryStrategy(QueryStrategy):
    def __call__(self,
                 ds_pool,
                 ds_train,
                 metadata,
                 n_query_instances,
                 current_iter,
                 seed=None,
                 mix_iteration_methods=None,
                 query_batch_size=32,
                 n_classes=None,
                 input_shape=None,
                 layers_to_analyze=None,
                 hypercolumn_features=None,
                 hypercolumn_reduction=None,
                 clustering=None,
                 k_features=(3,5),
                 use_gpu=False,
                 augment=False,
                 base_strategy=None,
                 query_limit=None,
                 augment_limit=None,
                 min_diff=None,
                 subset=None,
                 **ebano_kwargs):

        current_method = mix_iteration_methods[current_iter]
        self.ds_augment = None

        if current_method == "margin-sampling":
            from query.margin_sampling import MarginSamplingQueryStrategy
            qs = MarginSamplingQueryStrategy()
            qs.set_model(self.model, self.preprocess_input_fn)
            return qs(ds_pool, ds_train, metadata, n_query_instances, current_iter, seed=seed, query_batch_size=query_batch_size)
        elif current_method == "entropy":
            from query.entropy import EntropyQueryStrategy
            qs = EntropyQueryStrategy()
            qs.set_model(self.model, self.preprocess_input_fn)
            return qs(ds_pool, ds_train, metadata, n_query_instances, current_iter, seed=seed, query_batch_size=query_batch_size)
        elif current_method == "least-confident":
            from query.least_confident import LeastConfidentQueryStrategy
            qs = LeastConfidentQueryStrategy()
            qs.set_model(self.model, self.preprocess_input_fn)
            return qs(ds_pool, ds_train, metadata, n_query_instances, current_iter, seed=seed, query_batch_size=query_batch_size)
        elif current_method == "random":
            from query.random import RandomQueryStrategy
            qs = RandomQueryStrategy()
            return qs(ds_pool, ds_train, metadata, n_query_instances, current_iter, seed=seed)
        elif current_method == "ebano":
            from query.ebano_query import EBAnOQueryStrategy
            qs = EBAnOQueryStrategy()
            qs.set_model(self.model, self.preprocess_input_fn)
            idx_query = qs(ds_pool,
                           ds_train,
                           metadata,
                           n_query_instances,
                           current_iter,
                           seed=seed,
                           query_batch_size=query_batch_size,
                           n_classes=n_classes,
                           input_shape=input_shape,
                           layers_to_analyze=layers_to_analyze,
                           hypercolumn_features=hypercolumn_features,
                           hypercolumn_reduction=hypercolumn_reduction,
                           clustering=clustering,
                           k_features=k_features,
                           use_gpu=use_gpu,
                           augment=augment,
                           base_strategy=base_strategy,
                           query_limit=query_limit,
                           augment_limit=augment_limit,
                           min_diff=min_diff,
                           subset=subset,
                           **ebano_kwargs)
            self.ds_augment = qs.get_ds_augment()
            return idx_query
        else:
            raise ValueError(f"Unknown query method {current_method}")
