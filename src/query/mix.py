import numpy as np

from qs import QueryStrategy

class MixQueryStrategy(QueryStrategy):
    def __call__(self,
                 X_pool,
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
                 min_features=2,
                 max_features=5,
                 use_gpu=False,
                 **ebano_kwargs):

        current_method = mix_iteration_methods[current_iter]

        if current_method == "margin-sampling":
            from query.margin_sampling import MarginSamplingQueryStrategy
            qs = MarginSamplingQueryStrategy()
            qs.set_model(self.model, self.preprocess_input_fn)
            return qs(X_pool, n_query_instances, current_iter, seed=seed, query_batch_size=query_batch_size)
        elif current_method == "entropy":
            from query.entropy import EntropyQueryStrategy
            qs = EntropyQueryStrategy()
            qs.set_model(self.model, self.preprocess_input_fn)
            return qs(X_pool, n_query_instances, current_iter, seed=seed, query_batch_size=query_batch_size)
        elif current_method == "random":
            from query.random import RandomQueryStrategy
            qs = RandomQueryStrategy()
            return qs(X_pool, n_query_instances, current_iter, seed=seed)
        elif current_method == "ebano":
            from query.ebano_query import EBAnOQueryStrategy
            qs = EBAnOQueryStrategy()
            qs.set_model(self.model, self.preprocess_input_fn)
            return qs(X_pool,
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
                      min_features=min_features,
                      max_features=max_features,
                      use_gpu=use_gpu,
                      **ebano_kwargs)
        else:
            raise ValueError(f"Unknown query method {current_method}")
