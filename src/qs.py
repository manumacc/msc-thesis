class QueryStrategy:
    def __init__(self, preprocess_input_fn=None):
        self.preprocess_input_fn = preprocess_input_fn

    def __call__(self, X_pool, metadata, n_query_instances, model, seed=None):
        raise NotImplementedError("Can't call a base class")
