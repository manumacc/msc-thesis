class QueryStrategy:
    def __init__(self, model, preprocess_input_fn=None):
        self.model = model
        self.preprocess_input_fn = preprocess_input_fn

    def __call__(self, X_pool, n_query_instances, seed=None):
        raise NotImplementedError("Can't call a base class")
