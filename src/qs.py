class QueryStrategy:
    def __init__(self, model):
        self.model = model

    def __call__(self, X_pool, n_query_instances):
        raise NotImplementedError("Can't call a base class")