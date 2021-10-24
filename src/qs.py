class QueryStrategy:
    def __init__(self):
        self.model = None
        self.preprocess_input_fn = None
        self.augmented_set = None

    def set_model(self, model, preprocess_input_fn=None):
        self.model = model
        self.preprocess_input_fn = preprocess_input_fn

    def __call__(self, X_pool, y_pool, n_query_instances, current_iter, seed=None):
        raise NotImplementedError("Can't call a base class")

    def get_augmented(self):
        return self.augmented_set
