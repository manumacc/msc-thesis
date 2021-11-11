from tensorflow.keras.callbacks import Callback


class ReloadCheckpointOnTrainEnd(Callback):
    def __init__(self,
                 best_model_path):
        super(ReloadCheckpointOnTrainEnd, self).__init__()
        self.best_model_path = best_model_path

    def on_train_end(self, logs=None):
        print("Reload best weights")
        self.model.load_weights(self.best_model_path)
        super().on_train_end(logs)
