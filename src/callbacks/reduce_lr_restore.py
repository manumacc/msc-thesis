from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.platform import tf_logging as logging

class ReduceLRRestoreOnPlateau(ReduceLROnPlateau):
    def __init__(self, best_model_path, decay_times, *args, **kwargs):
        super(ReduceLRRestoreOnPlateau, self).__init__(*args, **kwargs)
        self.best_model_path = best_model_path
        self.decay_times = decay_times
        self.decay_counter = 0

    def _reset(self):
        self.decay_counter = 0

        super()._reset()

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)

        if current is None:
            logging.warning('Reduce LR on plateau conditioned on metric `%s` '
                            'which is not available. Available metrics are: %s',
                             self.monitor, ','.join(list(logs.keys())))

        if self.decay_counter < self.decay_times:
            if not self.monitor_op(current, self.best):
                if not self.in_cooldown():
                    if self.wait+1 >= self.patience:
                        print("Reload best weights")
                        self.model.load_weights(self.best_model_path)
                        print(f"Decay counter: {self.decay_counter+1}/{self.decay_times}")
                        self.decay_counter += 1

            super().on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        print("Reload best weights")
        self.model.load_weights(self.best_model_path)

        super().on_train_end(logs)
