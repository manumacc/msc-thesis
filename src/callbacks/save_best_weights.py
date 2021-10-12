import numpy as np
from tensorflow.keras.callbacks import Callback

class SaveBestWeights(Callback):
    def __init__(self,
                 monitor='val_loss',
                 verbose=1):
        super(SaveBestWeights, self).__init__()
        self.monitor = monitor
        self.monitor_op = np.less

        self.verbose = verbose

        self.best = np.Inf
        self.best_epoch = None
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.best = np.Inf
        self.best_epoch = None
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        if self.best_weights is None:
            # Restore the weights from first epoch if no progress is ever done
            self.best_weights = self.model.get_weights()

        if self._is_improvement(current, self.best) and epoch > 0:  # Ignore first epoch
            self.best = current
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.verbose > 0:
            print(f"Restoring model weights from the end of the best epoch (epoch {self.best_epoch+1}).")
        self.model.set_weights(self.best_weights)

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            print('WARNING: Early stopping conditioned on metric `%s` '
                  'which is not available. Available metrics are: %s',
                  self.monitor, ','.join(list(logs.keys())))
        return monitor_value

    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value, reference_value)
