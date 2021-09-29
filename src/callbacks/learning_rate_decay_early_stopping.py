import numpy as np
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend

class LearningRateDecayEarlyStopping(Callback):
    """Learning rate decay with early stopping.

    Decays the learning rate `n_decay` times by a factor of 10 when the model
    does not get any better for `patience` epochs. After `n_decay` times, stops
    training.

    """

    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 n_decay=0,
                 restore_best_weights=False,
                 verbose=1):
        super(LearningRateDecayEarlyStopping, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose

        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0

        self.n_decay = n_decay
        self.current_decay_level = 0

        self.best = np.Inf

        self.monitor_op = np.less

        self.restore_best_weights = restore_best_weights
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.current_decay_level = 0
        self.best = np.Inf
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        current = self.get_monitor_value(logs)
        if current is None:
            return

        if self.restore_best_weights and self.best_weights is None and epoch > 0:
            # Restore the weights after first epoch if no progress is ever made.
            self.best_weights = self.model.get_weights()

        self.wait += 1

        if self._is_improvement(current, self.best) and epoch > 0:  # Ignore first epoch
            self.best = current
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            self.wait = 0

        if self.wait >= self.patience and epoch > 0:
            if self.current_decay_level >= self.n_decay:
                self.model.stop_training = True
                self.stopped_epoch = epoch
            else:
                self.wait = 0
                self.current_decay_level += 1
                lr = float(backend.get_value(self.model.optimizer.lr))
                lr = lr * 0.1
                backend.set_value(self.model.optimizer.lr, backend.get_value(lr))
                if self.verbose > 0:
                    print(f"Epoch {epoch + 1}: learning rate decayed")

            if self.restore_best_weights and self.best_weights is not None:
                if self.verbose > 0:
                    print('Restoring model weights from the end of the best epoch.')
                self.model.set_weights(self.best_weights)

        logs['lr'] = backend.get_value(self.model.optimizer.lr)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f"Epoch {self.stopped_epoch + 1}: early stopping")

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            print('WARNING: Early stopping conditioned on metric `%s` '
                  'which is not available. Available metrics are: %s',
                  self.monitor, ','.join(list(logs.keys())))
        return monitor_value

    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value - self.min_delta, reference_value)
