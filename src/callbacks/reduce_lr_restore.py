import numpy as np

from tensorflow.keras import backend
from tensorflow.keras.callbacks import Callback
from tensorflow.python.platform import tf_logging as logging

class ReduceLRRestoreOnPlateau(Callback):
    def __init__(self,
                 best_model_path,
                 decay_schedule=None,
                 monitor='val_loss',
                 patience=10,
                 verbose=0,
                 mode='auto',
                 min_delta=1e-4):
        super(ReduceLRRestoreOnPlateau, self).__init__()
        self.best_model_path = best_model_path
        self.decay_schedule = decay_schedule
        self.decay_counter = 0
        self.decay_times = len(decay_schedule)

        print(f"Decay schedule: {self.decay_schedule}")

        self.monitor = monitor

        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None

        self._reset()

    def _reset(self):
        if self.mode not in ['auto', 'min', 'max']:
            logging.warning('Learning rate reduction mode %s is unknown, '
                            'fallback to auto mode.', self.mode)
            self.mode = 'auto'
        if (self.mode == 'min' or
                (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf

        self.wait = 0
        self.decay_counter = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = backend.get_value(self.model.optimizer.lr)
        current = logs.get(self.monitor)

        if current is None:
            logging.warning('Reduce LR on plateau conditioned on metric `%s` '
                            'which is not available. Available metrics are: %s',
                             self.monitor, ','.join(list(logs.keys())))

        else:
            if self.decay_counter < self.decay_times:
                if self.monitor_op(current, self.best):
                    self.best = current
                    self.wait = 0

                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        old_lr = backend.get_value(self.model.optimizer.lr)
                        new_lr = np.float32(self.decay_schedule[self.decay_counter])

                        print("Reload best weights")
                        self.model.load_weights(self.best_model_path)

                        backend.set_value(self.model.optimizer.lr, new_lr)
                        if self.verbose > 0:
                            print(f"Epoch {epoch + 1}: Set learning rate from {old_lr} to {new_lr}")

                        print(f"Decay counter: {self.decay_counter+1}/{self.decay_times}")
                        self.decay_counter += 1
                        self.wait = 0

    def on_train_end(self, logs=None):
        print("Reload best weights")
        self.model.load_weights(self.best_model_path)

        super().on_train_end(logs)
