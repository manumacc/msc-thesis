import pathlib
import datetime
import pickle

from timeit import default_timer

import tensorflow as tf
import tensorflow_addons as tfa

import al

class Experiment:
    def __init__(self, config):
        self.config = config

    def run(self, name, query_strategy=None, train_base=False):
        self.config["name"] = name
        self.config["query_strategy"] = query_strategy

        start_dt = datetime.datetime.now()

        print("Configuration:")
        for k, v in self.config.items():
            print(f"- {k} = {v}")

        # Model build
        def model_initializer(base=False):
            model = None
            if self.config["model"] == "VGG16":
                from network.vgg16 import VGG16
                print("Instantiating VGG16 model")
                model = VGG16(
                    n_classes=self.config["n_classes"],
                    dropout_rate=self.config["fc_dropout_rate"],
                    dense_units=self.config["dense_units"],
                    freeze_extractor=self.config["freeze_extractor"]
                )
            if self.config["model"] == "ResNet50":
                from network.resnet50 import ResNet50
                print("Instantiating ResNet50 model")
                model = ResNet50(
                    n_classes=self.config["n_classes"],
                    freeze_extractor=self.config["freeze_extractor"]
                )
            if self.config["model"] == "SimpleCNN":
                from network.simplecnn import SimpleCNN
                print("Instantiating SimpleCNN model")
                model = SimpleCNN(
                    n_classes=self.config["n_classes"]
                )

            loss_fn = None
            if self.config["loss"] == "categorical_crossentropy":
                loss_fn = tf.keras.losses.CategoricalCrossentropy()

            optimizer = None
            if self.config["optimizer"] == "SGDW":
                optimizer = tfa.optimizers.SGDW(
                    learning_rate=self.config["lr_init"] if not base else self.config["base_lr_init"],
                    momentum=self.config["momentum"],
                    weight_decay=self.config["weight_decay"] if not base else self.config["base_weight_decay"]
                )
            if self.config["optimizer"] == "RMSprop":
                optimizer = tf.keras.optimizers.RMSprop(
                    learning_rate=self.config["lr_init"] if not base else self.config["base_lr_init"],
                    decay=self.config["weight_decay"] if not base else self.config["base_weight_decay"]
                )

            return model, loss_fn, optimizer

        model_initialization_fn = model_initializer

        preprocess_fn = None
        target_size = None
        if self.config["model"] == "VGG16":
            preprocess_fn = tf.keras.applications.vgg16.preprocess_input
            target_size = (224, 224)
        if self.config["model"] == "ResNet50":
            preprocess_fn = tf.keras.applications.resnet50.preprocess_input
            target_size = (224, 224)
        if self.config["model"] == "SimpleCNN":
            preprocess_fn = lambda x: x / x.max()
            target_size = (32, 32)

        query_strategy = None
        query_kwargs = {}
        if not train_base:
            # Query strategy
            if self.config["query_strategy"] == "random":
                from query.random import RandomQueryStrategy
                query_strategy = RandomQueryStrategy()
            elif self.config["query_strategy"] == "least-confident":
                from query.least_confident import LeastConfidentQueryStrategy
                query_strategy = LeastConfidentQueryStrategy()
                query_kwargs = {
                    "query_batch_size": self.config["query_batch_size"],
                }
            elif self.config["query_strategy"] == "margin-sampling":
                from query.margin_sampling import MarginSamplingQueryStrategy
                query_strategy = MarginSamplingQueryStrategy()
                query_kwargs = {
                    "query_batch_size": self.config["query_batch_size"],
                }
            elif self.config["query_strategy"] == "entropy":
                from query.entropy import EntropyQueryStrategy
                query_strategy = EntropyQueryStrategy()
                query_kwargs = {
                    "query_batch_size": self.config["query_batch_size"],
                }

        # Initialize active learning loop
        al_loop = al.ActiveLearning(
            path_train=self.config["data_path_train"],
            path_test=self.config["data_path_test"],
            query_strategy=query_strategy,
            model_initialization_fn=model_initialization_fn,
            preprocess_input_fn=preprocess_fn,
            target_size=target_size,
            class_sample_size_train=self.config["class_sample_size_train"],
            class_sample_size_test=self.config["class_sample_size_test"],
            init_size=self.config["base_init_size"],
            val_size=self.config["val_size"],
            dataset=self.config["dataset"],
            save_models=self.config["save_models"],
            dataset_seed=self.config["dataset_seed"],
        )

        if train_base:
            callbacks = []
            if "reduce_lr_restore_on_plateau" in self.config["base_callbacks"]:
                from callbacks.reduce_lr_restore import ReduceLRRestoreOnPlateau

                best_model_checkpoint_path = pathlib.Path("models", "checkpoints", name, name)
                callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                    best_model_checkpoint_path,
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=True,
                    verbose=0,
                )
                callbacks.append(callback_checkpoint)
                callback_decay = ReduceLRRestoreOnPlateau(
                    best_model_path=best_model_checkpoint_path,
                    monitor='val_loss',
                    factor=self.config["base_reduce_lr_factor"],
                    patience=self.config["base_reduce_lr_patience"],
                    cooldown=self.config["base_reduce_lr_cooldown"],
                    min_delta=self.config["base_reduce_lr_min_delta"],
                    decay_times=self.config["base_reduce_lr_decay_times"],
                    verbose=1,
                )
                callbacks.append(callback_decay)

            start = default_timer()
            logs = al_loop.train_base(
                model_name=name,
                batch_size=self.config["batch_size"],
                n_epochs=self.config["base_n_epochs"],
                callbacks=callbacks,
                seed=self.config["experiment_seed"],
            )
            end = default_timer()

            with open(pathlib.Path("models", name, "config.pkl"), "wb") as f:
                pickle.dump(self.config, f)
            with open(pathlib.Path("models", name, "logs.pkl"), "wb") as f:
                pickle.dump(logs, f)
            with open(pathlib.Path("models", name, "stats.txt"), "w") as f:
                f.write(f"Elapsed time (s): {end - start}")

        else:
            dir_logs = f"{name}_{start_dt.strftime('%Y%m%d_%H%M%S')}"

            callbacks = []
            if "reduce_lr_restore_on_plateau" in self.config["callbacks"]:
                from callbacks.reduce_lr_restore import ReduceLRRestoreOnPlateau

                best_model_checkpoint_path = pathlib.Path("models", "checkpoints", dir_logs, name)
                callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                    best_model_checkpoint_path,
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=True,
                    verbose=0,
                )
                callbacks.append(callback_checkpoint)
                callback_decay = ReduceLRRestoreOnPlateau(
                    best_model_path=best_model_checkpoint_path,
                    monitor='val_loss',
                    factor=self.config["reduce_lr_factor"],
                    patience=self.config["reduce_lr_patience"],
                    cooldown=self.config["reduce_lr_cooldown"],
                    min_delta=self.config["reduce_lr_min_delta"],
                    decay_times=self.config["reduce_lr_decay_times"],
                    verbose=1,
                )
                callbacks.append(callback_decay)

            start = default_timer()
            logs = al_loop.learn(
                n_loops=self.config["n_loops"],
                n_query_instances=self.config["n_query_instances"],
                batch_size=self.config["batch_size"],
                n_epochs=self.config["n_epochs"],
                callbacks=callbacks,
                base_model_name=self.config["base_model_name"],
                dir_logs=dir_logs,
                seed=self.config["experiment_seed"],
                **query_kwargs,
            )
            end = default_timer()

            path_logs = pathlib.Path("logs", dir_logs)
            with open(pathlib.Path(path_logs, "config.pkl"), "wb") as f:
                pickle.dump(self.config, f)
            with open(pathlib.Path(path_logs, "logs.pkl"), "wb") as f:
                pickle.dump(logs, f)
            with open(pathlib.Path(path_logs, "stats.txt"), "w") as f:
                f.write(f"Elapsed time (s): {end - start}")
