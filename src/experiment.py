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
                model = VGG16(n_classes=self.config["n_classes"],
                              dropout_rate=self.config["fc_dropout_rate"],
                              dense_units=self.config["dense_units"],
                              freeze_extractor=self.config["freeze_extractor"])
            if self.config["model"] == "ResNet50":
                from network.resnet50 import ResNet50
                print("Instantiating ResNet50 model")
                model = ResNet50(n_classes=self.config["n_classes"],
                                 freeze_extractor=self.config["freeze_extractor"])

            loss_fn = None
            if self.config["loss"] == "categorical_crossentropy":
                loss_fn = tf.keras.losses.CategoricalCrossentropy()

            optimizer = None
            if self.config["optimizer"] == "SGDW":
                optimizer = tfa.optimizers.SGDW(learning_rate=self.config["lr_init"] if not base else self.config["base_lr_init"],
                                                momentum=self.config["momentum"],
                                                weight_decay=self.config["weight_decay"] if not base else self.config["base_weight_decay"])

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

        callbacks = []
        query_strategy = None
        query_kwargs = {}

        if train_base:
            if "decay_early_stopping" in self.config["base_callbacks"]:
                from callbacks.learning_rate_decay_early_stopping import LearningRateDecayEarlyStopping
                callback = LearningRateDecayEarlyStopping(patience=self.config["base_decay_early_stopping_patience"],
                                                          n_decay=self.config["base_decay_early_stopping_times"],
                                                          min_delta=self.config["base_decay_early_stopping_min_delta"],
                                                          restore_best_weights=self.config["base_decay_early_stopping_restore_best_weights"],
                                                          verbose=1)
                callbacks.append(callback)

        else:
            if "decay_early_stopping" in self.config["callbacks"]:
                from callbacks.learning_rate_decay_early_stopping import LearningRateDecayEarlyStopping
                callback = LearningRateDecayEarlyStopping(patience=self.config["decay_early_stopping_patience"],
                                                          n_decay=self.config["decay_early_stopping_times"],
                                                          min_delta=self.config["decay_early_stopping_min_delta"],
                                                          restore_best_weights=self.config[
                                                              "decay_early_stopping_restore_best_weights"],
                                                          verbose=1)
                callbacks.append(callback)

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
            model_callbacks=callbacks,
            save_models=self.config["save_models"],
            dataset_seed=self.config["dataset_seed"],
        )

        if train_base:
            start = default_timer()
            logs = al_loop.train_base(
                model_name=name,
                batch_size=self.config["batch_size"],
                n_epochs=self.config["base_n_epochs"],
                seed=self.config["experiment_seed"],
            )
            end = default_timer()

            with open(pathlib.Path("models", name, "logs.pkl"), "wb") as f:
                pickle.dump(logs, f)
            with open(pathlib.Path("models", name, "stats.txt"), "w") as f:
                f.write(f"Elapsed time (s): {end - start}")

        else:
            dir_logs = f"{name}_{start_dt.strftime('%Y%m%d_%H%M%S')}"

            start = default_timer()
            logs = al_loop.learn(
                n_loops=self.config["n_loops"],
                n_query_instances=self.config["n_query_instances"],
                batch_size=self.config["batch_size"],
                n_epochs=self.config["n_epochs"],
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
