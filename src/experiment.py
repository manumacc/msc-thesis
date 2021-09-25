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

    def run(self, name, query_strategy):
        self.config["name"] = name
        self.config["query_strategy"] = query_strategy

        print("CONFIGURATION:")
        for k, v in self.config.items():
            print(f"- {k} = {v}")

        start = default_timer()
        start_dt = datetime.datetime.now()

        # Model build
        def model_init():
            model = None
            if self.config["model"] == "VGG16":
                from network.vgg16 import VGG16
                print("Instantiating VGG16 model")
                model = VGG16(n_classes=self.config["n_classes"],
                              dropout_rate=self.config["fc_dropout_rate"],
                              dense_units=self.config["dense_units"],
                              load_imagenet=self.config["load_imagenet_weights"],
                              feature_extractor_trainable=self.config["feature_extractor_trainable"])

            loss_fn = None
            if self.config["loss"] == "categorical_crossentropy":
                loss_fn = tf.keras.losses.CategoricalCrossentropy()

            optimizer = None
            if self.config["optimizer"] == "SGDW":
                optimizer = tfa.optimizers.SGDW(learning_rate=self.config["lr_init"],
                                                momentum=self.config["momentum"],
                                                weight_decay=self.config["weight_decay"])

            print("Compiling model")
            model.compile(optimizer=optimizer,
                          loss=loss_fn,
                          metrics=["accuracy"])

            return model

        model_initialization_fn = model_init

        preprocess_fn = None
        target_size = None
        if self.config["model"] == "VGG16":
            preprocess_fn = tf.keras.applications.vgg16.preprocess_input
            target_size = (224, 224)

        callbacks = []
        if "decay_early_stopping" in self.config["callbacks"]:
            from callbacks.learning_rate_decay_early_stopping import LearningRateDecayEarlyStopping
            callback = LearningRateDecayEarlyStopping(patience=self.config["decay_early_stopping_patience"],
                                                      n_decay=self.config["decay_early_stopping_times"],
                                                      verbose=1)
            callbacks.append(callback)

        # Query strategy
        query_strategy = None
        query_kwargs = {}
        if self.config["query_strategy"] == "random":
            from query.random import RandomQueryStrategy
            query_strategy = RandomQueryStrategy()
        elif self.config["query_strategy"] == "least-confident":
            from query.least_confident import LeastConfidentQueryStrategy
            query_strategy = LeastConfidentQueryStrategy(preprocess_input_fn=preprocess_fn)
            query_kwargs = {
                "query_batch_size": self.config["query_batch_size"],
            }
        elif self.config["query_strategy"] == "margin-sampling":
            from query.margin_sampling import MarginSamplingQueryStrategy
            query_strategy = MarginSamplingQueryStrategy(preprocess_input_fn=preprocess_fn)
            query_kwargs = {
                "query_batch_size": self.config["query_batch_size"],
            }
        elif self.config["query_strategy"] == "entropy":
            from query.entropy import EntropyQueryStrategy
            query_strategy = EntropyQueryStrategy(preprocess_input_fn=preprocess_fn)
            query_kwargs = {
                "query_batch_size": self.config["query_batch_size"],
            }

        # Active learning loop
        if self.config["save_logs"]:
            path_logs = pathlib.Path("logs", f"{name}_{start_dt.strftime('%Y%m%d_%H%M%S')}")
            path_logs.mkdir(parents=True, exist_ok=False)
        else:
            path_logs = None

        al_loop = al.ActiveLearning(
            path_train=self.config["data_path_train"],
            path_test=self.config["data_path_test"],
            query_strategy=query_strategy,
            model_initialization_fn=model_initialization_fn,
            preprocess_input_fn=preprocess_fn,
            target_size=target_size,
            class_sample_size_train=self.config["class_sample_size_train"],
            class_sample_size_test=self.config["class_sample_size_test"],
            init_size=self.config["init_size"],
            val_size=self.config["val_size"],
            seed=self.config["seed"],
            model_callbacks=callbacks,
            path_logs=path_logs,
            save_models=self.config["save_models"]
        )

        al_loop.learn(
            n_loops=self.config["n_loops"],
            n_query_instances=self.config["n_query_instances"],
            batch_size=self.config["batch_size"],
            n_epochs=self.config["n_epochs"],
            seed=self.config["seed"],
            **query_kwargs,
        )

        end = default_timer()

        if self.config["save_logs"]:
            with open(pathlib.Path(path_logs, "config.pkl"), "wb") as f:
                pickle.dump(self.config, f)
            with open(pathlib.Path(path_logs, "al_logs.pkl"), "wb") as f:
                pickle.dump(al_loop.al_logs, f)
            with open(pathlib.Path(path_logs, "stats.txt"), "w") as f:
                f.write(f"Elapsed time (s): {end - start}")
