import os
import datetime
import pickle

from timeit import default_timer

import tensorflow as tf
import tensorflow_addons as tfa

import al

class Experiment:
    def __init__(self, config):
        self.config = config

    def run(self, name):
        start = default_timer()
        start_dt = datetime.datetime.now()

        # Model build
        model = None
        preprocess_fn = None
        target_size = None
        if self.config["model"] == "VGG16":
            from network.vgg16 import VGG16
            model = VGG16(n_classes=self.config["n_classes"],
                          dropout_rate=self.config["fc_dropout_rate"],
                          dense_units=self.config["dense_units"],
                          load_imagenet=self.config["load_imagenet_weights"],
                          feature_extractor_trainable=self.config["feature_extractor_trainable"])

            preprocess_fn = tf.keras.applications.vgg16.preprocess_input

            target_size = (224, 224)

        loss_fn = None
        if self.config["loss"] == "categorical_crossentropy":
            loss_fn = tf.keras.losses.CategoricalCrossentropy()

        optimizer = None
        if self.config["optimizer"] == "SGDW":
            optimizer = tfa.optimizers.SGDW(learning_rate=self.config["lr_init"],
                                            momentum=self.config["momentum"],
                                            weight_decay=self.config["weight_decay"])

        callbacks = []
        if "learning_rate_decay_early_stopping" in self.config["callbacks"]:
            from callbacks.learning_rate_decay_early_stopping import LearningRateDecayEarlyStopping
            callback = LearningRateDecayEarlyStopping(patience=3,
                                                      n_decay=3,
                                                      verbose=1)
            callbacks.append(callback)

        model.compile(optimizer=optimizer,
                      loss=loss_fn,
                      metrics=["accuracy"])

        # Query strategy
        query_strategy = None
        if self.config["query_strategy"] == "random":
            from query.random import RandomQueryStrategy
            query_strategy = RandomQueryStrategy(model)
        elif self.config["query_strategy"] == "least_confident":
            from query.least_confident import LeastConfidentQueryStrategy
            query_strategy = LeastConfidentQueryStrategy(model)

        # Active learning loop
        al_loop = al.ActiveLearning(
            path_train=self.config["data_path_train"],
            path_test=self.config["data_path_test"],
            query_strategy=query_strategy,
            model=model,
            preprocess_input_fn=preprocess_fn,
            batch_size=self.config["batch_size"],
            target_size=target_size,
            class_sample_size_train=self.config["class_sample_size_train"],
            class_sample_size_test=self.config["class_sample_size_test"],
            init_size=self.config["init_size"],
            val_size=self.config["val_size"],
            seed=self.config["seed"],
            model_callbacks=callbacks
        )

        al_loop.learn(
            n_loops=self.config["n_loops"],
            n_query_instances=self.config["n_query_instances"],
            n_epochs=self.config["n_epochs"],
            seed=self.config["seed"],
            require_raw_pool=self.config["require_raw_pool"]
        )

        end = default_timer()

        # TODO: better logging, redirect stdout to file via python
        #  see https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python
        logs = {
            "config": self.config,
            "al": al_loop.logs,
            "elapsed_s": end - start
        }

        log_path = "logs"
        fname = f"LOGS_{name}_{start_dt.strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(os.path.join(log_path, fname), "wb") as f:
            pickle.dump(logs, f)
