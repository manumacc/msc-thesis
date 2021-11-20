import pathlib
import datetime
import pickle
import shutil

import tensorflow as tf
import tensorflow_addons as tfa

import al
from dataset.metadata import get_labels_by_name

class Experiment:
    def __init__(self, config):
        self.config = config

    def run(self, name, seed=None, train_base=False, query_strategy=None, **kwargs):
        self.config["name"] = name
        self.config["query_strategy"] = query_strategy
        self.config["n_classes"] = len(get_labels_by_name(self.config["dataset_name"]))

        resume_job_dir = kwargs["resume_job"]
        self.config["resume_job_dir_logs"] = resume_job_dir

        if seed is not None:
            self.config["experiment_seed"] = seed

        if kwargs["dataset_hpc"] is not None:
            self.config["dataset_name"] = f"{kwargs['dataset_hpc']}"
            self.config["dataset_path"] = f"data/{kwargs['dataset_hpc']}"
            self.config["n_classes"] = len(get_labels_by_name(self.config["dataset_name"]))

            if kwargs["dataset_hpc"] == "imagenet-25":
                self.config["n_query_instances"] = 1500
                self.config["n_loops"] = 10
                self.config["n_epochs"] = 100
                self.config["base_model_name"] = "resnet50_imagenet25_base"
                self.config["reduce_lr_patience"] = 25
            elif kwargs["dataset_hpc"] == "imagenet-100":
                self.config["n_query_instances"] = 3000
                self.config["n_loops"] = 8
                self.config["n_epochs"] = 75
                self.config["base_model_name"] = "resnet50_imagenet100_base"
                self.config["reduce_lr_patience"] = 20
            elif kwargs["dataset_hpc"] == "imagenet-250":
                self.config["n_query_instances"] = 7500
                self.config["n_loops"] = 6
                self.config["n_epochs"] = 60
                self.config["base_model_name"] = "resnet50_imagenet250_base"
                self.config["reduce_lr_patience"] = 20

            print(f"Set dataset to {self.config['dataset_name']} at {self.config['dataset_path']}")

        start_dt = datetime.datetime.now()

        # Model build
        def model_initializer(base=False):
            model = None
            if self.config["model"] == "VGG16":
                from network.vgg16 import VGG16
                print("Instantiating VGG16 model")
                model = VGG16(
                    n_classes=self.config["n_classes"],
                    dropout_rate=0.5,
                    dense_units=4096,
                    freeze_extractor=False
                )
            elif self.config["model"] == "ResNet50":
                from network.resnet50 import ResNet50
                print("Instantiating ResNet50 model")
                model = ResNet50(
                    n_classes=self.config["n_classes"],
                    freeze_extractor=False
                )

            loss_fn = tf.keras.losses.CategoricalCrossentropy()
            lr_init = self.config["lr_init"] if not base else self.config["base_lr_init"]

            weight_decay = None
            if self.config["model"] == "VGG16":
                weight_decay = 5e-4
            elif self.config["model"] == "ResNet50":
                weight_decay = 1e-4

            print(f"Weight decay set to {weight_decay}")

            optimizer = tfa.optimizers.SGDW(
                learning_rate=lr_init,
                momentum=0.9,
                weight_decay=weight_decay
            )

            return model, loss_fn, optimizer, lr_init

        model_initialization_fn = model_initializer

        preprocess_fn = None
        target_size = None
        if self.config["model"] == "VGG16":
            preprocess_fn = tf.keras.applications.vgg16.preprocess_input
            target_size = (224, 224)
        elif self.config["model"] == "ResNet50":
            preprocess_fn = tf.keras.applications.resnet50.preprocess_input
            target_size = (224, 224)

        query_strategy = None
        query_kwargs = {}
        if not train_base:
            qs = self.config["query_strategy"]  # shorthand

            # Query strategy
            if qs == "random":
                from query.random import RandomQueryStrategy
                query_strategy = RandomQueryStrategy()
            elif qs == "least-confident":
                from query.least_confident import LeastConfidentQueryStrategy
                query_strategy = LeastConfidentQueryStrategy()
                query_kwargs = {
                    "query_batch_size": self.config["query_batch_size"],
                }
            elif qs == "margin-sampling":
                from query.margin_sampling import MarginSamplingQueryStrategy
                query_strategy = MarginSamplingQueryStrategy()
                query_kwargs = {
                    "query_batch_size": self.config["query_batch_size"],
                }
            elif qs == "entropy":
                from query.entropy import EntropyQueryStrategy
                query_strategy = EntropyQueryStrategy()
                query_kwargs = {
                    "query_batch_size": self.config["query_batch_size"],
                }
            elif qs in ["early-mix",
                        "augment-early-mix",
                        "late-mix",
                        "augment-late-mix"]:
                from query.mix import MixQueryStrategy
                query_strategy = MixQueryStrategy()

                self.config["query_batch_size"] = 32  # Stay on the safe side with BatchEBAnO

                if kwargs["ebano_mix_base_strategy"] is not None:
                    ebano_mix_base_strategy = kwargs["ebano_mix_base_strategy"]
                else:
                    ebano_mix_base_strategy = self.config["ebano_mix_default_base_strategy"]

                if kwargs["ebano_mix_query_limit"] is not None:
                    ebano_mix_query_limit = kwargs["ebano_mix_query_limit"]
                else:
                    ebano_mix_query_limit = self.config["ebano_mix_default_query_limit"]

                if kwargs["ebano_mix_augment_limit"] is not None:
                    ebano_mix_augment_limit = kwargs["ebano_mix_augment_limit"]
                else:
                    ebano_mix_augment_limit = self.config["ebano_mix_default_augment_limit"]

                if kwargs["ebano_mix_min_diff"] is not None:
                    ebano_mix_min_diff = kwargs["ebano_mix_min_diff"]
                else:
                    ebano_mix_min_diff = self.config["ebano_mix_default_min_diff"]

                if kwargs["ebano_mix_subset"] is not None:
                    ebano_mix_subset = kwargs["ebano_mix_subset"]
                else:
                    ebano_mix_subset = self.config["ebano_mix_default_subset"]

                mix_iteration_methods = None
                if qs in ["early-mix", "augment-early-mix"]:
                    if kwargs["dataset_hpc"] == "imagenet-100":
                        mix_iteration_methods = {
                            0: "ebano",
                            1: "ebano",
                            2: "ebano",
                            3: "ebano",
                            4: ebano_mix_base_strategy,
                            5: ebano_mix_base_strategy,
                            6: ebano_mix_base_strategy,
                            7: ebano_mix_base_strategy,
                        }
                    elif kwargs["dataset_hpc"] == "imagenet-250":
                        mix_iteration_methods = {
                            0: "ebano",
                            1: "ebano",
                            2: "ebano",
                            3: ebano_mix_base_strategy,
                            4: ebano_mix_base_strategy,
                            5: ebano_mix_base_strategy,
                        }
                    else:
                        mix_iteration_methods = {
                            0: "ebano",
                            1: "ebano",
                            2: "ebano",
                            3: "ebano",
                            4: "ebano",
                            5: ebano_mix_base_strategy,
                            6: ebano_mix_base_strategy,
                            7: ebano_mix_base_strategy,
                            8: ebano_mix_base_strategy,
                            9: ebano_mix_base_strategy,
                        }
                elif qs in ["late-mix", "augment-late-mix"]:
                    if kwargs["dataset_hpc"] == "imagenet-100":
                        mix_iteration_methods = {
                            0: ebano_mix_base_strategy,
                            1: ebano_mix_base_strategy,
                            2: ebano_mix_base_strategy,
                            3: ebano_mix_base_strategy,
                            4: "ebano",
                            5: "ebano",
                            6: "ebano",
                            7: "ebano",
                        }
                    elif kwargs["dataset_hpc"] == "imagenet-250":
                        mix_iteration_methods = {
                            0: ebano_mix_base_strategy,
                            1: ebano_mix_base_strategy,
                            2: ebano_mix_base_strategy,
                            3: "ebano",
                            4: "ebano",
                            5: "ebano",
                        }
                    else:
                        mix_iteration_methods = {
                            0: ebano_mix_base_strategy,
                            1: ebano_mix_base_strategy,
                            2: ebano_mix_base_strategy,
                            3: ebano_mix_base_strategy,
                            4: ebano_mix_base_strategy,
                            5: "ebano",
                            6: "ebano",
                            7: "ebano",
                            8: "ebano",
                            9: "ebano",
                        }

                ebano_mix_augment = None
                if qs in ["early-mix", "late-mix"]:
                    ebano_mix_augment = False
                elif qs in ["augment-early-mix", "augment-late-mix"]:
                    ebano_mix_augment = True

                print("Mix iteration methods:", mix_iteration_methods)
                print("Augment flag:", ebano_mix_augment)
                print("ebano_mix base_strategy", ebano_mix_base_strategy)
                print("ebano_mix query_limit", ebano_mix_query_limit)
                print("ebano_mix augment_limit", ebano_mix_augment_limit)
                print("ebano_mix min_diff", ebano_mix_min_diff)
                print("ebano_mix subset", ebano_mix_subset)

                query_kwargs = {
                    "mix_iteration_methods": mix_iteration_methods,
                    "query_batch_size": self.config["query_batch_size"],
                    "n_classes": self.config["n_classes"],
                    "input_shape": target_size,
                    "layers_to_analyze": self.config["layers_to_analyze"],
                    "hypercolumn_features": self.config["hypercolumn_features"],
                    "hypercolumn_reduction": self.config["hypercolumn_reduction"],
                    "clustering": self.config["clustering"],
                    "min_features": self.config["min_features"],
                    "max_features": self.config["max_features"],
                    "use_gpu": self.config["ebano_use_gpu"],
                    "augment": ebano_mix_augment,
                    "base_strategy": ebano_mix_base_strategy,
                    "query_limit": ebano_mix_query_limit,
                    "augment_limit": ebano_mix_augment_limit,
                    "min_diff": ebano_mix_min_diff,
                    "subset": ebano_mix_subset,
                    "niter": self.config["kmeans_niter"]
                }

        print("Configuration:")
        for k, v in self.config.items():
            print(f"- {k} = {v}")

        # Initialize active learning loop
        al_loop = al.ActiveLearning(
            dataset_name=self.config["dataset_name"],
            dataset_path=self.config["dataset_path"],
            query_strategy=query_strategy,
            model_initialization_fn=model_initialization_fn,
            preprocess_input_fn=preprocess_fn,
            target_size=target_size,
            save_models=self.config["save_models"],
        )

        # Logic for training base model
        if train_base:
            callbacks = []
            from callbacks.reload_checkpoint import ReloadCheckpointOnTrainEnd

            best_model_checkpoint_path = pathlib.Path("models", "checkpoints", name, name)
            callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                best_model_checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=0
            )
            callbacks.append(callback_checkpoint)
            callback_reload_checkpoint = ReloadCheckpointOnTrainEnd(
                best_model_checkpoint_path
            )
            callbacks.append(callback_reload_checkpoint)
            callback_decay = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=self.config["base_reduce_lr_patience"],
                min_delta=self.config["base_reduce_lr_min_delta"],
                min_lr=self.config["base_reduce_lr_min"],
                verbose=0,
            )
            callbacks.append(callback_decay)

            logs = al_loop.train_base(
                model_name=name,
                batch_size=self.config["batch_size"],
                n_epochs=self.config["base_n_epochs"],
                callbacks=callbacks,
                seed=self.config["experiment_seed"],
            )

            with open(pathlib.Path("models", name, "config.pkl"), "wb") as f:
                pickle.dump(self.config, f)
            with open(pathlib.Path("models", name, "logs.pkl"), "wb") as f:
                pickle.dump(logs, f)

        # Logic for training AL iterations
        else:
            dir_logs = f"{name}_{start_dt.strftime('%Y%m%d_%H%M%S')}"

            print(f"dir_logs (argument for resuming): {dir_logs}")

            logs_path = pathlib.Path("logs", dir_logs)
            logs_path.mkdir(parents=True, exist_ok=False)
            logs_partial_path = pathlib.Path(logs_path, "partial")
            logs_partial_path.mkdir(parents=True, exist_ok=False)

            callbacks = []
            from callbacks.reload_checkpoint import ReloadCheckpointOnTrainEnd

            best_model_checkpoint_path = pathlib.Path("models", "checkpoints", dir_logs, name)
            callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                best_model_checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=0
            )
            callbacks.append(callback_checkpoint)
            callback_reload_checkpoint = ReloadCheckpointOnTrainEnd(
                best_model_checkpoint_path
            )
            callbacks.append(callback_reload_checkpoint)
            callback_decay = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=self.config["reduce_lr_patience"],
                min_delta=self.config["reduce_lr_min_delta"],
                min_lr=self.config["reduce_lr_min"],
                verbose=0,
            )
            callbacks.append(callback_decay)

            print("Saving configuration")
            with open(pathlib.Path(logs_path, "config.pkl"), "wb") as f:
                pickle.dump(self.config, f)

            logs = al_loop.learn(
                n_loops=self.config["n_loops"],
                n_query_instances=self.config["n_query_instances"],
                batch_size=self.config["batch_size"],
                n_epochs=self.config["n_epochs"],
                callbacks=callbacks,
                base_model_name=self.config["base_model_name"],
                resume_job_dir=resume_job_dir,
                logs_path=logs_path,
                logs_partial_path=logs_partial_path,
                seed=self.config["experiment_seed"],
                **query_kwargs,
            )

            with open(pathlib.Path(logs_path, "logs.pkl"), "wb") as f:
                pickle.dump(logs, f)

            # Delete partials dir
            print("Delete partials directory")
            shutil.rmtree(logs_partial_path, ignore_errors=True)
