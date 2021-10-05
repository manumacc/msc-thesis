config_dict = {
    ## Logging
    "save_models": False,

    ## Seed
    "dataset_seed": 0,
    "experiment_seed": 42,

    ## Dataset
    "n_classes": 10,

    # Built-in dataset name
    "builtin_dataset": None,  # "cifar-10"

    # If builtin_dataset is None, accept data from directory
    "data_path_train": "data/imagenette2/train",
    "data_path_test": "data/imagenette2/val",
    "class_sample_size_train": 800,
    "class_sample_size_test": 300,

    ## Model
    "model": "",  # SimpleCNN; ResNet50; VGG16

    "freeze_extractor": False,  # if True, load ImageNet weights for feature extractor (where available)

    "optimizer": "SGDW",  # SGDW; RMSprop
    "loss": "categorical_crossentropy",

    "batch_size": 128,

    # VGG16
    "fc_dropout_rate": 0.5,
    "dense_units": 4096,

    ## Active learning
    "n_loops": 10,

    "val_size": 0.1,  # With respect to initial training subset

    "lr_init": 0.01,  # VGG16: 1e-2; ResNet: 0.1; SimpleCNN w/ RMSprop 0.0001
    "momentum": 0.9,
    "weight_decay": 1e-6,  # VGG16: 5e-4; ResNet: 1e-4; SimpleCNN w/ RMSprop 1e-6
    "n_epochs": 100,

    "callbacks": [
        "decay_early_stopping"
    ],

    "decay_early_stopping_patience": 10,
    "decay_early_stopping_times": 3,
    "decay_early_stopping_min_delta": 0.01,  # Empirically set for ResNet in AL loop
    "decay_early_stopping_restore_best_weights": True,

    # Query strategy arguments
    "n_query_instances": 256,  # Number of instances to add at each iteration
    "query_batch_size": 128,  # Batch size for unlabeled pool iterator

    ## Base model
    "base_model_name": "",
    "base_init_size": 0.2,  # With respect to whole training set

    # Base model training
    "base_lr_init": 0.1,
    "base_weight_decay": 1e-4,
    "base_n_epochs": 1000,

    "base_callbacks": [
        "decay_early_stopping"
    ],

    "base_decay_early_stopping_patience": 50,
    "base_decay_early_stopping_times": 3,
    "base_decay_early_stopping_min_delta": 0.1,  # Empirically set for ResNet
    "base_decay_early_stopping_restore_best_weights": False,
}
