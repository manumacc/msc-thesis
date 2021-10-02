config_dict = {
    ## Logging
    "save_logs": True,
    "save_models": False,

    ## Seed
    "dataset_seed": 0,
    "experiment_seed": 42,

    ## Dataset
    "data_path_train": "data/imagenette2/train",
    "data_path_test": "data/imagenette2/val",

    "n_classes": 10,

    "class_sample_size_train": 800,
    "class_sample_size_test": 300,

    ## Model
    "model": "ResNet50",
    "freeze_extractor": False,  # if True, load ImageNet weights for feature extractor

    "optimizer": "SGDW",
    "loss": "categorical_crossentropy",

    "batch_size": 256,

    # VGG16
    "fc_dropout_rate": 0.5,
    "dense_units": 4096,

    ## Active learning
    "n_loops": 10,

    "val_size": 0.1,  # With respect to current training set

    "lr_init": 0.1,  # VGG16: 1e-2; ResNet: 0.1
    "momentum": 0.9,
    "weight_decay": 1e-4,  # VGG16: 5e-4; ResNet: 1e-4
    "n_epochs": 100,

    "callbacks": [
        "decay_early_stopping"
    ],

    "decay_early_stopping_patience": 10,
    "decay_early_stopping_times": 3,
    "decay_early_stopping_min_delta": 0.1,  # Empirically set for ResNet
    "decay_early_stopping_restore_best_weights": False,

    # Query strategy arguments
    "n_query_instances": 256,  # Number of instances to add at each iteration
    "query_batch_size": 256,  # Batch size for unlabeled pool iterator

    # Base model
    "base_model_name": "resnet-1",

    "base_init_size": 0.5,  # With respect to whole training set

    "base_lr_init": 0.1,
    "base_weight_decay": 1e-4,
    "base_n_epochs": 1000,

    "base_callbacks": [
        "decay_early_stopping"
    ],

    "base_decay_early_stopping_patience": 20,
    "base_decay_early_stopping_times": 3,
    "base_decay_early_stopping_min_delta": 0.1,  # Empirically set for ResNet
    "base_decay_early_stopping_restore_best_weights": False,
}
