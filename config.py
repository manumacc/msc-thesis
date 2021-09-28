config_dict = {
    ## Logging
    "save_logs": True,
    "save_models": False,

    ## Seed
    "seed": 123,

    ## Dataset
    "data_path_train": "data/imagenette2/train",
    "data_path_test": "data/imagenette2/val",

    "n_classes": 10,

    "class_sample_size_train": 800,  # imagenette limit: 800
    "class_sample_size_test": 300,  # imagenette limit: 300

    ## Model
    "model": "VGG16",
    "freeze_extractor": False,  # if True, load ImageNet weights for feature extractor

    "fc_dropout_rate": 0.5,
    "dense_units": 4096,

    ## Learning
    "optimizer": "SGDW",

    "lr_init": 1e-2,
    "momentum": 0.9,
    "weight_decay": 5e-4,

    "batch_size": 256,  # VGG16 paper: 256
    "n_epochs": 50,

    "loss": "categorical_crossentropy",

    "callbacks": [
        "decay_early_stopping"
    ],

    "decay_early_stopping_patience": 5,
    "decay_early_stopping_times": 3,

    ## Active learning
    "n_loops": 10,

    "init_size": 0.1,
    "val_size": 0.1,  # With respect to current training set

    "n_query_instances": 256,  # Number of instances to add at each iteration

    ## Query strategy arguments
    "query_batch_size": 256,  # Batch size for unlabeled pool iterator
}
