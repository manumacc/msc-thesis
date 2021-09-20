config_dict = {
    # Seed
    "seed": 42,

    # Dataset
    "data_path_train": "data/imagenette2/train",
    "data_path_test": "data/imagenette2/val",

    "n_classes": 10,

    "class_sample_size_train": 800,
    "class_sample_size_test": 300,

    # Model
    "model": "VGG16",

    "fc_dropout_rate": 0.5,
    "dense_units": 4096,
    "load_imagenet_weights": False,
    "feature_extractor_trainable": True,

    # Learning
    "optimizer": "SGDW",

    "lr_init": 1e-2,
    "momentum": 0.9,
    "weight_decay": 5e-4,

    "batch_size": 64,  # VGG16 paper: 256
    "n_epochs": 50,

    "loss": "categorical_crossentropy",

    "callbacks": [
        "learning_rate_decay_early_stopping"
    ],

    # Active learning
    "n_loops": 10,

    "init_size": 0.2,
    "val_size": 0.1,  # With respect to current training set

    "query_strategy": "random",
    "require_raw_pool": False,

    "n_query_instances": 64,  # Number of instances to add at each iteration
}
