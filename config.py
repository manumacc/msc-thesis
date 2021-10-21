config_dict = {
    ## Logging
    "save_models": False,

    ## Seed
    "dataset_seed": 0,
    "experiment_seed": 42,

    ## Dataset
    "n_classes": 25,

    # Dataset name
    "dataset": "imagenet-25",  # None, "cifar-10", "imagenet-25"

    # If dataset is None, accept data from directory
    "data_path_train": "",  # data/imagenette2/train
    "data_path_test": "",  # data/imagenette2/val
    "class_sample_size_train": 0,  # 800
    "class_sample_size_test": 0,  # 300

    ## Model
    "model": "ResNet50",  # SimpleCNN; ResNet50; VGG16

    "freeze_extractor": False,  # if True, load ImageNet weights for feature extractor (where available)

    "optimizer": "SGDW",  # SGDW; RMSprop
    "loss": "categorical_crossentropy",

    "batch_size": 256,

    # VGG16
    "fc_dropout_rate": 0.5,
    "dense_units": 4096,

    ## Active learning
    "n_loops": 10,

    "val_size": 0.1,  # With respect to initial training subset; does not apply to datasets with dedicated val split

    "lr_init": 0.01,  # VGG16: 1e-2; ResNet: 0.1; SimpleCNN w/ RMSprop 0.0001
    "momentum": 0.9,
    "weight_decay": 1e-6,  # VGG16: 5e-4; ResNet: 1e-4; SimpleCNN w/ RMSprop 1e-6
    "n_epochs": 100,

    "callbacks": [
        "reduce_lr_restore_on_plateau",
    ],

    # decay_early_stopping
    "reduce_lr_patience": 20,
    "reduce_lr_factor": 0.1,
    "reduce_lr_min_delta": 0.01,  # Empirically set for ResNet
    "reduce_lr_cooldown": 0,
    "reduce_lr_decay_times": 3,  # Decay LR N times

    # Query strategy arguments
    "n_query_instances": 2000,  # Number of instances to add at each iteration
    "query_batch_size": 256,  # Batch size for unlabeled pool iterator. Set to a low size for EBAnO.

    # EBAnO query strategy arguments
    "layers_to_analyze": 3,
    "hypercolumn_features": 10,
    "hypercolumn_reduction": "sampletsvd",
    "clustering": "faisskmeans",
    "kmeans_niter": 100,
    "min_features": 2,
    "max_features": 5,
    "ebano_use_gpu": False,

    # Mix query strategy arguments
    "mix_iteration_methods": {
        0: "margin-sampling",  # 20000 -> 18000
        1: "margin-sampling",  # 18000 -> 16000
        2: "margin-sampling",  # 16000 -> 14000
        3: "margin-sampling",  # 14000 -> 12000
        4: "margin-sampling",  # 12000 -> 10000
        5: "ebano",  # 10000 -> 8000
        6: "ebano",  # 8000 -> 6000
        7: "ebano",  # 6000 -> 4000
        8: "margin-sampling",  # 4000 -> 2000
        9: "random",  # 2000 -> 0 (i.e., take all)
    },

    ## Base model
    "base_model_name": "resnet_imagenet_25_base",
    "base_init_size": 7500,  # With respect to whole training set; either 0-1 (percent) or >1 (number of samples)

    # Base model training
    "base_lr_init": 0.1,
    "base_weight_decay": 1e-4,
    "base_n_epochs": 1000,

    "base_callbacks": [
        "reduce_lr_restore_on_plateau",
    ],

    "base_reduce_lr_patience": 50,
    "base_reduce_lr_factor": 0.1,
    "base_reduce_lr_min_delta": 0.01,  # Empirically set for ResNet
    "base_reduce_lr_cooldown": 0,
    "base_reduce_lr_decay_times": 2,  # Decay LR N times
}
