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
    "weight_decay": 1e-4,  # VGG16: 5e-4; ResNet: 1e-4; SimpleCNN w/ RMSprop 1e-6
    "n_epochs": 100,

    "callbacks": [
        "reduce_lr_restore_on_plateau",
    ],

    # decay_early_stopping
    "reduce_lr_patience": 20,
    "reduce_lr_decay_schedule": [0.001, 0.0001, 0.00001],
    "reduce_lr_min_delta": 0.01,  # Empirically set for ResNet

    # Query strategy arguments
    "n_query_instances": 2000,  # Number of instances to add at each iteration
    "query_batch_size": 64,  # Batch size for unlabeled pool iterator. Set to a low size for EBAnO.

    # EBAnO query strategy arguments
    "layers_to_analyze": 3,
    "hypercolumn_features": 10,
    "hypercolumn_reduction": "sampletsvd",
    "clustering": "faisskmeans",
    "kmeans_niter": 100,
    "min_features": 2,
    "max_features": 5,
    "ebano_use_gpu": False,
    "ebano_mix_default_strategy": "rank",
    "ebano_mix_default_base_strategy": "random",
    "ebano_mix_default_query_limit": None,
    "ebano_mix_default_augment_limit": None,
    "ebano_mix_default_min_diff": 0.,
    "ebano_mix_default_eps": 0.,

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
    "base_reduce_lr_decay_schedule": [0.01, 0.001],
    "base_reduce_lr_min_delta": 0.01,  # Empirically set for ResNet
}
