config_dict = {
    ## Logging
    "save_models": False,

    ## Seed
    "experiment_seed": 42,

    ## Dataset
    "dataset_name": "caltech",  # None, "cifar-10", "imagenet-25", ...
    "dataset_path": "data/caltech-20-10-p10",

    ## Model
    "model": "ResNet50",  # ResNet50; VGG16
    "batch_size": 128,

    ## Active learning
    "n_loops": 5,
    "n_epochs": 50,

    "base_model_name": None,  # Base model for AL loop, can be None if --base
    "lr_init": 0.01,

    # decay_early_stopping
    "reduce_lr_patience": 10,
    "reduce_lr_min_delta": 0.001,
    "reduce_lr_min": 0.001,

    # Query strategy arguments
    "n_query_instances": 1000,  # Number of instances to add at each iteration
    "query_batch_size": 128,  # Batch size for unlabeled pool iterator. Set to a low size (64) for EBAnO.

    # EBAnO query strategy arguments
    "layers_to_analyze": 3,
    "hypercolumn_features": 10,
    "hypercolumn_reduction": "sampletsvd",
    "clustering": "faisskmeans",
    "kmeans_niter": 100,
    "min_features": 2,
    "max_features": 5,
    "ebano_use_gpu": False,
    "ebano_base_strategy": "entropy",
    "ebano_query_limit": 1000,
    "ebano_augment_limit": 1000,
    "ebano_min_diff": 0.5,
    "ebano_subset": None,

    ## Base model training
    "base_lr_init": 0.001,
    "base_n_epochs": 300,

    "base_reduce_lr_patience": 10,
    "base_reduce_lr_min_delta": 0.001,
    "base_reduce_lr_min": 0.00001,
}
