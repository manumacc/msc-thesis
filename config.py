config_dict = {
    ## Logging
    "save_models": False,

    ## Seed
    "experiment_seed": 42,

    ## Dataset
    "dataset_name": "imagenet-25",  # None, "cifar-10", "imagenet-25", ...
    "dataset_path": "data/imagenet-25",

    ## Model
    "model": "ResNet50",  # ResNet50; VGG16
    "batch_size": 256,

    ## Active learning
    "n_loops": 10,
    "n_epochs": 100,

    "base_model_name": None,  # Base model for AL loop, can be None if --base
    "lr_init": 0.01,  # ResNet: 0.01; VGG16: 0.001

    # decay_early_stopping
    "reduce_lr_patience": 20,
    "reduce_lr_min_delta": 0.001,  # Empirically set. ResNet: 0.001
    "reduce_lr_min": 0.0001,  # ResNet: 0.0001 (0.01/(10*2)); VGG16: 0.00001 (0.001/(10*2))

    # Query strategy arguments
    "n_query_instances": 1500,  # Number of instances to add at each iteration
    "query_batch_size": 64,  # Batch size for unlabeled pool iterator. Set to a low size (64) for EBAnO.

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
    "ebano_mix_default_subset": None,

    ## Base model training
    "base_lr_init": 0.1,  # VGG16: 0.01; ResNet: 0.1
    "base_n_epochs": 500,

    "base_reduce_lr_patience": 50,
    "base_reduce_lr_min_delta": 0.01,  # Empirically set for ResNet
    "base_reduce_lr_min": 0.001,  # ResNet: 0.001 (0.1/(10*2)); VGG16: 0.0001 (0.01/(10*2))
}
