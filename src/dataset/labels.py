IMAGENET_25_LABELS = [
    0, 19, 97, 217, 388, 389, 393, 402, 440, 482, 497, 504, 508, 545, 559, 563,
    566, 569, 571, 574, 628, 701, 759, 933, 963
]

IMAGENET_50_LABELS = []

IMAGENET_100_LABELS = []

IMAGENET_250_LABELS = []

def get_labels_by_name(dataset_name):
    if dataset_name == "imagenet-25":
        labels = IMAGENET_25_LABELS
    elif dataset_name == "imagenet-50":
        raise NotImplementedError
    elif dataset_name == "imagenet-100":
        raise NotImplementedError
    elif dataset_name == "imagenet-250":
        raise NotImplementedError
    else:
        raise ValueError(f"Dataset {dataset_name} mapper is not available.")

    return labels
