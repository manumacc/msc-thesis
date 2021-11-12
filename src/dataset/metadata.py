IMAGENET_INIT_SIZE = 300
IMAGENET_TEST_SIZE = 100
IMAGENET_TRAIN_TOTAL = 1300


IMAGENETTE_LABELS = [
    0, 217, 482, 491, 497, 566, 569, 571, 574, 701
]

IMAGENET_25_LABELS = [
    0, 19, 97, 217, 388, 389, 393, 402, 440, 482, 497, 504, 508, 545, 559, 563,
    566, 569, 571, 574, 628, 701, 759, 933, 963
]

IMAGENET_100_LABELS = [
      3,   5,   7,  13,  18,  24,  28,  33,  45,  63,  66,  69,  76,
     77, 109, 120, 144, 149, 220, 233, 235, 242, 244, 253, 265, 270,
    289, 302, 310, 317, 332, 342, 344, 353, 356, 365, 371, 385, 415,
    427, 432, 433, 450, 469, 470, 473, 477, 484, 495, 527, 538, 540,
    546, 554, 562, 568, 570, 598, 601, 608, 617, 642, 647, 648, 659,
    664, 685, 690, 694, 698, 700, 715, 725, 742, 750, 752, 763, 779,
    784, 796, 802, 816, 820, 833, 840, 849, 866, 878, 883, 908, 913,
    918, 936, 949, 954, 968, 970, 980, 995, 998
]

IMAGENET_250_LABELS = [
      2,   6,   8,  12,  15,  16,  17,  22,  26,  30,  39,  41,  44,
     48,  52,  56,  57,  64,  65,  67,  70,  75,  87,  88,  89, 100,
    101, 106, 115, 117, 119, 125, 130, 132, 133, 136, 155, 156, 159,
    169, 173, 176, 182, 184, 186, 187, 192, 198, 201, 205, 208, 209,
    213, 227, 240, 245, 247, 248, 250, 255, 261, 266, 272, 273, 275,
    280, 281, 285, 287, 290, 291, 293, 295, 304, 306, 313, 316, 320,
    321, 322, 325, 326, 330, 334, 338, 347, 348, 349, 350, 355, 357,
    362, 366, 368, 372, 375, 377, 378, 382, 384, 395, 396, 400, 405,
    407, 411, 413, 414, 420, 422, 428, 442, 443, 445, 446, 447, 449,
    452, 455, 457, 459, 460, 462, 467, 471, 479, 485, 487, 492, 493,
    496, 498, 506, 510, 516, 518, 519, 523, 525, 529, 534, 541, 542,
    544, 547, 552, 553, 561, 576, 586, 587, 591, 593, 594, 602, 606,
    611, 615, 619, 620, 621, 625, 626, 629, 632, 640, 643, 645, 649,
    650, 656, 660, 674, 683, 691, 692, 696, 703, 705, 713, 718, 732,
    736, 741, 755, 756, 757, 760, 761, 762, 764, 768, 775, 777, 780,
    792, 793, 794, 797, 799, 803, 814, 823, 828, 829, 830, 832, 835,
    836, 843, 844, 845, 847, 856, 862, 870, 873, 874, 875, 880, 886,
    887, 893, 898, 900, 905, 909, 911, 916, 922, 923, 928, 932, 942,
    948, 951, 959, 960, 961, 964, 967, 971, 974, 976, 981, 983, 985,
    988, 992, 996
]


def get_labels_by_name(dataset_name):
    if dataset_name == "imagenet-10" or dataset_name == "imagenette":
        labels = IMAGENETTE_LABELS
    elif dataset_name == "imagenet-25":
        labels = IMAGENET_25_LABELS
    elif dataset_name == "imagenet-100":
        labels = IMAGENET_100_LABELS
    elif dataset_name == "imagenet-250":
        labels = IMAGENET_250_LABELS
    else:
        raise ValueError(f"Dataset {dataset_name} mapper is not available.")

    return labels


def get_size_by_name(dataset_name, split):
    if dataset_name.startswith("imagenet"):
        if dataset_name == "imagenet-10" or dataset_name == "imagenette":
            n_classes = 10
        elif dataset_name == "imagenet-25":
            n_classes = 25
        elif dataset_name == "imagenet-100":
            n_classes = 100
        elif dataset_name == "imagenet-250":
            n_classes = 250
        else:
            raise ValueError(f"ImageNet dataset {dataset_name} is not available.")

        train_total_size = IMAGENET_TRAIN_TOTAL * n_classes
        init_size = IMAGENET_INIT_SIZE * n_classes
        test_size = IMAGENET_TEST_SIZE * n_classes

        if split == "test":
            size = test_size
        elif split == "init":
            size = init_size
        elif split == "pool":
            size = train_total_size - init_size - test_size
        elif split == "total":
            size = train_total_size
        else:
            raise ValueError(f"ImageNet split {split} is not available.")
    else:
        raise ValueError(f"Dataset {dataset_name} is not available")

    return size
