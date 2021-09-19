# Seed
SEED = 42

# Dataset
DATA_PATH_TRAIN = "data/imagenette2/train"
DATA_PATH_TEST = "data/imagenette2/val"

N_CLASSES = 10

CLASS_SAMPLE_SIZE_TRAIN = 800
CLASS_SAMPLE_SIZE_TEST = 300

# Model
MODEL = "VGG16"

FC_DROPOUT_RATE = 0.5
DENSE_UNITS = 4096
LOAD_IMAGENET_WEIGHTS = False
FEATURE_EXTRACTOR_TRAINABLE = True

# Learning
OPTIMIZER = "SGDW"

LR_INIT = 1e-2
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

BATCH_SIZE = 64
N_EPOCHS = 50

LOSS = "categorical_crossentropy"

CALLBACKS = [
    "learning_rate_decay_with_early_stopping"
]

# Active learning
N_LOOPS = 10

INIT_SIZE = 0.2
VAL_SIZE = 0.1  # With respect to current training set

QUERY_STRATEGY = "random"
REQUIRE_RAW_POOL = False

N_QUERY_INSTANCES = 64  # Number of instances to add at each iteration
