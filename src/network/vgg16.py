from tensorflow.keras import layers, initializers
from tensorflow.keras import Model

def VGG16(n_classes,
          input_shape=(224, 224, 3),
          dropout_rate=0.5,
          dense_units=4096,
          freeze_extractor=False):
    """VGG16 model for training and inference, based on the original paper.

    Args:
        n_classes: Number of output classes to classify images into.
        input_shape: Shape of the input.
        dropout_rate: Rate of dropout for fully connected dropout layers.
        dense_units: Number of units of fully connected layers.
        freeze_extractor: If True, initialize the feature extractor layers to
            ImageNet weights and freeze them. Otherwise, randomly initialize
            feature extractor layers and set them as trainable.

    Returns:
        model: VGG16 model with classifier.
    """

    model_base = vgg16.VGG16(
        include_top=False,
        weights='imagenet' if freeze_extractor else None,
        input_shape=input_shape
    )

    if freeze_extractor:
        model_base.trainable = False

    x = model_base.output

    # Dense block
    x = layers.Flatten(name='flatten')(x)

    x = layers.Dense(dense_units, activation='relu', name='fc1')(x)
    x = layers.Dropout(dropout_rate, name='fc1_dropout')(x)
    x = layers.Dense(dense_units, activation='relu', name='fc2')(x)
    x = layers.Dropout(dropout_rate, name='fc2_dropout')(x)

    output = layers.Dense(n_classes, activation='softmax', name='predictions')(x)

    # Model definition
    model = Model(inputs=model_base.input, outputs=output, name='vgg16')

    return model
