from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications import vgg16

def VGG16(n_classes,
          input_shape=(224, 224, 3),
          dropout_rate=0.5,
          dense_units=4096,
          load_imagenet=True,
          feature_extractor_trainable=False):
    """VGG16 model for training and inference, based on the original paper.

    Args:
        n_classes: Number of output classes to classify images into.
        input_shape: Shape of the input.
        dropout_rate: Rate of dropout for fully connected dropout layers.
        dense_units: Number of units of fully connected layers.
        load_imagenet: If True, load imagenet weights for the feature extractor.
            Note that the dense classifier block is always randomly initialized.
        feature_extractor_trainable: If False, freeze the feature extractor
            layers. Otherwise, set them to trainable layers.

    Returns:
        model: VGG16 model with classifier.
    """

    model_base = vgg16.VGG16(
        include_top=False,
        weights='imagenet' if load_imagenet else None,
        input_shape=input_shape
    )

    if not feature_extractor_trainable:
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
