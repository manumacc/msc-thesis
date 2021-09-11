from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications import vgg16

def VGG16(n_classes,
          input_shape=(224, 224, 3),
          load_imagenet=True,
          dropout_rate=0.5):
    """
    VGG16 model for training and inference based on the original paper.
    """

    model_base = vgg16.VGG16(
        include_top=False,
        weights='imagenet' if load_imagenet else None,
        input_shape=input_shape
    )

    x = model_base.output

    # Dense block
    x = layers.Flatten(name='flatten')(x)

    x = layers.Dense(4096, activation='relu', name='fc1')(x)
    x = layers.Dropout(dropout_rate, name='fc1_dropout')(x)
    x = layers.Dense(4096, activation='relu', name='fc2')(x)
    x = layers.Dropout(dropout_rate, name='fc2_dropout')(x)

    output = layers.Dense(n_classes, activation='softmax', name='predictions')(x)

    # Model definition
    model = Model(inputs=model_base.input, outputs=output, name='vgg16')

    return model
