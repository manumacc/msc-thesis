from tensorflow.keras import layers, initializers
from tensorflow.keras import Model

def VGG16(n_classes,
          input_shape=(224, 224, 3),
          dropout_rate=0.5,
          dense_units=4096,
          seed=None):
    """VGG16 model for training and inference, based on the original paper.

    Args:
        n_classes: Number of output classes to classify images into.
        input_shape: Shape of the input.
        dropout_rate: Rate of dropout for fully connected dropout layers.
        dense_units: Number of units of fully connected layers.
        seed: Seed for kernel initializer

    Returns:
        model: VGG16 model with classifier.
    """

    x_input = layers.Input(shape=input_shape)

    # Convolutional block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu', padding='same', name='block1_conv1',
                      kernel_initializer=initializers.GlorotUniform(seed=seed+1 if seed else None))(x_input)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu', padding='same', name='block1_conv2',
                      kernel_initializer=initializers.GlorotUniform(seed=seed+2 if seed else None))(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Convolutional block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu', padding='same', name='block2_conv1',
                      kernel_initializer=initializers.GlorotUniform(seed=seed+1 if seed else None))(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu', padding='same', name='block2_conv2',
                      kernel_initializer=initializers.GlorotUniform(seed=seed+2 if seed else None))(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Convolutional block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu', padding='same', name='block3_conv1',
                      kernel_initializer=initializers.GlorotUniform(seed=seed+1 if seed else None))(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu', padding='same', name='block3_conv2',
                      kernel_initializer=initializers.GlorotUniform(seed=seed+2 if seed else None))(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu', padding='same', name='block3_conv3',
                      kernel_initializer=initializers.GlorotUniform(seed=seed+3 if seed else None))(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Convolutional block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu', padding='same', name='block4_conv1',
                      kernel_initializer=initializers.GlorotUniform(seed=seed+1 if seed else None))(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu', padding='same', name='block4_conv2',
                      kernel_initializer=initializers.GlorotUniform(seed=seed+2 if seed else None))(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu', padding='same', name='block4_conv3',
                      kernel_initializer=initializers.GlorotUniform(seed=seed+3 if seed else None))(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Convolutional block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu', padding='same', name='block5_conv1',
                      kernel_initializer=initializers.GlorotUniform(seed=seed+1 if seed else None))(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu', padding='same', name='block5_conv2',
                      kernel_initializer=initializers.GlorotUniform(seed=seed+2 if seed else None))(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu', padding='same', name='block5_conv3',
                      kernel_initializer=initializers.GlorotUniform(seed=seed+3 if seed else None))(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Dense block
    x = layers.Flatten(name='flatten')(x)

    x = layers.Dense(dense_units, activation='relu', name='fc1',
                     kernel_initializer=initializers.GlorotUniform(seed=seed+1 if seed else None))(x)
    x = layers.Dropout(dropout_rate, name='fc1_dropout')(x)
    x = layers.Dense(dense_units, activation='relu', name='fc2',
                     kernel_initializer=initializers.GlorotUniform(seed=seed+2 if seed else None))(x)
    x = layers.Dropout(dropout_rate, name='fc2_dropout')(x)

    x_output = layers.Dense(n_classes, activation='softmax', name='predictions',
                                kernel_initializer=initializers.GlorotUniform(seed=seed+3 if seed else None))(x)

    # Model definition
    model = Model(inputs=x_input, outputs=x_output, name='VGG16')

    return model
