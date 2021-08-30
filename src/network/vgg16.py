from tensorflow.keras import layers
from tensorflow.keras import Model

def VGG16(n_classes,
          dropout_rate=0.5,
          input_shape=(224, 224, 3)):
    """
    VGG16 model for training and inference based on the original paper.
    """

    inp = layers.Input(shape=input_shape)

    # Convolutional block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu', padding='same', name='block1_conv1')(inp)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu', padding='same', name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Convolutional block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu', padding='same', name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Convolutional block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu', padding='same', name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu', padding='same', name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu', padding='same', name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Convolutional block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu', padding='same', name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu', padding='same', name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Convolutional block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu', padding='same', name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu', padding='same', name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu', padding='same', name='block5_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Dense block
    x = layers.Flatten(name='flatten')(x)

    x = layers.Dense(4096, activation='relu', name='fc1')(x)
    x = layers.Dropout(dropout_rate, name='fc1_dropout')(x)
    x = layers.Dense(4096, activation='relu', name='fc2')(x)
    x = layers.Dropout(dropout_rate, name='fc2_dropout')(x)

    out = layers.Dense(n_classes, activation='softmax', name='predictions')(x)

    # Model definition
    model = Model(inputs=inp, outputs=out, name='vgg16')

    return model
