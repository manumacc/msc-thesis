from tensorflow.keras import layers
from tensorflow.keras import Model

def SimpleCNN(n_classes,
              input_shape=(32, 32, 3)):
    x_in = layers.Input(shape=input_shape)

    # Convolutional block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x_in)
    x = layers.MaxPool2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPool2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Convolutional block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPool2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPool2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Classifier
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x_out = layers.Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=x_in, outputs=x_out, name='simplecnn')

    return model
