from tensorflow.keras.applications import resnet50
from tensorflow.keras import layers
from tensorflow.keras import Model

def ResNet50(n_classes,
             input_shape=(224, 224, 3),
             freeze_extractor=False):

    model_base = resnet50.ResNet50(
        include_top=False,
        weights='imagenet' if freeze_extractor else None,
        input_shape=input_shape
    )

    if freeze_extractor:
        model_base.trainable = False

    x = model_base.output

    # Classifier
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    output = layers.Dense(n_classes, activation='softmax', name='predictions')(x)

    # Model definition
    model = Model(inputs=model_base.input, outputs=output, name='resnet50')

    return model
