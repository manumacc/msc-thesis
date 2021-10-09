"""
Create an ImageNet subset of specified size containing only specified
classes. Save as pickle containing numpy arrays X, y.
"""

import pickle
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

ds = tfds.load('imagenet2012', split='train', shuffle_files=True, as_supervised=True)
print(ds)

# Script
TARGET_SIZE = (224, 224)

FILTER_CLASSES = False

BATCH_SIZE = 100000
NUM_BATCHES = 1
OUTPUT = "batch"

if FILTER_CLASSES is not None:
    ds = ds.filter(lambda image, label:
                   label == 0 or
                   label == 217 or
                   label == 482 or
                   label == 491 or
                   label == 497 or
                   label == 566 or
                   label == 569 or
                   label == 571 or
                   label == 574 or
                   label == 701)


def resize(image, label):
    return tf.image.resize(image, TARGET_SIZE), label


ds = ds.map(resize)
ds = ds.shuffle(1000)
ds = ds.batch(BATCH_SIZE)
np_iter = ds.as_numpy_iterator()

print("Fetching data into ndarray(s)")
image_arrays, label_arrays = [], []
for i, data_array_i in enumerate(np_iter):
    print(f"Fetching batch {i}")
    image_arrays.append(data_array_i[0])
    label_arrays.append(data_array_i[1])

    if i >= NUM_BATCHES - 1:
        break

image_arrays = np.concatenate(image_arrays)
label_arrays = np.concatenate(label_arrays)
print(len(image_arrays), len(label_arrays))

print(np.unique(label_arrays, return_counts=True))

print(f"Memory usage (in bytes): {image_arrays.nbytes} + {label_arrays.nbytes}")

print("Write pickle")
with open(f"{OUTPUT}.pkl", "wb") as f:
    pickle.dump([image_arrays, label_arrays], f)
