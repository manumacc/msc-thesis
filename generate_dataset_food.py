import sys

sys.path.append('src')

import pathlib
import random

import tensorflow as tf

DATA_GENERATED_PATH = "data/food-101-val100-p10"
PATH_ORIG = "data/food-101"
PATH_TRAIN = "images-train"
PATH_TEST = "images-test"

# Food-101 contains 750 training images and 250 test images per class,
# as defined by the .txt files detailing the split.

# Total images: 101000 - Training images: 75750 - Test images: 25250

# > Version 150 val, init p10
#   Total elements for AL loop: 600*101 = 60600 elements - 10% = 54540 =~ 55000
#   ==> ~55000 total elements for AL loop, starting from 10%.
#       1100 per loop: ~ +2%

# > Version 250 val, init p25
#   Total elements for AL loop: 500*101 = 50500 elements - 25% = 37875 =~ 38000
#   ==> ~38000 total elements for AL loop, starting from 25%.
#       2280 per loop: ~ +6%


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _serialize_tensor(value):
    return tf.io.serialize_tensor(value)


def image_to_tf_example(image, label, feature_id=None):
    feature = {
        "height": _int64_feature(image.shape[0]),
        "width": _int64_feature(image.shape[1]),
        "image": _bytes_feature(_serialize_tensor(image)),
        "label": _int64_feature(label),
    }

    if feature_id is not None:
        feature["id"] = _int64_feature(feature_id)

    return tf.train.Example(features=tf.train.Features(feature=feature))


def get_shards():
    n_shards_init, n_shards_val, n_shards_pool, n_shards_test = 10, 10, 60, 25
    return n_shards_init, n_shards_val, n_shards_pool, n_shards_test


def generate_dataset():
    tf.random.set_seed(0)

    # Initialize dataset
    print("Initialize dataset")

    ds_orig = tf.keras.preprocessing.image_dataset_from_directory(
        directory=pathlib.Path(PATH_ORIG, PATH_TRAIN),
        labels="inferred",
        label_mode="int",
        batch_size=256,
        image_size=(224, 224),
        interpolation="bilinear",
        shuffle=True,
    )

    ds_orig = ds_orig.unbatch()

    n_shards_init, n_shards_val, n_shards_pool, n_shards_test = get_shards()
    n_shards = {
        "init": n_shards_init,
        "val": n_shards_val,
        "pool": n_shards_pool,
        "test": n_shards_test,
    }
    writers = {
        "init": [],
        "val": [],
        "pool": [],
        "test": []
    }

    path = DATA_GENERATED_PATH
    n_classes = 101

    print("Initializing writers")
    for subset in writers:
        part_filename_format = f"food-{subset}.tfrecord"
        print(part_filename_format)
        for i in range(n_shards[subset]):
            part_filename = f"{part_filename_format}-{i + 1:05d}-of-{n_shards[subset]:05d}"
            writers[subset].append(
                tf.io.TFRecordWriter(str(pathlib.Path(path, part_filename)))
            )

    def _select_random_shard(sset):
        return random.choice(list(range(n_shards[sset])))

    n_samples_val = 150  # number of samples for val per class

    p_samples_init = 0.20  # percent of samples to reserve for initial training set

    init_label_count = {k: 0 for k in range(n_classes)}
    pool_label_count = {k: 0 for k in range(n_classes)}
    val_label_count = {k: 0 for k in range(n_classes)}
    current_pool_id = 0

    total_count = 0
    for tf_x, tf_y in ds_orig:
        total_count += 1
        if total_count % 100 == 0:
            print(f">>> Processing image {total_count}")

        y = tf_y.numpy()

        # First of all, fill val
        if val_label_count[y] < n_samples_val:
            tf_example = image_to_tf_example(tf_x, tf_y, feature_id=None)
            shard_id = _select_random_shard("val")
            writers["val"][shard_id].write(tf_example.SerializeToString())
            val_label_count[y] += 1
            if val_label_count[y] == n_samples_val:
                print(f"Wrote {val_label_count[y]} samples of class {y} to val")
            continue

        # Now fill init and pool, in the proportions specified
        n_samples_init_per_class = {
            k: int((750 - n_samples_val) * p_samples_init)
            for k in range(n_classes)
        }

        if init_label_count[y] < n_samples_init_per_class[y]:
            tf_example = image_to_tf_example(tf_x, tf_y, feature_id=None)
            shard_id = _select_random_shard("init")
            writers["init"][shard_id].write(tf_example.SerializeToString())
            init_label_count[y] += 1
            if init_label_count[y] == n_samples_init_per_class[y]:
                print(f"Wrote {init_label_count[y]} samples of class {y} to init")

        else:  # pool
            tf_example = image_to_tf_example(tf_x, tf_y, feature_id=current_pool_id)
            shard_id = _select_random_shard("pool")
            writers["pool"][shard_id].write(tf_example.SerializeToString())
            current_pool_id += 1
            pool_label_count[y] += 1

    print("Val label count")
    print(val_label_count)
    print("Init label count")
    print(init_label_count)
    print("Pool label count")
    print(pool_label_count)

    print("Test set")
    ds_test = tf.keras.preprocessing.image_dataset_from_directory(
        directory=pathlib.Path(PATH_ORIG, PATH_TEST),
        labels="inferred",
        label_mode="int",
        batch_size=256,
        image_size=(224, 224),
        interpolation="bilinear",
        shuffle=True,
    )

    ds_test = ds_test.unbatch()

    for tf_x, tf_y in ds_test:
        tf_example = image_to_tf_example(tf_x, tf_y, feature_id=None)
        shard_id = _select_random_shard("test")
        writers["test"][shard_id].write(tf_example.SerializeToString())
    print("Done")


if __name__ == '__main__':
    generate_dataset()
