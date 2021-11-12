import sys
sys.path.append('src')

import pathlib
import argparse
import gc
import random

import tensorflow as tf
import tensorflow_datasets as tfds

from src.dataset.metadata import get_labels_by_name
from src.utils import Profiling


DATA_GENERATED_PATH = pathlib.Path("data", "generated")


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


def get_shards(dataset_name):
    if dataset_name == "imagenet-10" or dataset_name == "imagenette":
        n_shards_init, n_shards_val, n_shards_pool, n_shards_test = 15, 5, 50, 10
    elif dataset_name == "imagenet-25":
        n_shards_init, n_shards_val, n_shards_pool, n_shards_test = 40, 10, 100, 20
    elif dataset_name == "imagenet-100":
        n_shards_init, n_shards_val, n_shards_pool, n_shards_test = 100, 30, 400, 50
    elif dataset_name == "imagenet-250":
        n_shards_init, n_shards_val, n_shards_pool, n_shards_test = 300, 100, 800, 100
    else:
        raise ValueError(f"Dataset {dataset_name} is not available.")

    return n_shards_init, n_shards_val, n_shards_pool, n_shards_test


def generate_dataset(dataset_name,
                     n_samples_init=None,
                     n_samples_test=None,
                     image_resize_shape=(224, 224)):
    if dataset_name.startswith("imagenet"):
        # Set default configuration for splitting
        if n_samples_init is None:
            n_samples_init = 300
        if n_samples_test is None:
            n_samples_test = 100

        # Set tensorflow global seed for reproducible results
        tf.random.set_seed(0)

        # Initialize dataset
        print("Initialize dataset")
        imagenet_builder = tfds.builder('imagenet2012')
        imagenet_builder.download_and_prepare()
        ds_info = imagenet_builder.info
        ds_train_imagenet, ds_val_imagenet = imagenet_builder.as_dataset(
            split=['train', 'validation'],
            as_supervised=True,
            shuffle_files=True
        )

        real_labels = get_labels_by_name(dataset_name)  # Get desired labels
        n_classes = len(real_labels)

        print(real_labels)
        print(n_classes)

        # Filter only required labels
        tf_keep_labels = tf.constant(real_labels, dtype=tf.int64)
        def filter_labels(x, y):
            keep_sample = tf.equal(tf_keep_labels, tf.cast(y, tf_keep_labels.dtype))
            reduced = tf.reduce_sum(tf.cast(keep_sample, tf.float32))
            return tf.greater(reduced, tf.constant(0.))

        ds_train = ds_train_imagenet.filter(filter_labels)
        ds_val = ds_val_imagenet.filter(filter_labels)

        # Reshape images
        if image_resize_shape is not None:
            def map_image_resize(x, y):  # returns tf.float32 images
                return tf.image.resize(x, list(image_resize_shape)), y

            ds_train = ds_train.map(map_image_resize, num_parallel_calls=tf.data.AUTOTUNE)
            ds_val = ds_val.map(map_image_resize, num_parallel_calls=tf.data.AUTOTUNE)

        # Map labels to sequential numbering (0 to len(real_labels)-1)
        def map_labels(x, y):
            label_pos = tf.where(tf.equal(tf_keep_labels, y))
            ny = tf.squeeze(label_pos)
            return x, tf.cast(ny, tf.int64)

        ds_train = ds_train.map(map_labels, num_parallel_calls=tf.data.AUTOTUNE)
        ds_val = ds_val.map(map_labels, num_parallel_calls=tf.data.AUTOTUNE)

        # Initialize writers
        path = DATA_GENERATED_PATH

        n_shards_init, n_shards_val, n_shards_pool, n_shards_test = get_shards(dataset_name)
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

        print("Initializing writers")
        for subset in writers:
            part_filename_format = f"{dataset_name}-{subset}.tfrecord"
            print(part_filename_format)
            for i in range(n_shards[subset]):
                part_filename = f"{part_filename_format}-{i+1:05d}-of-{n_shards[subset]:05d}"
                writers[subset].append(
                    tf.io.TFRecordWriter(str(pathlib.Path(path, part_filename)))
                )

        def _select_random_shard(sset):
            return random.choice(list(range(n_shards[sset])))

        yet_to_fill = {k: ["init", "test", "pool"] for k in range(n_classes)}
        init_label_count = {k: 0 for k in range(n_classes)}
        test_label_count = {k: 0 for k in range(n_classes)}
        pool_label_count = {k: 0 for k in range(n_classes)}
        current_pool_id = 0
        for tf_x, tf_y in ds_train:
            y = tf_y.numpy()

            chosen_subset = random.choice(yet_to_fill[y])
            if chosen_subset == "init":
                tf_example = image_to_tf_example(tf_x, tf_y, feature_id=None)
                shard_id = _select_random_shard("init")
                writers["init"][shard_id].write(tf_example.SerializeToString())
                init_label_count[y] += 1
                if init_label_count[y] == n_samples_init:
                    print(f"Wrote {init_label_count[y]} samples of class {y} to init")
                    yet_to_fill[y] = [x for x in yet_to_fill[y] if x != "init"]
            elif chosen_subset == "test":
                tf_example = image_to_tf_example(tf_x, tf_y, feature_id=None)
                shard_id = _select_random_shard("test")
                writers["test"][shard_id].write(tf_example.SerializeToString())
                test_label_count[y] += 1
                if test_label_count[y] == n_samples_test:
                    print(f"Wrote {test_label_count[y]} samples of class {y} to test")
                    yet_to_fill[y] = [x for x in yet_to_fill[y] if x != "test"]
            else:
                tf_example = image_to_tf_example(tf_x, tf_y, feature_id=current_pool_id)
                shard_id = _select_random_shard("pool")
                writers["pool"][shard_id].write(tf_example.SerializeToString())
                current_pool_id += 1
                pool_label_count[y] += 1

        print("Init label count")
        print(init_label_count)
        print("Test label count")
        print(test_label_count)
        print("Pool label count")
        print(pool_label_count)

        print("Validation set")
        for tf_x, tf_y in ds_val:
            tf_example = image_to_tf_example(tf_x, tf_y, feature_id=None)
            shard_id = _select_random_shard("val")
            writers["val"][shard_id].write(tf_example.SerializeToString())
        print("Done")
    else:
        raise ValueError(f"Dataset {dataset_name} is not available")


arg_parser = argparse.ArgumentParser(description="generate a dataset")
arg_parser.add_argument("dataset_name", type=str, help="name of the dataset to generate")


if __name__ == '__main__':
    args = arg_parser.parse_args()
    generate_dataset(args.dataset_name)
