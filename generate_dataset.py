import sys
sys.path.append('src')

import pathlib
import argparse

import tensorflow as tf
import tensorflow_datasets as tfds

from src.dataset.record import save_dataset_to_tfrecord
from src.dataset.labels import get_labels_by_name
from src.utils import Profiling


DATA_GENERATED_PATH = pathlib.Path("data", "generated")


def generate_dataset(dataset_name,
                     n_samples_init=None,
                     n_samples_test=None,
                     shard_size=None,
                     cycle_length=8,
                     block_length=3,
                     dataset_seed=None):
    if dataset_name.startswith("imagenet"):
        # Set default configuration for splitting
        if n_samples_init is None:
            n_samples_init = 300
        if n_samples_test is None:
            n_samples_test = 100
        if shard_size is None:
            shard_size = 1500

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
            shuffle_files=False
        )

        real_labels = get_labels_by_name(dataset_name)  # Get desired labels

        # Create single-class datasets of chosen labels (training set)
        ds_init_list = []
        ds_test_list = []
        ds_pool_list = []
        for i, l in enumerate(real_labels):
            def filter_label(x, y, keep_label=tf.constant(l, dtype=tf.int64)):
                return tf.equal(keep_label, tf.cast(y, keep_label.dtype))

            ds_label = ds_train_imagenet.filter(filter_label)
            ds_init_list.append(ds_label.take(n_samples_init))
            ds_test_list.append(ds_label.skip(n_samples_init).take(n_samples_test))
            ds_pool_list.append(ds_label.skip(n_samples_init + n_samples_test).take(-1))

        # Create datasets where each element is a `tf.data.Dataset` object.
        # Then, extract all elements from datasets and interleave the results.
        # (!) NOTE: This does not thoroughly shuffle the datasets. The first
        # classes will still be at the beginning, and the other classes at the end.
        ds_init = tf.data.Dataset.from_tensor_slices(ds_init_list)
        ds_init = ds_init.interleave(lambda x: x, cycle_length=cycle_length, block_length=block_length, num_parallel_calls=tf.data.AUTOTUNE)
        ds_pool = tf.data.Dataset.from_tensor_slices(ds_pool_list)
        ds_pool = ds_pool.interleave(lambda x: x, cycle_length=cycle_length, block_length=block_length, num_parallel_calls=tf.data.AUTOTUNE)
        ds_test = tf.data.Dataset.from_tensor_slices(ds_test_list)
        ds_test = ds_test.interleave(lambda x: x, cycle_length=cycle_length, block_length=block_length, num_parallel_calls=tf.data.AUTOTUNE)

        # Create validation dataset by filtering all desired classes
        def filter_labels(x, y, keep_labels=tf.constant(real_labels)):
            keep_sample = tf.equal(keep_labels, tf.cast(y, keep_labels.dtype))
            reduced = tf.reduce_sum(tf.cast(keep_sample, tf.float32))
            return tf.greater(reduced, tf.constant(0.))

        ds_val = ds_val_imagenet.filter(filter_labels)

        # Map labels to sequential numbering (0 to len(real_labels)-1)
        def map_labels(x, y, labels=tf.constant(real_labels, dtype=tf.int64)):
            label_pos = tf.where(tf.equal(labels, y))
            ny = tf.squeeze(label_pos)
            return x, tf.cast(ny, tf.int64)

        ds_init = ds_init.map(map_labels)
        ds_pool = ds_pool.map(map_labels)
        ds_val = ds_val.map(map_labels)
        ds_test = ds_test.map(map_labels)

        # Map sequential labels to one-hot encoding
        # def map_one_hot(x, y, depth=len(real_labels)):
        #     return x, tf.one_hot(y, depth=depth)
        #
        # ds_init = ds_init.map(map_one_hot)
        # ds_pool = ds_pool.map(map_one_hot)
        # ds_val = ds_val.map(map_one_hot)
        # ds_test = ds_test.map(map_one_hot)
    else:
        raise ValueError(f"Dataset {dataset_name} is not available")

    print("Save datasets to TFRecords")
    with Profiling("Write ds_init"):
        save_dataset_to_tfrecord(ds_init, f"{dataset_name}_init", path=DATA_GENERATED_PATH, shard_size=shard_size, write_id=False)
    with Profiling("Write ds_pool"):
        save_dataset_to_tfrecord(ds_pool, f"{dataset_name}_pool", path=DATA_GENERATED_PATH, shard_size=shard_size, write_id=True)
    with Profiling("Write ds_val"):
        save_dataset_to_tfrecord(ds_val, f"{dataset_name}_val", path=DATA_GENERATED_PATH, shard_size=shard_size, write_id=False)
    with Profiling("Write ds_test"):
        save_dataset_to_tfrecord(ds_test, f"{dataset_name}_test", path=DATA_GENERATED_PATH, shard_size=shard_size, write_id=False)


arg_parser = argparse.ArgumentParser(description="generate a dataset")
arg_parser.add_argument("dataset_name", type=str, help="name of the dataset to generate")
arg_parser.add_argument('--shardsize', nargs='?', default=None, type=int)
arg_parser.add_argument('--seed', nargs='?', default=None, type=int)


if __name__ == '__main__':
    args = arg_parser.parse_args()
    generate_dataset(args.dataset_name, shard_size=args.shardsize, dataset_seed=args.seed)
