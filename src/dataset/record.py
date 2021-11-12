import pathlib

import tensorflow as tf


def parse_tf_example(tf_example, feature_description):
    return tf.io.parse_single_example(tf_example, feature_description)


def load_dataset_from_tfrecord(filename, path=None, load_id=False, num_parallel_reads=None, deterministic=True):
    if path is None:
        path = pathlib.Path(".")

    image_feature_description = {
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64)
    }

    if load_id:
        image_feature_description["id"] = tf.io.FixedLenFeature([], tf.int64)

        def parse_image_tf_example(tf_example):
            example = parse_tf_example(tf_example, image_feature_description)
            image_tensor = tf.io.parse_tensor(example["image"], out_type=tf.float32)
            return example["id"], (
                tf.reshape(image_tensor, shape=[example["height"], example["width"], 3]), example["label"])
    else:
        def parse_image_tf_example(tf_example):
            example = parse_tf_example(tf_example, image_feature_description)
            image_tensor = tf.io.parse_tensor(example["image"], out_type=tf.float32)
            return tf.reshape(image_tensor, shape=[example["height"], example["width"], 3]), example["label"]

    if deterministic:
        ### VERSION 1 (NO DIRECT INTERLEAVING; NO FURTHER SHUFFLING)
        parts = sorted([str(p) for p in list(pathlib.Path(path).rglob(f"{filename}.tfrecord-*"))])
        if not parts:
            raise ValueError(f"No records found at {path}")
        dataset = tf.data.TFRecordDataset(parts, num_parallel_reads=num_parallel_reads)
        print(f"Loaded {len(parts)} TFRecord files at {path}")

    else:
        ### VERSION 2 (WITH INTERLEAVING)
        fname_datasets = tf.data.TFRecordDataset.list_files(str(pathlib.Path(path, f"{filename}.tfrecord-*")), shuffle=True)  # non-deterministic order!

        c = 0
        for f in fname_datasets:
            c += 1

        if c == 0:
            raise ValueError(f"No records found at {path}")
        else:
            print(f"Loaded {c} TFRecord files at {path}")

        dataset = fname_datasets.interleave(
            tf.data.TFRecordDataset,
            cycle_length=num_parallel_reads,
            block_length=1,
            num_parallel_calls=num_parallel_reads,
            deterministic=False,
        )

    parsed_dataset = dataset.map(parse_image_tf_example)
    return parsed_dataset
