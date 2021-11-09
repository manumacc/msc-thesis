import pathlib
import tensorflow as tf


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


def save_dataset_to_tfrecord(dataset, filename, path=None, shard_size=None, write_id=False):
    part_filename_format = f"{filename}.tfrecord"

    if write_id:
        current_id = 0
    else:
        current_id = None

    current_part = 0
    current_part_count = 0
    writer = None

    if path is None:
        path = pathlib.Path(".")

    for x, y in dataset:
        if writer is None:
            part_filename = f"{part_filename_format}-{current_part:05d}"
            writer = tf.io.TFRecordWriter(str(pathlib.Path(path, part_filename)))

        tf_example = image_to_tf_example(x, y, feature_id=current_id)
        writer.write(tf_example.SerializeToString())
        current_part_count += 1
        if write_id:
            current_id += 1

        if current_part_count >= shard_size:
            print(f"Wrote {current_part_count} records to {part_filename}")
            current_part_count = 0
            current_part += 1
            writer = None

    if current_part_count > 0:
        print(f"Wrote {current_part_count} records to {part_filename} (last part)")


def parse_tf_example(tf_example, feature_description):
    return tf.io.parse_single_example(tf_example, feature_description)


def load_dataset_from_tfrecord(filename, path=None, load_id=False, num_parallel_reads=None):
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
            return example["id"], (tf.reshape(image_tensor, shape=[example["height"], example["width"], 3]), example["label"])
    else:
        def parse_image_tf_example(tf_example):
            example = parse_tf_example(tf_example, image_feature_description)
            image_tensor = tf.io.parse_tensor(example["image"], out_type=tf.float32)
            return tf.reshape(image_tensor, shape=[example["height"], example["width"], 3]), example["label"]

    # Get filenames
    parts = sorted([str(p) for p in list(pathlib.Path(path).rglob(f"{filename}.tfrecord-*"))])
    if not parts:
        raise ValueError(f"No records found at {path}")

    dataset = tf.data.TFRecordDataset(parts, num_parallel_reads=num_parallel_reads)
    parsed_dataset = dataset.map(parse_image_tf_example)

    return parsed_dataset
