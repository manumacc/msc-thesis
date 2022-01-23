import sys

sys.path.append('src')

import pathlib
import random

import tensorflow as tf

DATA_GENERATED_PATH = pathlib.Path("data", "caltech-20-10-p10")
PATH_ORIG = "data/256_ObjectCategories"

LABEL_COUNTS = {0: 98, 1: 97, 2: 151, 3: 127, 4: 148, 5: 90, 6: 106, 7: 232, 8: 102, 9: 94, 10: 278, 11: 216, 12: 98,
                13: 86, 14: 122, 15: 91, 16: 104, 17: 101, 18: 124, 19: 83, 20: 142, 21: 97, 22: 110, 23: 112, 24: 114,
                25: 106, 26: 100, 27: 110, 28: 103, 29: 104, 30: 90, 31: 101, 32: 102, 33: 100, 34: 87, 35: 106,
                36: 120, 37: 110, 38: 85, 39: 124, 40: 87, 41: 87, 42: 124, 43: 121, 44: 85, 45: 133, 46: 94, 47: 103,
                48: 106, 49: 97, 50: 114, 51: 85, 52: 82, 53: 118, 54: 98, 55: 102, 56: 106, 57: 93, 58: 83, 59: 87,
                60: 102, 61: 83, 62: 122, 63: 131, 64: 101, 65: 83, 66: 83, 67: 110, 68: 99, 69: 84, 70: 99, 71: 118,
                72: 100, 73: 115, 74: 83, 75: 84, 76: 92, 77: 90, 78: 99, 79: 116, 80: 95, 81: 81, 82: 95, 83: 84,
                84: 112, 85: 80, 86: 93, 87: 98, 88: 110, 89: 212, 90: 95, 91: 201, 92: 112, 93: 104, 94: 86, 95: 285,
                96: 89, 97: 100, 98: 80, 99: 93, 100: 138, 101: 88, 102: 111, 103: 97, 104: 270, 105: 87, 106: 89,
                107: 85, 108: 156, 109: 85, 110: 84, 111: 84, 112: 116, 113: 120, 114: 88, 115: 107, 116: 121, 117: 108,
                118: 87, 119: 130, 120: 82, 121: 103, 122: 111, 123: 91, 124: 101, 125: 242, 126: 128, 127: 105,
                128: 190, 129: 91, 130: 92, 131: 190, 132: 136, 133: 119, 134: 93, 135: 93, 136: 156, 137: 192, 138: 86,
                139: 89, 140: 117, 141: 107, 142: 130, 143: 82, 144: 798, 145: 82, 146: 202, 147: 174, 148: 103,
                149: 111, 150: 109, 151: 120, 152: 93, 153: 103, 154: 92, 155: 96, 156: 105, 157: 149, 158: 209,
                159: 83, 160: 103, 161: 91, 162: 90, 163: 101, 164: 88, 165: 92, 166: 86, 167: 140, 168: 92, 169: 102,
                170: 84, 171: 99, 172: 106, 173: 84, 174: 83, 175: 110, 176: 96, 177: 98, 178: 80, 179: 102, 180: 100,
                181: 120, 182: 100, 183: 84, 184: 103, 185: 81, 186: 95, 187: 88, 188: 119, 189: 112, 190: 111,
                191: 112, 192: 174, 193: 112, 194: 87, 195: 107, 196: 100, 197: 108, 198: 105, 199: 100, 200: 81,
                201: 97, 202: 91, 203: 80, 204: 87, 205: 98, 206: 115, 207: 109, 208: 102, 209: 111, 210: 95, 211: 136,
                212: 101, 213: 139, 214: 84, 215: 98, 216: 105, 217: 81, 218: 84, 219: 94, 220: 103, 221: 91, 222: 80,
                223: 110, 224: 90, 225: 99, 226: 147, 227: 95, 228: 95, 229: 94, 230: 112, 231: 358, 232: 100, 233: 122,
                234: 114, 235: 97, 236: 90, 237: 97, 238: 84, 239: 201, 240: 95, 241: 93, 242: 90, 243: 91, 244: 91,
                245: 101, 246: 92, 247: 84, 248: 100, 249: 96, 250: 800, 251: 116, 252: 435, 253: 95, 254: 103,
                255: 108}


# Caltech-256 total elements: 29783 -- no background noise class

# > Version 20 test, 10 val, p10 init (caltech-20-10-p10)
#   Total elements for AL loop: 29783 - 20*256 - 10*256 = 22103 - 10% =~ 20000 elements
#   ==> ~20000 elements total for AL loop, starting from 10%.
#       1200 per loop: ~ +5%.

# > Version 20 test, 10 val, p20 init (caltech-test20-val10-p20)
#   Total elements for AL loop: 29783 - 20*256 - 10*256 = 22103 - 20% =~ 17500 elements
#   ==> ~17500 elements total for AL loop, starting from 20%.
#       900 per loop: ~ +5%.


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
    n_shards_init, n_shards_val, n_shards_pool, n_shards_test = 10, 5, 100, 20
    return n_shards_init, n_shards_val, n_shards_pool, n_shards_test


def generate_dataset():
    tf.random.set_seed(0)

    # Initialize dataset
    print("Initialize dataset")

    ds_orig = tf.keras.preprocessing.image_dataset_from_directory(
        directory=PATH_ORIG,
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
    n_classes = 256

    print("Initializing writers")
    for subset in writers:
        part_filename_format = f"caltech-{subset}.tfrecord"
        print(part_filename_format)
        for i in range(n_shards[subset]):
            part_filename = f"{part_filename_format}-{i + 1:05d}-of-{n_shards[subset]:05d}"
            writers[subset].append(
                tf.io.TFRecordWriter(str(pathlib.Path(path, part_filename)))
            )

    def _select_random_shard(sset):
        return random.choice(list(range(n_shards[sset])))

    n_samples_test = 20  # number of samples for test per class
    n_samples_val = 10  # number of samples for val per class

    p_samples_init = 0.10  # percent of samples to reserve for initial training set

    init_label_count = {k: 0 for k in range(n_classes)}
    pool_label_count = {k: 0 for k in range(n_classes)}
    test_label_count = {k: 0 for k in range(n_classes)}
    val_label_count = {k: 0 for k in range(n_classes)}
    current_pool_id = 0

    total_count = 0
    for tf_x, tf_y in ds_orig:
        total_count += 1
        if total_count % 100 == 0:
            print(f">>> Processing image {total_count}")

        y = tf_y.numpy()

        # First of all, fill test and val
        if test_label_count[y] < n_samples_test:
            tf_example = image_to_tf_example(tf_x, tf_y, feature_id=None)
            shard_id = _select_random_shard("test")
            writers["test"][shard_id].write(tf_example.SerializeToString())
            test_label_count[y] += 1
            if test_label_count[y] == n_samples_test:
                print(f"Wrote {test_label_count[y]} samples of class {y} to test")
            continue

        elif val_label_count[y] < n_samples_val:
            tf_example = image_to_tf_example(tf_x, tf_y, feature_id=None)
            shard_id = _select_random_shard("val")
            writers["val"][shard_id].write(tf_example.SerializeToString())
            val_label_count[y] += 1
            if val_label_count[y] == n_samples_val:
                print(f"Wrote {val_label_count[y]} samples of class {y} to val")
            continue

        # Now fill init and pool, in the proportions specified
        n_samples_init_per_class = {
            k: int((LABEL_COUNTS[k] - n_samples_test - n_samples_val) * p_samples_init)
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

    print("Test label count")
    print(test_label_count)
    print("Val label count")
    print(val_label_count)
    print("Init label count")
    print(init_label_count)
    print("Pool label count")
    print(pool_label_count)


if __name__ == '__main__':
    generate_dataset()
