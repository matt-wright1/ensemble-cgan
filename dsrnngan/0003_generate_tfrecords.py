import argparse
import glob
import os

import tensorflow as tf
import yaml

import read_config
from tfrecords_generator import write_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate TFRecords for cGAN training."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML file."
    )

    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2018, 2019, 2020, 2021],
        help="Years to generate TFRecords for."
    )

    return parser.parse_args()


def load_experiment_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_tfrecords(years, constants_list):

    data_paths = read_config.get_data_paths()
    records_folder = data_paths["TFRecords"]["tfrecords_path"]

    os.makedirs(records_folder, exist_ok=True)

    print("Constants:", constants_list)
    print("Number of constant fields:", len(constants_list))

    # Generate TFRecords
    for year in years:
        print(f"Writing {year}")

        write_data(
            int(year),
            constants_list=constants_list
        )

    # Inspect generated files
    for year in years:

        print(year)

        files = glob.glob(
            f"{records_folder}/{year}_1.*.tfrecords"
        )

        for f in files:

            print(f)

            try:
                ds = tf.data.TFRecordDataset(
                    f,
                    compression_type="GZIP"
                )

                n = sum(1 for _ in ds)

                print("  total records:", n)
                print("  size MB:", os.path.getsize(f) / 1e6)

                # Re-open to inspect first record
                ds = tf.data.TFRecordDataset(
                    f,
                    compression_type="GZIP"
                )

                first_raw = next(iter(ds.take(1)))

                first_ex = tf.train.Example.FromString(
                    first_raw.numpy()
                )

                print(
                    "  first record keys:",
                    list(first_ex.features.feature.keys())
                )

                for k, v in first_ex.features.feature.items():

                    kind = v.WhichOneof("kind")

                    if kind == "bytes_list":
                        print(
                            f"    {k}: "
                            f"bytes_list len={len(v.bytes_list.value)}"
                        )

                    elif kind == "float_list":
                        print(
                            f"    {k}: "
                            f"float_list len={len(v.float_list.value)}"
                        )

                    elif kind == "int64_list":
                        print(
                            f"    {k}: "
                            f"int64_list len={len(v.int64_list.value)}"
                        )

            except Exception as e:
                print("  BAD:", e)


if __name__ == "__main__":

    args = parse_args()

    config = load_experiment_config(args.config)

    constants_list = config["CONSTANTS"]["constants_list"]

    generate_tfrecords(
        years=args.years,
        constants_list=constants_list
    )