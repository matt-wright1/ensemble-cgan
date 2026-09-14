import argparse
import glob
import os

import tensorflow as tf
import yaml


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


def generate_tfrecords(
        years,
        write_data,
        read_config):

    data_paths = read_config.get_data_paths()
    records_folder = data_paths["TFRecords"]["tfrecords_path"]

    os.makedirs(records_folder, exist_ok=True)

    print(f"TFRecord folder: {records_folder}")

    # ------------------------------------------------------------
    # Generate TFRecords
    # ------------------------------------------------------------

    for year in years:

        print(f"Writing {year}")

        write_data(int(year))

    # ------------------------------------------------------------
    # Inspect generated files
    # ------------------------------------------------------------

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
                print(
                    "  size MB:",
                    os.path.getsize(f) / 1e6
                )

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
                            f"bytes_list len="
                            f"{len(v.bytes_list.value)}"
                        )

                    elif kind == "float_list":
                        print(
                            f"    {k}: "
                            f"float_list len="
                            f"{len(v.float_list.value)}"
                        )

                    elif kind == "int64_list":
                        print(
                            f"    {k}: "
                            f"int64_list len="
                            f"{len(v.int64_list.value)}"
                        )

            except Exception as e:
                print("  BAD:", e)


if __name__ == "__main__":

    # ------------------------------------------------------------
    # Read experiment config FIRST
    # ------------------------------------------------------------

    args = parse_args()

    config = load_experiment_config(args.config)

    # ------------------------------------------------------------
    # Select local config for THIS process
    # ------------------------------------------------------------

    local_config_path = config["GENERAL"]["local_config_path"]

    if not os.path.isabs(local_config_path):
        local_config_path = os.path.join(
            os.path.dirname(os.path.abspath(args.config)),
            local_config_path,
        )

    if not os.path.isfile(local_config_path):
        raise FileNotFoundError(
            f"Local config does not exist: {local_config_path}"
        )

    os.environ["CGAN_LOCAL_CONFIG"] = local_config_path

    print(f"Experiment config: {os.path.abspath(args.config)}")
    print(f"Local config:      {local_config_path}")

    # ------------------------------------------------------------
    # NOW import modules that read local_config
    # ------------------------------------------------------------

    import read_config
    from tfrecords_generator import write_data

    # ------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------

    generate_tfrecords(
        years=args.years,
        write_data=write_data,
        read_config=read_config,
    )