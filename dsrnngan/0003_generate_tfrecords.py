"""
Generate and validate TFRecord files for cGAN training.

This script loads an experiment configuration from a YAML file, configures
the environment variables required by the data-processing modules, selects
the experiment-specific local configuration, and generates compressed
TFRecord files for one or more years.

The experiment and local configurations are established before importing
``read_config`` and ``tfrecords_generator`` because these modules depend on
configuration values supplied through environment variables at import time.

After generation, the script inspects each TFRecord file and reports the
number of records, file size, feature keys in the first record, and the
type and length of each feature. Files that cannot be read or parsed are
reported as invalid.

Usage
-----
Generate TFRecords using the default years (2018--2021):

    python generate_tfrecords.py --config path/to/config.yaml

Generate TFRecords for specific years:

    python generate_tfrecords.py \
        --config path/to/config.yaml \
        --years 2019 2020 2021

Arguments
---------
--config : str
    Path to the experiment configuration YAML file.

--years : int, optional
    Years for which TFRecords should be generated. Multiple years may be
    supplied. Defaults to 2018, 2019, 2020, and 2021.

Configuration
-------------
The experiment YAML file is expected to define the following entries:

    GENERAL.local_config_path
    DATA.crop_to_bounds
    DATA.bounds
    DATA.all_fcst_fields
    DATA.accumulated_fields
    DATA.nonnegative_fields
    DATA.leadtime
    DATA.accumulation

``DATA.accumulation`` must be either 6 or 24 hours.

Output
------
TFRecord files are written to the directory specified by
``TFRecords.tfrecords_path`` in the selected local configuration.

For each generated file, the script prints basic validation information,
including record count, file size, and the structure of the first
serialized ``tf.train.Example``.

Raises
------
FileNotFoundError
    If the local configuration file specified by
    ``GENERAL.local_config_path`` does not exist.

ValueError
    If ``DATA.accumulation`` is not 6 or 24 hours.
"""

import argparse
import glob
import os
import json

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
        leadtime,
        accumulation,
        constants_list,
        write_data,
        read_config):

    data_paths = read_config.get_data_paths()
    records_folder = data_paths["TFRecords"]["tfrecords_path"]

    os.makedirs(records_folder, exist_ok=True)

    print(f"TFRecord folder: {records_folder}")
    print(f"Constants: {constants_list}")
    print(f"Number of constant fields: {len(constants_list)}")

    # ------------------------------------------------------------
    # Generate TFRecords
    # ------------------------------------------------------------

    for year in years:

        print(f"Writing {year}")

        write_data(
            int(year),
            leadtime=leadtime,
            accumulation=accumulation,
            constants_list=constants_list
        )

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

    constants_list = config["CONSTANTS"]["constants_list"]

    #Set crop_to_bounds
    os.environ["CGAN_CROP_TO_BOUNDS"] = str(
        config["DATA"]["crop_to_bounds"]
    )

    os.environ["CGAN_BOUNDS"] = ",".join(
        str(x) for x in config["DATA"]["bounds"]
    )

    os.environ["CGAN_ALL_FCST_FIELDS"] = json.dumps(
        config["DATA"]["all_fcst_fields"]
    )

    os.environ["CGAN_ACCUMULATED_FIELDS"] = json.dumps(
        config["DATA"]["accumulated_fields"]
    )

    os.environ["CGAN_NONNEGATIVE_FIELDS"] = json.dumps(
        config["DATA"]["nonnegative_fields"]
    )

    all_fcst_fields = json.loads(
        os.environ.get("CGAN_ALL_FCST_FIELDS", "[]")
    )

    accumulated_fields = json.loads(
        os.environ.get("CGAN_ACCUMULATED_FIELDS", "[]")
    )

    nonnegative_fields = json.loads(
        os.environ.get("CGAN_NONNEGATIVE_FIELDS", "[]")
    )

    # ------------------------------------------------------------
    # Select local config for THIS process
    # ------------------------------------------------------------

    leadtime = config["DATA"]["leadtime"]
    if isinstance(leadtime, int):
        leadtime = [leadtime]
    else:
        leadtime = list(leadtime)
    accumulation = config["DATA"]["accumulation"]
    local_config_path = config["GENERAL"]["local_config_path"]

    if accumulation not in (6, 24):
        raise ValueError(
            f"accumulation must be 6 or 24 hours, got {accumulation}"
        )

    os.environ["LEADTIME"] = str(leadtime[0])
    os.environ["ACCUMULATION"] = str(accumulation)
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
        leadtime=leadtime,
        accumulation=accumulation,
        constants_list=constants_list,
        write_data=write_data,
        read_config=read_config,
    )