"""
Generate forecast normalisation data for configured experiment years.

This script loads an experiment configuration from a YAML file, configures
the environment variables required by the data-loading modules, selects the
experiment-specific local configuration, and generates forecast
normalisations for one or more years.

The experiment configuration is read before importing ``read_config`` and
``data`` because these modules depend on configuration values supplied
through environment variables at import time.

Usage
-----
Generate normalisations using the default years (2018--2021):

    python generate_normalisations.py --config path/to/config.yaml

Generate normalisations for specific years:

    python generate_normalisations.py \
        --config path/to/config.yaml \
        --years 2019 2020 2021

Arguments
---------
--config : str
    Path to the experiment configuration YAML file.

--years : int, optional
    Years for which forecast normalisations should be generated.
    Multiple years may be supplied. Defaults to 2018, 2019, 2020, and 2021.

Configuration
-------------
The experiment YAML file is expected to define the following entries:

    GENERAL.local_config_path
    DATA.crop_to_bounds
    DATA.bounds
    DATA.all_fcst_fields
    DATA.accumulated_fields
    DATA.nonnegative_fields

These values are exported through environment variables before importing
configuration-dependent modules.

Output
------
Normalisation files are written to the directory specified by
``GENERAL.NORMALISATION_PATH`` in the selected local configuration.

Raises
------
FileNotFoundError
    If the local configuration file specified by
    ``GENERAL.local_config_path`` does not exist.
"""

import argparse
import os
import yaml
import json

# ------------------------------------------------------------
# Parse command-line arguments
# ------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--config",
    required=True,
    help="Path to experiment configuration YAML file"
)

parser.add_argument(
    "--years",
    type=int,
    nargs="+",
    default=[2018, 2019, 2020, 2021],
    help="Years for which to generate forecast normalisations"
)

args = parser.parse_args()


# ------------------------------------------------------------
# Read experiment config BEFORE importing data/read_config
# ------------------------------------------------------------

with open(args.config, "r") as f:
    setup_params = yaml.safe_load(f)

os.environ["CGAN_CROP_TO_BOUNDS"] = str(
    setup_params["DATA"]["crop_to_bounds"]
)

os.environ["CGAN_BOUNDS"] = ",".join(
    str(x) for x in setup_params["DATA"]["bounds"]
)

os.environ["CGAN_ALL_FCST_FIELDS"] = json.dumps(
    setup_params["DATA"]["all_fcst_fields"]
)

os.environ["CGAN_ACCUMULATED_FIELDS"] = json.dumps(
    setup_params["DATA"]["accumulated_fields"]
)

os.environ["CGAN_NONNEGATIVE_FIELDS"] = json.dumps(
    setup_params["DATA"]["nonnegative_fields"]
)


# ------------------------------------------------------------
# Select local config for THIS process
# ------------------------------------------------------------

local_config_path = setup_params["GENERAL"]["local_config_path"]

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
# NOW import modules which depend on local_config
# ------------------------------------------------------------

import read_config
from data import gen_fcst_norm


def do_normalisations(years):

    data_paths = read_config.get_data_paths()
    norm_folder = data_paths["GENERAL"]["NORMALISATION_PATH"]

    os.makedirs(norm_folder, exist_ok=True)

    print(f"Normalisation folder: {norm_folder}")

    for year in years:
        print(f"Doing year {year}")
        gen_fcst_norm(year)
        print(f"Finished year {year}")

    print("Done")


if __name__ == "__main__":
    do_normalisations(args.years)