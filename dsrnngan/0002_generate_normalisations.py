import argparse
import os
import yaml


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