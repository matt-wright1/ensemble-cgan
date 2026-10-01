#!/usr/bin/env python
# coding: utf-8

# Big warning:
# This is not a general-purpose forecast script.
# This is for forecasting on the pre-defined 'ICPAC region' (e.g., the latitudes
# and longitudes are hard-coded), and assumes the input forecast data starts at
# time 0, with time steps of data.HOURS.
# A more robust version of this script would parse the latitudes, longitudes, and
# forecast time info from the input file.
# The forecast data fields must match those defined in data.all_fcst_fields

import os
import argparse
import json
import pathlib
import yaml
from datetime import datetime, date, timedelta
import properscoring as ps

import netCDF4 as nc
import numpy as np
from tensorflow.keras.utils import Progbar

#Imports
from data import HOURS, denormalise, load_hires_constants, load_truth_and_mask, load_fcst_norm, logprec
import read_config
from noise import NoiseGenerator
from setupmodel import setup_model

log_precip = True

#Parse command line args
parser = argparse.ArgumentParser()
parser.add_argument(
    "--date",
    required=True,
    help="Forecast initialisation date YYYYMMDD"
)
parser.add_argument(
    "--time",
    type=str,
    choices=["0000", "0600", "1200", "1800"],
    default="0000",
    help="Forecast initialisation time in hours UTC (default: 0)"
)
parser.add_argument(
    "--leadtime",
    nargs="+",
    type=int,
    default=None,
    help="Forecast lead times in hours"
)
parser.add_argument(
    "--accumulation",
    choices=[6, 24],
    type=int,
    default=6,
    help="Accumulation of forecasts in hours (default: 6)"
)
parser.add_argument(
    "--fcst_yaml_file",
    default=None,
    help="Path to forecast configuration YAML file. If omitted, selected automatically based on accumulation and leadtime."
)
parser.add_argument(
    "--save_crps",
    action="store_true",
    help="Whether to save CRPS for these forecasts or not (default: False)."
)
parser.add_argument(
    "--n_ens",
    default=1000,
    type=int,
    help="Number of ensemble members to produce (default: 1000)."
)
args = parser.parse_args()

d = datetime.strptime(args.date, "%Y%m%d").date()
hour = int(args.time[:2])

accumulation = args.accumulation

if args.leadtime is None:
    if args.accumulation == 6:
        args.leadtime = [30, 36, 42, 48]
    elif args.accumulation == 24:
        args.leadtime = [6, 30, 54, 78, 102, 126, 150]

leadtime = args.leadtime

if args.fcst_yaml_file is None:
    if args.accumulation == 6:
        args.fcst_yaml_file = "forecast_6haccum_default.yaml"
    elif args.accumulation == 24:
        args.fcst_yaml_file = "forecast_24haccum_default.yaml"

ensemble_members = args.n_ens
save_crps = args.save_crps

with open(args.fcst_yaml_file, "r") as f:
    try:
        fcst_params = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)
        raise

# Setup
read_config.set_gpu_mode()  # set up whether to use GPU, and mem alloc mode
downscaling_steps = read_config.read_downscaling_factor()["steps"]

model_folder = fcst_params["MODEL"]["folder"]
checkpoint = fcst_params["MODEL"]["checkpoint"]

fcst_input_folder = fcst_params["INPUT"]["fcst_folder"]
truth_input_folder = fcst_params["INPUT"]["truth_folder"]
constants_folder = fcst_params["INPUT"]["constants_folder"]
normalisation_folder = fcst_params["INPUT"]["normalisation_folder"]

all_fcst_fields = fcst_params["DATA"]["all_fcst_fields"]
accumulated_fields = fcst_params["DATA"]["accumulated_fields"]
nonnegative_fields = fcst_params["DATA"]["nonnegative_fields"]
crop_to_bounds = fcst_params["DATA"]["crop_to_bounds"]
bounds = fcst_params["DATA"]["bounds"]

output_folder = fcst_params["OUTPUT"]["folder"]

if truth_input_folder == '/to/edit/if/you/want/to/calculate/crps' and save_crps:
    raise ValueError("If you want to calculate and save the CRPS, you need to download and specify the location of truth data. Remove --save_crps flag to run forecasts without calculating CRPS.")


local_fcst_norm = load_fcst_norm(year=2018, normalisation_path=normalisation_folder)
assert local_fcst_norm is not None

# Open and parse GAN config file
config_path = os.path.join(model_folder, "setup_params.yaml")
with open(config_path, "r") as f:
    try:
        setup_params = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

mode = setup_params["GENERAL"]["mode"]
arch = setup_params["MODEL"]["architecture"]
padding = setup_params["MODEL"]["padding"]
filters_gen = setup_params["GENERATOR"]["filters_gen"]
noise_channels = setup_params["GENERATOR"]["noise_channels"]
latent_variables = setup_params["GENERATOR"]["latent_variables"]
filters_disc = setup_params["DISCRIMINATOR"]["filters_disc"]

constant_fields = 2

assert mode == "GAN", "standalone forecast script only for GAN, not VAE-GAN or deterministic model"

# Set up pre-trained GAN
weights_fn = os.path.join(model_folder, "models", f"gen_weights-{checkpoint:07}.h5")
input_channels = 2*len(all_fcst_fields)

model = setup_model(mode=mode,
                    arch=arch,
                    downscaling_steps=downscaling_steps,
                    input_channels=input_channels,
                    constant_fields=constant_fields,
                    filters_gen=filters_gen,
                    filters_disc=filters_disc,
                    noise_channels=noise_channels,
                    latent_variables=latent_variables,
                    padding=padding)
gen = model.gen
print(weights_fn)
gen.load_weights(weights_fn)

network_const_input = load_hires_constants(batch_size=1, constants_path=constants_folder)  # 1 x lats x lons x 2


def create_output_file(nc_out_path):
    netcdf_dict = {}
    rootgrp = nc.Dataset(nc_out_path, "w", format="NETCDF4")
    netcdf_dict["rootgrp"] = rootgrp
    rootgrp.description = "GAN 24-hour rainfall ensemble members in the ICPAC region."

    # Create output file dimensions
    rootgrp.createDimension("latitude", len(latitude))
    rootgrp.createDimension("longitude", len(longitude))
    rootgrp.createDimension("time", None)
    rootgrp.createDimension("valid_time", None)
    rootgrp.createDimension("member", ensemble_members)

    # Create coordinate variables
    latitude_data = rootgrp.createVariable("latitude", "f4", ("latitude",))
    latitude_data.units = "degrees_north"
    latitude_data[:] = latitude

    longitude_data = rootgrp.createVariable("longitude", "f4", ("longitude",))
    longitude_data.units = "degrees_east"
    longitude_data[:] = longitude

    ensemble_data = rootgrp.createVariable("member", "i4", ("member",))
    ensemble_data.units = "ensemble member"
    ensemble_data[:] = range(1, ensemble_members + 1)

    netcdf_dict["time_data"] = rootgrp.createVariable("time", "f4", ("time",))
    netcdf_dict["time_data"].units = "hours since 1900-01-01 00:00:00.0"

    netcdf_dict["valid_time_data"] = rootgrp.createVariable(
        "fcst_valid_time", "f4", ("time", "valid_time")
    )
    netcdf_dict["valid_time_data"].units = "hours since 1900-01-01 00:00:00.0"

    # Ensemble precipitation output
    netcdf_dict["precipitation"] = rootgrp.createVariable(
        "precipitation",
        "f4",
        ("time", "member", "valid_time", "latitude", "longitude"),
        compression="zlib",
        chunksizes=(1, 1, 1, len(latitude), len(longitude)),
    )
    netcdf_dict["precipitation"].units = "mm/h"
    netcdf_dict["precipitation"].long_name = "Precipitation"

    # CRPS output
    if save_crps:
        netcdf_dict["crps"] = rootgrp.createVariable(
            "crps",
            "f4",
            ("time", "valid_time", "latitude", "longitude"),
            compression="zlib",
            chunksizes=(1, 1, len(latitude), len(longitude)),
        )
        netcdf_dict["crps"].units = "mm/h"
        netcdf_dict["crps"].long_name = "CRPS for precipitation"

    return netcdf_dict

#Make the forecast for the right data

# Open input netCDF file to get the times
file_name = os.path.join(
    fcst_input_folder,
    str(d.year),
    f"IFS_{args.date}_{hour:02d}Z.nc"
)

print(f"Opening forecast file: {file_name}")
nc_in = nc.Dataset(file_name, mode="r")
start_times = nc_in["time"][:]
valid_times = nc_in["valid_time"][:]
latitude = nc_in["latitude"][:]
longitude = nc_in["longitude"][:]

if crop_to_bounds:
    lat_min, lon_min, lat_max, lon_max = bounds

    # Boolean masks work whether coordinates are ascending or descending
    lat_mask = (latitude >= lat_min) & (latitude <= lat_max)
    lon_mask = (longitude >= lon_min) & (longitude <= lon_max)

    if not lat_mask.any():
        raise ValueError(
            f"No latitude values found within bounds {lat_min} to {lat_max}."
        )

    if not lon_mask.any():
        raise ValueError(
            f"No longitude values found within bounds {lon_min} to {lon_max}."
        )

    latitude = latitude[lat_mask]
    longitude = longitude[lon_mask]

# Create output netCDF file
pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
nc_out_path = os.path.join(output_folder, f"GAN_{d.year}{d.month:02d}{d.day:02d}_{hour:02d}Z.nc")
netcdf_dict = create_output_file(nc_out_path)

netcdf_dict["time_data"][0] = start_times[0]

# Get the valid-time indices corresponding to each requested lead time
valid_time_idx = [int(lt / HOURS) for lt in leadtime]

valid_times_forecast = valid_times[valid_time_idx]

print("Lead times:", leadtime)
print("Valid time indices:", valid_time_idx)
print("Valid times:", valid_times_forecast)

# Store all valid times in the output file
netcdf_dict["valid_time_data"][0, :] = valid_times_forecast

# For each valid time
for valid_time_num, current_leadtime in enumerate(leadtime):
    print(
        f"Producing forecast {valid_time_num + 1}/{len(leadtime)}: "
        f"lead time {current_leadtime}h"
    )

    field_arrays = []
    
    for field in all_fcst_fields:

        # Ensemble mean and standard deviation are already stored in the
        # combined IFS forecast file.
        all_data_mean = nc_in[f"{field}_ensemble_mean"]
        all_data_sd = nc_in[f"{field}_ensemble_standard_deviation"]

        # Convert lead time to 6-hourly index:
        # 6h -> 1, 30h -> 5, 36h -> 6, etc.
        lead_idx = int(current_leadtime / HOURS)

        if accumulation == 24:

            if field in accumulated_fields:
                # Four 6-hour periods covering the 24-hour target
                temp_data_mean = all_data_mean[
                    lead_idx:lead_idx + 4, :, :
                ]

                temp_data_var = (
                    all_data_sd[
                        lead_idx:lead_idx + 4, :, :
                    ] ** 2
                )

                data1 = np.mean(temp_data_mean, axis=0)
                data2 = np.sqrt(np.mean(temp_data_var, axis=0))

                data = np.stack([data1, data2], axis=-1)

            else:
                # Five instantaneous values:
                # t, t+6, t+12, t+18, t+24
                temp_data_mean = all_data_mean[
                    lead_idx:lead_idx + 5, :, :
                ]

                temp_data_var = (
                    all_data_sd[
                        lead_idx:lead_idx + 5, :, :
                    ] ** 2
                )

                data1 = (
                    temp_data_mean[0] / 2
                    + np.sum(temp_data_mean[1:4], axis=0)
                    + temp_data_mean[4] / 2
                ) / 4

                data2 = (
                    temp_data_var[0] / 2
                    + np.sum(temp_data_var[1:4], axis=0)
                    + temp_data_var[4] / 2
                ) / 4

                data = np.stack(
                    [data1, np.sqrt(data2)],
                    axis=-1
                )

        elif accumulation == 6:

            if field in accumulated_fields:
                # Accumulated fields already represent the 6-hour interval.
                data1 = all_data_mean[lead_idx, :, :]
                data2 = all_data_sd[lead_idx, :, :]

            else:
                # Instantaneous fields: average the start and end of the
                # 6-hour interval using the trapezium rule.
                temp_data_mean = all_data_mean[
                    lead_idx:lead_idx + 2, :, :
                ]

                temp_data_var = (
                    all_data_sd[
                        lead_idx:lead_idx + 2, :, :
                    ] ** 2
                )

                data1 = (
                    temp_data_mean[0] + temp_data_mean[1]
                ) / 2

                data2 = (
                    temp_data_var[0] + temp_data_var[1]
                ) / 2

                data2 = np.sqrt(data2)

            data = np.stack([data1, data2], axis=-1)

        else:
            raise ValueError(
                f"Unsupported accumulation period: {accumulation}"
            )

        # Crop the forecast fields to the same region as the output grid.
        if crop_to_bounds:
            data = data[lat_mask, :, :]
            data = data[:, lon_mask, :]

        # Eliminate any small negative values caused by regridding/numerics.
        if field in nonnegative_fields:
            data = np.maximum(data, 0.0)

        # Unit conversions
        if field in ["tp", "cp"]:
            # Precipitation is in metres accumulated over 6 hours.
            # Convert to mm/hr.
            data *= 1000
            data /= HOURS

        elif field in accumulated_fields:
            # Other accumulated fields (e.g. ssr):
            # convert 6-hour accumulation to per-second rate.
            data /= HOURS * 3600

        # Normalisation
        if field in ["tp", "cp"]:
            data = logprec(data, True)

        elif field in ["mcc", "tcc"]:
            # Already bounded between 0 and 1.
            pass

        elif field in ["sp", "t2m"]:
            # Subtract the historical mean from the ensemble mean only.
            # Do not subtract it from the ensemble standard deviation.
            data[:, :, 0] -= local_fcst_norm[field]["mean"]

            # Scale both ensemble mean and ensemble standard deviation.
            data /= local_fcst_norm[field]["std"]

        elif field in nonnegative_fields:
            data /= local_fcst_norm[field]["max"]

        else:
            # Wind fields
            data /= max(
                -local_fcst_norm[field]["min"],
                local_fcst_norm[field]["max"]
            )

        field_arrays.append(data)
    
    network_fcst_input = np.concatenate(field_arrays, axis=-1)  # lat x lon x 2*len(all_fcst_fields)
    network_fcst_input = np.expand_dims(network_fcst_input, axis=0)  # 1 x lat x lon x 2*len(...)
    
    noise_shape = network_fcst_input.shape[1:-1] + (noise_channels,)
    noise_gen = NoiseGenerator(noise_shape, batch_size=1)
    z = noise_gen()
    progbar = Progbar(ensemble_members)

    ens_cgan_preds = []
    for ii in range(ensemble_members):
        gan_inputs = [network_fcst_input, network_const_input, noise_gen()]
        gan_prediction = gen.predict(gan_inputs, verbose=False)  # 1 x lat x lon x 1
        pred = denormalise(gan_prediction[0, :, :, 0])
        netcdf_dict["precipitation"][0, ii, valid_time_num, :, :] = pred
        
        ens_cgan_preds.append(pred)
        progbar.add(1)
        
    #Calculate and save CRPS if requested
    if save_crps:

        ens_cgan_preds_stacked = np.stack(ens_cgan_preds, axis=0)
        
        #load relevant truth data
        truth_data, _ = load_truth_and_mask(d.strftime('%Y%m%d'), leadtime=current_leadtime, log_precip=log_precip, truth_path=truth_input_folder)
        if truth_data.ndim == 3 and truth_data.shape[0] == 1:
            truth_data = truth_data[0]
        print(f"shape truth = {np.shape(truth_data)}")
        print(f"shape ens_cgan_preds_stacked = {np.shape(ens_cgan_preds_stacked)}")
        crps = ps.crps_ensemble(
            truth_data,
            ens_cgan_preds_stacked,
            axis=0
        )

        netcdf_dict["crps"][0, valid_time_num, :, :] = crps

#Close file after all leadtimes are done
netcdf_dict["rootgrp"].close()






