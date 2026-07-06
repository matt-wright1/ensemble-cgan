#!/usr/bin/env python
# coding: utf-8


# Same as forecast.py, but the date to process is given as a command line argument

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
import pathlib
import yaml
from datetime import datetime, date, timedelta
import properscoring as ps

import netCDF4 as nc
import numpy as np
from tensorflow.keras.utils import Progbar

from data import HOURS, LEADTIME, all_fcst_fields, fcst_norm, denormalise, load_hires_constants, load_fcst, load_truth_and_mask, load_fcst_norm
import read_config
from noise import NoiseGenerator
from setupmodel import setup_model

import xesmf as xe

start_date = date(2020, 1, 1)
end_date   = date(2020, 12, 31)
log_precip = True

# Some setup
read_config.set_gpu_mode()  # set up whether to use GPU, and mem alloc mode
data_paths = read_config.get_data_paths()  # need the constants directory
downscaling_steps = read_config.read_downscaling_factor()["steps"]

# Open and parse forecast.yaml
parser = argparse.ArgumentParser()
parser.add_argument(
    "yaml_file",
    nargs="?",
    default="forecast.yaml",
    help="Path to forecast configuration YAML file."
)
args = parser.parse_args()

with open(args.yaml_file, "r") as f:
    try:
        fcst_params = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)
        raise

model_folder = fcst_params["MODEL"]["folder"]
checkpoint = fcst_params["MODEL"]["checkpoint"]
include_cape = fcst_params["MODEL"]["include_cape"]
fcst_input_folder = fcst_params["INPUT"]["fcst_folder"]
truth_input_folder = fcst_params["INPUT"]["truth_folder"]
constants_folder = fcst_params["INPUT"]["constants_folder"]
normalisation_folder = fcst_params["INPUT"]["normalisation_folder"]
output_folder = fcst_params["OUTPUT"]["folder"]
ensemble_members = fcst_params["OUTPUT"]["ensemble_members"]
save_crps_only = fcst_params["OUTPUT"]["save_crps_only"]

local_fcst_norm = load_fcst_norm(year=2018, normalisation_path=normalisation_folder)
assert local_fcst_norm is not None

#Set up fcst_fields --     # needed for now as Fenwick only has 13 variables
if include_cape:
    all_fcst_fields = ['cape', 'cp', 'mcc', 'sp', 'ssr', 't2m', 'tciw', 'tclw', 'tcrw', 'tcw', 'tcwv', 'tp', 'u700', 'v700']
    accumulated_fields = ['cp', 'ssr', 'tp']
    nonnegative_fields = ['cape', 'cp', 'mcc', 'sp', 'ssr', 't2m', 'tciw', 'tclw', 'tcrw', 'tcw', 'tcwv', 'tp'] #MW: things that can't be below 0
else:
    all_fcst_fields = ['cp', 'mcc', 'sp', 'ssr', 't2m', 'tciw', 'tclw', 'tcrw', 'tcw', 'tcwv', 'tp', 'u700', 'v700']
    accumulated_fields = ['cp', 'ssr', 'tp']
    nonnegative_fields = ['cp', 'mcc', 'sp', 'ssr', 't2m', 'tciw', 'tclw', 'tcrw', 'tcw', 'tcwv', 'tp'] #MW: things that can't be below 0

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
# TODO: avoid setting up discriminator in forecast mode?
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

    if not save_crps_only:
        rootgrp.createDimension("member", ensemble_members)

    # Create coordinate variables
    latitude_data = rootgrp.createVariable("latitude", "f4", ("latitude",))
    latitude_data.units = "degrees_north"
    latitude_data[:] = latitude

    longitude_data = rootgrp.createVariable("longitude", "f4", ("longitude",))
    longitude_data.units = "degrees_east"
    longitude_data[:] = longitude

    if not save_crps_only:
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
    if not save_crps_only:
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

def get_crop_slices(forecast_lat, forecast_lon, truth_lat, truth_lon, tol=1e-6):
    # Find overlapping coordinate range
    lat_min = max(forecast_lat.min(), truth_lat.min())
    lat_max = min(forecast_lat.max(), truth_lat.max())
    lon_min = max(forecast_lon.min(), truth_lon.min())
    lon_max = min(forecast_lon.max(), truth_lon.max())

    # Handle ascending or descending latitude
    if forecast_lat[0] < forecast_lat[-1]:
        f_lat_idx = np.where((forecast_lat >= lat_min - tol) & (forecast_lat <= lat_max + tol))[0]
    else:
        f_lat_idx = np.where((forecast_lat <= lat_max + tol) & (forecast_lat >= lat_min - tol))[0]

    f_lon_idx = np.where((forecast_lon >= lon_min - tol) & (forecast_lon <= lon_max + tol))[0]

    if truth_lat[0] < truth_lat[-1]:
        t_lat_idx = np.where((truth_lat >= lat_min - tol) & (truth_lat <= lat_max + tol))[0]
    else:
        t_lat_idx = np.where((truth_lat <= lat_max + tol) & (truth_lat >= lat_min - tol))[0]

    t_lon_idx = np.where((truth_lon >= lon_min - tol) & (truth_lon <= lon_max + tol))[0]

    return f_lat_idx, f_lon_idx, t_lat_idx, t_lon_idx

def iter_dates(start, end):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)

#Iterate through all dates that we want to forecast/CRPS
for d in iter_dates(start_date, end_date):

    # Open input netCDF file to get the times
    file_name = os.path.join(fcst_input_folder, str(d.year), "tp.nc")
    with nc.Dataset(file_name, mode="r") as nc_in:
        start_times = nc_in["time"][:]
        valid_times = nc_in["fcst_valid_time"][:]
        fcst_lat = nc_in["latitude"][:]
        fcst_lon = nc_in["longitude"][:]
    #Truth file name
    truth_file_name = os.path.join(truth_input_folder, f"{str(d.year)}/{d.strftime('%Y%m%d')}_06.nc")
    with nc.Dataset(truth_file_name, mode="r") as nc_truth:
        latitude = nc_truth["latitude"][:]
        longitude = nc_truth["longitude"][:]
    # nc_in.close()

    # The datetime corresponding to this start time
    # d = datetime(1900,1,1) + timedelta(hours=int(start_times[0]))

    # Create output netCDF file
    pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
    if not save_crps_only:
        nc_out_path = os.path.join(output_folder, f"GAN_fcst_crps_{d.year}{d.month:02d}{d.day:02d}_00Z.nc")
    elif save_crps_only:
        nc_out_path = os.path.join(output_folder, f"GAN_crps_vs_enact_{d.year}{d.month:02d}{d.day:02d}_00Z.nc")
    netcdf_dict = create_output_file(nc_out_path)

    # copy across valid_time from input file
    # For 7x 24h forecasts with lead times of 6, 30, 54, 78, 102, 126, 150 hours
    # in_time_idx = ([1,5,9,13,17,21,25],)
    fcst_idx = d.toordinal() - date(d.year, 1, 1).toordinal()
    netcdf_dict["time_data"][0] = start_times[fcst_idx]

    valid_time_idx = ([LEADTIME/HOURS],) #for 1x24h forecast with lead time 30h
    valid_times_forecast = valid_times[fcst_idx, valid_time_idx]
    print(np.shape(valid_times_forecast))
    netcdf_dict["valid_time_data"][0,:] = valid_times_forecast

    # For each valid time
    for valid_time_num in range(len(valid_times_forecast)):
        
        # the contents of the next loop are v. similar to load_fcst from data.py,
        # but not quite the same, since that has different assumptions on how the
        # forecast data is stored.  TODO: unify the data normalisation between these?
        field_arrays = []
        for field in all_fcst_fields:
            data = load_fcst(field, d.strftime('%Y%m%d'), 0, log_precip=log_precip, norm=True, fcst_path=fcst_input_folder, fcst_norm_dict=local_fcst_norm)
            field_arrays.append(data)

        # for j, field in enumerate(all_fcst_fields):
        #     arr = field_arrays[j]
        #     print(field, arr.min(), arr.max(), arr.mean())

        # print("const input:", network_const_input.min(), network_const_input.max(), network_const_input.mean())
        
        network_fcst_input = np.concatenate(field_arrays, axis=-1)  # lat x lon x 2*len(all_fcst_fields)
        network_fcst_input = np.expand_dims(network_fcst_input, axis=0)  # 1 x lat x lon x 2*len(...)
        
        noise_shape = network_fcst_input.shape[1:-1] + (noise_channels,)
        noise_gen = NoiseGenerator(noise_shape, batch_size=1)
        z = noise_gen()
        # print("noise:", z.min(), z.max(), z.mean())
        progbar = Progbar(ensemble_members)

        ens_cgan_preds = []
        for ii in range(ensemble_members):
            gan_inputs = [network_fcst_input, network_const_input, noise_gen()]
            gan_prediction = gen.predict(gan_inputs, verbose=False)  # 1 x lat x lon x 1
            pred = denormalise(gan_prediction[0, :, :, 0])
            if not save_crps_only:
                netcdf_dict["precipitation"][0, ii, valid_time_num, :, :] = pred
            
            ens_cgan_preds.append(pred)
            progbar.add(1)
            

        ens_cgan_preds_stacked = np.stack(ens_cgan_preds, axis=0)

        #load relevant truth data
        truth_data, _ = load_truth_and_mask(d.strftime('%Y%m%d'), 0, log_precip=log_precip, truth_path=truth_input_folder)

        # f_lat_idx, f_lon_idx, t_lat_idx, t_lon_idx = get_crop_slices(
        #     fcst_lat, fcst_lon, latitude, longitude
        # )

        # forecast_crop = ens_cgan_preds_stacked[:, f_lat_idx[:, None], f_lon_idx]
        # truth_crop = truth_data[np.ix_(t_lat_idx, t_lon_idx)]

        grid_in = {
            "lat": fcst_lat,
            "lon": fcst_lon,
        }
        grid_out = {
            "lat": latitude,
            "lon": longitude,
        }

        regridder = xe.Regridder(grid_in, grid_out, "bilinear", periodic=False)

        # If forecast is (member, lat, lon)
        forecast_regridded = regridder(ens_cgan_preds_stacked)
        
        print(f"shape truth = {np.shape(truth_data)}")
        print(f"shape ens_cgan_preds_stacked = {np.shape(ens_cgan_preds_stacked)}")
        crps = ps.crps_ensemble(
            truth_data,
            forecast_regridded,
            axis=0
        )

        netcdf_dict["crps"][0, valid_time_num, :, :] = crps

    print("network_fcst_input finite:", np.isfinite(network_fcst_input).all())
    print("gan_prediction finite:", np.isfinite(gan_prediction).all())
    print("pred finite:", np.isfinite(pred).all())
    print("pred min/max:", np.nanmin(pred), np.nanmax(pred))
    netcdf_dict["rootgrp"].close()

    # Close the ECMWF forecasts NetCDF file
    # nc_in.close()






