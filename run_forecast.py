#!/usr/bin/env python

# Python script to run the forecasts for a given date and time
#
# Requirements:
#
#    Conda environment for cGAN installed and activated.
#
# Before use:
# 
#    conda activate tf215gpu
#
# Usage examples:
#
# Run todays 6h forecasts initialised at 0000
#    python run_forecast.py
#    python run_forecast.py --accumulation 6h
#
# Run the 6h forecasts for the 10th of February 2025 initialised at 1800
#    python run_forecast.py --date 20250210 --time 1800
#
# Run the most recent 24h forecasts
#    python run_forecast.py --accumulation 24h
#
# Run the 24h forecasts for the 10th of February 2025
#    python run_forecast.py --accumulation 24h --date 20250210
#
# Run todays 6h forecasts initialised at 0000 and delete the forecasts once
# statistics have been computed
#    python run_forecast.py --delete_forecasts Y

import argparse
import sys
import os
import subprocess
import pathlib
import datetime
import platform


# Parse arguments to this script
def parseArguments():

    parser = argparse.ArgumentParser(description="""Requirements:

    Conda environment for cGAN installed and activated.

 Before use:
 
    conda activate tf215gpu

 Usage examples:

 Run todays 6h forecasts initialised at 0000
    python run_forecast.py
    python run_forecast.py --accumulation 6h

 Run the 6h forecasts for the 10th of February 2025 initialised at 1800
    python run_forecast.py --date 20250210 --time 1800

 Run the most recent 24h forecasts
    python run_forecast.py --accumulation 24h

 Run the 24h forecasts for the 10th of February 2025
    python run_forecast.py --accumulation 24h --date 20250210  
    
 Run todays 6h forecasts initialised at 0000 and delete the forecasts once
 statistics have been computed
    python run_forecast.py --delete_forecasts Y  
    """, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '--accumulation',
        help='How long rainfall is accumulated for, either 6h or 24h',
        default=None,
        type=str
    )

    parser.add_argument(
        '--date',
        help='Forecast initialisation date (YYYYMMDD)',
        default=None,
        type=str
    )

    parser.add_argument(
        '--time',
        help='Forecast initialisation time (HHMM)',
        default=None,
        type=str
    )

    parser.add_argument(
        '--delete_forecasts',
        help='Should forecasts be deleted or not (Y/N)',
        default=None,
        type=str
    )

    # New cGAN arguments
    parser.add_argument(
        '--save_crps',
        action='store_true',
        help='Calculate and save CRPS in the cGAN forecast file.'
    )

    parser.add_argument(
        '--n_ens',
        type=int,
        default=1000,
        help='Number of cGAN ensemble members to produce (default: 1000).'
    )

    parser.add_argument(
        '--leadtime',
        nargs='+',
        type=int,
        default=None,
        help='Forecast lead times in hours. '
            'If omitted, forecast_date.py selects defaults based on accumulation.'
    )

    parser.add_argument(
        '--fcst_yaml_file',
        type=str,
        default=None,
        help='Optional forecast YAML file. '
            'If omitted, forecast_date.py selects the default based on accumulation.'
    )

    args = parser.parse_args()
    
    # Parse the accumulation
    if (args.accumulation is not None):
        
        if (args.accumulation == '6h') or (args.accumulation == '6'):
            accumulation_time = 6
            
        elif (args.accumulation == '24h') or (args.accumulation == '24'):
            accumulation_time = 24
            
        else:
            print("ERROR: Incorrect accumulation.")
            print("       Available accumulations 6h, 24h.")
            parser.print_help()
            sys.exit()
        
    else:
        
        # Default 6h accumulation
        accumulation_time = 6
    
    # Parse the date
    if (args.date is not None):
    
        if (len(args.date) != 8):
            print("ERROR: Incorrect date.")
            parser.print_help()
            sys.exit()
        
        year = int(args.date[0:4])
        month = int(args.date[4:6])
        day = int(args.date[6:8])
        
    else:
    
        # Default today
        d = datetime.datetime.today()
        year = d.year
        month = d.month
        day = d.day
    
    # Parse the time
    if (args.time is not None):
    
        if (len(args.time) != 4):
            print("ERROR: Incorrect time.")
            parser.print_help()
            sys.exit()
            
        hour = int(args.time[0:2])
        minute = int(args.time[2:4])
        
        if (accumulation_time == 6):
            if (hour not in [0,6,12,18]) or (minute != 0):
                print("ERROR: Incorrect time.")
                print("       Available initialisation times are 0000, 0600, 1200, 1800 for 6h accumulations.")
                parser.print_help()
                sys.exit()
                
        elif (accumulation_time == 24):
            if (hour != 0):
                print("ERROR: Incorrect time.")
                print("       Available initialisation time is 0000 for 24h accumulations.")
                parser.print_help()
                sys.exit()
                
    else:
        
        # Default 0000
        hour = 0
        minute = 0
        
    # Parse delete_forecasts
    delete_forecasts = False  # Default
    if (args.delete_forecasts is not None):
        
        if ((args.delete_forecasts == "T") or (args.delete_forecasts == "t") or
            (args.delete_forecasts == "Y") or (args.delete_forecasts == "y")):
            delete_forecasts = True

    run_ELR = False
    
    return accumulation_time, year, month, day, hour, minute, delete_forecasts, run_ELR, args.save_crps, args.n_ens, args.leadtime, args.fcst_yaml_file

# Checks that all of the histogram counts files for this date and time are there or not.
# Arguments:
#    counts_path - The directory to check.
#    date_str    - A string containing the initialisation date (YYYYMMDD).
#    hour        - An integer corresponding to initialisation hour of the day.
#    valid_hours - A list of the valid_hours that the forecast is computed at.
# Returns:
#    If all files that should be there, are there.
def check_counts_files(counts_path, date_str, hour, valid_hours):
    
    # Extract the year from date_str
    year = date_str[0:4]
    
    # Check to see if each file exists
    num_files_exist = 0
    for i in valid_hours:
        file_name = f"counts_{date_str}_{hour:02d}_{i}h.nc"
        exists = os.path.isfile(f"{counts_path}/{year}/{file_name}")
        if (exists):
            num_files_exist += 1
            # print(f"{counts_path}/{year}/{file_name} already exists.")
    
    # Does the number of files that exist equal the number that should exist?
    correct_num_files_exist = (num_files_exist == len(valid_hours))
    
    return correct_num_files_exist


# Checks that all of the ELR output files for this date and time are there or not.
# Returns:
#    If all files that should be there, are there.
def check_ELR_files(model_path, save_path, accumulation_time, countries, 
                    country_admin_regions, date, model='GAN', day=[1]):
    num_files_expected = len(ELR_countries)*len(day)
    num_files_actual = 0
    for country in ELR_countries:
        for d in day:
            if os.path.exists(\
                save_path+f'{accumulation_time}h_accumulations/{country}/{country_admin_regions[country]}/{model}_{date}_ELR_v{d}.nc'):
                num_files_actual+=1
    
    correct_num_files_exist = (num_files_expected==num_files_actual)
    return correct_num_files_exist


if __name__=='__main__':
    
    # Parse arguments to this script
    accumulation_time, year, month, day, hour, minute, delete_forecasts, run_ELR, save_crps, n_ens, leadtime, fcst_yaml_file = parseArguments()
    
    print(f"Producing forecasts of {accumulation_time}h accumulations")
    print(f"initialised on {year}-{month:02d}-{day:02d} at {hour:02d}{minute:02d}.")
    
    # What are the valid hours of the forecast
    # What are the valid hours of the forecast
    if leadtime is not None:
        # User supplied custom lead times
        valid_hours = leadtime
    elif accumulation_time == 6:
        valid_hours = [30, 36, 42, 48]
    elif accumulation_time == 24:
        valid_hours = [6, 30, 54, 78, 102, 126, 150]
    else:
        print("ERROR: Incorrect accumulation time.")
        sys.exit()
    
    # Shorthand
    date_str = f"{year}{month:02d}{day:02d}"
    time_str = f"{hour:02d}{minute:02d}"
    
    # The SEWAA-forecasts directory
    root_dir = "."
    
    # Where IFS data for 6 hour accumulations will be stored
    IFS_data_path_6h = f"{root_dir}/6h_accumulations/IFS_forecast_data"
    
    # Where IFS data for 24 hour accumulations will be stored
    IFS_data_path_24h = f"{root_dir}/24h_accumulations/IFS_forecast_data"
    
    # Where cGAN 6h forecasts will be stored
    cGAN_forecast_path_6h = f"{root_dir}/6h_accumulations/cGAN_forecasts_6h"
    
    # Where cGAN 24h forecasts will be stored
    cGAN_forecast_path_24h = f"{root_dir}/24h_accumulations/cGAN_forecasts_24h"
    
    # Where the cGAN model forecast script is located
    cGAN_forecast_script_path = f"{root_dir}/ifs-cgan"

    # Where the ELR model script is located
    ELR_script_path = f"{root_dir}/ELR/"

    # Where the ELR models are located
    ELR_model_path = f"{root_dir}/ELR/models/"

    # Where the ELR predictions are saved
    ELR_predictions_path = f"{root_dir}/interface/ensemble_logistic_regression/ELR_predictions/"

    # Countries for ELR
    ELR_countries = ["Rwanda","Kenya","Ethiopia"]
    ELR_country_admin_regions = {"Rwanda":"county","Kenya":"subcounty","Ethiopia":"subcounty"}
    
    # Where all of the cGAN histogram counts will be stored
    cGAN_counts_path = f"{root_dir}/interface/view_forecasts/data"
    
    # Where the cGAN 6h histogram counts will be stored
    cGAN_counts_path_6h = f"{cGAN_counts_path}/counts_6h"
    
    # Where the cGAN 24h histogram counts will be stored
    cGAN_counts_path_24h = f"{cGAN_counts_path}/counts_24h"
    
    
    # Download the IFS data
    
    if accumulation_time == 6:
        IFS_data_path = IFS_data_path_6h
    elif accumulation_time == 24:
        IFS_data_path = IFS_data_path_24h
    else:
        print("ERROR: Incorrect accumulation time.")
        sys.exit()

    pathlib.Path(IFS_data_path).mkdir(exist_ok=True)

    file_name = f"IFS_{date_str}_{hour:02d}Z.nc"

    if os.path.isfile(f"{IFS_data_path}/{file_name}"):
        print(f"{IFS_data_path}/{file_name} already exists.")

    else:
        if platform.system() == "Windows":
            oblivion = "nul"
        else:
            oblivion = "/dev/null"

        file_URL = (
            f"https://rain.physics.ox.ac.uk/South_East_Africa/IFS_forecast_data/"
            f"IFS_forecast_data/{year}/{file_name}"
        )

        print(f"Checking University of Oxford for {file_name}")

        return_value = subprocess.run(
            ["curl", "-Isw", "%{http_code}", file_URL, "-o", oblivion],
            capture_output=True,
            text=True
        )

        if return_value.stdout == "200":

            print(
                f"Copying {accumulation_time}h accumulation data, "
                f"{file_name}, from University of Oxford."
            )
            print(f"to {IFS_data_path}/.")

            subprocess.run([
                "curl",
                file_URL,
                "-o",
                f"{IFS_data_path}/{file_name}"
            ])

        else:

            print(
                f"Unable to copy {file_name} from {file_URL}. "
                f"HTTP error {return_value.stdout}."
            )

            file_URL = (
                f"http://megacorr.dynu.net/South_East_Africa/IFS_forecast_data/"
                f"IFS_forecast_data/{year}/{file_name}"
            )

            print(f"Checking Fenwick's home for {file_name}")

            return_value = subprocess.run(
                ["curl", "-Isw", "%{http_code}", file_URL, "-o", oblivion],
                capture_output=True,
                text=True
            )

            if return_value.stdout == "200":

                print(
                    f"Copying {accumulation_time}h accumulation data, "
                    f"{file_name}, from Fenwick's home."
                )
                print(f"to {IFS_data_path}/.")

                subprocess.run([
                    "curl",
                    file_URL,
                    "-o",
                    f"{IFS_data_path}/{file_name}"
                ])

            else:
                print(
                    f"Unable to copy {file_name} from {file_URL}. "
                    f"HTTP error {return_value.stdout}."
                )
                sys.exit()
    
    
    # Run cGAN on this data
    if accumulation_time == 6:
        cGAN_forecast_path = cGAN_forecast_path_6h
        cGAN_counts_path_current = cGAN_counts_path_6h

    elif accumulation_time == 24:
        cGAN_forecast_path = cGAN_forecast_path_24h
        cGAN_counts_path_current = cGAN_counts_path_24h

    else:
        print("ERROR: Incorrect accumulation time.")
        sys.exit()
    
    # Check whether all expected histogram files already exist
    correct_num_counts_files = check_counts_files(
        cGAN_counts_path_current,
        date_str,
        hour,
        valid_hours
    )


    # If the histogram files already exist and forecasts are normally
    # deleted afterwards, there is no need to recreate the forecast
    if not (correct_num_counts_files and delete_forecasts):

        # Create forecast output directory if necessary
        pathlib.Path(cGAN_forecast_path).mkdir(
            parents=True,
            exist_ok=True
        )

        # All lead times are now stored in one forecast file
        file_name = f"GAN_{date_str}_{hour:02d}Z.nc"
        forecast_file = f"{cGAN_forecast_path}/{file_name}"

        # Check whether forecast has already been produced
        if os.path.isfile(forecast_file):
            print(f"{forecast_file} already exists.")

        else:
            print(
                f"Running {accumulation_time}h cGAN for "
                f"{date_str} {time_str}."
            )

            # forecast_date.py now lives in the single ifs-cgan directory
            run_dir = cGAN_forecast_script_path

            # Required/common arguments
            forecast_cmd = [
                "python",
                "forecast_date.py",
                "--date", date_str,
                "--time", time_str,
                "--accumulation", str(accumulation_time),
                "--n_ens", str(n_ens),
            ]

            # Optional CRPS calculation
            if save_crps:
                forecast_cmd.append("--save_crps")

            # Optional custom lead times.
            # If omitted, forecast_date.py selects defaults based on accumulation.
            if leadtime is not None:
                forecast_cmd.append("--leadtime")
                forecast_cmd.extend(
                    str(lt) for lt in leadtime
                )

            # Optional custom forecast YAML.
            # If omitted, forecast_date.py selects the appropriate default.
            if fcst_yaml_file is not None:
                forecast_cmd.extend([
                    "--fcst_yaml_file",
                    os.path.abspath(fcst_yaml_file)
                ])

            print("Running command:")
            print(" ".join(forecast_cmd))

            subprocess.run(
                forecast_cmd,
                cwd=run_dir,
                check=True
            )

    else:
        print(
            "Histogram counts files already exist and "
            "delete_forecasts is True; no forecast required."
        )

    # Compute the histogram counts

    # Create the counts directory if it doesn't exist
    pathlib.Path(cGAN_counts_path_current).mkdir(
        parents=True,
        exist_ok=True
    )

    # Create the year directory if it doesn't exist
    pathlib.Path(
        f"{cGAN_counts_path_current}/{year}"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    # Check whether all required histogram files already exist
    correct_num_counts_files = check_counts_files(
        cGAN_counts_path_current,
        date_str,
        hour,
        valid_hours
    )

    if not correct_num_counts_files:

        print(
            f"Computing {accumulation_time}h histograms "
            f"for {date_str} {time_str}."
        )

        run_dir = cGAN_forecast_script_path

        histogram_cmd = [
            "python",
            "forecast2histogram.py",
            date_str,
            str(hour),
            "--accumulation",
            str(accumulation_time),
        ]

        subprocess.run(
            histogram_cmd,
            cwd=root_dir,
            check=True
        )

    else:
        print("Histogram counts files already exist.")
    
    # Run ELR forecasts
    # ELR is currently disabled / not implemented for this configuration

    if run_ELR:

        # ELR currently only supports 24h accumulation forecasts
        if accumulation_time != 24:
            print("Skipping ELR: only implemented for 24h accumulations.")

        # ELR currently only supports forecasts initialised at 00Z
        elif hour != 0:
            print("Skipping ELR: only implemented for 00Z forecasts.")

        else:
            # Check whether ELR output already exists
            correct_num_ELR_files = check_ELR_files(
                ELR_model_path,
                ELR_predictions_path,
                accumulation_time,
                ELR_countries,
                ELR_country_admin_regions,
                date_str
            )

            if not correct_num_ELR_files:

                print("Running ELR 24h forecasts.")

                subprocess.run(
                    [
                        "python",
                        "run_ELR.py",
                        "--date", date_str,
                        "--model", "GAN",
                        "--accumulation", "24h_accumulations"
                    ],
                    cwd=ELR_script_path,
                    check=True
                )

            else:
                print("ELR files already exist.")

    else:
        print("ELR forecasts disabled.")
        
    # Update .JSON file for the interface

    print(
        f"Listing {accumulation_time}h counts "
        f"for the interface."
    )

    subprocess.run(
        [
            "python",
            "find_available_dates.py",
            "--accumulation",
            str(accumulation_time)
        ],
        cwd=root_dir,
        check=True
    )
    
    # ELR forecasts run only when time is 0
    if ((hour == 0) and (run_ELR)):
        print("Listing ELR available dates.")
        run_dir = "ELR"
        subprocess.run(["python", f"ELR_available_dates.py"], cwd=run_dir)
    
    # Delete the cGAN forecast
    if delete_forecasts:

        file_to_delete = os.path.join(
            cGAN_forecast_path,
            f"GAN_{date_str}_{hour:02d}Z.nc"
        )

        if os.path.isfile(file_to_delete):
            print(f"Deleting {file_to_delete}")
            os.remove(file_to_delete)

        else:
            print(
                f"Forecast file {file_to_delete} "
                f"does not exist; nothing to delete."
            )
    
    
    # Show that we are done (and haven't crashed)
    print("Script run_forecast.py is done!")