#!/usr/bin/env python
# coding: utf-8

# From the contents of the directory create the .json file detailing the avaliable dates.
# Year object contains month objects
# Month objects contain day objects
# Day objects contain time objects
# time objects hold a list of valid times

import os
import numpy as np
import json
import argparse

#Parse command line args
parser = argparse.ArgumentParser()
parser.add_argument(
    "--accumulation",
    choices=[6, 24],
    type=int,
    default=6,
    help="Accumulation of forecasts in hours (default: 6)"
)
args = parser.parse_args()
accumulation = args.accumulation

# The directory with the counts data in
counts_dir = "interface/view_forecasts/data/IFS_cGAN_counts_{accumulation}h"

# Find the years to include
counts_years = []
dirs = os.listdir(counts_dir)
for dir_name in dirs:
    if dir_name.isdigit():
        if (int(dir_name) > 2020) and (int(dir_name) < 3000):   # Optimistic!
            counts_years.append(int(dir_name))

# Where to write the .json file with the list of counts files
output_dir = counts_dir

# The list of all files in counts_dir for all years in counts_years
file_list = []
for i in range(len(counts_years)):
    file_list = file_list + os.listdir(f"{counts_dir}/{counts_years[i]}")

# Extract the dates and times from each compatible file name
times_list = []
for i in range(len(file_list)):
    file_name = file_list[i]

    # "counts_" must be the first part of the file
    if (file_name[0:7] == "counts_"):

        # Is the next part an 8 digit integer, underscore, 2 digit integer, underscore?
        if file_name[7:15].isdecimal() and (file_name[15]=='_') and file_name[16:18].isdecimal() and (file_name[18]=='_'):

            # Start time
            year = int(file_name[7:11])
            month = int(file_name[11:13])
            day = int(file_name[13:15])
            hour = int(file_name[16:18])

            # Read the next integer
            idx = 19
            valid_hour_str = ""
            while file_name[idx].isdecimal() and (idx < 22):
                valid_hour_str += file_name[idx]
                idx += 1
            valid_hour = int(valid_hour_str)

            # Is the final part of the string "h.nc"
            if (file_name[idx:idx+4] == "h.nc") and (len(file_name) == idx+4):

                # Include this file in the list
                times_list.append([year,month,day,hour,valid_hour])

# Define the sort criteria
def sortFunc(e):
    return e[0]*100*100*100*1000 + e[1]*100*100*1000 + e[2]*100*1000 + e[3]*1000 + e[4];

# Sort the list of times
times_list.sort(reverse=False, key=sortFunc)
if not times_list:
    raise FileNotFoundError(
        f"No compatible histogram count files found in {counts_dir}"
    )

# Make dictionaries
year = str(times_list[0][0])
month = str(times_list[0][1])
day = str(times_list[0][2])
time = str(times_list[0][3])

# While the start date remains the same
year_new = year
month_new = month
day_new = day
time_new = time
idx = 0
available_dates = {}
while (idx < len(times_list)):
    valid_times = []
    while (year_new==year) and (month_new==month) and (day_new==day) and (time_new==time):

        # Add to the list of valid times at this start date
        valid_times.append(times_list[idx][4])
        idx += 1

        # If we haven't reached the end of the list
        if (idx < len(times_list)):
            # Get the time of the next file in times_list
            year_new = str(times_list[idx][0])
            month_new = str(times_list[idx][1])
            day_new = str(times_list[idx][2])
            time_new = str(times_list[idx][3])
        else:
            # There are no more files in times_list to include
            break

    # Build the dictionary
    if year not in available_dates.keys():
        available_dates[year] = {}
    if month not in available_dates[year].keys():
        available_dates[year][month] = {}
    if day not in available_dates[year][month].keys():
        available_dates[year][month][day] = {}
    available_dates[year][month][day][time] = valid_times

    # Move to the next date
    year = year_new
    month = month_new
    day = day_new
    time = time_new

# convert into a JSON string
available_dates_json = json.dumps(available_dates)

# Write the JSON to a file
text_file = open(f"{output_dir}/available_dates.json", "w")
text_file.write(available_dates_json)
text_file.close()