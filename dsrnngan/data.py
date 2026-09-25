""" File for handling data loading and saving. """
import os
import datetime
import pickle
import json

import numpy as np
import netCDF4 as nc
import xarray as xr

import read_config

data_paths = read_config.get_data_paths()
TRUTH_PATH = data_paths["GENERAL"]["TRUTH_PATH"]
FCST_PATH = data_paths["GENERAL"]["FORECAST_PATH"]
CONSTANTS_PATH = data_paths["GENERAL"]["CONSTANTS_PATH"]
NORMALISATION_PATH = data_paths["GENERAL"]["NORMALISATION_PATH"]

HOURS = 6  #6 hour data

crop_to_bounds = (
    os.environ.get("CGAN_CROP_TO_BOUNDS", "False").lower()
    == "true"
)

bounds_str = os.environ.get("CGAN_BOUNDS")

if bounds_str is not None:
    bounds = [float(x) for x in bounds_str.split(",")]
else:
    bounds = None

all_fcst_fields = json.loads(
    os.environ.get("CGAN_ALL_FCST_FIELDS", "[]")
)

accumulated_fields = json.loads(
    os.environ.get("CGAN_ACCUMULATED_FIELDS", "[]")
)

nonnegative_fields = json.loads(
    os.environ.get("CGAN_NONNEGATIVE_FIELDS", "[]")
)

LEADTIME = int(os.environ.get("LEADTIME", 30))
ACCUMULATION = int(os.environ.get("ACCUMULATION", 24))

# utility function; generator to iterate over a range of dates
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(days=n)


def denormalise(x):
    """
    Undo log-transform of rainfall.  Also cap at 100 (feel free to adjust according to application!)
    """
    return np.minimum(10**x - 1.0, 100.0)


def logprec(y, log_precip=False):
    if log_precip:
        return np.log10(1.0+y)
    else:
        return y


def get_dates(year,
              leadtime=LEADTIME,
              accumulation=ACCUMULATION):
    """
    Return forecast start dates for which the corresponding truth data exists.

    `leadtime` refers to the START of the target accumulation interval.

    For example:
        leadtime=30, accumulation=6
        -> target interval is +30 h to +36 h
        -> truth file is timestamped at +30 h

    Dates are returned as YYYYMMDD strings.
    """

    # Sanity checks
    assert leadtime >= 0
    assert leadtime <= 168
    assert leadtime % HOURS == 0

    if accumulation not in (6, 24):
        raise ValueError(
            f"Unsupported accumulation period: {accumulation} hours"
        )

    start_date = datetime.date(year, 1, 1)
    end_date = datetime.date(year + 1, 1, 1)

    valid_dates = []

    for curdate in daterange(start_date, end_date):

        # Forecasts initialise at 00 UTC
        fcst_dt = datetime.datetime.combine(
            curdate,
            datetime.time(0, 0)
        )

        # leadtime denotes the START of the target interval
        target_start_dt = fcst_dt + datetime.timedelta(
            hours=leadtime
        )

        if accumulation == 24:
            # 24-hour truth files use the _06 naming convention
            truth_fname = target_start_dt.strftime("%Y%m%d_06")

        elif accumulation == 6:
            # 6-hour truth files are timestamped by the
            # start of the accumulation interval
            truth_fname = target_start_dt.strftime("%Y%m%d_%H")

        # Use the year of the target interval, not necessarily
        # the forecast initialisation year (important around New Year)
        truth_path = os.path.join(
            TRUTH_PATH,
            str(target_start_dt.year),
            f"{truth_fname}.nc"
        )

        if os.path.exists(truth_path):
            valid_dates.append(curdate.strftime("%Y%m%d"))

    return valid_dates

def load_truth_and_mask(date,
                        leadtime=LEADTIME,
                        log_precip=False,
                        truth_path=TRUTH_PATH):
    '''
    Returns a single (truth, mask) item of data.
    Parameters:
        date: forecast start date
        time_idx: forecast 'valid time' array index
        log_precip: whether to apply log10(1+x) transformation
    '''
    # convert date and time_idx to get the correct truth file
    fcst_date = datetime.datetime.strptime(date, "%Y%m%d")
    valid_dt = fcst_date + datetime.timedelta(hours=leadtime)
    year = str(valid_dt.year)
    fname = valid_dt.strftime('%Y%m%d_%H')
    data_path = os.path.join(truth_path, f"{year}/{fname}.nc")

    df = xr.open_dataset(data_path)
    if crop_to_bounds and 'latitude' in df.coords:
        lat0, lon0, lat1, lon1 = bounds
        lat_slice = slice(lat0, lat1) if df.latitude[0] < df.latitude[-1] else slice(lat1, lat0)
        lon_slice = slice(lon0, lon1) if df.longitude[0] < df.longitude[-1] else slice(lon1, lon0)
        df = df.sel(latitude=lat_slice, longitude=lon_slice)

    elif crop_to_bounds and 'lat' in df.coords:
        lat0, lon0, lat1, lon1 = bounds
        lat_slice = slice(lat0, lat1) if df.lat[0] < df.lat[-1] else slice(lat1, lat0)
        lon_slice = slice(lon0, lon1) if df.lon[0] < df.lon[-1] else slice(lon1, lon0)
        df = df.sel(lat=lat_slice, lon=lon_slice)
    da = df["precipitation"]
    y = da.values
    df.close()

    # mask: False for valid truth data, True for invalid truth data
    # (compatible with the NumPy masked array functionality)
    # if all data is valid:
    mask = np.full(y.shape, False, dtype=bool)

    if log_precip:
        return np.log10(1+y), mask
    else:
        return y, mask

#MW: needs changing if data source changes
def load_hires_constants(batch_size=1, constants_path=CONSTANTS_PATH):
    oro_path = os.path.join(constants_path, "elev.nc")
    df = xr.load_dataset(oro_path)
    if crop_to_bounds and 'latitude' in df.coords:
        lat0, lon0, lat1, lon1 = bounds
        lat_slice = slice(lat0, lat1) if df.latitude[0] < df.latitude[-1] else slice(lat1, lat0)
        lon_slice = slice(lon0, lon1) if df.longitude[0] < df.longitude[-1] else slice(lon1, lon0)
        df = df.sel(latitude=lat_slice, longitude=lon_slice)

    elif crop_to_bounds and 'lat' in df.coords:
        lat0, lon0, lat1, lon1 = bounds
        lat_slice = slice(lat0, lat1) if df.lat[0] < df.lat[-1] else slice(lat1, lat0)
        lon_slice = slice(lon0, lon1) if df.lon[0] < df.lon[-1] else slice(lon1, lon0)
        df = df.sel(lat=lat_slice, lon=lon_slice)
    # Orography in m.  Divide by 10,000 to give O(1) normalisation
    z = df["elevation"].values
    if z.ndim == 3 and z.shape[0] == 1:
        z = z[0]    
    z /= 10000.0
    df.close()

    lsm_path = os.path.join(constants_path, "lsm.nc")
    df = xr.load_dataset(lsm_path)
    if crop_to_bounds and 'latitude' in df.coords:
        lat0, lon0, lat1, lon1 = bounds
        lat_slice = slice(lat0, lat1) if df.latitude[0] < df.latitude[-1] else slice(lat1, lat0)
        lon_slice = slice(lon0, lon1) if df.longitude[0] < df.longitude[-1] else slice(lon1, lon0)
        df = df.sel(latitude=lat_slice, longitude=lon_slice)

    elif crop_to_bounds and 'lat' in df.coords:
        lat0, lon0, lat1, lon1 = bounds
        lat_slice = slice(lat0, lat1) if df.lat[0] < df.lat[-1] else slice(lat1, lat0)
        lon_slice = slice(lon0, lon1) if df.lon[0] < df.lon[-1] else slice(lon1, lon0)
        df = df.sel(lat=lat_slice, lon=lon_slice)
    # LSM is already 0:1
    lsm = df["lsm"].values
    if lsm.ndim == 3 and lsm.shape[0] == 1:
        lsm = lsm[0]
    df.close()

    temp = np.stack([z, lsm], axis=-1)  # shape H x W x 2
    return np.repeat(temp[np.newaxis, ...], batch_size, axis=0)  # shape batch_size x H x W x 2


def load_fcst_truth_batch(dates_batch,
                          time_idx_batch,
                          fcst_fields=all_fcst_fields,
                          leadtime=LEADTIME,
                          accumulation=ACCUMULATION,
                          log_precip=False,
                          norm=False,
                          fcst_norm_dict=None
                          ):
    '''
    Returns a batch of (forecast, truth, mask) data, although usually the batch size is 1
    Parameters:
        dates_batch (iterable of strings): Dates of forecasts
        time_idx_batch (iterable of ints): Corresponding 'valid_time' array indices
        fcst_fields (list of strings): The fields to be used
        log_precip (bool): Whether to apply log10(1+x) transform to precip-related forecast fields, and truth
        norm (bool): Whether to apply normalisation to forecast fields to make O(1)
    '''
    batch_x = []  # forecast
    batch_y = []  # truth
    batch_mask = []  # mask

    for time_idx, date in zip(time_idx_batch, dates_batch):
        batch_x.append(load_fcst_stack(fcst_fields, date, leadtime=leadtime, accumulation=accumulation, log_precip=log_precip, norm=norm, fcst_norm_dict=fcst_norm_dict))
        truth, mask = load_truth_and_mask(date, leadtime=leadtime, log_precip=log_precip)
        batch_y.append(truth)
        batch_mask.append(mask)

    return np.array(batch_x), np.array(batch_y), np.array(batch_mask)

def load_fcst(field,
              date,
              leadtime=LEADTIME,
              accumulation=ACCUMULATION,
              log_precip=False,
              norm=False,
              fcst_path=FCST_PATH,
              fcst_norm_dict=None):
    """
    Returns forecast field data for the given date and accumulation interval.

    Two channels are returned for each field:
        0: temporal mean of the ensemble mean
        1: temporal RMS/combined ensemble standard deviation
    """
    # print(f"Loading forecast {field} on {date}")

    #Normalisation
    norm_dict = fcst_norm if fcst_norm_dict is None else fcst_norm_dict

    yearstr = date[:4]
    year = int(yearstr)
    ds_path = os.path.join(fcst_path, yearstr, f"{field}.nc")

    # open using netCDF
    nc_file = nc.Dataset(ds_path, mode="r")
    all_data_mean = nc_file[f"{field}_mean"]
    all_data_sd = nc_file[f"{field}_sd"]
    # data is stored as [day of year, valid time index, lat, lon]

    lat_slice = slice(None)
    lon_slice = slice(None)
    if crop_to_bounds and bounds is not None:
        if 'latitude' in nc_file.variables and 'longitude' in nc_file.variables:
            lat_vals = np.asarray(nc_file.variables['latitude'][:])
            lon_vals = np.asarray(nc_file.variables['longitude'][:])
            lat_idx = np.where((lat_vals >= min(bounds[0], bounds[2])) & (lat_vals <= max(bounds[0], bounds[2])))[0]
            lon_idx = np.where((lon_vals >= min(bounds[1], bounds[3])) & (lon_vals <= max(bounds[1], bounds[3])))[0]
            lat_slice = slice(lat_idx[0], lat_idx[-1] + 1)
            lon_slice = slice(lon_idx[0], lon_idx[-1] + 1)
        elif 'lat' in nc_file.variables and 'lon' in nc_file.variables:
            lat_vals = np.asarray(nc_file.variables['lat'][:])
            lon_vals = np.asarray(nc_file.variables['lon'][:])
            lat_idx = np.where((lat_vals >= min(bounds[0], bounds[2])) & (lat_vals <= max(bounds[0], bounds[2])))[0]
            lon_idx = np.where((lon_vals >= min(bounds[1], bounds[3])) & (lon_vals <= max(bounds[1], bounds[3])))[0]
            lat_slice = slice(lat_idx[0], lat_idx[-1] + 1)
            lon_slice = slice(lon_idx[0], lon_idx[-1] + 1)

    # Find this date's actual index in THIS field's NetCDF file.
    # Different forecast fields may contain different sets of dates.
    time_var = nc_file.variables["time"]

    times = nc.num2date(
        time_var[:],
        units=time_var.units,
        calendar=getattr(time_var, "calendar", "standard")
    )

    target_date = datetime.datetime.strptime(date, "%Y%m%d").date()

    matches = [
        i for i, t in enumerate(times)
        if t.year == target_date.year
        and t.month == target_date.month
        and t.day == target_date.day
    ]

    if not matches:
        nc_file.close()
        raise FileNotFoundError(
            f"No forecast date {date} for field {field}"
        )

    fcst_idx = matches[0]
    #Check that new time_idx logic is workign as intended
    # print(f"{field}: requested={date}, index={fcst_idx}, actual={times[fcst_idx]}")

    lead_idx1 = int(leadtime / HOURS)

    if accumulation == 24:
        if field in accumulated_fields:
            # Four 6-hour accumulation periods:
            # [t,t+6], [t+6,t+12], [t+12,t+18], [t+18,t+24]
            lead_idx2 = lead_idx1 + 4
        else:
            # Five instantaneous times:
            # t, t+6, t+12, t+18, t+24
            lead_idx2 = lead_idx1 + 5

    elif accumulation == 6:
        if field in accumulated_fields:
            # One 6-hour accumulation period:
            # [t,t+6]
            lead_idx2 = lead_idx1 + 1
        else:
            # Two instantaneous times:
            # t, t+6
            lead_idx2 = lead_idx1 + 2

    else:
        raise ValueError(
            f"Unsupported accumulation period: {accumulation} hours"
        )

    if field in accumulated_fields:
        # Forecast accumulated fields contain one value per 6-hour interval.
        temp_data_mean = all_data_mean[
            fcst_idx, lead_idx1:lead_idx2, lat_slice, lon_slice
        ]
        temp_data_var = all_data_sd[
            fcst_idx, lead_idx1:lead_idx2, lat_slice, lon_slice
        ] ** 2

        if accumulation == 24:
            # Mean across the four 6-hour periods.
            data1 = np.mean(temp_data_mean, axis=0)

            # RMS standard deviation across the four periods.
            data2 = np.sqrt(np.mean(temp_data_var, axis=0))

        elif accumulation == 6:
            # Only one 6-hour period, so no temporal averaging is needed.
            data1 = temp_data_mean[0, :, :]
            data2 = np.sqrt(temp_data_var[0, :, :])

        data = np.stack([data1, data2], axis=-1)

        if not np.isfinite(data).all():
            nc_file.close()
            raise FileNotFoundError(
                f"Invalid forecast data for {date}, field {field}"
            )

    else:
        # Instantaneous forecast fields.
        temp_data_mean = all_data_mean[
            fcst_idx, lead_idx1:lead_idx2, lat_slice, lon_slice
        ]
        temp_data_var = all_data_sd[
            fcst_idx, lead_idx1:lead_idx2, lat_slice, lon_slice
        ] ** 2

        if accumulation == 24:
            # Trapezoidal average over:
            # t, t+6, t+12, t+18, t+24
            data1 = (
                temp_data_mean[0, :, :] / 2
                + np.sum(temp_data_mean[1:4, :, :], axis=0)
                + temp_data_mean[4, :, :] / 2
            ) / 4

            data2 = (
                temp_data_var[0, :, :] / 2
                + np.sum(temp_data_var[1:4, :, :], axis=0)
                + temp_data_var[4, :, :] / 2
            ) / 4

        elif accumulation == 6:
            # Trapezoidal average over the two endpoints:
            # t and t+6
            data1 = (
                temp_data_mean[0, :, :]
                + temp_data_mean[1, :, :]
            ) / 2

            data2 = (
                temp_data_var[0, :, :]
                + temp_data_var[1, :, :]
            ) / 2

        data = np.stack([data1, np.sqrt(data2)], axis=-1)

        if not np.isfinite(data).all():
            nc_file.close()
            raise FileNotFoundError(
                f"Invalid forecast data for {date}, field {field}"
            )

    nc_file.close()

    if field in nonnegative_fields:
        data = np.maximum(data, 0.0)  # eliminate any data weirdness/regridding issues

    if field in ["tp", "cp"]:
        # precip is measured in metres, so multiply to get mm
        data *= 1000
        data /= HOURS  # convert to mm/hr
    elif field in accumulated_fields:
        # for all other accumulated fields [just ssr for us]
        data /= (HOURS*3600)  # convert from a 6-hr difference to a per-second rate

    if field in ["tp", "cp"] and log_precip:
        return logprec(data, log_precip)
    elif norm:
        # apply transformation to make fields O(1), based on historical
        # forecast data from one of the training years
        if norm_dict is None:
            raise RuntimeError("Forecast normalisation dictionary has not been loaded")
        if field in ["mcc", "tcc"]:
            # already 0-1
            return data
        elif field in ["sp", "t2m"]:
            # these are bounded well away from zero, so subtract mean from ens mean (but NOT from ens sd!)
            data[:, :, 0] -= norm_dict[field]["mean"]
            # data[:, :, 2] -= norm_dict[field]["mean"] #TO CHECK
            return data/norm_dict[field]["std"]
        elif field in nonnegative_fields:
            return data/norm_dict[field]["max"]
        else:
            # winds
            return data/max(-norm_dict[field]["min"], norm_dict[field]["max"])
    else:
        return data


def load_fcst_stack(fields,
                    date,
                    leadtime=LEADTIME,
                    accumulation=ACCUMULATION,
                    log_precip=False,
                    norm=False,
                    fcst_norm_dict=None):
    '''
    Returns forecast fields, for the given date and time interval.
    Each field returned by load_fcst has two channels (see load_fcst for details),
    then these are concatentated to form an array of H x W x 2*len(fields)
    '''
    field_arrays = []
    for f in fields:
        field_arrays.append(load_fcst(f, date, leadtime=leadtime, accumulation=accumulation, log_precip=log_precip, norm=norm, fcst_norm_dict=fcst_norm_dict))
    return np.concatenate(field_arrays, axis=-1)


def get_fcst_stats_slow(field, leadtime=LEADTIME, year=2018):
    '''
    Calculates and returns min, max, mean, std per field,
    which can be used to generate normalisation parameters.

    These are done via the data loading routines, which is
    slightly inefficient.
    '''
    dates = get_dates(year, leadtime=leadtime)

    mi = 0.0
    mx = 0.0
    dsum = 0.0
    dsqrsum = 0.0
    nsamples = 0
    for datestr in dates:
        for time_idx in range(28):
            data = load_fcst(field, datestr, leadtime=leadtime)[:, :, 0]
            mi = min(mi, data.min())
            mx = max(mx, data.max())
            dsum += np.mean(data)
            dsqrsum += np.mean(np.square(data))
            nsamples += 1
    mn = dsum / nsamples
    sd = (dsqrsum/nsamples - mn**2)**0.5
    return mi, mx, mn, sd


def get_fcst_stats_fast(field, year=2018):
    '''
    Calculates and returns min, max, mean, std per field,
    which can be used to generate normalisation parameters.

    These are done directly from the forecast netcdf file,
    which is somewhat faster, as long as it fits into memory.
    '''
    ds_path = os.path.join(FCST_PATH, str(year), f"{field}.nc")
    nc_file = nc.Dataset(ds_path, mode="r")

    if field in accumulated_fields:
        data = nc_file[f"{field}_mean"][:, :-1, :, :]  # last time_idx is full of zeros
    else:
        data = nc_file[f"{field}_mean"][:, :, :, :]

    if crop_to_bounds and bounds is not None:
        if 'latitude' in nc_file.variables and 'longitude' in nc_file.variables:
            lat_vals = np.asarray(nc_file.variables['latitude'][:])
            lon_vals = np.asarray(nc_file.variables['longitude'][:])
            lat_idx = np.where((lat_vals >= min(bounds[0], bounds[2])) & (lat_vals <= max(bounds[0], bounds[2])))[0]
            lon_idx = np.where((lon_vals >= min(bounds[1], bounds[3])) & (lon_vals <= max(bounds[1], bounds[3])))[0]
            data = data[:, :, lat_idx[0]:lat_idx[-1] + 1, lon_idx[0]:lon_idx[-1] + 1]
        elif 'lat' in nc_file.variables and 'lon' in nc_file.variables:
            lat_vals = np.asarray(nc_file.variables['lat'][:])
            lon_vals = np.asarray(nc_file.variables['lon'][:])
            lat_idx = np.where((lat_vals >= min(bounds[0], bounds[2])) & (lat_vals <= max(bounds[0], bounds[2])))[0]
            lon_idx = np.where((lon_vals >= min(bounds[1], bounds[3])) & (lon_vals <= max(bounds[1], bounds[3])))[0]
            data = data[:, :, lat_idx[0]:lat_idx[-1] + 1, lon_idx[0]:lon_idx[-1] + 1]

    nc_file.close()

    if field in ["tp", "cp"]:
        # precip is measured in metres, so multiply to get mm
        data *= 1000
        data /= HOURS  # convert to mm/hr
        data = np.maximum(data, 0.0)  # shouldn't be necessary, but just in case
    elif field in accumulated_fields:
        # for all other accumulated fields [just ssr for us]
        data /= (HOURS*3600)  # convert from a 6-hr difference to a per-second rate

    if np.ma.isMaskedArray(data):
        data = data.filled(np.nan)

    mi = np.nanmin(data)
    mx = np.nanmax(data)
    mn = np.nanmean(data, dtype=np.float64)
    sd = np.nanstd(data, dtype=np.float64)
    return mi, mx, mn, sd

#MW: do it once -- for one year -- and save. Can be used very simply.
def gen_fcst_norm(year=2018):
    '''
    One-off function, used to generate normalisation constants, which
    are used to normalise the various input fields for training/inference.
    '''

    stats_dic = {}
    fcstnorm_path = os.path.join(NORMALISATION_PATH, f"FCSTNorm{year}.pkl")
    os.makedirs(os.path.dirname(fcstnorm_path), exist_ok=True)

    # make sure we can actually write there, before doing computation!!!
    with open(fcstnorm_path, 'wb') as f:
        pickle.dump(stats_dic, f)

    for field in all_fcst_fields:
        print(field)
        mi, mx, mn, sd = get_fcst_stats_fast(field, year)
        stats_dic[field] = {}
        stats_dic[field]['min'] = mi
        stats_dic[field]['max'] = mx
        stats_dic[field]['mean'] = mn
        stats_dic[field]['std'] = sd

    with open(fcstnorm_path, 'wb') as f:
        pickle.dump(stats_dic, f)


def load_fcst_norm(year=2018, normalisation_path=NORMALISATION_PATH):
    print("In load_fcst_norm")
    fcstnorm_path = os.path.join(normalisation_path, f"FCSTNorm{year}.pkl")
    print(f"fcstnorm_path = {fcstnorm_path}")
    with open(fcstnorm_path, 'rb') as f:
        return pickle.load(f)


try:
    print("Loading forecast normalisations")
    fcst_norm = load_fcst_norm(2018)
except:  # noqa
    fcst_norm = None
    print("******************************************")
    print("*** FORECAST NORMALISATIONS NOT LOADED ***")
    print("******************************************")
