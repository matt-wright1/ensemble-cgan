import glob
import os
import random
import math
from pathlib import Path
import re

import numpy as np
import tensorflow as tf
import xarray as xr

import read_config
from data import all_fcst_fields, denormalise, get_dates, HOURS, crop_to_bounds, bounds


data_paths = read_config.get_data_paths()
records_folder = data_paths["TFRecords"]["tfrecords_path"]
truth_folder = Path(data_paths["GENERAL"]["TRUTH_PATH"])
ds_fac = read_config.read_downscaling_factor()["downscaling_factor"]

CLASSES = 4
# Find H/W of image
# Find the first file anywhere under the directory
truth_files = sorted(
    p
    for p in truth_folder.rglob("*.nc")
    if any(re.fullmatch(r"\d{4}", parent.name) for parent in p.parents)
)
truth_file = truth_files[0] if truth_files else None

if truth_file is None:
    raise FileNotFoundError(f"No files found in {truth_folder}")

with xr.open_dataset(truth_file) as ds:
    # Handle either lat/lon or latitude/longitude
    # if crop_to_bounds and "latitude" in ds.coords:
    #     lat0, lon0, lat1, lon1 = bounds
    #     lat_slice = slice(lat0, lat1) if ds.latitude[0] < ds.latitude[-1] else slice(lat1, lat0)
    #     lon_slice = slice(lon0, lon1) if ds.longitude[0] < ds.longitude[-1] else slice(lon1, lon0)
    #     ds = ds.sel(latitude=lat_slice, longitude=lon_slice)

    # elif crop_to_bounds and "lat" in ds.coords:
    #     lat0, lon0, lat1, lon1 = bounds
    #     lat_slice = slice(lat0, lat1) if ds.lat[0] < ds.lat[-1] else slice(lat1, lat0)
    #     lon_slice = slice(lon0, lon1) if ds.lon[0] < ds.lon[-1] else slice(lon1, lon0)
        # ds = ds.sel(lat=lat_slice, lon=lon_slice)

    lat_name = "y"
    lon_name = "x"

    IMAGE_SIZE_H = ds.sizes[lat_name]
    IMAGE_SIZE_W = ds.sizes[lon_name]

def choose_square_dim(h: int, w: int, close_px: int = 4) -> int:
    m = min(h, w, 128)

    # Largest power of 2 strictly below m
    p2 = 1 << (m.bit_length() - 1)

    # Use the power-of-2 size only if it's above 50 and close enough
    if p2 > 50 and (m - p2) <= close_px:
        return p2

    return m

S = choose_square_dim(IMAGE_SIZE_H, IMAGE_SIZE_W)
print(f"Using square image size {S}x{S} for training, from original {IMAGE_SIZE_H}x{IMAGE_SIZE_W}")

assert S % ds_fac == 0
fcst_shape = S // ds_fac
print(f"S % ds_fac == 0, fcst_shape = {fcst_shape}")

DEFAULT_FCST_SHAPE = (fcst_shape, fcst_shape, 2*len(all_fcst_fields))
DEFAULT_CON_SHAPE = (S, S, 2)
DEFAULT_OUT_SHAPE = (S, S, 1)

def DataGenerator(years, batch_size, repeat=True, autocoarsen=False, weights=None):
    return create_mixed_dataset(years, batch_size, repeat=repeat, autocoarsen=autocoarsen, weights=weights)


def create_mixed_dataset(years,
                         batch_size,
                         fcst_shape=DEFAULT_FCST_SHAPE,
                         con_shape=DEFAULT_CON_SHAPE,
                         out_shape=DEFAULT_OUT_SHAPE,
                         repeat=True,
                         autocoarsen=False,
                         folder=records_folder,
                         shuffle_size=64,
                         weights=None):

    if weights is None:
        weights = [1./CLASSES]*CLASSES
    datasets = [create_dataset(years,
                               ii,
                               fcst_shape=fcst_shape,
                               con_shape=con_shape,
                               out_shape=out_shape,
                               folder=folder,
                               shuffle_size=shuffle_size,
                               repeat=repeat)
                for ii in range(CLASSES)]
    sampled_ds = tf.data.Dataset.sample_from_datasets(datasets,
                                                      weights=weights).batch(batch_size)

    if autocoarsen:
        sampled_ds = sampled_ds.map(_dataset_autocoarsener)
    sampled_ds = sampled_ds.prefetch(2)
    return sampled_ds

# Note, if we wanted fewer classes, we can use glob syntax to grab multiple classes as once
# e.g. create_dataset(2015,"[67]")
# will take classes 6 & 7 together


def _dataset_autocoarsener(inputs, outputs):
    image = outputs['output']
    kernel_tf = tf.constant(1.0/(ds_fac*ds_fac), shape=(ds_fac, ds_fac, 1, 1), dtype=tf.float32)
    image = tf.nn.conv2d(image, filters=kernel_tf, strides=[1, ds_fac, ds_fac, 1], padding='VALID',
                         name='conv_debug', data_format='NHWC')
    inputs['lo_res_inputs'] = image
    return inputs, outputs


def _parse_batch(record_batch,
                 insize=DEFAULT_FCST_SHAPE,
                 consize=DEFAULT_CON_SHAPE,
                 outsize=DEFAULT_OUT_SHAPE):
    # Create a description of the features
    feature_description = {
        'generator_input': tf.io.FixedLenFeature(insize, tf.float32),
        'constants': tf.io.FixedLenFeature(consize, tf.float32),
        'generator_output': tf.io.FixedLenFeature(outsize, tf.float32),
    }

    # Parse the input `tf.Example` proto using the dictionary above
    example = tf.io.parse_example(record_batch, feature_description)
    return ({'lo_res_inputs': example['generator_input'],
             'hi_res_inputs': example['constants']},
            {'output': example['generator_output']})


def create_dataset(years,
                   clss,
                   fcst_shape=DEFAULT_FCST_SHAPE,
                   con_shape=DEFAULT_CON_SHAPE,
                   out_shape=DEFAULT_OUT_SHAPE,
                   folder=records_folder,
                   shuffle_size=64,
                   repeat=True):
    # TODO: tf.data.Dataset.list_files should accept the list of glob patterns,
    # not the list of globbed filenames

    # "The file_pattern argument should be a small number of glob patterns. If your
    # filenames have already been globbed, use Dataset.from_tensor_slices(filenames)
    # instead, as re-globbing every filename with list_files may result in poor
    # performance with remote storage systems."

    # however, tried this on EWC and it was marginally slower!
    # But may want to change in future
    filelist = []
    for yr in years:
        fpattern = os.path.join(folder, f"{yr}_*.{clss}.tfrecords")
        filelist += glob.glob(fpattern)

    files_ds = tf.data.Dataset.list_files(filelist)
    ds = tf.data.TFRecordDataset(files_ds,
                                 compression_type="GZIP",
                                 num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.shuffle(shuffle_size)
    ds = ds.map(lambda x: _parse_batch(x,
                                       insize=fcst_shape,
                                       consize=con_shape,
                                       outsize=out_shape))
    if repeat:
        return ds.repeat()
    else:
        return ds


# create_fixed_dataset currently unused; full image dataset used for validation
def create_fixed_dataset(year=None,
                         mode='validation',
                         batch_size=16,
                         autocoarsen=False,
                         fcst_shape=DEFAULT_FCST_SHAPE,
                         con_shape=DEFAULT_CON_SHAPE,
                         out_shape=DEFAULT_OUT_SHAPE,
                         name=None,
                         folder=records_folder):
    assert year is not None or name is not None, "Must specify year or file name"
    if name is None:
        name = os.path.join(folder, f"{mode}{year}.tfrecords")
    else:
        name = os.path.join(folder, name)
    fl = glob.glob(name)
    files_ds = tf.data.Dataset.list_files(fl)
    ds = tf.data.TFRecordDataset(files_ds,
                                 num_parallel_reads=1)
    ds = ds.map(lambda x: _parse_batch(x,
                                       insize=fcst_shape,
                                       consize=con_shape,
                                       outsize=out_shape))
    ds = ds.batch(batch_size)
    if autocoarsen:
        ds = ds.map(_dataset_autocoarsener)
    return ds


def _float_feature(list_of_floats):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


def write_data(year,
               folder=records_folder,
               fcst_fields=all_fcst_fields,
               num_class=CLASSES,
               log_precip=True,
               fcst_norm=True):

    from data_generator import DataGenerator as DataGeneratorFull

    year = int(year)

    # Binning based on mean rainfall over the full domain
    bins = [0.2, 0.3, 0.45]
    assert num_class == 4

    def report_nonfinite(name, arr):
        """
        Report NaN/+Inf/-Inf values in an array.
        Returns True if any non-finite values are present.
        """
        arr = np.asarray(arr)

        n_nan = np.isnan(arr).sum()
        n_posinf = np.isposinf(arr).sum()
        n_neginf = np.isneginf(arr).sum()

        if n_nan or n_posinf or n_neginf:
            print(
                f"    {name}: shape={arr.shape}, "
                f"NaN={n_nan}, "
                f"+Inf={n_posinf}, "
                f"-Inf={n_neginf}, "
                f"total={arr.size}"
            )
            return True

        return False

    # ------------------------------------------------------------
    # Lead times
    # ------------------------------------------------------------

    # Currently just time_idx=1.
    # Change this range when you want additional lead times.
    for time_idx in range(1, 2):

        print(f"\nDoing time index {time_idx}")

        s_hour = time_idx * HOURS
        e_hour = s_hour

        print(f"start_hour={s_hour}, end_hour={e_hour}")

        dates = get_dates(
            year,
            start_hour=s_hour,
            end_hour=e_hour
        )

        print(f"Number of dates: {len(dates)}")

        dgc = DataGeneratorFull(
            dates,
            fcst_fields=fcst_fields,
            start_hour=s_hour,
            end_hour=e_hour,
            batch_size=1,
            log_precip=log_precip,
            shuffle=False,
            constants=True,
            fcst_norm=fcst_norm
        )

        print(f"Generator length: {len(dgc)}")

        # --------------------------------------------------------
        # Create one TFRecord file for each rainfall class
        # --------------------------------------------------------

        fle_hdles = []

        for fh in range(num_class):

            flename = os.path.join(
                folder,
                f"{year}_{time_idx}.{fh}.tfrecords"
            )

            options = tf.io.TFRecordOptions(
                compression_type="GZIP"
            )

            fle_hdles.append(
                tf.io.TFRecordWriter(
                    flename,
                    options=options
                )
            )

        # --------------------------------------------------------
        # Counters
        # --------------------------------------------------------

        class_counts = np.zeros(num_class, dtype=int)

        skipped_mask = 0
        skipped_empty = 0
        skipped_nonfinite = 0
        skipped_missing_file = 0

        # --------------------------------------------------------
        # Generator
        # --------------------------------------------------------

        for batch in range(len(dgc)):

            if batch % 10 == 0:
                print(f"\ntime_idx={time_idx}, batch={batch}")

            try:
                sample = dgc.__getitem__(batch)

            except FileNotFoundError as e:

                print(
                    f"Skipping batch {batch} because "
                    f"source file is missing: {e}"
                )

                skipped_missing_file += 1
                continue

            # ----------------------------------------------------
            # FULL DOMAIN -- NO SUBSAMPLING
            # ----------------------------------------------------

            forecast = np.asarray(
                sample[0]['lo_res_inputs'][0, ...]
            )

            const = np.asarray(
                sample[0]['hi_res_inputs'][0, ...]
            )

            mask = np.asarray(
                sample[1]['mask']
            )

            mask = np.squeeze(mask)

            truth = np.asarray(
                sample[1]['output']
            )

            truth = np.squeeze(truth)

            nan_pixels = np.isnan(truth)

            print("truth NaNs:", np.count_nonzero(nan_pixels))
            print("mask True:", np.count_nonzero(mask))
            print("NaNs covered by mask:", np.count_nonzero(nan_pixels & mask))
            print("NaNs NOT covered by mask:", np.count_nonzero(nan_pixels & ~mask))

            # ----------------------------------------------------
            # Print shapes for first batch
            # ----------------------------------------------------

            if batch == 0:

                print("\nFull-domain shapes:")
                print("  forecast: ", forecast.shape)
                print("  constants:", const.shape)
                print("  truth:    ", truth.shape)
                print("  mask:     ", mask.shape)

                print("\nExpected flattened sizes:")
                print("  forecast: ", forecast.size)
                print("  constants:", const.size)
                print("  truth:    ", truth.size)

            # ----------------------------------------------------
            # Empty-array check
            # ----------------------------------------------------

            if (
                forecast.size == 0
                or const.size == 0
                or truth.size == 0
            ):

                print(
                    f"Skipping batch {batch}: empty array"
                )

                print(
                    f"    forecast={forecast.shape}, "
                    f"const={const.shape}, "
                    f"truth={truth.shape}"
                )

                skipped_empty += 1
                continue

            # ----------------------------------------------------
            # Diagnose NaN / Inf
            # ----------------------------------------------------

            bad_forecast = report_nonfinite(
                "forecast",
                forecast
            )

            bad_const = report_nonfinite(
                "constants",
                const
            )

            bad_truth = report_nonfinite(
                "truth",
                truth
            )

            # ----------------------------------------------------
            # If forecast is bad, identify the bad channel(s)
            # ----------------------------------------------------

            if bad_forecast:

                print(
                    f"  Batch {batch}: "
                    "non-finite forecast channels:"
                )

                for ch in range(forecast.shape[-1]):

                    x = forecast[..., ch]

                    n_bad = np.count_nonzero(
                        ~np.isfinite(x)
                    )

                    if n_bad:

                        print(
                            f"    channel {ch}: "
                            f"bad={n_bad}/{x.size}, "
                            f"NaN={np.isnan(x).sum()}, "
                            f"+Inf={np.isposinf(x).sum()}, "
                            f"-Inf={np.isneginf(x).sum()}"
                        )

            # ----------------------------------------------------
            # More diagnostics for constants
            # ----------------------------------------------------

            if bad_const and const.ndim >= 3:

                print(
                    f"  Batch {batch}: "
                    "non-finite constant channels:"
                )

                for ch in range(const.shape[-1]):

                    x = const[..., ch]

                    n_bad = np.count_nonzero(
                        ~np.isfinite(x)
                    )

                    if n_bad:

                        print(
                            f"    channel {ch}: "
                            f"bad={n_bad}/{x.size}, "
                            f"NaN={np.isnan(x).sum()}, "
                            f"+Inf={np.isposinf(x).sum()}, "
                            f"-Inf={np.isneginf(x).sum()}"
                        )

            # ----------------------------------------------------
            # Mask check
            # ----------------------------------------------------

            if mask.shape != truth.shape:

                if (
                    mask.ndim == truth.ndim - 1
                    and truth.shape[-1] == 1
                    and mask.shape == truth.shape[:-1]
                ):
                    mask = mask[..., np.newaxis]

                else:
                    raise ValueError(
                        f"Mask/truth shape mismatch: "
                        f"mask={mask.shape}, truth={truth.shape}"
                    )

            n_masked = np.count_nonzero(mask)

            if n_masked:
                print(
                    f"Batch {batch}: "
                    f"{n_masked}/{mask.size} pixels masked "
                    f"({100.0 * n_masked / mask.size:.3f}%)"
                )

            # ----------------------------------------------------
            # Replace invalid masked radar pixels
            # ----------------------------------------------------

            mask = mask.astype(bool)

            # Find non-finite truth pixels that are NOT masked.
            # These indicate a problem beyond normal missing radar data.
            bad_valid_pixels = (~np.isfinite(truth)) & (~mask)

            if np.any(bad_valid_pixels):
                print(
                    f"Skipping batch {batch}: "
                    f"{np.count_nonzero(bad_valid_pixels)} non-finite "
                    "truth pixels are outside the radar mask"
                )
                skipped_nonfinite += 1
                continue

            # Masked radar pixels may contain NaN.
            # Do not allow these NaNs into the TFRecord / GAN.
            truth = truth.copy()
            truth[mask] = 0.0

            # Final safety check
            if not np.all(np.isfinite(truth)):
                print(
                    f"Skipping batch {batch}: "
                    "truth still contains NaN/Inf after masking"
                )
                skipped_nonfinite += 1
                continue
            # ----------------------------------------------------
            # Flatten AFTER all checks
            # ----------------------------------------------------

            forecast_flat = forecast.flatten()
            const_flat = const.flatten()
            truth_flat = truth.flatten()

            # ----------------------------------------------------
            # Denormalise truth for rainfall classification
            # ----------------------------------------------------

            truth_raw = denormalise(truth_flat)

            if truth_raw.size == 0:

                print(
                    f"Skipping batch {batch}: "
                    "denormalised truth is empty"
                )

                skipped_empty += 1
                continue

            truth_mean = np.nanmean(truth_raw)

            if not np.isfinite(truth_mean):

                print(
                    f"Skipping batch {batch}: "
                    f"truth_mean={truth_mean}"
                )

                skipped_nonfinite += 1
                continue

            # ----------------------------------------------------
            # Rainfall class
            # ----------------------------------------------------

            if truth_mean < bins[0]:

                clss = 0

            elif truth_mean < bins[1]:

                clss = 1

            elif truth_mean < bins[2]:

                clss = 2

            else:

                clss = 3

            # ----------------------------------------------------
            # TFRecord
            # ----------------------------------------------------

            feature = {

                'generator_input':
                    _float_feature(forecast_flat),

                'constants':
                    _float_feature(const_flat),

                'generator_output':
                    _float_feature(truth_flat)
            }

            features = tf.train.Features(
                feature=feature
            )

            example = tf.train.Example(
                features=features
            )

            fle_hdles[clss].write(
                example.SerializeToString()
            )

            class_counts[clss] += 1

        # --------------------------------------------------------
        # Close TFRecord files
        # --------------------------------------------------------

        for fh in fle_hdles:
            fh.close()

        # --------------------------------------------------------
        # Summary
        # --------------------------------------------------------

        print("\n" + "=" * 60)

        print(
            f"Finished year={year}, "
            f"time_idx={time_idx}"
        )

        print("\nWritten per class:")

        for clss, count in enumerate(class_counts):
            print(
                f"  class {clss}: {count}"
            )

        print("\nSkipped:")

        print(
            f"  missing source file: {skipped_missing_file}"
        )

        print(
            f"  empty arrays:        {skipped_empty}"
        )

        print(
            f"  NaN/Inf:             {skipped_nonfinite}"
        )

        print(
            f"  mask:                {skipped_mask}"
        )

        print("=" * 60)


# currently unused; was previously used to make small-image validation dataset,
# but this is now obsolete
def save_dataset(tfrecords_dataset, flename, max_batches=None):
    flename = os.path.join(records_folder, flename)
    fle_hdle = tf.io.TFRecordWriter(flename)
    for ii, sample in enumerate(tfrecords_dataset):
        print(ii)
        if max_batches is not None:
            if ii == max_batches:
                break
        for k in range(sample[1]['output'].shape[0]):
            feature = {
                'generator_input': _float_feature(sample[0]['lo_res_inputs'][k, ...].numpy().flatten()),
                'constants': _float_feature(sample[0]['hi_res_inputs'][k, ...].numpy().flatten()),
                'generator_output': _float_feature(sample[1]['output'][k, ...].numpy().flatten())
            }
            features = tf.train.Features(feature=feature)
            example = tf.train.Example(features=features)
            example_to_string = example.SerializeToString()
            fle_hdle.write(example_to_string)
    fle_hdle.close()
    return
