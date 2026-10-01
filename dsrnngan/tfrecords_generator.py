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
    if crop_to_bounds and "latitude" in ds.coords:
        lat0, lon0, lat1, lon1 = bounds
        lat_slice = slice(lat0, lat1) if ds.latitude[0] < ds.latitude[-1] else slice(lat1, lat0)
        lon_slice = slice(lon0, lon1) if ds.longitude[0] < ds.longitude[-1] else slice(lon1, lon0)
        ds = ds.sel(latitude=lat_slice, longitude=lon_slice)

    elif crop_to_bounds and "lat" in ds.coords:
        lat0, lon0, lat1, lon1 = bounds
        lat_slice = slice(lat0, lat1) if ds.lat[0] < ds.lat[-1] else slice(lat1, lat0)
        lon_slice = slice(lon0, lon1) if ds.lon[0] < ds.lon[-1] else slice(lon1, lon0)
        ds = ds.sel(lat=lat_slice, lon=lon_slice)

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
DEFAULT_OUT_SHAPE = (S, S, 1)

def DataGenerator(years, batch_size, repeat=True, autocoarsen=False, weights=None, constant_fields=None):
    if constant_fields is None:
        raise ValueError("constant_fields must be provided")

    con_shape = (S, S, constant_fields)

    return create_mixed_dataset(
        years,
        batch_size,
        repeat=repeat,
        autocoarsen=autocoarsen,
        weights=weights,
        con_shape=con_shape
    )


def create_mixed_dataset(years,
                         batch_size,
                         fcst_shape=DEFAULT_FCST_SHAPE,
                         con_shape=None,
                         out_shape=DEFAULT_OUT_SHAPE,
                         repeat=True,
                         autocoarsen=False,
                         folder=records_folder,
                         shuffle_size=64,
                         weights=None,
    ):

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
                 consize=None,
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
                   con_shape=None,
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
                         con_shape=None,
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
               leadtime=30,
               accumulation=24,
               img_chunk_width=DEFAULT_FCST_SHAPE[0],
               num_class=CLASSES,
               log_precip=True,
               fcst_norm=True,
               constants_list=None):

    """
    Write full-domain radar training samples to TFRecords.

    Multiple lead times are handled by DataGenerator. For example,

        leadtime=[30, 54, 78, 102]

    causes DataGenerator to construct every (date, leadtime) sample.
    All lead times are written into the SAME set of rainfall-class
    TFRecord files:

        {year}_1.0.tfrecords
        {year}_1.1.tfrecords
        {year}_1.2.tfrecords
        {year}_1.3.tfrecords

    Radar pixels marked invalid by the radar mask are replaced by 0
    before serialisation. Non-finite truth pixels outside the mask
    cause the sample to be skipped.

    Parameters
    ----------
    year : int
        Forecast initialisation year.

    folder : str
        Output directory.

    fcst_fields : list
        Forecast fields passed to DataGenerator.

    leadtime : int or sequence of int
        Start hour(s) of the target accumulation interval.

    accumulation : int
        Target accumulation period, e.g. 6 or 24 hours.

    img_chunk_width :
        Retained for interface compatibility. This version writes the
        FULL domain and does not perform spatial subsampling.

    num_class : int
        Number of rainfall classes. Currently must be 4.

    log_precip : bool
        Passed to DataGenerator.

    fcst_norm : bool
        Passed to DataGenerator.

    constants_list : list or None
        Constants to request from DataGenerator. The exact argument
        passed to DataGenerator may need adapting depending on the
        constants interface used by your generator.
    """

    from data_generator import DataGenerator as DataGeneratorFull

    # ------------------------------------------------------------
    # Basic setup
    # ------------------------------------------------------------

    year = int(year)

    if constants_list is None:
        constants_list = []

    # Accept either:
    #
    #     leadtime=30
    #
    # or:
    #
    #     leadtime=[30, 54, 78, 102]
    #
    if np.isscalar(leadtime):
        leadtime = [int(leadtime)]
    else:
        leadtime = [int(x) for x in leadtime]

    if len(leadtime) == 0:
        raise ValueError("At least one lead time must be supplied")

    # Rainfall-class boundaries, mm/hr
    bins = [0.2, 0.3, 0.45]

    assert num_class == 4

    print("=" * 70)
    print(f"Generating radar TFRecords for year {year}")
    print(f"Lead times:   {leadtime}")
    print(f"Accumulation: {accumulation} h")
    print(f"Constants:    {constants_list}")
    print("=" * 70)

    # ------------------------------------------------------------
    # Helper for NaN / Inf diagnostics
    # ------------------------------------------------------------

    def report_nonfinite(name, arr):
        """
        Report NaN/+Inf/-Inf values.

        Returns
        -------
        bool
            True if any non-finite values are present.
        """

        arr = np.asarray(arr)

        n_nan = np.isnan(arr).sum()
        n_posinf = np.isposinf(arr).sum()
        n_neginf = np.isneginf(arr).sum()

        if n_nan or n_posinf or n_neginf:

            print(
                f"    {name}: "
                f"shape={arr.shape}, "
                f"NaN={n_nan}, "
                f"+Inf={n_posinf}, "
                f"-Inf={n_neginf}, "
                f"total={arr.size}"
            )

            return True

        return False

    # ------------------------------------------------------------
    # Get dates
    # ------------------------------------------------------------
    #
    # IMPORTANT:
    #
    # get_dates() should return forecast initialisation dates for
    # which the requested target data exist.
    #
    # This assumes get_dates() accepts multiple lead times.
    #
    # DataGenerator then expands:
    #
    #     dates x leadtime
    #
    # into individual training samples.
    # ------------------------------------------------------------

    dates = get_dates(
        year,
        leadtime=leadtime,
        accumulation=accumulation
    )

    print(f"\nNumber of initialisation dates: {len(dates)}")

    if len(dates) == 0:
        print("No dates found. Nothing to write.")
        return

    # ------------------------------------------------------------
    # Create ONE generator containing ALL lead times
    # ------------------------------------------------------------
    #
    # With, for example:
    #
    #     dates = [date1, date2]
    #     leadtime = [30, 54, 78]
    #
    # DataGenerator produces:
    #
    #     date1, 30
    #     date1, 54
    #     date1, 78
    #     date2, 30
    #     date2, 54
    #     date2, 78
    #
    # ------------------------------------------------------------

    dgc = DataGeneratorFull(
        dates,
        fcst_fields=fcst_fields,
        leadtime=leadtime,
        accumulation=accumulation,
        batch_size=1,
        log_precip=log_precip,
        shuffle=False,
        fcst_norm=fcst_norm,
        constants_list=constants_list
    )

    print(f"Generator samples: {len(dgc)}")

    expected_samples = len(dates) * len(leadtime)

    print(
        f"Expected date x leadtime combinations: "
        f"{len(dates)} x {len(leadtime)} "
        f"= {expected_samples}"
    )

    if len(dgc) != expected_samples:

        print(
            "WARNING: generator length does not equal "
            "dates x leadtime."
        )

    # ------------------------------------------------------------
    # ONE set of TFRecords for ALL lead times
    # ------------------------------------------------------------

    time_idx = 1

    fle_hdles = []

    try:

        for clss in range(num_class):

            flename = os.path.join(
                folder,
                f"{year}_{time_idx}.{clss}.tfrecords"
            )

            print(
                f"Opening class {clss}: "
                f"{flename}"
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

        class_counts = np.zeros(
            num_class,
            dtype=int
        )

        skipped_empty = 0
        skipped_nonfinite = 0
        skipped_missing_file = 0
        masked_pixels_total = 0

        # ========================================================
        # Generator loop
        # ========================================================

        for batch in range(len(dgc)):

            if batch % 10 == 0:

                print(
                    f"\nBatch {batch}/{len(dgc)}"
                )

                # Since batch_size=1, this identifies exactly
                # which date/leadtime sample is being processed.
                if hasattr(dgc, "dates"):
                    print(
                        f"  date:     {dgc.dates[batch]}"
                    )

                if hasattr(dgc, "leadtime"):
                    print(
                        f"  leadtime: {dgc.leadtime[batch]} h"
                    )

            # ----------------------------------------------------
            # Load sample
            # ----------------------------------------------------

            try:

                sample = dgc.__getitem__(batch)

            except FileNotFoundError as e:

                print(
                    f"Skipping batch {batch}: "
                    f"source file is missing: {e}"
                )

                skipped_missing_file += 1
                continue

            # ----------------------------------------------------
            # FULL DOMAIN -- NO SPATIAL SUBSAMPLING
            # ----------------------------------------------------

            forecast = np.asarray(
                sample[0]["lo_res_inputs"][0, ...]
            )

            # Constants may either be present or absent depending
            # on the requested constants.
            if "hi_res_inputs" in sample[0]:

                const = np.asarray(
                    sample[0]["hi_res_inputs"][0, ...]
                )

            else:

                # Preserve a valid empty constants feature if no
                # constants have been requested.
                const = np.asarray(
                    [],
                    dtype=np.float32
                )

            mask = np.asarray(
                sample[1]["mask"]
            )

            truth = np.asarray(
                sample[1]["output"]
            )

            # ----------------------------------------------------
            # Remove batch/singleton dimensions carefully
            # ----------------------------------------------------

            # DataGenerator batch_size=1.
            #
            # Remove ONLY the batch dimension first rather than
            # blindly squeezing all dimensions.
            if mask.ndim > 0 and mask.shape[0] == 1:
                mask = mask[0]

            if truth.ndim > 0 and truth.shape[0] == 1:
                truth = truth[0]

            # Some radar loaders return:
            #
            #     truth: (1, H, W)
            #
            # after the batch dimension has been removed.
            #
            # Convert this to:
            #
            #     (H, W)
            #
            if (
                truth.ndim == 3
                and truth.shape[0] == 1
            ):
                truth = truth[0]

            if (
                mask.ndim == 3
                and mask.shape[0] == 1
            ):
                mask = mask[0]

            # ----------------------------------------------------
            # Shape information for first sample
            # ----------------------------------------------------

            if batch == 0:

                print("\nFull-domain shapes:")

                print(
                    "  forecast: ",
                    forecast.shape
                )

                print(
                    "  constants:",
                    const.shape
                )

                print(
                    "  truth:    ",
                    truth.shape
                )

                print(
                    "  mask:     ",
                    mask.shape
                )

                print("\nFlattened sizes:")

                print(
                    "  forecast: ",
                    forecast.size
                )

                print(
                    "  constants:",
                    const.size
                )

                print(
                    "  truth:    ",
                    truth.size
                )

                if forecast.ndim >= 3:

                    print(
                        "  forecast channels:",
                        forecast.shape[-1]
                    )

                if const.ndim >= 3:

                    print(
                        "  constant channels:",
                        const.shape[-1]
                    )

            # ----------------------------------------------------
            # Empty-array check
            # ----------------------------------------------------

            if (
                forecast.size == 0
                or truth.size == 0
            ):

                print(
                    f"Skipping batch {batch}: "
                    "empty forecast/truth array"
                )

                print(
                    f"    forecast={forecast.shape}, "
                    f"truth={truth.shape}"
                )

                skipped_empty += 1
                continue

            # Constants are allowed to be empty when none were
            # requested.
            if (
                len(constants_list) > 0
                and const.size == 0
            ):

                print(
                    f"Skipping batch {batch}: "
                    "constants were requested but "
                    "constant array is empty"
                )

                skipped_empty += 1
                continue

            # ----------------------------------------------------
            # Check forecast/constants for NaN / Inf
            # ----------------------------------------------------

            bad_forecast = report_nonfinite(
                "forecast",
                forecast
            )

            bad_const = False

            if const.size > 0:

                bad_const = report_nonfinite(
                    "constants",
                    const
                )

            # ----------------------------------------------------
            # Forecast channel diagnostics
            # ----------------------------------------------------

            if (
                bad_forecast
                and forecast.ndim >= 3
            ):

                print(
                    f"  Batch {batch}: "
                    "non-finite forecast channels:"
                )

                for ch in range(
                    forecast.shape[-1]
                ):

                    x = forecast[..., ch]

                    n_bad = np.count_nonzero(
                        ~np.isfinite(x)
                    )

                    if n_bad:

                        field_name = (
                            fcst_fields[ch]
                            if ch < len(fcst_fields)
                            else f"channel_{ch}"
                        )

                        print(
                            f"    channel {ch} "
                            f"({field_name}): "
                            f"bad={n_bad}/{x.size}, "
                            f"NaN={np.isnan(x).sum()}, "
                            f"+Inf={np.isposinf(x).sum()}, "
                            f"-Inf={np.isneginf(x).sum()}"
                        )

            # ----------------------------------------------------
            # Constant channel diagnostics
            # ----------------------------------------------------

            if (
                bad_const
                and const.ndim >= 3
            ):

                print(
                    f"  Batch {batch}: "
                    "non-finite constant channels:"
                )

                for ch in range(
                    const.shape[-1]
                ):

                    x = const[..., ch]

                    n_bad = np.count_nonzero(
                        ~np.isfinite(x)
                    )

                    if n_bad:

                        const_name = (
                            constants_list[ch]
                            if ch < len(constants_list)
                            else f"channel_{ch}"
                        )

                        print(
                            f"    channel {ch} "
                            f"({const_name}): "
                            f"bad={n_bad}/{x.size}, "
                            f"NaN={np.isnan(x).sum()}, "
                            f"+Inf={np.isposinf(x).sum()}, "
                            f"-Inf={np.isneginf(x).sum()}"
                        )

            # Forecast/constants have no radar-style validity
            # mask. Therefore non-finite values invalidate the
            # sample.
            if bad_forecast or bad_const:

                print(
                    f"Skipping batch {batch}: "
                    "forecast/constants contain NaN/Inf"
                )

                skipped_nonfinite += 1
                continue

            # ====================================================
            # RADAR-SPECIFIC HANDLING
            # ====================================================

            # ----------------------------------------------------
            # Align mask and truth shapes
            # ----------------------------------------------------

            if mask.shape != truth.shape:

                # Common case:
                #
                #     truth = (H, W, 1)
                #     mask  = (H, W)
                #
                if (
                    truth.ndim == mask.ndim + 1
                    and truth.shape[-1] == 1
                    and truth.shape[:-1] == mask.shape
                ):

                    mask = mask[..., np.newaxis]

                # Opposite singleton-channel case.
                elif (
                    mask.ndim == truth.ndim + 1
                    and mask.shape[-1] == 1
                    and mask.shape[:-1] == truth.shape
                ):

                    mask = mask[..., 0]

                else:

                    raise ValueError(
                        "Mask/truth shape mismatch: "
                        f"mask={mask.shape}, "
                        f"truth={truth.shape}"
                    )

            mask = mask.astype(bool)

            # ----------------------------------------------------
            # Radar diagnostics
            # ----------------------------------------------------

            nan_pixels = np.isnan(truth)

            n_nan = np.count_nonzero(
                nan_pixels
            )

            n_masked = np.count_nonzero(
                mask
            )

            n_nan_masked = np.count_nonzero(
                nan_pixels & mask
            )

            n_nan_unmasked = np.count_nonzero(
                nan_pixels & ~mask
            )

            if (
                n_nan > 0
                or n_masked > 0
            ):

                print(
                    f"Batch {batch} radar:"
                )

                print(
                    f"    truth NaNs:          "
                    f"{n_nan}"
                )

                print(
                    f"    mask True:           "
                    f"{n_masked}"
                )

                print(
                    f"    NaNs covered by mask:"
                    f" {n_nan_masked}"
                )

                print(
                    f"    NaNs outside mask:   "
                    f"{n_nan_unmasked}"
                )

            if n_masked:

                masked_pixels_total += n_masked

                print(
                    f"    masked fraction: "
                    f"{100.0 * n_masked / mask.size:.3f}%"
                )

            # ----------------------------------------------------
            # Find invalid radar values OUTSIDE radar mask
            # ----------------------------------------------------
            #
            # NaNs/Inf are acceptable where the radar mask says
            # that observations are invalid.
            #
            # They are NOT acceptable in nominally valid radar
            # pixels.
            # ----------------------------------------------------

            bad_valid_pixels = (
                ~np.isfinite(truth)
            ) & (
                ~mask
            )

            n_bad_valid = np.count_nonzero(
                bad_valid_pixels
            )

            if n_bad_valid:

                print(
                    f"Skipping batch {batch}: "
                    f"{n_bad_valid} non-finite truth "
                    "pixels are outside the radar mask"
                )

                skipped_nonfinite += 1
                continue

            # ----------------------------------------------------
            # Replace masked radar values
            # ----------------------------------------------------
            #
            # Masked pixels may contain NaNs. Never serialise
            # those NaNs into the TFRecord.
            # ----------------------------------------------------

            truth = truth.copy()

            truth[mask] = 0.0

            # Final radar safety check.
            if not np.all(
                np.isfinite(truth)
            ):

                print(
                    f"Skipping batch {batch}: "
                    "truth still contains NaN/Inf "
                    "after applying radar mask"
                )

                skipped_nonfinite += 1
                continue

            # ====================================================
            # Flatten only AFTER validation/masking
            # ====================================================

            forecast_flat = (
                forecast
                .astype(np.float32, copy=False)
                .flatten()
            )

            const_flat = (
                const
                .astype(np.float32, copy=False)
                .flatten()
            )

            truth_flat = (
                truth
                .astype(np.float32, copy=False)
                .flatten()
            )

            # ----------------------------------------------------
            # Denormalise radar truth for rainfall classification
            # ----------------------------------------------------

            truth_raw = np.asarray(
                denormalise(truth_flat)
            )

            if truth_raw.size == 0:

                print(
                    f"Skipping batch {batch}: "
                    "denormalised truth is empty"
                )

                skipped_empty += 1
                continue

            if not np.all(
                np.isfinite(truth_raw)
            ):

                print(
                    f"Skipping batch {batch}: "
                    "denormalised truth contains "
                    "NaN/Inf"
                )

                skipped_nonfinite += 1
                continue

            # ----------------------------------------------------
            # Mean rainfall
            # ----------------------------------------------------

            truth_mean = truth_raw.mean()

            if not np.isfinite(
                truth_mean
            ):

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
            # Construct TFRecord example
            # ----------------------------------------------------

            feature = {

                "generator_input":
                    _float_feature(
                        forecast_flat
                    ),

                "constants":
                    _float_feature(
                        const_flat
                    ),

                "generator_output":
                    _float_feature(
                        truth_flat
                    )
            }

            features = tf.train.Features(
                feature=feature
            )

            example = tf.train.Example(
                features=features
            )

            # ----------------------------------------------------
            # Write into rainfall class
            # ----------------------------------------------------

            fle_hdles[clss].write(
                example.SerializeToString()
            )

            class_counts[clss] += 1

    # ============================================================
    # Always close TFRecord files
    # ============================================================

    finally:

        for fh in fle_hdles:
            fh.close()

    # ============================================================
    # Summary
    # ============================================================

    total_written = int(
        class_counts.sum()
    )

    total_skipped = (
        skipped_empty
        + skipped_nonfinite
        + skipped_missing_file
    )

    print("\n" + "=" * 70)

    print(
        f"Finished radar TFRecords for year={year}"
    )

    print(
        f"Lead times included: {leadtime}"
    )

    print(
        f"Accumulation: {accumulation} h"
    )

    print(
        f"Generator samples: {len(dgc)}"
    )

    print("\nWritten per rainfall class:")

    for clss, count in enumerate(
        class_counts
    ):

        print(
            f"  class {clss}: {count}"
        )

    print(
        f"\nTotal written: {total_written}"
    )

    print("\nSkipped:")

    print(
        f"  missing source file: "
        f"{skipped_missing_file}"
    )

    print(
        f"  empty arrays:        "
        f"{skipped_empty}"
    )

    print(
        f"  NaN/Inf:             "
        f"{skipped_nonfinite}"
    )

    print(
        f"  total skipped:       "
        f"{total_skipped}"
    )

    print(
        f"\nTotal masked radar pixels replaced: "
        f"{masked_pixels_total}"
    )

    print("\nOutput files:")

    for clss in range(num_class):

        print(
            "  "
            + os.path.join(
                folder,
                f"{year}_{time_idx}.{clss}.tfrecords"
            )
        )

    print("=" * 70)

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
