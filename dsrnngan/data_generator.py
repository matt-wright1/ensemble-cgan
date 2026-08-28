""" Data generator class for full-image evaluation of precipitation downscaling network """

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

from data import (
    load_fcst_truth_batch,
    load_hires_constants,
    load_truth_and_mask,
    HOURS,
)
import read_config


class DataGenerator(Sequence):
    '''
    Data generator class that returns (forecast, constants, mask, truth) data.

    Class will return forecast data at the start and end of each interval
    (for non-accumulated fields) and accumulated fields over the interval.
    The truth data is averaged over the interval.

    DataGenerator(
        ["20180409", "20200607"],
        fcst_fields=["cape", "tp"],
        start_hour=12,
        end_hour=24
    )
    will return data over two periods: 12-18 and 18-24 hours for the
    forecasts initialised on 20180409 and 20200607.
    '''

    def __init__(
        self,
        dates,
        fcst_fields,
        start_hour=6,
        end_hour=6,
        batch_size=1,
        log_precip=True,
        shuffle=True,
        constants=True,
        fcst_norm=True,
        autocoarsen=False,
        seed=9999,
    ):
        '''
        Forecast: input forecast data
        Constants: geographic fields; LSM and orography
        Mask: False where truth data is valid, True where truth data is invalid
        Truth: precipitation data

        Parameters:
            dates (list of YYYYMMDD strings):
                The forecast start dates to be used
            fcst_fields (list of strings):
                The forecast fields to be used
            start_hour (int):
                Lead time of first forecast/truth hour to use
            end_hour (int):
                Lead time of last forecast/truth hour to use
            batch_size (int):
                Batch size
            log_precip (bool):
                Whether to apply log10(1+x) transform to precip-related fields
            shuffle (bool):
                Whether to shuffle data
            constants (bool):
                Whether to return orography/LSM fields
            fcst_norm (bool):
                Whether to apply normalisation to fields to make O(1)
            autocoarsen (bool):
                Whether to replace forecast data by coarsened truth
            seed (int):
                Random seed given to NumPy
        '''

        # ----------------------------------------------------
        # Sanity checks
        # ----------------------------------------------------

        assert start_hour >= 0
        assert end_hour <= 168
        assert start_hour % HOURS == 0
        assert end_hour % HOURS == 0
        assert end_hour >= start_hour
        assert autocoarsen is False

        self.fcst_fields = fcst_fields
        self.batch_size = batch_size
        self.log_precip = log_precip
        self.shuffle = shuffle
        self.fcst_norm = fcst_norm
        self.autocoarsen = autocoarsen
        self.seed = seed

        # ----------------------------------------------------
        # Autocoarsening
        # ----------------------------------------------------

        if self.autocoarsen:
            df_dict = read_config.read_downscaling_factor()
            self.ds_factor = df_dict["downscaling_factor"]

        # ----------------------------------------------------
        # High-resolution constants
        # ----------------------------------------------------

        if constants:
            self.constants = load_hires_constants(self.batch_size)
        else:
            self.constants = None

        # ----------------------------------------------------
        # Construct requested date/time samples
        # ----------------------------------------------------

        temp_dates = np.array(dates)

        # There is only one valid time in this set of forecasts.
        #
        # 0 represents the first valid interval used by
        # load_fcst_truth_batch/load_truth_and_mask.
        temp_time_idxs = np.array([0])

        all_dates = np.repeat(
            temp_dates,
            len(temp_time_idxs)
        )

        all_time_idxs = np.tile(
            temp_time_idxs,
            len(temp_dates)
        )

        # ----------------------------------------------------
        # Remove samples with missing radar truth
        # ----------------------------------------------------
        #
        # IMPORTANT:
        #
        # Use load_truth_and_mask() itself here rather than
        # reconstructing the radar filename in this class.
        #
        # This means the availability check uses exactly the
        # same date/time/path logic as evaluation.
        # ----------------------------------------------------

        valid_dates = []
        valid_time_idxs = []

        missing_count = 0

        print("Checking radar availability...")

        for date, time_idx in zip(all_dates, all_time_idxs):

            try:
                load_truth_and_mask(
                    date,
                    time_idx,
                    log_precip=self.log_precip,
                )

            except FileNotFoundError as e:

                missing_count += 1

                print(
                    f"Skipping missing radar sample: "
                    f"date={date}, time_idx={time_idx}"
                )
                print(f"  {e}")

                continue

            valid_dates.append(date)
            valid_time_idxs.append(time_idx)

        # ----------------------------------------------------
        # Store only valid samples
        # ----------------------------------------------------

        self.dates = np.asarray(valid_dates)
        self.time_idxs = np.asarray(valid_time_idxs)

        print(
            f"Radar availability: "
            f"{len(self.dates)}/{len(all_dates)} samples available"
        )

        if missing_count:
            print(
                f"Skipped {missing_count} samples because "
                "radar truth files were missing"
            )

        if len(self.dates) == 0:
            raise ValueError(
                "No valid radar samples available for DataGenerator"
            )

        # ----------------------------------------------------
        # Shuffle
        # ----------------------------------------------------

        if self.shuffle:
            rng = np.random.default_rng(seed)
            self.shuffle_data(rng)

    def __len__(self):
        """Number of batches in dataset."""

        return len(self.dates) // self.batch_size

    def _dataset_autocoarsener(self, truth):

        kernel_tf = tf.constant(
            1.0 / (self.ds_factor * self.ds_factor),
            shape=(
                self.ds_factor,
                self.ds_factor,
                1,
                1,
            ),
            dtype=tf.float32,
        )

        image = tf.nn.conv2d(
            truth,
            filters=kernel_tf,
            strides=[
                1,
                self.ds_factor,
                self.ds_factor,
                1,
            ],
            padding="VALID",
            name="conv_debug",
            data_format="NHWC",
        )

        return image

    def __getitem__(self, idx):
        """Get batch at index idx."""

        # ----------------------------------------------------
        # Select dates/time indices for this batch
        # ----------------------------------------------------

        dates_batch = self.dates[
            idx * self.batch_size:(idx + 1) * self.batch_size
        ]

        time_idx_batch = self.time_idxs[
            idx * self.batch_size:(idx + 1) * self.batch_size
        ]

        # ----------------------------------------------------
        # Load forecast, truth and mask
        # ----------------------------------------------------

        data_x_batch, data_y_batch, data_mask_batch = (
            load_fcst_truth_batch(
                dates_batch,
                time_idx_batch,
                fcst_fields=self.fcst_fields,
                log_precip=self.log_precip,
                norm=self.fcst_norm,
            )
        )

        # ----------------------------------------------------
        # Fix single-radar truth/mask shape
        #
        # Current single-radar loader returns:
        #
        #     (B, 1, H, W)
        #
        # Evaluation expects:
        #
        #     (B, H, W)
        #
        # ----------------------------------------------------

        if (
            data_y_batch.ndim == 4
            and data_y_batch.shape[1] == 1
        ):
            data_y_batch = np.squeeze(
                data_y_batch,
                axis=1,
            )

        if (
            data_mask_batch.ndim == 4
            and data_mask_batch.shape[1] == 1
        ):
            data_mask_batch = np.squeeze(
                data_mask_batch,
                axis=1,
            )

        # ----------------------------------------------------
        # Check expected evaluation shapes
        # ----------------------------------------------------

        assert data_y_batch.ndim == 3, (
            f"Expected truth shape (B,H,W), "
            f"got {data_y_batch.shape}"
        )

        assert data_mask_batch.shape == data_y_batch.shape, (
            f"Mask/truth shape mismatch: "
            f"mask={data_mask_batch.shape}, "
            f"truth={data_y_batch.shape}"
        )

        # ----------------------------------------------------
        # Autocoarsening
        # ----------------------------------------------------

        if self.autocoarsen:

            truth_temp = data_y_batch.copy()

            truth_temp[data_mask_batch] = 0.0

            data_x_batch = self._dataset_autocoarsener(
                truth_temp[..., np.newaxis]
            )

        # ----------------------------------------------------
        # Return model inputs / outputs
        # ----------------------------------------------------

        if self.constants is None:

            return (
                {
                    "lo_res_inputs": data_x_batch,
                },
                {
                    "output": data_y_batch,
                    "mask": data_mask_batch,
                },
            )

        else:

            return (
                {
                    "lo_res_inputs": data_x_batch,
                    "hi_res_inputs": self.constants,
                },
                {
                    "output": data_y_batch,
                    "mask": data_mask_batch,
                },
            )

    def shuffle_data(self, rng):
        """Shuffle dates and time indices together."""

        assert len(self.time_idxs) == len(self.dates)

        p = rng.permutation(len(self.dates))

        self.dates = self.dates[p]
        self.time_idxs = self.time_idxs[p]

    def on_epoch_end(self):

        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            self.shuffle_data(rng)


if __name__ == "__main__":
    pass