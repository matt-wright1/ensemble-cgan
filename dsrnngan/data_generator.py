"""
Data generator class for full-image evaluation of precipitation
downscaling network.

Supports:
    - one or multiple lead times
    - 6 h or 24 h accumulation
    - configurable high-resolution constants
    - radar-specific missing-file filtering
    - radar truth/mask shape correction
"""

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
    """
    Data generator returning forecast, constants, mask and truth data.

    `leadtime` denotes the START of the target accumulation interval.

    For example:

        leadtime=30
        accumulation=6

    corresponds to the +30 to +36 hour target interval.

    Multiple lead times may be supplied:

        leadtime=[30, 54, 78]

    In that case, every combination of:

        date x leadtime

    is constructed as an individual sample.

    For example:

        dates = ["20180409", "20200607"]
        leadtime = [30, 54]

    produces:

        20180409, 30
        20180409, 54
        20200607, 30
        20200607, 54

    Samples for which the required radar truth file does not exist
    are removed during initialisation.
    """

    def __init__(
        self,
        dates,
        fcst_fields,
        leadtime,
        accumulation,
        batch_size=1,
        log_precip=True,
        shuffle=True,
        constants_list=None,
        fcst_norm=True,
        autocoarsen=False,
        seed=9999,
    ):
        """
        Parameters
        ----------
        dates : list of str
            Forecast initialisation dates, e.g. YYYYMMDD.

        fcst_fields : list of str
            Forecast fields to use.

        leadtime : int or sequence of int
            Start hour(s) of the target accumulation interval.

            For example:

                leadtime=30

            or:

                leadtime=[30, 54, 78]

        accumulation : int
            Target accumulation period in hours.
            Currently 6 or 24.

        batch_size : int
            Batch size.

        log_precip : bool
            Whether to apply log10(1+x) transformation to
            precipitation-related fields.

        shuffle : bool
            Whether to shuffle samples.

        constants_list : list of str or None
            High-resolution constant fields to use.

            None means no high-resolution constants are returned.

        fcst_norm : bool
            Whether forecast fields are normalised.

        autocoarsen : bool
            Whether forecast input should be replaced by coarsened
            radar truth.

        seed : int
            Random seed used for reproducible shuffling.
        """

        # ========================================================
        # Lead-time handling
        # ========================================================

        if np.isscalar(leadtime):

            leadtime = [
                int(leadtime)
            ]

        else:

            leadtime = [
                int(x)
                for x in leadtime
            ]

        if len(leadtime) == 0:

            raise ValueError(
                "At least one lead time must be supplied"
            )

        # --------------------------------------------------------
        # Validate lead times
        # --------------------------------------------------------

        for ld in leadtime:

            if ld < 0:

                raise ValueError(
                    f"Lead time must be >= 0; got {ld}"
                )

            if ld > 168:

                raise ValueError(
                    f"Lead time must be <= 168 h; got {ld}"
                )

            if ld % HOURS != 0:

                raise ValueError(
                    f"Lead time {ld} is not divisible "
                    f"by HOURS={HOURS}"
                )

        # --------------------------------------------------------
        # Validate accumulation
        # --------------------------------------------------------

        if accumulation not in (6, 24):

            raise ValueError(
                f"Unsupported accumulation period: "
                f"{accumulation} hours"
            )

        # ========================================================
        # Store configuration
        # ========================================================

        self.fcst_fields = fcst_fields

        self.requested_leadtime = np.asarray(
            leadtime,
            dtype=int,
        )

        self.accumulation = int(
            accumulation
        )

        self.batch_size = int(
            batch_size
        )

        self.log_precip = log_precip
        self.shuffle = shuffle
        self.fcst_norm = fcst_norm
        self.autocoarsen = autocoarsen
        self.seed = seed

        self.constants_list = constants_list

        # ========================================================
        # Autocoarsening
        # ========================================================

        if self.autocoarsen:

            df_dict = (
                read_config
                .read_downscaling_factor()
            )

            self.ds_factor = (
                df_dict[
                    "downscaling_factor"
                ]
            )

        # ========================================================
        # High-resolution constants
        # ========================================================

        if self.constants_list is not None:

            self.constants = (
                load_hires_constants(
                    batch_size=self.batch_size,
                    constants_list=self.constants_list,
                )
            )

        else:

            self.constants = None

        # ========================================================
        # Construct every:
        #
        #     date x leadtime
        #
        # combination
        # ========================================================

        temp_dates = np.asarray(
            dates
        )

        if len(temp_dates) == 0:

            raise ValueError(
                "No dates supplied to DataGenerator"
            )

        # Example:
        #
        # dates:
        #
        #   20200101
        #   20200102
        #
        # leadtime:
        #
        #   30
        #   54
        #   78
        #
        # becomes:
        #
        # dates:
        #
        #   20200101
        #   20200101
        #   20200101
        #   20200102
        #   20200102
        #   20200102
        #
        # leadtime:
        #
        #   30
        #   54
        #   78
        #   30
        #   54
        #   78
        #

        all_dates = np.repeat(
            temp_dates,
            len(self.requested_leadtime),
        )

        all_leadtimes = np.tile(
            self.requested_leadtime,
            len(temp_dates),
        )

        # ========================================================
        # Remove samples with missing radar truth
        # ========================================================
        #
        # IMPORTANT:
        #
        # Use load_truth_and_mask() itself for this check rather
        # than reconstructing radar filenames here.
        #
        # This ensures that availability checking uses the same
        # radar date/time/path logic as the actual radar loader.
        #
        # Each (date, leadtime) pair is checked separately because
        # different target intervals require different radar data.
        # ========================================================

        valid_dates = []
        valid_leadtimes = []

        missing_count = 0

        total_samples = len(
            all_dates
        )

        print(
            "Checking radar availability "
            f"for {total_samples} "
            "(date, leadtime) samples..."
        )

        for date, ld in zip(
            all_dates,
            all_leadtimes,
        ):

            try:

                load_truth_and_mask(
                    date,
                    leadtime=int(ld),
                    accumulation=self.accumulation,
                    log_precip=self.log_precip,
                )

            except FileNotFoundError as e:

                missing_count += 1

                print(
                    "Skipping missing radar sample: "
                    f"date={date}, "
                    f"leadtime={ld} h, "
                    f"accumulation={self.accumulation} h"
                )

                print(
                    f"  {e}"
                )

                continue

            valid_dates.append(
                date
            )

            valid_leadtimes.append(
                ld
            )

        # ========================================================
        # Store valid samples only
        # ========================================================

        self.dates = np.asarray(
            valid_dates
        )

        self.leadtime = np.asarray(
            valid_leadtimes,
            dtype=int,
        )

        print(
            "Radar availability: "
            f"{len(self.dates)}/"
            f"{total_samples} "
            "samples available"
        )

        if missing_count:

            print(
                f"Skipped {missing_count} samples "
                "because radar truth files were missing"
            )

        if len(self.dates) == 0:

            raise ValueError(
                "No valid radar samples available "
                "for DataGenerator"
            )

        # ========================================================
        # Random-number generator
        # ========================================================

        self.rng = np.random.default_rng(
            self.seed
        )

        # ========================================================
        # Shuffle
        # ========================================================

        if self.shuffle:

            self.shuffle_data(
                self.rng
            )

    # ============================================================
    # Length
    # ============================================================

    def __len__(self):
        """
        Number of complete batches in the dataset.
        """

        return (
            len(self.dates)
            // self.batch_size
        )

    # ============================================================
    # Autocoarsening
    # ============================================================

    def _dataset_autocoarsener(
        self,
        truth,
    ):

        kernel_tf = tf.constant(
            1.0
            / (
                self.ds_factor
                * self.ds_factor
            ),
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

    # ============================================================
    # Get batch
    # ============================================================

    def __getitem__(
        self,
        idx,
    ):
        """
        Return one batch.

        Each sample corresponds to one specific:

            forecast initialisation date
            +
            lead time
            +
            accumulation period
        """

        # --------------------------------------------------------
        # Batch indices
        # --------------------------------------------------------

        start = (
            idx
            * self.batch_size
        )

        end = (
            (idx + 1)
            * self.batch_size
        )

        # --------------------------------------------------------
        # Select date/leadtime pairs
        # --------------------------------------------------------

        dates_batch = self.dates[
            start:end
        ]

        leadtime_batch = self.leadtime[
            start:end
        ]

        # --------------------------------------------------------
        # Load forecast, radar truth and radar mask
        # --------------------------------------------------------

        (
            data_x_batch,
            data_y_batch,
            data_mask_batch,
        ) = load_fcst_truth_batch(
            dates_batch,
            leadtime_batch,
            accumulation=self.accumulation,
            fcst_fields=self.fcst_fields,
            log_precip=self.log_precip,
            norm=self.fcst_norm,
        )

        # ========================================================
        # RADAR-SPECIFIC SHAPE FIX
        # ========================================================
        #
        # The single-radar loader may return:
        #
        #     truth: (B, 1, H, W)
        #     mask:  (B, 1, H, W)
        #
        # Evaluation/training code expects:
        #
        #     truth: (B, H, W)
        #     mask:  (B, H, W)
        #
        # Only remove axis 1 when it is explicitly singleton.
        # Do not use an unrestricted np.squeeze(), because batch
        # size or other meaningful singleton dimensions could
        # otherwise disappear.
        # ========================================================

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

        # --------------------------------------------------------
        # Radar shape validation
        # --------------------------------------------------------

        if data_y_batch.ndim != 3:

            raise ValueError(
                "Expected radar truth shape "
                "(B, H, W), "
                f"got {data_y_batch.shape}"
            )

        if (
            data_mask_batch.shape
            != data_y_batch.shape
        ):

            raise ValueError(
                "Radar mask/truth shape mismatch: "
                f"mask={data_mask_batch.shape}, "
                f"truth={data_y_batch.shape}"
            )

        # Ensure mask has boolean semantics.
        data_mask_batch = (
            data_mask_batch.astype(
                bool,
                copy=False,
            )
        )

        # ========================================================
        # Autocoarsening
        # ========================================================

        if self.autocoarsen:

            # Never allow missing radar pixels to propagate
            # through the convolution as NaNs.
            truth_temp = (
                data_y_batch.copy()
            )

            truth_temp[
                data_mask_batch
            ] = 0.0

            data_x_batch = (
                self._dataset_autocoarsener(
                    truth_temp[
                        ...,
                        np.newaxis
                    ]
                )
            )

        # ========================================================
        # Return model inputs / outputs
        # ========================================================

        output_dict = {
            "output": data_y_batch,
            "mask": data_mask_batch,
        }

        # --------------------------------------------------------
        # No high-resolution constants
        # --------------------------------------------------------

        if self.constants is None:

            input_dict = {
                "lo_res_inputs":
                    data_x_batch,
            }

        # --------------------------------------------------------
        # Forecast + flexible high-resolution constants
        # --------------------------------------------------------

        else:

            input_dict = {
                "lo_res_inputs":
                    data_x_batch,

                "hi_res_inputs":
                    self.constants,
            }

        return (
            input_dict,
            output_dict,
        )

    # ============================================================
    # Shuffle
    # ============================================================

    def shuffle_data(
        self,
        rng,
    ):
        """
        Shuffle dates and lead times together.

        It is essential that the same permutation is applied to
        both arrays so that each date remains paired with its
        corresponding lead time.
        """

        if (
            len(self.leadtime)
            != len(self.dates)
        ):

            raise ValueError(
                "dates and leadtime arrays "
                "have different lengths"
            )

        p = rng.permutation(
            len(self.dates)
        )

        self.dates = (
            self.dates[p]
        )

        self.leadtime = (
            self.leadtime[p]
        )

    # ============================================================
    # End of epoch
    # ============================================================

    def on_epoch_end(
        self,
    ):

        if self.shuffle:

            # Continue using the generator's state rather than
            # recreating a generator with the same seed each
            # epoch. This gives deterministic but different
            # permutations from epoch to epoch.
            self.shuffle_data(
                self.rng
            )


if __name__ == "__main__":
    pass