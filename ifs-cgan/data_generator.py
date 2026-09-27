""" Data generator class for full-image evaluation of precipitation downscaling network """
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

from data import load_fcst_truth_batch, load_hires_constants, HOURS
import read_config


class DataGenerator(Sequence):
    """
    Data generator that returns forecast, constants, mask and truth data.

    `leadtime` denotes the START of the target interval.

    For example:
        leadtime=30, accumulation=6

    corresponds to the +30 to +36 hour target interval.
    """
    def __init__(self, dates, fcst_fields,
                 leadtime, accumulation,
                 batch_size=1, log_precip=True,
                 shuffle=True, constants=True, fcst_norm=True,
                 autocoarsen=False, seed=9999):
        '''
        Forecast: input forecast data
        Constants: geographic fields; LSM and orography
        Mask: False where truth data is valid, True where truth data is invalid
        Truth: precipitation data
        Parameters:
            dates (list of YYYYMMDD strings): The forecast start dates to be used
            fcst_fields (list of strings): The forecast fields to be used
            start_hour (int): Lead time of first forecast/truth hour to use
            end_hour (int): Lead time of last forecast/truth hour to use
            batch size (int): Batch size
            log_precip (bool): Whether to apply log10(1+x) transform to precip-related fields
            shuffle (bool): Whether to shuffle data (else return sorted by date then lead time)
            constants (bool): Whether to return orography/LSM fields
            fcst_norm (bool): Whether to apply normalisation to fields to make O(1)
            autocoarsen (bool): Whether to replace forecast data by coarsened truth
            seed (int): Random seed given to NumPy, used for repeatable shuffles
        '''

        if np.isscalar(leadtime):
            leadtime = [int(leadtime)]
        else:
            leadtime = [int(x) for x in leadtime]

        if not leadtime:
            raise ValueError("At least one lead time must be supplied")

        for ld in leadtime:
            assert ld >= 0
            assert ld <= 168
            assert ld % HOURS == 0

        if accumulation not in (6, 24):
            raise ValueError(
                f"Unsupported accumulation period: {accumulation} hours"
            )

        self.fcst_fields = fcst_fields
        self.requested_leadtime = np.asarray(leadtime, dtype=int)
        self.accumulation = accumulation
        self.batch_size = batch_size
        self.log_precip = log_precip
        self.shuffle = shuffle
        self.fcst_norm = fcst_norm
        self.autocoarsen = autocoarsen
        self.seed = seed

        if self.autocoarsen:
            # read downscaling factor from file
            df_dict = read_config.read_downscaling_factor()  # read downscaling params
            self.ds_factor = df_dict["downscaling_factor"]

        if constants:
            self.constants = load_hires_constants(self.batch_size)
        else:
            self.constants = None

        # convert to numpy array for easy use of np.repeat
        temp_dates = np.array(dates)
        
        # Construct every (date, leadtime) training sample
        self.dates = np.repeat(
            temp_dates,
            len(self.requested_leadtime)
        )

        self.leadtime = np.tile(
            self.requested_leadtime,
            len(temp_dates)
        )
        self.rng = np.random.default_rng(seed)

        if self.shuffle:
            self.shuffle_data(self.rng)

    def __len__(self):
        # Number of batches in dataset
        return len(self.dates) // self.batch_size

    def _dataset_autocoarsener(self, truth):
        kernel_tf = tf.constant(1.0/(self.ds_factor*self.ds_factor), shape=(self.ds_factor, self.ds_factor, 1, 1), dtype=tf.float32)
        image = tf.nn.conv2d(truth, filters=kernel_tf, strides=[1, self.ds_factor, self.ds_factor, 1], padding='VALID',
                             name='conv_debug', data_format='NHWC')
        return image

    def __getitem__(self, idx):
        # Get batch at index idx
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size

        dates_batch = self.dates[start:end]
        leadtime_batch = self.leadtime[start:end]

        # Load and return this batch of data
        data_x_batch, data_y_batch, data_mask_batch = load_fcst_truth_batch(
            dates_batch,
            leadtime_batch,
            accumulation=self.accumulation,
            fcst_fields=self.fcst_fields,
            log_precip=self.log_precip,
            norm=self.fcst_norm
        )

        if self.autocoarsen:
            # replace forecast data by coarsened truth data!
            truth_temp = data_y_batch.copy()
            truth_temp[data_mask_batch] = 0.0
            data_x_batch = self._dataset_autocoarsener(truth_temp[..., np.newaxis])

        if self.constants is None:
            return {"lo_res_inputs": data_x_batch},\
                   {"output": data_y_batch,
                    "mask": data_mask_batch}
        else:
            return {"lo_res_inputs": data_x_batch,
                    "hi_res_inputs": self.constants},\
                   {"output": data_y_batch,
                    "mask": data_mask_batch}

    def shuffle_data(self, rng):
        assert len(self.leadtime) == len(self.dates)

        p = rng.permutation(len(self.dates))

        self.dates = self.dates[p]
        self.leadtime = self.leadtime[p]

    def on_epoch_end(self):
        if self.shuffle:
            self.shuffle_data(self.rng)


if __name__ == "__main__":
    pass
