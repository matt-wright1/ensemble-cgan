
import numpy as np
from tfrecords_generator import write_data
import tensorflow as tf
import glob
import os
import read_config

years = [2018, 2019, 2020, 2021]


def generate_tfrecords(years):
    data_paths = read_config.get_data_paths()
    records_folder = data_paths["TFRecords"]["tfrecords_path"]

    os.makedirs(records_folder, exist_ok=True)

    for year in years:
        print(f'Writing {year}')
        write_data(int(year))

    for year in years:
        print(year)
        files = glob.glob(f"{records_folder}/{year}_1.*.tfrecords")

        for f in files:
            print(f)
            try:
                n = sum(1 for _ in tf.data.TFRecordDataset(f, compression_type="GZIP"))
                print("  total records:", n)
                print("  size MB:", os.path.getsize(f) / 1e6)
            except Exception as e:
                print("  BAD:", e)

    for year in years:
        print(year)
        files = glob.glob(f"{records_folder}/{year}_1.*.tfrecords")
        
        for f in files:
            print(f)
            try:
                ds = tf.data.TFRecordDataset(f, compression_type="GZIP")
                n = sum(1 for _ in ds)
                print("  total records:", n)
                print("  size MB:", os.path.getsize(f) / 1e6)

                # Re-open to read the first record
                ds = tf.data.TFRecordDataset(f, compression_type="GZIP")
                first_raw = next(iter(ds.take(1)))
                first_ex = tf.train.Example.FromString(first_raw.numpy())

                print("  first record keys:", list(first_ex.features.feature.keys()))
                for k, v in first_ex.features.feature.items():
                    kind = v.WhichOneof("kind")
                    if kind == "bytes_list":
                        print(f"    {k}: bytes_list len={len(v.bytes_list.value)}")
                    elif kind == "float_list":
                        print(f"    {k}: float_list len={len(v.float_list.value)}")
                    elif kind == "int64_list":
                        print(f"    {k}: int64_list len={len(v.int64_list.value)}")

            except Exception as e:
                print("  BAD:", e)


if __name__ == "__main__":
    generate_tfrecords(years)