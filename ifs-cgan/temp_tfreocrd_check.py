import glob
import numpy as np
import tensorflow as tf

# CHANGE THIS
pattern = "/network/group/aopp/predict/AWH024_COOPERNATH_IFS/IFS/IFS-Zambia/tfrecords/*"

S = 128
N_FIELDS = 12

feature_description = {
    "generator_input": tf.io.FixedLenFeature(
        (S, S, 2 * N_FIELDS), tf.float32
    ),
    "constants": tf.io.FixedLenFeature(
        (S, S, 2), tf.float32
    ),
    "generator_output": tf.io.FixedLenFeature(
        (S, S, 1), tf.float32
    ),
}

files = sorted(glob.glob(pattern))
print("Files:", len(files))

n = 0

for fname in files:
    print("\nChecking:", fname)

    ds = tf.data.TFRecordDataset(
        fname,
        compression_type="GZIP"
    )

    for raw in ds:
        ex = tf.io.parse_single_example(raw, feature_description)

        for name, arr in ex.items():
            x = arr.numpy()

            if not np.all(np.isfinite(x)):
                print(
                    f"\nBAD RECORD: file={fname}, record={n}, "
                    f"field={name}"
                )
                print("shape:", x.shape)
                print("NaNs:", np.isnan(x).sum())
                print("Infs:", np.isinf(x).sum())
                print("min/max:", np.nanmin(x), np.nanmax(x))
                for ch in range(x.shape[-1]):
                    xx = x[:, :, ch]
                    print(
                        f"channel {ch}: "
                        f"NaNs={np.isnan(xx).sum()}, "
                        f"finite={np.isfinite(xx).all()}"
                    )
                raise SystemExit

        n += 1

print("\nALL OK")
print("Total records checked:", n)