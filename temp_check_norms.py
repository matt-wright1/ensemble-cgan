import netCDF4 as nc
import numpy as np

path = "/network/group/aopp/predict/AWH024_COOPERNATH_IFS/IFS/IFS-Zambia/hindcasts/2018/sp.nc"

with nc.Dataset(path) as ds:
    x = ds["sp_mean"][:]

    print("shape:", x.shape)
    print("type:", type(x))
    print("masked:", np.ma.isMaskedArray(x))

    if np.ma.isMaskedArray(x):
        print("masked count:", np.ma.count_masked(x))

    a = np.asarray(x.filled(np.nan) if np.ma.isMaskedArray(x) else x)

    print("NaN count:", np.isnan(a).sum())
    print("Inf count:", np.isinf(a).sum())
    print("total:", a.size)

    # Which forecast dates contain bad values?
    bad = np.any(~np.isfinite(a), axis=(1, 2, 3))
    print("bad time indices:", np.where(bad)[0])

    time = ds["time"]
    dates = nc.num2date(
        time[:],
        units=time.units,
        calendar=getattr(time, "calendar", "standard")
    )

    print("bad index 19:", dates[19])