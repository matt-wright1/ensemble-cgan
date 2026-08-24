from pathlib import Path
import numpy as np
import xarray as xr
import os
import argparse


def find_any_nan_mask(
    folder,
    variable,
    x_dim="x",
    y_dim="y",
    save_mask=True
):
    """
    Find grid cells that are NaN in at least one NetCDF file.

    Searches for .nc files inside subdirectories named YYYY.

    Parameters
    ----------
    folder : str or Path
        Root directory containing YYYY subdirectories.

    variable : str
        Name of the variable to check.

    x_dim, y_dim : str
        Names of the spatial dimensions.

    save_mask : bool
        If True, save the mask to folder/mask/mask.nc.

    Returns
    -------
    xarray.DataArray
        Boolean mask if save_mask=False.
    """

    folder = Path(folder)

    # Find .nc files inside YYYY directories
    files = []

    for year_dir in folder.iterdir():
        if (
            year_dir.is_dir()
            and year_dir.name.isdigit()
            and len(year_dir.name) == 4
        ):
            files.extend(year_dir.glob("*.nc"))

    files = sorted(files)

    if not files:
        raise FileNotFoundError(
            f"No .nc files found inside YYYY folders in {folder}"
        )

    print(f"Found {len(files)} files.")

    # ---------------------------------------------------------
    # First file establishes the grid
    # ---------------------------------------------------------

    with xr.open_dataset(files[0]) as ds:

        if variable not in ds:
            raise KeyError(
                f"'{variable}' not found in {files[0]}"
            )

        da = ds[variable]

        nan_mask = da.isnull()

        other_dims = [
            d for d in nan_mask.dims
            if d not in (y_dim, x_dim)
        ]

        if other_dims:
            nan_mask = nan_mask.any(dim=other_dims)

        nan_mask = nan_mask.load()

    # ---------------------------------------------------------
    # Loop through remaining files
    # ---------------------------------------------------------

    for i, file in enumerate(files[1:], start=2):

        print(f"[{i}/{len(files)}] {file}")

        with xr.open_dataset(file) as ds:

            if variable not in ds:
                raise KeyError(
                    f"'{variable}' not found in {file}"
                )

            file_mask = ds[variable].isnull()

            other_dims = [
                d for d in file_mask.dims
                if d not in (y_dim, x_dim)
            ]

            if other_dims:
                file_mask = file_mask.any(dim=other_dims)

            file_mask = file_mask.load()

            nan_mask = nan_mask | file_mask

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    n_bad = int(nan_mask.sum())
    n_total = nan_mask.size

    print("\nDone.")
    print(f"Grid cells:       {n_total:,}")
    print(f"NaN in any file:  {n_bad:,}")
    print(f"Fraction masked:  {n_bad / n_total:.2%}")

    if save_mask:
        out_dir = folder / "mask"
        out_dir.mkdir(exist_ok=True)

        out_path = out_dir / "mask.nc"

        nan_mask.to_netcdf(out_path)

        print(f"Mask saved to:    {out_path}")
    else:
        return nan_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find grid cells that are NaN in at least one NetCDF file."
    )

    parser.add_argument(
        "folder",
        type=Path,
        help="Root directory containing YYYY subdirectories."
    )

    parser.add_argument(
        "variable",
        type=str,
        help="Name of the variable to check."
    )

    args = parser.parse_args()

    find_any_nan_mask(
        folder=args.folder,
        variable=args.variable,
    )