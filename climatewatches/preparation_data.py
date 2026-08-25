import pandas as pd
import xarray as xr


NOAA_ERSST_MONTHLY_URL = (
    "http://psl.noaa.gov/thredds/dodsC/"
    "Datasets/noaa.ersst.v5/sst.mnmean.nc"
)

COORD_ALIASES = {
    "lat": ("lat", "latitude", "y"),
    "lon": ("lon", "longitude", "x"),
}

VARIABLE_ALIASES = {
    "precip": ("rfe", "rainfall", "precip", "prcp", "rain"),
    "temperature": ("tmean", "temp", "temperature", "tm"),
    "sst": ("sst", "tos", "sea_surface_temperature"),
    "mslp": ("slp", "mslp", "msl", "mean_sea_level_pressure"),
}


def _matching_name(names, aliases):
    lower_names = {
        str(name).lower(): name
        for name in names
    }

    for alias in aliases:
        match = lower_names.get(alias.lower())

        if match is not None:
            return match

    return None


def normalize_lat_lon(data):
    """
    Rename common latitude/longitude dimension aliases to ``lat`` and ``lon``.
    """

    rename = {}

    for target, aliases in COORD_ALIASES.items():
        if target in data.dims:
            continue

        match = _matching_name(data.dims, aliases)

        if match is not None and match != target:
            rename[match] = target
            continue

        match = _matching_name(data.coords, aliases)

        if match is not None and match != target:
            rename[match] = target

    if rename:
        data = data.rename(rename)

    missing = [
        coord
        for coord in ("lat", "lon")
        if coord not in data.dims
    ]

    if missing:
        raise ValueError(
            "Could not identify NetCDF spatial dimension(s): "
            f"{', '.join(missing)}. Supported latitude names are "
            f"{COORD_ALIASES['lat']}; supported longitude names are "
            f"{COORD_ALIASES['lon']}."
        )

    return data


def resolve_variable(ds, kind, variable=None):
    """
    Return a data variable name by explicit name or known aliases.
    """

    if variable is not None:
        if variable in ds:
            return variable

        match = _matching_name(ds.data_vars, (variable,))

        if match is not None:
            return match

        aliases = VARIABLE_ALIASES[kind]

        if str(variable).lower() in aliases:
            match = _matching_name(ds.data_vars, aliases)

            if match is not None:
                return match

        raise ValueError(
            f"Variable {variable!r} not found. Available variables: "
            f"{', '.join(map(str, ds.data_vars))}."
        )

    aliases = VARIABLE_ALIASES[kind]
    match = _matching_name(ds.data_vars, aliases)

    if match is not None:
        return match

    raise ValueError(
        f"Could not identify a {kind} variable. Supported names are "
        f"{aliases}. Available variables: {', '.join(map(str, ds.data_vars))}."
    )


def is_precip_variable(variable):
    return str(variable).lower() in VARIABLE_ALIASES["precip"]


def apply_scientific_plot_style():
    """
    Apply a consistent scientific plotting style.
    """

    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [
            "DejaVu Serif",
            "Liberation Serif",
            "serif",
        ],
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.grid": False,
    })


def set_scientific_title(ax, title, fontsize=14, pad=12):
    ax.set_title(
        title,
        fontsize=fontsize,
        weight="bold",
        fontfamily="DejaVu Serif",
        pad=pad,
    )


def style_scientific_grid(ax):
    ax.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=0.5,
        alpha=0.45,
    )


def style_cartopy_gridlines(gl):
    gl.xlabel_style = {
        "size": 10,
        "family": "DejaVu Serif",
    }
    gl.ylabel_style = {
        "size": 10,
        "family": "DejaVu Serif",
    }
    return gl


def _read_custom_cdt_csv(file_path, value_name):
    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:
        lines = [
            line.strip()
            for line in file.readlines()
            if line.strip()
        ]

    if len(lines) < 4:
        raise ValueError(
            f"{file_path!r} does not contain the expected CDT header/data."
        )

    lon = float(
        lines[1].split(",")[1]
    )
    lat = float(
        lines[2].split(",")[1]
    )

    data = []

    for line in lines[3:]:
        parts = line.split(",")

        if len(parts) < 2:
            continue

        date_text = parts[0].strip()
        value_text = parts[1].strip()
        value = pd.to_numeric(
            value_text,
            errors="coerce",
        )

        data.append(
            [
                date_text,
                value,
            ]
        )

    df = pd.DataFrame(
        data,
        columns=[
            "DATE",
            value_name,
        ],
    )
    df["DATE"] = pd.to_datetime(
        df["DATE"],
        format="%Y%m%d",
        errors="coerce",
    )
    df = df.dropna(
        subset=[
            "DATE",
        ]
    )

    return df, lon, lat


def blend_stn_cdt_data(
    rain_file,
    tmin_file,
    tmax_file,
    output_file="merged_daily_data.csv",
):
    """
    Merge CDT-style rainfall, Tmin, and Tmax station files by date.

    Each input file is expected to use this format:

    ID,Point
    LON,<longitude>
    DATE/LAT,<latitude>
    YYYYMMDD,<value>

    The output keeps the same metadata header and writes daily rows as:
    YYYYMMDD,Rainfall,Tmin,Tmax
    """

    rain_df, lon, lat = _read_custom_cdt_csv(
        rain_file,
        "Rainfall",
    )
    tmin_df, tmin_lon, tmin_lat = _read_custom_cdt_csv(
        tmin_file,
        "Tmin",
    )
    tmax_df, tmax_lon, tmax_lat = _read_custom_cdt_csv(
        tmax_file,
        "Tmax",
    )

    metadata_values = [
        (
            tmin_lon,
            tmin_lat,
        ),
        (
            tmax_lon,
            tmax_lat,
        ),
    ]

    for other_lon, other_lat in metadata_values:
        if round(other_lon, 6) != round(lon, 6) or round(other_lat, 6) != round(lat, 6):
            raise ValueError(
                "Input files do not have matching longitude/latitude metadata."
            )

    merged = pd.merge(
        rain_df,
        tmin_df,
        on="DATE",
        how="outer",
    )
    merged = pd.merge(
        merged,
        tmax_df,
        on="DATE",
        how="outer",
    )
    merged = merged.sort_values(
        "DATE"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:
        file.write("ID,Point\n")
        file.write(f"LON,{lon}\n")
        file.write(f"DATE/LAT,{lat}\n")

        for _, row in merged.iterrows():
            date_text = row["DATE"].strftime(
                "%Y%m%d"
            )
            rainfall = (
                ""
                if pd.isna(row["Rainfall"])
                else f"{row['Rainfall']:.3f}"
            )
            tmin = (
                ""
                if pd.isna(row["Tmin"])
                else f"{row['Tmin']:.3f}"
            )
            tmax = (
                ""
                if pd.isna(row["Tmax"])
                else f"{row['Tmax']:.3f}"
            )
            file.write(
                f"{date_text},{rainfall},{tmin},{tmax}\n"
            )

    print(f"Merged file saved as: {output_file}")

    return merged


def download_sst_mean_data(
    output_file=None,
    url=NOAA_ERSST_MONTHLY_URL,
    start_time=None,
    end_time=None,
    drop_time_bounds=True,
):
    """
    Open NOAA ERSST v5 monthly mean SST data for climatology/monitoring.

    Parameters
    ----------
    output_file : str, optional
        If provided, save the selected dataset to this local NetCDF file.

    url : str
        OPeNDAP URL for the SST monthly mean dataset.

    start_time, end_time : str, optional
        Optional time bounds, for example ``"1950-01-01"`` and
        ``"2026-07-01"``.

    drop_time_bounds : bool
        Drop ``time_bnds`` when it exists, matching the climate-driver scripts.
    """

    dataset = xr.open_dataset(url)

    if drop_time_bounds and "time_bnds" in dataset.variables:
        dataset = dataset.drop_vars(
            "time_bnds"
        )

    if start_time is not None or end_time is not None:
        dataset = dataset.sel(
            time=slice(
                start_time,
                end_time,
            )
        )

    print(dataset)

    if output_file is not None:
        dataset.to_netcdf(
            output_file
        )
        print(f"SST monthly mean data saved as: {output_file}")

    return dataset
