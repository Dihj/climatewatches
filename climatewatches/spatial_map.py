"""Spatial climate maps."""

import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import rioxarray  # noqa: F401 - registers the xarray ``.rio`` accessor

from matplotlib.colors import LinearSegmentedColormap
from rasterio.features import geometry_mask
from shapely.geometry import mapping

try:
    from .preparation_data import (
        set_plot_defaults,
        normalize_lat_lon,
        resolve_variable,
        set_map_grid,
        set_title,
    )
except ImportError:
    from preparation_data import (
        set_plot_defaults,
        normalize_lat_lon,
        resolve_variable,
        set_map_grid,
        set_title,
    )

season_months_dict = {
    "ONDJ": [10, 11, 12, 1],
    "OND": [10, 11, 12],
    "NDJ": [11, 12, 1],
    "DJF": [12, 1, 2],
    "JFM": [1, 2, 3],
    "FMA": [2, 3, 4],
    "MAM": [3, 4, 5],
    "AMJ": [4, 5, 6],
    "MJJ": [5, 6, 7],
    "JJA": [6, 7, 8],
    "JAS": [7, 8, 9],
    "ASO": [8, 9, 10],
    "SON": [9, 10, 11],
    "NDJFM": [11, 12, 1, 2, 3],
    "ONDJFMA": [10, 11, 12, 1, 2, 3, 4],
}


TRANSLATION = {
    "fr": {
        "cumulative_title": "Cumul des précipitations",
        "cumulative_label": "Cumul des précipitations (mm)",
        "anomaly_title": "Anomalie standardisée des précipitations",
        "anomaly_label": "Anomalie standardisée des précipitations (σ)",
        "temperature_title": "Température moyenne",
        "temperature_label": "Température moyenne",
        "temperature_anomaly_title": "Anomalie standardisée de température",
        "temperature_anomaly_label": "Anomalie standardisée de température (σ)",
    },
    "en": {
        "cumulative_title": "Cumulative rainfall",
        "cumulative_label": "Cumulative rainfall (mm)",
        "anomaly_title": "Standardized rainfall anomaly",
        "anomaly_label": "Standardized rainfall anomaly (σ)",
        "temperature_title": "Mean temperature",
        "temperature_label": "Mean temperature",
        "temperature_anomaly_title": "Standardized temperature anomaly",
        "temperature_anomaly_label": "Standardized temperature anomaly (σ)",
    },
    "mg": {
        "cumulative_title": "Rotsakorana",
        "cumulative_label": "Rotsakorana (mm)",
        "anomaly_title": "Tahan'ny fironan'ny rotsakorana",
        "anomaly_label": "Tahan'ny fironan'ny rotsakorana (σ)",
        "temperature_title": "Maripana ankapobeny",
        "temperature_label": "Maripana ankapobeny",
        "temperature_anomaly_title": "Tahan'ny fironan'ny maripana",
        "temperature_anomaly_label": "Tahan'ny fironan'ny maripana (σ)",
    },
}


# ============================================================
# Rainfall colormap
# ============================================================

coulPREC_colors = [
    "#CB9362",
    "#DABE90",
    "#D2D179",
    "#91D47D",
    "#5CC247",
    "#49A136",
    "#287733",
    "#2A7E61",
    "#309181",
    "#327295",
    "#5B86C8",
    "#9D8CD9",
    "#CC79D2",
    "#C24799",
    "#7E2A73",
]

coulPREC = LinearSegmentedColormap.from_list(
    "coulPREC",
    coulPREC_colors,
)

anomaly_colors = [
    "orange",
    "white",
    "green",
]

cmap_rain = LinearSegmentedColormap.from_list(
    "anomaly_cmap",
    anomaly_colors,
    N=256,
)

temp_colors = [
    "#313695",
    "#4575b4",
    "#74add1",
    "#abd9e9",
    "#e0f3f8",
    "#ffffbf",
    "#fee090",
    "#fdae61",
    "#f46d43",
    "#d73027",
    "#a50026",
]

cmap_temp = LinearSegmentedColormap.from_list(
    "cmap_temp",
    temp_colors,
)


def mask_raster_with_shape(data, gdf):
    """
    Masque le raster pour ne garder que l'intérieur du shapefile
    sans effet de bord.
    """

    data = normalize_lat_lon(data)
    transform = data.rio.transform()

    out_shape = (
        data.sizes["lat"],
        data.sizes["lon"],
    )

    geometries = [
        mapping(geom)
        for geom in gdf.geometry
    ]

    mask = geometry_mask(
        geometries,
        transform=transform,
        invert=True,
        out_shape=out_shape,
    )

    masked_data = data.where(mask)

    return masked_data


def _aggregate_period(data, analysis, timescale, year, reducer):
    """Select and aggregate a monthly or cross-year seasonal period."""

    if analysis == "month":
        selected_month = int(timescale)
        selected = data.sel(
            time=(
                (data["time.month"] == selected_month)
                & (data["time.year"] == year)
            )
        )
        period_label = f"{selected_month:02d}/{year}"
        filename_period = f"month_{selected_month:02d}_{year}"
    elif analysis == "season":
        selected_season = str(timescale).upper()
        months = season_months_dict[selected_season]
        mask = xr.zeros_like(data["time"], dtype=bool)

        for month in months:
            season_year = year - 1 if month >= 10 else year
            mask |= (
                (data["time.month"] == month)
                & (data["time.year"] == season_year)
            )

        selected = data.sel(time=mask)
        period_label = f"{selected_season} {year}"
        filename_period = f"season_{selected_season}_{year}"
    else:
        raise ValueError(
            "Le type d'analyse doit être 'month' ou 'season'."
        )

    aggregated = getattr(selected, reducer)(dim="time")
    return aggregated, period_label, filename_period


def _render_aggregated_map(
    data,
    districts,
    lon,
    lat,
    title,
    colorbar_label,
    filename,
    cmap,
):
    """Render the common rainfall/temperature map layout."""

    bounds = districts.total_bounds
    data = data.rio.write_crs("EPSG:4326", inplace=True)
    districts = districts.to_crs("EPSG:4326")
    masked = mask_raster_with_shape(data, districts)

    fig, ax = plt.subplots(
        figsize=(9, 7),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    set_title(ax, title, fontsize=15, pad=15)
    ax.add_feature(cfeature.LAND, facecolor="white")
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    districts.boundary.plot(ax=ax, edgecolor="black", linewidth=1)

    pcm = ax.contourf(
        lon,
        lat,
        masked,
        cmap=cmap,
        extend="both",
        levels=15,
        transform=ccrs.PlateCarree(),
    )
    cbar = plt.colorbar(
        pcm,
        ax=ax,
        orientation="horizontal",
        fraction=0.046,
        pad=0.07,
    )
    cbar.set_label(colorbar_label, fontsize=12)

    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    set_map_grid(gl)
    ax.set_extent(
        [bounds[0], bounds[2], bounds[1], bounds[3]],
        crs=ccrs.PlateCarree(),
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

    return data


def _plot_aggregated_map(
    file,
    shapefile,
    analysis,
    timescale,
    year,
    language,
    *,
    variable_kind,
    reducer,
    title_key,
    label_key,
    filename_prefix,
    cmap,
):
    """Load, aggregate, and render a spatial climate field."""

    language = str(language).strip().lower()
    text = TRANSLATION[language]
    set_plot_defaults()

    ds = normalize_lat_lon(xr.open_dataset(file))
    data = ds[resolve_variable(ds, variable_kind)]
    districts = gpd.read_file(shapefile)
    aggregated, period_label, filename_period = _aggregate_period(
        data,
        analysis,
        timescale,
        year,
        reducer,
    )

    return _render_aggregated_map(
        aggregated,
        districts,
        ds["lon"],
        ds["lat"],
        f"{text[title_key]} - {period_label}",
        text[label_key],
        f"{filename_prefix}_{filename_period}.png",
        cmap,
    )


def plot_precip_map(
    file,
    shapefile,
    analysis="month",
    timescale=7,
    year=2026,
    language="fr",
):
    """
    Plot cumulative rainfall for one month or one season.

    Parameters
    ----------
    file : str
        NetCDF file path.

    shapefile : str
        Shapefile path.

    analysis : str
        Choose "month" or "season".

    timescale : int or str
        Month number for monthly analysis or season name for
        seasonal analysis.

    year : int
        Selected year.
    """

    return _plot_aggregated_map(
        file,
        shapefile,
        analysis,
        timescale,
        year,
        language,
        variable_kind="precip",
        reducer="sum",
        title_key="cumulative_title",
        label_key="cumulative_label",
        filename_prefix="Rainfall_sum",
        cmap=coulPREC,
    )


def plot_temp_map(
    file,
    shapefile,
    analysis="month",
    timescale=7,
    year=2026,
    language="fr",
):
    """
    Plot mean temperature for one month or one season.

    Parameters
    ----------
    file : str
        NetCDF file path.

    shapefile : str
        Shapefile path.

    analysis : str
        Choose "month" or "season".

    timescale : int or str
        Month number for monthly analysis or season name for
        seasonal analysis.

    year : int
        Selected year.
    """

    return _plot_aggregated_map(
        file,
        shapefile,
        analysis,
        timescale,
        year,
        language,
        variable_kind="temperature",
        reducer="mean",
        title_key="temperature_title",
        label_key="temperature_label",
        filename_prefix="Temperature_mean",
        cmap=cmap_temp,
    )


def compute_standardized_monthly_anomaly(
    rainfall,
    selected_year,
    selected_month,
    clim_years=(1991, 2020),
):
    """
    Compute standardized anomaly for a specific month.
    """


    rainfall_month = rainfall.sel(
        time=(
            rainfall["time.month"] == selected_month
        )
    )


    rainfall_clim = rainfall_month.sel(
        time=(
            (
                rainfall_month["time.year"]
                >= clim_years[0]
            )
            & (
                rainfall_month["time.year"]
                <= clim_years[1]
            )
        )
    )


    clim_mean = rainfall_clim.mean(
        dim="time"
    )

    clim_std = rainfall_clim.std(
        dim="time"
    )

    rainfall_target = rainfall.sel(
        time=(
            (
                rainfall["time.year"]
                == selected_year
            )
            & (
                rainfall["time.month"]
                == selected_month
            )
        )
    )

    if len(rainfall_target["time"]) == 0:
        raise ValueError(
            f"No data found for "
            f"{selected_month}/{selected_year}."
        )

    monthly_mean = rainfall_target.mean(
        dim="time"
    )

    anomaly_std = (
        monthly_mean - clim_mean
    ) / clim_std

    return anomaly_std


def _compute_standardized_seasonal_anomaly(
    rainfall,
    season_months,
    selected_year,
    clim_years,
    reducer,
):
    """Compute a standardized seasonal anomaly with the given reducer."""

    start_month = season_months[0]


    time_df = pd.DataFrame({
        "time": rainfall["time"].values,
        "month": rainfall["time.month"].values,
        "year": rainfall["time.year"].values,
    })


    time_df["season_year"] = time_df.apply(
        lambda row: (
            row["year"] - 1
            if row["month"] < start_month
            else row["year"]
        ),
        axis=1,
    )


    season_years = xr.DataArray(
        time_df["season_year"].values,
        coords=[rainfall["time"]],
        name="season_year",
    )


    rainfall_season = rainfall.where(
        rainfall["time.month"].isin(
            season_months
        ),
        drop=True,
    )

    season_years = season_years.where(
        rainfall["time.month"].isin(
            season_months
        ),
        drop=True,
    )


    clim_mask = (
        (season_years >= clim_years[0])
        & (season_years <= clim_years[1])
    )

    rainfall_clim = rainfall_season.where(
        clim_mask,
        drop=True,
    )

    season_years_clim = season_years.where(
        clim_mask,
        drop=True,
    )


    rainfall_clim.coords["season_year"] = (
        season_years_clim
    )


    seasonal_stack = getattr(
        rainfall_clim.groupby("season_year"),
        reducer,
    )(
        dim="time"
    )

    seasonal_clim_mean = seasonal_stack.mean(
        dim="season_year"
    )

    seasonal_clim_std = seasonal_stack.std(
        dim="season_year"
    )


    selected_time_df = time_df[
        (
            (time_df["month"] >= start_month)
            & (time_df["year"] == selected_year)
        )
        | (
            (time_df["month"] < start_month)
            & (time_df["year"] == selected_year + 1)
        )
    ]

    selected_times = selected_time_df[
        "time"
    ].values

    rainfall_selected = rainfall.sel(
        time=selected_times
    )

    seasonal_value = getattr(rainfall_selected, reducer)(
        dim="time"
    )

    anomaly_std = (
        seasonal_value - seasonal_clim_mean
    ) / seasonal_clim_std

    return anomaly_std


def compute_standardized_seasonal_anomaly(
    rainfall,
    season_months,
    selected_year,
    clim_years=(1991, 2020),
):
    """Compute standardized seasonal anomaly using rainfall sums."""

    return _compute_standardized_seasonal_anomaly(
        rainfall,
        season_months,
        selected_year,
        clim_years,
        reducer="sum",
    )


def compute_standardized_monthly_temperature_anomaly(
    temperature,
    selected_year,
    selected_month,
    clim_years=(1991, 2020),
):
    """
    Compute standardized temperature anomaly for a specific month.
    """


    return compute_standardized_monthly_anomaly(
        temperature,
        selected_year,
        selected_month,
        clim_years=clim_years,
    )


def compute_standardized_seasonal_temperature_anomaly(
    temperature,
    season_months,
    selected_year,
    clim_years=(1991, 2020),
):
    """
    Compute standardized seasonal anomaly using temperature means.
    """

    return _compute_standardized_seasonal_anomaly(
        temperature,
        season_months,
        selected_year,
        clim_years,
        reducer="mean",
    )


def _render_anomaly_map(
    anomaly,
    districts,
    lon,
    lat,
    title,
    colorbar_label,
    filename,
    cmap,
):
    """Render the common standardized-anomaly map layout."""

    bounds = districts.total_bounds
    anomaly = anomaly.rio.write_crs("EPSG:4326", inplace=True)
    districts = districts.to_crs("EPSG:4326")
    anomaly_masked = mask_raster_with_shape(anomaly, districts)

    fig, ax = plt.subplots(
        figsize=(9, 7),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    set_title(ax, title, fontsize=15, pad=15)
    districts.boundary.plot(ax=ax, edgecolor="black", linewidth=1)
    pcm = ax.contourf(
        lon,
        lat,
        anomaly_masked,
        cmap=cmap,
        levels=np.linspace(-3, 3, 13),
        extend="both",
        transform=ccrs.PlateCarree(),
    )
    ax.set_extent(
        [bounds[0], bounds[2], bounds[1], bounds[3]],
        crs=ccrs.PlateCarree(),
    )

    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    set_map_grid(gl)

    cbar = plt.colorbar(
        pcm,
        ax=ax,
        orientation="horizontal",
        fraction=0.05,
        pad=0.05,
    )
    cbar.set_label(colorbar_label)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

    return anomaly


def plot_precip_anomaly(
    file,
    shapefile,
    analysis="month",
    timescale=7,
    year=2026,
    clim_years=(1991, 2020),
    language="fr",
):
    """
    Plot the standardized rainfall anomaly for a month or season.
    """

    language = str(language).strip().lower()
    text = TRANSLATION[language]
    set_plot_defaults()


    ds = normalize_lat_lon(xr.open_dataset(file))

    rainfall = ds[resolve_variable(ds, "precip")]
    lat = ds["lat"]
    lon = ds["lon"]

    districts = gpd.read_file(
        shapefile
    )

    if analysis == "season":
        selected_season = str(
            timescale
        ).upper()

        selected_months = (
            season_months_dict[selected_season]
        )

        anomaly_std = (
            compute_standardized_seasonal_anomaly(
                rainfall,
                selected_months,
                year,
                clim_years=clim_years,
            )
        )

        title = (
            f"{text['anomaly_title']} - "
            f"{selected_season} {year}"
        )

        filename = (
            f"Rainfall_anomaly_season_"
            f"{selected_season}_{year}.png"
        )

    elif analysis == "month":
        selected_month = int(
            timescale
        )

        anomaly_std = (
            compute_standardized_monthly_anomaly(
                rainfall,
                year,
                selected_month,
                clim_years=clim_years,
            )
        )

        title = (
            f"{text['anomaly_title']} - "
            f"{selected_month} {year}"
        )

        filename = (
            f"Rainfall_Anomaly_season_"
            f"{selected_month}_{year}.png"
        )

    else:
        raise ValueError(
            "Le type d'analyse doit être 'month' ou 'season'."
        )


    return _render_anomaly_map(
        anomaly_std,
        districts,
        lon,
        lat,
        title,
        text["anomaly_label"],
        filename,
        cmap_rain,
    )


def plot_temp_anomaly(
    file,
    shapefile,
    analysis="month",
    timescale=7,
    year=2026,
    clim_years=(1991, 2020),
    language="fr",
):
    """
    Plot the standardized temperature anomaly for a month or season.
    """

    language = str(language).strip().lower()
    text = TRANSLATION[language]
    set_plot_defaults()


    ds = normalize_lat_lon(xr.open_dataset(file))

    temperature = ds[resolve_variable(ds, "temperature")]
    lat = ds["lat"]
    lon = ds["lon"]

    districts = gpd.read_file(
        shapefile
    )

    if analysis == "season":
        selected_season = str(
            timescale
        ).upper()

        selected_months = (
            season_months_dict[selected_season]
        )

        anomaly_std = (
            compute_standardized_seasonal_temperature_anomaly(
                temperature,
                selected_months,
                year,
                clim_years=clim_years,
            )
        )

        title = (
            f"{text['temperature_anomaly_title']} - "
            f"{selected_season} {year}"
        )

        filename = (
            f"Temperature_anomaly_season_"
            f"{selected_season}_{year}.png"
        )

    elif analysis == "month":
        selected_month = int(
            timescale
        )

        anomaly_std = (
            compute_standardized_monthly_temperature_anomaly(
                temperature,
                year,
                selected_month,
                clim_years=clim_years,
            )
        )

        title = (
            f"{text['temperature_anomaly_title']} - "
            f"{selected_month} {year}"
        )

        filename = (
            f"Temperature_anomaly_month_"
            f"{selected_month}_{year}.png"
        )

    else:
        raise ValueError(
            "Le type d'analyse doit être 'month' ou 'season'."
        )


    return _render_anomaly_map(
        anomaly_std,
        districts,
        lon,
        lat,
        title,
        text["temperature_anomaly_label"],
        filename,
        cmap_temp,
    )


if __name__ == "__main__":
    pass
