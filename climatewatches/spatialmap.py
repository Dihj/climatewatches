import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import rioxarray

from matplotlib.colors import LinearSegmentedColormap
from rasterio.features import geometry_mask
from shapely.geometry import mapping

try:
    from .preparation_data import (
        apply_scientific_plot_style,
        normalize_lat_lon,
        resolve_variable,
        set_scientific_title,
        style_cartopy_gridlines,
    )
except ImportError:
    from preparation_data import (
        apply_scientific_plot_style,
        normalize_lat_lon,
        resolve_variable,
        set_scientific_title,
        style_cartopy_gridlines,
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
        "anomaly_title": "Tahan'ny fivilian'ny rotsakorana",
        "anomaly_label": "Tahan'ny fivilian'ny rotsakorana (σ)",
        "temperature_title": "Maripana ankapobeny",
        "temperature_label": "Maripana ankapobeny",
        "temperature_anomaly_title": "Tahan'ny fivilian'ny maripana",
        "temperature_anomaly_label": "Tahan'ny fivilian'ny maripana (σ)",
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

    language = str(language).strip().lower()
    text = TRANSLATION[language]
    apply_scientific_plot_style()

    ds = normalize_lat_lon(xr.open_dataset(file))

    rainfall = ds[resolve_variable(ds, "precip")]
    lat = ds["lat"]
    lon = ds["lon"]

    districts = gpd.read_file(shapefile)

    bounds = districts.total_bounds


    if analysis == "month":
        selected_month = int(timescale)
        selected_year = year

        rainfall_selected = rainfall.sel(
            time=(
                (rainfall["time.month"] == selected_month)
                & (rainfall["time.year"] == selected_year)
            )
        )

        rainfall_sum = rainfall_selected.sum(
            dim="time"
        )

        title = (
            f"{text['cumulative_title']} - "
            f"{selected_month:02d}/{selected_year}"
        )

        filename = (
            f"Rainfall_sum_month_"
            f"{selected_month:02d}_{selected_year}.png"
        )

    elif analysis == "season":
        selected_season = str(timescale).upper()
        selected_year = year

        months = season_months_dict[selected_season]

        times = []

        for month in months:
            season_year = (
                selected_year - 1
                if month >= 10
                else selected_year
            )

            times.append({
                "month": month,
                "year": season_year,
            })

        mask = xr.zeros_like(
            rainfall["time"],
            dtype=bool,
        )

        for selected_time in times:
            mask |= (
                (
                    rainfall["time.month"]
                    == selected_time["month"]
                )
                & (
                    rainfall["time.year"]
                    == selected_time["year"]
                )
            )

        rainfall_selected = rainfall.sel(
            time=mask
        )

        rainfall_sum = rainfall_selected.sum(
            dim="time"
        )

        title = (
            f"{text['cumulative_title']} - "
            f"{selected_season} {selected_year}"
        )

        filename = (
            f"Rainfall_sum_season_"
            f"{selected_season}_{selected_year}.png"
        )

    else:
        raise ValueError(
            "Le type d'analyse doit être 'month' ou 'season'."
        )


    rainfall_sum = rainfall_sum.rio.write_crs(
        "EPSG:4326",
        inplace=True,
    )

    districts = districts.to_crs(
        "EPSG:4326"
    )

    rainfall_masked = mask_raster_with_shape(
        rainfall_sum,
        districts,
    )


    fig, ax = plt.subplots(
        figsize=(9, 7),
        subplot_kw={
            "projection": ccrs.PlateCarree(),
        },
    )

    set_scientific_title(
        ax,
        title,
        fontsize=15,
        pad=15,
    )

    ax.add_feature(
        cfeature.LAND,
        facecolor="white",
    )

    ax.add_feature(
        cfeature.BORDERS,
        linestyle=":",
    )

    districts.boundary.plot(
        ax=ax,
        edgecolor="black",
        linewidth=1,
    )

    pcm = ax.contourf(
        lon,
        lat,
        rainfall_masked,
        cmap=coulPREC,
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

    cbar.set_label(
        text["cumulative_label"],
        fontsize=12,
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
    style_cartopy_gridlines(gl)

    ax.set_extent(
        [
            bounds[0],
            bounds[2],
            bounds[1],
            bounds[3],
        ],
        crs=ccrs.PlateCarree(),
    )

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    return rainfall_sum


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

    language = str(language).strip().lower()
    text = TRANSLATION[language]
    apply_scientific_plot_style()


    ds = normalize_lat_lon(xr.open_dataset(file))

    temperature = ds[resolve_variable(ds, "temperature")]
    lat = ds["lat"]
    lon = ds["lon"]

    districts = gpd.read_file(shapefile)

    bounds = districts.total_bounds


    if analysis == "month":
        selected_month = int(timescale)
        selected_year = year

        temperature_selected = temperature.sel(
            time=(
                (temperature["time.month"] == selected_month)
                & (temperature["time.year"] == selected_year)
            )
        )

        temperature_mean = temperature_selected.mean(
            dim="time"
        )

        title = (
            f"{text['temperature_title']} - "
            f"{selected_month:02d}/{selected_year}"
        )

        filename = (
            f"Temperature_mean_month_"
            f"{selected_month:02d}_{selected_year}.png"
        )

    elif analysis == "season":
        selected_season = str(timescale).upper()
        selected_year = year

        months = season_months_dict[selected_season]

        times = []

        for month in months:
            season_year = (
                selected_year - 1
                if month >= 10
                else selected_year
            )

            times.append({
                "month": month,
                "year": season_year,
            })

        mask = xr.zeros_like(
            temperature["time"],
            dtype=bool,
        )

        for selected_time in times:
            mask |= (
                (
                    temperature["time.month"]
                    == selected_time["month"]
                )
                & (
                    temperature["time.year"]
                    == selected_time["year"]
                )
            )

        temperature_selected = temperature.sel(
            time=mask
        )

        temperature_mean = temperature_selected.mean(
            dim="time"
        )

        title = (
            f"{text['temperature_title']} - "
            f"{selected_season} {selected_year}"
        )

        filename = (
            f"Temperature_mean_season_"
            f"{selected_season}_{selected_year}.png"
        )

    else:
        raise ValueError(
            "Le type d'analyse doit être 'month' ou 'season'."
        )


    temperature_mean = temperature_mean.rio.write_crs(
        "EPSG:4326",
        inplace=True,
    )

    districts = districts.to_crs(
        "EPSG:4326"
    )

    temperature_masked = mask_raster_with_shape(
        temperature_mean,
        districts,
    )


    fig, ax = plt.subplots(
        figsize=(9, 7),
        subplot_kw={
            "projection": ccrs.PlateCarree(),
        },
    )

    set_scientific_title(
        ax,
        title,
        fontsize=15,
        pad=15,
    )

    ax.add_feature(
        cfeature.LAND,
        facecolor="white",
    )

    ax.add_feature(
        cfeature.BORDERS,
        linestyle=":",
    )

    districts.boundary.plot(
        ax=ax,
        edgecolor="black",
        linewidth=1,
    )

    pcm = ax.contourf(
        lon,
        lat,
        temperature_masked,
        cmap=cmap_temp,
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

    cbar.set_label(
        text["temperature_label"],
        fontsize=12,
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
    style_cartopy_gridlines(gl)

    ax.set_extent(
        [
            bounds[0],
            bounds[2],
            bounds[1],
            bounds[3],
        ],
        crs=ccrs.PlateCarree(),
    )

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    return temperature_mean


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


def compute_standardized_seasonal_anomaly(
    rainfall,
    season_months,
    selected_year,
    clim_years=(1991, 2020),
):
    """
    Compute standardized seasonal anomaly using rainfall sums.
    """

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


    seasonal_stack = rainfall_clim.groupby(
        "season_year"
    ).sum(
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

    seasonal_sum = rainfall_selected.sum(
        dim="time"
    )

    anomaly_std = (
        seasonal_sum - seasonal_clim_mean
    ) / seasonal_clim_std

    return anomaly_std

def compute_standardized_monthly_temperature_anomaly(
    temperature,
    selected_year,
    selected_month,
    clim_years=(1991, 2020),
):
    """
    Compute standardized temperature anomaly for a specific month.
    """


    temperature_month = temperature.sel(
        time=(
            temperature["time.month"] == selected_month
        )
    )


    temperature_clim = temperature_month.sel(
        time=(
            (
                temperature_month["time.year"]
                >= clim_years[0]
            )
            & (
                temperature_month["time.year"]
                <= clim_years[1]
            )
        )
    )


    clim_mean = temperature_clim.mean(
        dim="time"
    )

    clim_std = temperature_clim.std(
        dim="time"
    )


    temperature_target = temperature.sel(
        time=(
            (
                temperature["time.year"]
                == selected_year
            )
            & (
                temperature["time.month"]
                == selected_month
            )
        )
    )

    if len(temperature_target["time"]) == 0:
        raise ValueError(
            f"No data found for "
            f"{selected_month}/{selected_year}."
        )

    monthly_mean = temperature_target.mean(
        dim="time"
    )


    anomaly_std = (
        monthly_mean - clim_mean
    ) / clim_std

    return anomaly_std

def compute_standardized_seasonal_temperature_anomaly(
    temperature,
    season_months,
    selected_year,
    clim_years=(1991, 2020),
):
    """
    Compute standardized seasonal anomaly using temperature means.
    """

    start_month = season_months[0]


    time_df = pd.DataFrame({
        "time": temperature["time"].values,
        "month": temperature["time.month"].values,
        "year": temperature["time.year"].values,
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
        coords=[temperature["time"]],
        name="season_year",
    )


    temperature_season = temperature.where(
        temperature["time.month"].isin(
            season_months
        ),
        drop=True,
    )

    season_years = season_years.where(
        temperature["time.month"].isin(
            season_months
        ),
        drop=True,
    )


    clim_mask = (
        (season_years >= clim_years[0])
        & (season_years <= clim_years[1])
    )

    temperature_clim = temperature_season.where(
        clim_mask,
        drop=True,
    )

    season_years_clim = season_years.where(
        clim_mask,
        drop=True,
    )


    temperature_clim.coords["season_year"] = (
        season_years_clim
    )


    seasonal_stack = temperature_clim.groupby(
        "season_year"
    ).mean(
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

    temperature_selected = temperature.sel(
        time=selected_times
    )

    seasonal_mean = temperature_selected.mean(
        dim="time"
    )


    anomaly_std = (
        seasonal_mean - seasonal_clim_mean
    ) / seasonal_clim_std

    return anomaly_std


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
    apply_scientific_plot_style()


    ds = normalize_lat_lon(xr.open_dataset(file))

    rainfall = ds[resolve_variable(ds, "precip")]
    lat = ds["lat"]
    lon = ds["lon"]

    districts = gpd.read_file(
        shapefile
    )

    bounds = districts.total_bounds


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


    anomaly_std = anomaly_std.rio.write_crs(
        "EPSG:4326",
        inplace=True,
    )

    districts = districts.to_crs(
        "EPSG:4326"
    )

    anomaly_masked = mask_raster_with_shape(
        anomaly_std,
        districts,
    )


    fig, ax = plt.subplots(
        figsize=(9, 7),
        subplot_kw={
            "projection": ccrs.PlateCarree(),
        },
    )

    set_scientific_title(
        ax,
        title,
        fontsize=15,
        pad=15,
    )

    districts.boundary.plot(
        ax=ax,
        edgecolor="black",
        linewidth=1,
    )

    pcm = ax.contourf(
        lon,
        lat,
        anomaly_masked,
        cmap=cmap_rain,
        levels=np.linspace(
            -3,
            3,
            13,
        ),
        extend="both",
        transform=ccrs.PlateCarree(),
    )

    ax.set_extent(
        [
            bounds[0],
            bounds[2],
            bounds[1],
            bounds[3],
        ],
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
    style_cartopy_gridlines(gl)

    cbar = plt.colorbar(
        pcm,
        ax=ax,
        orientation="horizontal",
        fraction=0.05,
        pad=0.05,
    )

    cbar.set_label(
        text["anomaly_label"]
    )

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    return anomaly_std


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
    apply_scientific_plot_style()


    ds = normalize_lat_lon(xr.open_dataset(file))

    temperature = ds[resolve_variable(ds, "temperature")]
    lat = ds["lat"]
    lon = ds["lon"]

    districts = gpd.read_file(
        shapefile
    )

    bounds = districts.total_bounds


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


    anomaly_std = anomaly_std.rio.write_crs(
        "EPSG:4326",
        inplace=True,
    )

    districts = districts.to_crs(
        "EPSG:4326"
    )

    anomaly_masked = mask_raster_with_shape(
        anomaly_std,
        districts,
    )


    fig, ax = plt.subplots(
        figsize=(9, 7),
        subplot_kw={
            "projection": ccrs.PlateCarree(),
        },
    )

    set_scientific_title(
        ax,
        title,
        fontsize=15,
        pad=15,
    )

    districts.boundary.plot(
        ax=ax,
        edgecolor="black",
        linewidth=1,
    )

    pcm = ax.contourf(
        lon,
        lat,
        anomaly_masked,
        cmap=cmap_temp,
        levels=np.linspace(
            -3,
            3,
            13,
        ),
        extend="both",
        transform=ccrs.PlateCarree(),
    )

    ax.set_extent(
        [
            bounds[0],
            bounds[2],
            bounds[1],
            bounds[3],
        ],
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
    style_cartopy_gridlines(gl)

    cbar = plt.colorbar(
        pcm,
        ax=ax,
        orientation="horizontal",
        fraction=0.05,
        pad=0.05,
    )

    cbar.set_label(
        text["temperature_anomaly_label"]
    )

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    return anomaly_std


if __name__ == "__main__":
    pass
