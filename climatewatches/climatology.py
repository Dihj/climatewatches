import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import rioxarray  # noqa: F401 - registers the xarray ``.rio`` accessor

from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import mapping

try:
    from .preparation_data import (
        set_plot_defaults,
        is_precip_variable,
        normalize_lat_lon,
        resolve_variable,
        set_grid,
        set_map_grid,
        set_title,
    )
except ImportError:
    from preparation_data import (
        set_plot_defaults,
        is_precip_variable,
        normalize_lat_lon,
        resolve_variable,
        set_grid,
        set_map_grid,
        set_title,
    )


season_months = {
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
    "OND": [10, 11, 12],
    "NDJ": [11, 12, 1],
}

'''
rainfall_colors = [
    "#f7fbff",
    "#deebf7",
    "#c6dbef",
    "#9ecae1",
    "#6baed6",
    "#4292c6",
    "#2171b5",
    "#084594",
]

cmap_clim = LinearSegmentedColormap.from_list(
    "clim_cmap",
    rainfall_colors,
    N=256,
)
'''

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

cmap_clim = LinearSegmentedColormap.from_list(
    "clim_cmap",
    coulPREC_colors,
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


TRANSLATIONS = {
    "fr": {
        "precip_month_title": "Climatologie des precipitations - Mois",
        "precip_season_title": "Climatologie des precipitations",
        "precip_label": "Precipitations saisonnieres moyennes (mm)",
        "temp_month_title": "Temperature moyenne climatologique - Mois",
        "temp_season_title": "Temperature moyenne climatologique - Saison",
        "temp_label": "Temperature moyenne (deg C)",
        "spatial_month_title": "Precipitations - Mois",
        "spatial_season_title": "Precipitations",
        "spatial_xlabel": "Annees",
        "spatial_ylabel": "Precipitations totales (mm)",
        "below_normal": "Inferieur a la normale",
        "normal": "Normal",
        "above_normal": "Superieur a la normale",
        "percentile_33": "33e percentile",
        "percentile_66": "66e percentile",
        "ombro_title": "Diagramme Ombrothermique",
        "ombro_temp_label": "Temperature (deg C)",
        "ombro_rain_label": "Precipitations (mm)",
        "ombro_rain_line": "Precipitations (mm)",
        "ombro_temp_line": "Temperature moyenne (deg C)",
        "ombro_wet_area": "Pluie > Temperature",
        "ombro_dry_area": "Temperature > Pluie",
        "composite_title": "Anomalie composite",
        "composite_label_precip": "Anomalie de precipitations (mm)",
        "composite_label_temp": "Anomalie de temperature (deg C)",
        "csv_saved": "CSV enregistre",
        "saved": "Graphique cree et enregistre",
    },
    "en": {
        "precip_month_title": "Rainfall climatology - Month",
        "precip_season_title": "Rainfall climatology",
        "precip_label": "Mean seasonal rainfall (mm)",
        "temp_month_title": "Mean temperature climatology - Month",
        "temp_season_title": "Mean temperature climatology - Season",
        "temp_label": "Mean temperature (deg C)",
        "spatial_month_title": "Rainfall - Month",
        "spatial_season_title": "Rainfall",
        "spatial_xlabel": "Years",
        "spatial_ylabel": "Total rainfall (mm)",
        "below_normal": "Below normal",
        "normal": "Normal",
        "above_normal": "Above normal",
        "percentile_33": "33rd percentile",
        "percentile_66": "66th percentile",
        "ombro_title": "Ombrothermic Diagram",
        "ombro_temp_label": "Temperature (deg C)",
        "ombro_rain_label": "Rainfall (mm)",
        "ombro_rain_line": "Rainfall (mm)",
        "ombro_temp_line": "Mean temperature (deg C)",
        "ombro_wet_area": "Rain > Temperature",
        "ombro_dry_area": "Temperature > Rain",
        "composite_title": "Composite anomaly",
        "composite_label_precip": "Rainfall anomaly (mm)",
        "composite_label_temp": "Temperature anomaly (deg C)",
        "csv_saved": "CSV saved",
        "saved": "Plot created and saved",
    },
    "mg": {
        "precip_month_title": "Toetran'ny rotsakorana - Volana",
        "precip_season_title": "Toetran'ny rotsakorana",
        "precip_label": "Salanisan'ny rotsakorana ara-potoana (mm)",
        "temp_month_title": "Toetran'ny maripana - Volana",
        "temp_season_title": "Toetran'ny maripana - Tonontaona",
        "temp_label": "Salanisan'ny maripana (deg C)",
        "spatial_month_title": "Rotsakorana - Volana",
        "spatial_season_title": "Rotsakorana",
        "spatial_xlabel": "Taona",
        "spatial_ylabel": "Fitambaran'ny Rotsakorana (mm)",
        "below_normal": "Latsaky ny mahazatra",
        "normal": "Ara-dalàna",
        "above_normal": "Mihoatra ny mahazatra",
        "percentile_33": "Percentile 33",
        "percentile_66": "Percentile 66",
        "ombro_title": "Diagramme Ombrothermique",
        "ombro_temp_label": "Maripana (deg C)",
        "ombro_rain_label": "Rotsakorana (mm)",
        "ombro_rain_line": "Rotsakorana (mm)",
        "ombro_temp_line": "Salanisan'ny maripana (deg C)",
        "ombro_wet_area": "Rotsakorana > Maripana",
        "ombro_dry_area": "Maripana > Rotsakorana",
        "composite_title": "Oharina amin'ny mahazatra",
        "composite_label_precip": "Rotsakorana oharina amin'ny mahazatra (mm)",
        "composite_label_temp": "Maripana oharina amin'ny mahazatra (deg C)",
        "csv_saved": "CSV voatahiry",
        "saved": "Vita sy voatahiry ny kisarisary",
    },
}


def _validate_language(language):
    language = str(language).strip().lower()

    if language not in TRANSLATIONS:
        raise ValueError(
            f"Unsupported language {language!r}. Choose 'fr', 'en', 'mg'."
        )

    return language


def _select_cross_year_season(data, months, year):
    start_month = months[0]
    selected = []

    for month in months:
        selected_year = year + 1 if month < start_month else year
        month_data = data.sel(
            time=(
                (data["time.month"] == month)
                & (data["time.year"] == selected_year)
            )
        )
        selected.append(month_data)

    return xr.concat(selected, dim="time")


def _compute_monthly_climatology(
    data,
    selected_month,
    start_year,
    end_year,
    reducer,
):
    climatology_period = data.sel(
        time=(
            (data["time.year"] >= start_year)
            & (data["time.year"] <= end_year)
        )
    )

    monthly_data = climatology_period.sel(
        time=climatology_period["time.month"] == selected_month
    )

    if len(monthly_data["time"]) == 0:
        raise ValueError(
            f"No data found for month {selected_month:02d} "
            f"during {start_year}-{end_year}."
        )

    return getattr(monthly_data, reducer)(dim="time")


def _compute_seasonal_climatology(
    data,
    selected_season,
    start_year,
    end_year,
    reducer,
):
    selected_season = str(selected_season).strip().upper()

    if selected_season not in season_months:
        raise ValueError(
            f"Unsupported season {selected_season!r}. "
            f"Choose one of: {', '.join(sorted(season_months))}."
        )

    months = season_months[selected_season]
    start_month = months[0]
    crosses_year = any(month < start_month for month in months)
    last_year = end_year if not crosses_year else end_year - 1
    seasonal_values = []

    for year in range(start_year, last_year + 1):
        season_data = _select_cross_year_season(
            data,
            months,
            year,
        )

        if len(season_data["time"]) == len(months):
            seasonal_value = getattr(season_data, reducer)(
                dim="time"
            )
            seasonal_values.append(
                seasonal_value.expand_dims(time=[year])
            )

    if not seasonal_values:
        raise ValueError(
            f"No complete {selected_season} seasons found for "
            f"{start_year}-{end_year}."
        )

    return xr.concat(
        seasonal_values,
        dim="time",
    ).mean(
        dim="time"
    )


def _mask_raster_with_shape(data, gdf):
    data = normalize_lat_lon(data)
    gdf = gdf.to_crs("EPSG:4326")
    data = data.rio.set_spatial_dims(
        x_dim="lon",
        y_dim="lat",
    ).rio.write_crs(
        "EPSG:4326",
        inplace=False,
    )

    return data.rio.clip(
        [
            mapping(geom)
            for geom in gdf.geometry
        ],
        gdf.crs,
        drop=False,
    )


def _compute_seasonal_spatial_rainfall(
    rainfall,
    months,
    region_gdf,
):
    time = pd.to_datetime(rainfall["time"].values)
    years = time.year
    time_months = time.month
    start_month = months[0]
    season_year = np.where(
        time_months < start_month,
        years - 1,
        years,
    )

    mask = np.isin(
        time_months,
        months,
    )
    rainfall_filtered = rainfall.sel(
        time=mask,
    )
    rainfall_filtered = rainfall_filtered.assign_coords(
        season_year=(
            "time",
            season_year[mask],
        )
    )

    rainfall_masked = _mask_raster_with_shape(
        rainfall_filtered,
        region_gdf,
    )

    mean_over_region = rainfall_masked.mean(
        dim=["lat", "lon"],
        skipna=True,
    )

    return mean_over_region.groupby(
        "season_year"
    ).sum(
        dim="time"
    )


def _compute_monthly_spatial_rainfall(
    rainfall,
    selected_month,
    region_gdf,
):
    rainfall_month = rainfall.where(
        rainfall["time.month"] == selected_month,
        drop=True,
    )
    rainfall_masked = _mask_raster_with_shape(
        rainfall_month,
        region_gdf,
    )
    mean_over_region = rainfall_masked.mean(
        dim=["lat", "lon"],
        skipna=True,
    )

    return mean_over_region.groupby(
        "time.year"
    ).sum(
        dim="time"
    )


def _plot_climatology_map(
    data,
    districts,
    lon,
    lat,
    title,
    colorbar_label,
    cmap,
    filename,
    levels=None,
    show=True,
):
    set_plot_defaults()

    fig, ax = plt.subplots(
        figsize=(8, 6),
        subplot_kw={
            "projection": ccrs.PlateCarree(),
        },
    )

    set_title(
        ax,
        title,
        fontsize=14,
    )

    ax.add_feature(
        cfeature.LAND,
        edgecolor="black",
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

    contour_kwargs = {
        "cmap": cmap,
        "extend": "both",
        "transform": ccrs.PlateCarree(),
    }

    if levels is not None:
        contour_kwargs["levels"] = levels

    pcm = ax.contourf(
        lon,
        lat,
        data,
        **contour_kwargs,
    )

    bounds = districts.total_bounds
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
    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def plot_precip_climatology_map(
    file,
    shapefile,
    analysis="season",
    timescale="ASO",
    start_year=1991,
    end_year=2020,
    language="fr",
    output_file=None,
    show=True,
):
    """
    Plot monthly or seasonal rainfall climatology from a NetCDF file.

    Rainfall uses variable ``rfe``. Monthly climatology is a mean rainfall
    field. Seasonal climatology is the mean seasonal rainfall sum.
    """

    language = _validate_language(language)
    text = TRANSLATIONS[language]
    set_plot_defaults()

    ds = normalize_lat_lon(xr.open_dataset(file))
    rainfall = ds[resolve_variable(ds, "precip")]
    lat = ds["lat"]
    lon = ds["lon"]
    districts = gpd.read_file(shapefile)

    analysis = str(analysis).strip().lower()

    if analysis == "month":
        selected_month = int(timescale)
        clim_data = _compute_monthly_climatology(
            rainfall,
            selected_month,
            start_year,
            end_year,
            "mean",
        )
        title = (
            f"{text['precip_month_title']} {selected_month:02d} "
            f"({start_year}-{end_year})"
        )
        filename_part = f"{selected_month:02d}"

    elif analysis == "season":
        selected_season = str(timescale).strip().upper()
        clim_data = _compute_seasonal_climatology(
            rainfall,
            selected_season,
            start_year,
            end_year,
            "sum",
        )
        title = (
            f"{text['precip_season_title']} - {selected_season} "
            f"({start_year}-{end_year})"
        )
        filename_part = selected_season

    else:
        raise ValueError("analysis must be 'month' or 'season'.")

    if output_file is None:
        output_file = (
            f"Rainfall_Climatology_{analysis.upper()}_"
            f"{filename_part}_{language}.png"
        )

    clim_data_masked = _mask_raster_with_shape(
        clim_data,
        districts,
    )

    fig, ax = _plot_climatology_map(
        clim_data_masked,
        districts,
        lon,
        lat,
        title,
        text["precip_label"],
        cmap_clim,
        output_file,
        levels=15,
        show=show,
    )

    return {
        "figure": fig,
        "axis": ax,
        "data": clim_data_masked,
        "unmasked_data": clim_data,
        "output_file": output_file,
    }


def plot_temperature_climatology_map(
    file,
    shapefile,
    analysis="season",
    timescale="ASO",
    start_year=1991,
    end_year=2020,
    language="fr",
    output_file=None,
    show=True,
):
    """
    Plot monthly or seasonal mean temperature climatology from a NetCDF file.

    Temperature uses variable ``tmean``. Both monthly and seasonal
    climatologies are means.
    """

    language = _validate_language(language)
    text = TRANSLATIONS[language]
    set_plot_defaults()

    ds = normalize_lat_lon(xr.open_dataset(file))
    temperature = ds[resolve_variable(ds, "temperature")]
    lat = ds["lat"]
    lon = ds["lon"]
    districts = gpd.read_file(shapefile)

    analysis = str(analysis).strip().lower()

    if analysis == "month":
        selected_month = int(timescale)
        clim_data = _compute_monthly_climatology(
            temperature,
            selected_month,
            start_year,
            end_year,
            "mean",
        )
        title = (
            f"{text['temp_month_title']} {selected_month:02d} "
            f"({start_year}-{end_year})"
        )
        filename_part = f"{selected_month:02d}"

    elif analysis == "season":
        selected_season = str(timescale).strip().upper()
        clim_data = _compute_seasonal_climatology(
            temperature,
            selected_season,
            start_year,
            end_year,
            "mean",
        )
        title = (
            f"{text['temp_season_title']} {selected_season} "
            f"({start_year}-{end_year})"
        )
        filename_part = selected_season

    else:
        raise ValueError("analysis must be 'month' or 'season'.")

    if output_file is None:
        output_file = (
            f"Temperature_Climatology_{analysis.upper()}_"
            f"{filename_part}_{language}.png"
        )

    clim_data_masked = _mask_raster_with_shape(
        clim_data,
        districts,
    )

    fig, ax = _plot_climatology_map(
        clim_data_masked,
        districts,
        lon,
        lat,
        title,
        text["temp_label"],
        cmap_temp,
        output_file,
        show=show,
    )

    return {
        "figure": fig,
        "axis": ax,
        "data": clim_data_masked,
        "unmasked_data": clim_data,
        "output_file": output_file,
    }


def plot_climatology_spatial_mean(
    file,
    shapefile,
    analysis="season",
    timescale="JJA",
    selected_year=2025,
    language="fr",
    output_file=None,
    show=True,
):
    """
    Plot spatially averaged rainfall totals with climatology percentile bands.

    Rainfall uses variable ``rfe``. For seasonal analysis, ``timescale`` must
    be a season name. For monthly analysis, ``timescale`` must be a month
    number.
    """

    language = _validate_language(language)
    text = TRANSLATIONS[language]
    set_plot_defaults()

    ds = normalize_lat_lon(xr.open_dataset(file))
    rainfall = ds[resolve_variable(ds, "precip")]
    districts = gpd.read_file(shapefile).to_crs(
        "EPSG:4326"
    )

    analysis = str(analysis).strip().lower()
    selected_year = int(selected_year)

    if analysis == "season":
        selected_season = str(timescale).strip().upper()

        if selected_season not in season_months:
            raise ValueError(
                f"Unsupported season {selected_season!r}. "
                f"Choose one of: {', '.join(sorted(season_months))}."
            )

        rainfall_series = _compute_seasonal_spatial_rainfall(
            rainfall,
            season_months[selected_season],
            districts,
        )
        year_coord = "season_year"
        title = (
            f"{text['spatial_season_title']} ({selected_season}) - "
            f"{int(rainfall_series[year_coord].min())}-"
            f"{int(rainfall_series[year_coord].max())}"
        )
        filename_part = selected_season

    elif analysis == "month":
        selected_month = int(timescale)
        rainfall_series = _compute_monthly_spatial_rainfall(
            rainfall,
            selected_month,
            districts,
        )
        year_coord = "year"
        title = (
            f"{text['spatial_month_title']} {selected_month:02d} - "
            f"{int(rainfall_series[year_coord].min())}-"
            f"{int(rainfall_series[year_coord].max())}"
        )
        filename_part = f"{selected_month:02d}"

    else:
        raise ValueError("analysis must be 'month' or 'season'.")

    rain_values = rainfall_series.values

    if np.isnan(rain_values).all():
        raise ValueError("Rainfall series contains only NaN values.")

    below_norm = np.nanpercentile(
        rain_values,
        33,
    )
    above_norm = np.nanpercentile(
        rain_values,
        66,
    )
    max_value = np.nanmax(rain_values)
    upper_limit = max_value * 1.05

    if upper_limit <= above_norm:
        upper_limit = above_norm * 1.05

        if upper_limit <= 0:
            upper_limit = 1

    years = rainfall_series[year_coord].values

    fig, ax = plt.subplots(
        figsize=(10, 6),
    )

    ax.axhspan(
        0,
        below_norm,
        color="lightcoral",
        alpha=0.3,
        label=text["below_normal"],
    )
    ax.axhspan(
        below_norm,
        above_norm,
        color="palegreen",
        alpha=0.3,
        label=text["normal"],
    )
    ax.axhspan(
        above_norm,
        upper_limit,
        color="skyblue",
        alpha=0.3,
        label=text["above_normal"],
    )

    first_year = years[0]
    ax.text(
        first_year - 0.5,
        below_norm,
        text["percentile_33"],
        color="red",
        fontsize=10,
        va="bottom",
        ha="left",
    )
    ax.text(
        first_year - 0.5,
        above_norm,
        text["percentile_66"],
        color="blue",
        fontsize=10,
        va="bottom",
        ha="left",
    )

    ax.bar(
        years,
        rain_values,
        color="steelblue",
        edgecolor="black",
    )

    if selected_year in years:
        selected_value = rainfall_series.sel(
            {
                year_coord: selected_year,
            }
        ).values
        ax.bar(
            selected_year,
            selected_value,
            color="navy",
            edgecolor="black",
            label=f"{selected_year}",
        )

    set_title(
        ax,
        title,
        fontsize=14,
        pad=12,
    )
    ax.set_xlabel(
        text["spatial_xlabel"],
        fontsize=12,
    )
    ax.set_ylabel(
        text["spatial_ylabel"],
        fontsize=12,
    )
    ax.legend(
        loc="upper left",
        frameon=True,
    )
    set_grid(ax)

    plt.tight_layout()

    if output_file is None:
        output_file = (
            f"Rainfall_bar_{analysis}_{filename_part}_{language}.png"
        )

    plt.savefig(
        output_file,
        dpi=300,
    )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "figure": fig,
        "axis": ax,
        "data": rainfall_series,
        "below_normal": below_norm,
        "above_normal": above_norm,
        "output_file": output_file,
    }


def plot_diagram_ombro_one_point(
    rain_file,
    temp_file,
    latitude,
    longitude,
    rain_var=None,
    temp_var=None,
    clim_years=(1991, 2020),
    language="fr",
    output_file=None,
    csv_file=None,
    show=True,
):
    """
    Plot a Walter-Lieth ombrothermic diagram for one grid point.

    Rainfall and temperature are extracted from the nearest NetCDF grid point.
    Rainfall above 100 mm is compressed by a factor of 5 on the rainfall axis.
    """

    language = _validate_language(language)
    text = TRANSLATIONS[language]
    set_plot_defaults()

    def compress_rain(value):
        if value <= 100:
            return value

        return 100 + (value - 100) / 5.0

    rain_ds = normalize_lat_lon(xr.open_dataset(rain_file))
    temp_ds = normalize_lat_lon(xr.open_dataset(temp_file))
    rain_var = resolve_variable(rain_ds, "precip", rain_var)
    temp_var = resolve_variable(temp_ds, "temperature", temp_var)

    rain_point = rain_ds[rain_var].sel(
        lat=latitude,
        lon=longitude,
        method="nearest",
    )
    temp_point = temp_ds[temp_var].sel(
        lat=latitude,
        lon=longitude,
        method="nearest",
    )

    start_date = f"{clim_years[0]}-01-01"
    end_date = f"{clim_years[1]}-12-31"

    rain_point = rain_point.sel(
        time=slice(
            start_date,
            end_date,
        )
    )
    temp_point = temp_point.sel(
        time=slice(
            start_date,
            end_date,
        )
    )
    rain_point, temp_point = xr.align(
        rain_point,
        temp_point,
        join="inner",
    )

    if len(rain_point["time"]) == 0:
        raise ValueError(
            f"No overlapping rain and temperature data found for "
            f"{clim_years[0]}-{clim_years[1]}."
        )

    df = pd.DataFrame({
        "date": pd.to_datetime(
            rain_point["time"].values
        ),
        "rain": rain_point.values,
        "temp": temp_point.values,
    })
    df["month"] = df["date"].dt.month

    clim = df.groupby(
        "month"
    ).mean(
        numeric_only=True,
    ).reset_index().sort_values(
        "month"
    )

    if len(clim) < 12:
        raise ValueError(
            "Monthly climatology is incomplete; fewer than 12 months found."
        )

    rainfall = clim["rain"].values
    temperature = clim["temp"].values
    rainfall_compressed = np.array([
        compress_rain(value)
        for value in rainfall
    ])

    x = np.arange(12) + 0.5
    months = [
        "Jan",
        "Fev",
        "Mar",
        "Avr",
        "Mai",
        "Juin",
        "Juil",
        "Aout",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    max_rain_compressed = compress_rain(600.0)
    temp_scale = max_rain_compressed / 100.0
    temperature_on_rain_axis = temperature * temp_scale

    rain_ticks_real = [
        0,
        20,
        40,
        60,
        80,
        100,
        200,
        300,
        400,
        500,
        600,
    ]
    rain_ticks_compressed = [
        compress_rain(value)
        for value in rain_ticks_real
    ]

    monthly_summary = pd.DataFrame({
        "month": months,
        "temp_mean_C": temperature,
        "rain_mean_mm": rainfall,
        "rain_compressed_mm": rainfall_compressed,
        "temp_on_rain_axis": temperature_on_rain_axis,
        "drought_T_gt_P": temperature > rainfall,
        "rain_above_100mm": rainfall > 100,
    })

    if csv_file is None:
        csv_file = (
            "ombrothermic_summary_"
            f"lat{latitude:.2f}_lon{longitude:.2f}.csv"
        )

    monthly_summary.to_csv(
        csv_file,
        index=False,
        encoding="utf-8",
    )
    print(f"{text['csv_saved']}: {csv_file}")

    fig, ax = plt.subplots(
        figsize=(11, 6),
    )

    ax.plot(
        x,
        rainfall_compressed,
        color="tab:blue",
        lw=2.5,
        marker="o",
        label=text["ombro_rain_line"],
    )
    ax.plot(
        x,
        temperature_on_rain_axis,
        color="tab:red",
        lw=2.5,
        marker="o",
        label=text["ombro_temp_line"],
    )

    ax.fill_between(
        x,
        temperature_on_rain_axis,
        rainfall_compressed,
        where=rainfall_compressed >= temperature_on_rain_axis,
        interpolate=True,
        color="cornflowerblue",
        alpha=0.45,
        label=text["ombro_wet_area"],
    )
    ax.fill_between(
        x,
        temperature_on_rain_axis,
        rainfall_compressed,
        where=rainfall_compressed < temperature_on_rain_axis,
        interpolate=True,
        color="#ffcccc",
        alpha=0.6,
        label=text["ombro_dry_area"],
    )

    big_rain_mask = rainfall > 100.0

    if big_rain_mask.any():
        ax.axhline(
            compress_rain(100.0),
            color="mediumpurple",
            linestyle="--",
            linewidth=0.8,
            alpha=0.8,
        )

    ax.set_xticks(
        x
    )
    ax.set_xticklabels(
        months,
        fontsize=11,
    )
    ax.set_xlim(
        0.0,
        12.0,
    )
    ax.set_ylim(
        0,
        max_rain_compressed,
    )
    ax.set_ylabel(
        text["ombro_temp_label"],
        color="tab:red",
        fontsize=12,
    )

    temp_tick_values = np.arange(
        0,
        101,
        10,
    )
    temp_tick_compressed = temp_tick_values * temp_scale
    ax.set_yticks(
        temp_tick_compressed
    )
    ax.set_yticklabels(
        [
            str(value)
            for value in temp_tick_values
        ],
        color="tab:red",
    )
    ax.tick_params(
        axis="y",
        labelcolor="tab:red",
    )

    ax_rain = ax.twinx()
    ax_rain.set_ylim(
        0,
        max_rain_compressed,
    )
    ax_rain.set_yticks(
        rain_ticks_compressed
    )
    ax_rain.set_yticklabels(
        [
            str(int(value))
            for value in rain_ticks_real
        ],
        color="tab:blue",
    )
    ax_rain.set_ylabel(
        text["ombro_rain_label"],
        color="tab:blue",
        fontsize=12,
    )
    ax_rain.tick_params(
        axis="y",
        labelcolor="tab:blue",
    )

    set_title(
        ax,
        (
            f"{text['ombro_title']}\n"
            f"Lat={latitude}, Lon={longitude} "
            f"({clim_years[0]}-{clim_years[1]})"
        ),
        fontsize=12,
    )

    avg_temp = temperature.mean()
    total_rain = rainfall.sum()

    ax.text(
        0.90,
        0.95,
        f"Ø {avg_temp:.1f} deg C",
        transform=ax.transAxes,
        color="red",
        fontsize=11,
        ha="left",
        va="top",
    )
    ax.text(
        0.90,
        0.88,
        f"Σ {total_rain:.0f} mm",
        transform=ax.transAxes,
        color="blue",
        fontsize=11,
        ha="left",
        va="top",
    )

    set_grid(ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="upper left",
        fontsize=9,
    )
    plt.tight_layout()

    if output_file is None:
        output_file = (
            "Diagram_ombro_"
            f"lat{latitude:.2f}_lon{longitude:.2f}_{language}.png"
        )

    plt.savefig(
        output_file,
        dpi=300,
    )

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"{text['saved']}: {output_file}")

    return {
        "figure": fig,
        "axis": ax,
        "rain_axis": ax_rain,
        "summary": monthly_summary,
        "output_file": output_file,
        "csv_file": csv_file,
    }


def plot_composite_map(
    file,
    shapefile,
    selected_years,
    analysis="season",
    timescale="NDJ",
    variable=None,
    reducer=None,
    clim_years=(1991, 2020),
    language="fr",
    output_file=None,
    show=True,
):
    """
    Plot individual event anomalies and their composite mean anomaly.

    The default behavior follows the climatology map logic in this module:
    rainfall-like variables are summed over the month/season, while
    temperature-like variables are averaged. Anomalies are computed against
    the selected climatology period, 1991-2020 by default.
    """

    language = _validate_language(language)
    text = TRANSLATIONS[language]
    set_plot_defaults()

    ds = normalize_lat_lon(xr.open_dataset(file))
    districts = gpd.read_file(shapefile).to_crs(
        "EPSG:4326"
    )

    if variable is None:
        try:
            variable = resolve_variable(ds, "precip")
        except ValueError:
            variable = resolve_variable(ds, "temperature")
    else:
        try:
            variable = resolve_variable(ds, "precip", variable)
        except ValueError:
            variable = resolve_variable(ds, "temperature", variable)

    data = ds[variable]
    lat = ds["lat"]
    lon = ds["lon"]

    if reducer is None:
        reducer = "sum" if is_precip_variable(variable) else "mean"

    if reducer not in ["sum", "mean"]:
        raise ValueError("reducer must be 'sum' or 'mean'.")

    if isinstance(selected_years, (int, np.integer)):
        selected_years = [
            int(selected_years),
        ]

    composite_seasons = {
        **season_months,
        "ONDJF": [10, 11, 12, 1, 2],
        "NDJFM": [11, 12, 1, 2, 3],
        "ONDJFMA": [10, 11, 12, 1, 2, 3, 4],
    }

    analysis = str(analysis).strip().lower()

    if analysis == "season":
        selected_season = str(timescale).strip().upper()

        if selected_season not in composite_seasons:
            raise ValueError(
                f"Unsupported season {selected_season!r}. "
                f"Choose one of: {', '.join(sorted(composite_seasons))}."
            )

        months = composite_seasons[selected_season]
        label_part = selected_season

    elif analysis == "month":
        selected_month = int(timescale)
        months = [selected_month]
        label_part = f"{selected_month:02d}"

    else:
        raise ValueError("analysis must be 'month' or 'season'.")

    def compute_value_for_year(year):
        start_month = months[0]
        selected_months = []

        for month in months:
            target_year = year + 1 if month < start_month else year
            month_data = data.sel(
                time=(
                    (data["time.month"] == month)
                    & (data["time.year"] == target_year)
                )
            )

            if len(month_data["time"]) == 0:
                return None

            selected_months.append(month_data)

        period_data = xr.concat(
            selected_months,
            dim="time",
        )

        return getattr(period_data, reducer)(
            dim="time",
            skipna=True,
        )

    def build_stack(years):
        values = []
        valid_years = []

        for year in years:
            period_value = compute_value_for_year(int(year))

            if period_value is None or not period_value.notnull().any():
                continue

            values.append(
                period_value.expand_dims(
                    year=[
                        int(year),
                    ]
                )
            )
            valid_years.append(
                int(year)
            )

        if not values:
            return None, []

        return xr.concat(
            values,
            dim="year",
        ), valid_years

    clim_stack, clim_valid_years = build_stack(
        range(
            int(clim_years[0]),
            int(clim_years[1]) + 1,
        )
    )

    if clim_stack is None:
        raise ValueError(
            f"No complete climatology periods found for "
            f"{clim_years[0]}-{clim_years[1]}."
        )

    selected_stack, selected_valid_years = build_stack(
        selected_years
    )

    if selected_stack is None:
        raise ValueError(
            "No selected years contain complete data for this month/season."
        )

    climatology = clim_stack.mean(
        dim="year",
        skipna=True,
    )
    anomalies_selected = selected_stack - climatology
    composite_mean = anomalies_selected.mean(
        dim="year",
        skipna=True,
    )

    all_anomaly_values = np.concatenate([
        np.ravel(anomalies_selected.values),
        np.ravel(composite_mean.values),
    ])
    max_abs = np.nanmax(
        np.abs(all_anomaly_values)
    )

    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1

    levels = np.linspace(
        -max_abs,
        max_abs,
        17,
    )

    bounds = districts.total_bounds
    extent = [
        bounds[0],
        bounds[2],
        bounds[1],
        bounds[3],
    ]

    ncols = len(selected_valid_years)
    fig = plt.figure(
        figsize=(
            max(8, 3 * ncols),
            10,
        )
    )
    grid = gridspec.GridSpec(
        nrows=2,
        ncols=ncols,
        height_ratios=[
            1,
            3,
        ],
        wspace=0.35,
        hspace=0.3,
    )

    for index, year in enumerate(selected_valid_years):
        ax_small = plt.subplot(
            grid[0, index],
            projection=ccrs.PlateCarree(),
        )
        districts.boundary.plot(
            ax=ax_small,
            edgecolor="black",
            linewidth=1,
        )
        ax_small.set_extent(
            extent,
            crs=ccrs.PlateCarree(),
        )
        anomaly_year = anomalies_selected.sel(
            year=year
        )
        ax_small.contourf(
            lon,
            lat,
            anomaly_year,
            cmap="RdBu",
            levels=levels,
            extend="both",
            transform=ccrs.PlateCarree(),
        )
        gl_small = ax_small.gridlines(
            draw_labels=False,
            linewidth=0.4,
            color="gray",
            alpha=0.6,
            linestyle="--",
        )
        set_map_grid(gl_small)
        set_title(
            ax_small,
            f"{year}",
            fontsize=9,
        )

    ax_main = plt.subplot(
        grid[1, :],
        projection=ccrs.PlateCarree(),
    )
    districts.boundary.plot(
        ax=ax_main,
        edgecolor="black",
        linewidth=1,
    )
    ax_main.set_extent(
        extent,
        crs=ccrs.PlateCarree(),
    )
    ax_main.add_feature(
        cfeature.LAND,
        facecolor="white",
    )
    ax_main.add_feature(
        cfeature.BORDERS,
        linestyle=":",
    )

    pcm_main = ax_main.contourf(
        lon,
        lat,
        composite_mean,
        cmap="RdBu",
        levels=levels,
        extend="both",
        transform=ccrs.PlateCarree(),
    )
    gl = ax_main.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.7,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    set_map_grid(gl)

    set_title(
        ax_main,
        (
            f"{text['composite_title']} ({label_part}) - "
            f"{selected_valid_years}"
        ),
        fontsize=14,
    )

    cbar = fig.colorbar(
        pcm_main,
        ax=ax_main,
        orientation="horizontal",
        pad=0.05,
        aspect=50,
    )
    colorbar_label = (
        text["composite_label_precip"]
        if reducer == "sum"
        else text["composite_label_temp"]
    )
    cbar.set_label(
        colorbar_label,
    )

    if output_file is None:
        output_file = (
            f"composite_map_{variable}_{label_part}_"
            f"{'_'.join(str(year) for year in selected_valid_years)}_"
            f"{language}.png"
        )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"{text['saved']}: {output_file}")

    return {
        "figure": fig,
        "axis": ax_main,
        "climatology": climatology,
        "anomalies": anomalies_selected,
        "composite": composite_mean,
        "climatology_years": clim_valid_years,
        "selected_years": selected_valid_years,
        "output_file": output_file,
    }
