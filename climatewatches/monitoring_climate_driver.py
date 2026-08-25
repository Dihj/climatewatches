import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from matplotlib.patches import Patch
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

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


TRANSLATIONS = {
    "fr": {
        "temp_axis": "Anomalie de temperature (deg C)",
        "soi_axis": "Indice d'oscillation australe (SOI)",
        "x_axis": "Annee",
        "title": "Variations des indices climatiques",
        "sst_map_colorbar": (
            "Anomalie de TSM (deg C) sur la base climatologique"
        ),
        "sst_map_title": (
            "TSM\nAnomalie (couleur) vs. TSM moyenne (contours)"
        ),
        "saved": "Graphique cree et enregistre",
    },
    "en": {
        "temp_axis": "Temperature anomaly (deg C)",
        "soi_axis": "Southern Oscillation Index (SOI)",
        "x_axis": "Year",
        "title": "Climate index variations",
        "sst_map_colorbar": (
            "SST anomaly (deg C) relative to climatology"
        ),
        "sst_map_title": (
            "SST\nAnomaly (colors) vs. mean SST (contours)"
        ),
        "saved": "Plot created and saved",
    },
    "mg": {
        "temp_axis": "Maripana oharina amin'ny mahazatra (deg C)",
        "soi_axis": "Southern Oscillation Index (SOI)",
        "x_axis": "Taona",
        "title": "Fironan'ny mpandrafitra ny toetrandro",
        "sst_map_colorbar": (
            "Tahan'ny hafanan'ny ranomasina (deg C) raha oharina amin'ny mahazatra"
        ),
        "sst_map_title": (
            "Hafanan'ny ranomasina\nFironana (loko) vs. salanisa (tsipika)"
        ),
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


def _enso_category(value):
    if np.isnan(value):
        return "Neutral", None

    if value >= 2.0:
        return "Very Strong El Nino", "#67000D"
    if value >= 1.5:
        return "Strong El Nino", "#CB181D"
    if value >= 1.0:
        return "Moderate El Nino", "#FB6A4A"
    if value >= 0.5:
        return "Weak El Nino", "#FCAE91"
    if value <= -2.0:
        return "Very Strong La Nina", "#08306B"
    if value <= -1.5:
        return "Strong La Nina", "#2171B5"
    if value <= -1.0:
        return "Moderate La Nina", "#6BAED6"
    if value <= -0.5:
        return "Weak La Nina", "#C6DBEF"

    return "Neutral", None


def _select_lat_lon(data, lat_bounds, lon_bounds):
    lat = data["lat"]
    lat_slice = (
        slice(lat_bounds[0], lat_bounds[1])
        if lat[0] > lat[-1]
        else slice(lat_bounds[1], lat_bounds[0])
    )

    return data.sel(
        lat=lat_slice,
        lon=slice(lon_bounds[0], lon_bounds[1]),
    )


def _spatial_mean(data, lat_bounds, lon_bounds):
    return _select_lat_lon(
        data,
        lat_bounds,
        lon_bounds,
    ).mean(
        dim=("lat", "lon")
    )


def _monthly_anomaly(data, variable, clim_start, clim_end):
    climatology = data.sel(
        time=slice(
            clim_start,
            clim_end,
        )
    ).groupby(
        "time.month"
    ).mean(
        dim="time"
    )

    anomaly = data.groupby("time.month") - climatology

    return anomaly[variable]


def _three_month_mean(data, name):
    return data.rolling(
        time=3,
        center=True,
    ).mean().dropna(
        dim="time"
    ).rename(
        name
    )


def _read_soi(soi_file, data_start, data_end):
    df_soi = pd.read_csv(
        soi_file,
        header=None,
        names=[
            "date_raw",
            "soi_raw",
        ],
    )
    df_soi["time"] = pd.to_datetime(
        df_soi["date_raw"].astype(str),
        format="%Y%m",
    )
    df_soi = df_soi.set_index(
        "time"
    ).sort_index()

    soi = xr.DataArray(
        df_soi["soi_raw"].values,
        dims=[
            "time",
        ],
        coords={
            "time": df_soi.index,
        },
    ).rename(
        "SOI"
    )

    soi = soi.sel(
        time=slice(
            data_start,
            data_end,
        )
    )

    return _three_month_mean(
        soi,
        "SOI",
    )


def _enso_legend_patches():
    return [
        Patch(facecolor="#67000D", alpha=0.30, label="Very Strong El Nino (>= 2.0 deg C)"),
        Patch(facecolor="#CB181D", alpha=0.30, label="Strong El Nino (1.5 to 1.9 deg C)"),
        Patch(facecolor="#FB6A4A", alpha=0.30, label="Moderate El Nino (1.0 to 1.4 deg C)"),
        Patch(facecolor="#FCAE91", alpha=0.30, label="Weak El Nino (0.5 to 0.9 deg C)"),
        Patch(facecolor="#C6DBEF", alpha=0.30, label="Weak La Nina (-0.5 to -0.9 deg C)"),
        Patch(facecolor="#6BAED6", alpha=0.30, label="Moderate La Nina (-1.0 to -1.4 deg C)"),
        Patch(facecolor="#2171B5", alpha=0.30, label="Strong La Nina (-1.5 to -1.9 deg C)"),
        Patch(facecolor="#08306B", alpha=0.30, label="Very Strong La Nina (<= -2.0 deg C)"),
    ]


def _shade_enso_categories(ax, index_data):
    times = pd.to_datetime(
        index_data["time"].values
    )
    values = index_data.values

    current_category = None
    current_color = None
    start_time = None

    for index, value in enumerate(values):
        category, color = _enso_category(value)

        if category == "Neutral":
            if current_category is not None:
                ax.axvspan(
                    start_time,
                    times[index],
                    facecolor=current_color,
                    alpha=0.30,
                    edgecolor="none",
                    zorder=0,
                )
                current_category = None
                current_color = None
                start_time = None
            continue

        if current_category is None:
            current_category = category
            current_color = color
            start_time = times[index]
        elif category != current_category:
            ax.axvspan(
                start_time,
                times[index],
                facecolor=current_color,
                alpha=0.30,
                edgecolor="none",
                zorder=0,
            )
            current_category = category
            current_color = color
            start_time = times[index]

    if current_category is not None:
        ax.axvspan(
            start_time,
            times[-1] + pd.offsets.MonthEnd(1),
            facecolor=current_color,
            alpha=0.30,
            edgecolor="none",
            zorder=0,
        )


def plot_climate_indices(
    sst_file,
    soi_file=None,
    include_roni=True,
    include_oni=True,
    include_iod=True,
    include_siod=True,
    include_soi=True,
    variable="sst",
    data_start="1950-01-01",
    data_end="2026-07-01",
    clim_start="1991-01-01",
    clim_end="2020-12-01",
    plot_start="1981-01-01",
    plot_end="2026-12-31",
    language="fr",
    include_enso_shading=True,
    shading_index="RONI",
    output_file=None,
    show=True,
):
    """
    Plot selected climate drivers from monthly SST and optional SOI data.

    Drivers can be enabled or disabled independently with ``include_roni``,
    ``include_oni``, ``include_iod``, ``include_siod``, and ``include_soi``.
    """

    language = _validate_language(language)
    text = TRANSLATIONS[language]
    apply_scientific_plot_style()

    if include_soi and soi_file is None:
        raise ValueError("soi_file is required when include_soi=True.")

    ds = normalize_lat_lon(xr.open_dataset(sst_file))

    if "time_bnds" in ds.variables:
        ds = ds.drop_vars("time_bnds")

    variable = resolve_variable(ds, "sst", variable)

    ds = ds.sel(
        time=slice(
            data_start,
            data_end,
        )
    )

    dat = ds[[variable]]
    dat = dat.assign_coords(
        lon=dat["lon"] % 360
    ).sortby(
        "lon"
    )

    indices = {}

    needs_nino = include_roni or include_oni

    if needs_nino:
        tropav = _spatial_mean(
            dat,
            (20, -20),
            (0, 360),
        ).rename(
            {
                variable: "tropav",
            }
        )
        nino34 = _spatial_mean(
            dat,
            (5, -5),
            (190, 240),
        ).rename(
            {
                variable: "nino34",
            }
        )
        nino34_anom = _monthly_anomaly(
            nino34,
            "nino34",
            clim_start,
            clim_end,
        )

        if include_oni:
            indices["ONI"] = _three_month_mean(
                nino34_anom,
                "ONI",
            )

        if include_roni:
            trop_anom = _monthly_anomaly(
                tropav,
                "tropav",
                clim_start,
                clim_end,
            )
            diff_index = nino34_anom - trop_anom
            nino_std = nino34_anom.groupby(
                "time.month"
            ).std(
                "time"
            )
            diff_std = diff_index.groupby(
                "time.month"
            ).std(
                "time"
            )
            scaling_factor = nino_std / diff_std
            scaled_relative_nino34 = diff_index.groupby(
                "time.month"
            ) * scaling_factor
            indices["RONI"] = _three_month_mean(
                scaled_relative_nino34,
                "RONI",
            )

    if include_iod:
        wtio = _spatial_mean(
            dat,
            (10, -10),
            (50, 70),
        )
        setio = _spatial_mean(
            dat,
            (0, -10),
            (90, 110),
        )
        wtio_anom = _monthly_anomaly(
            wtio,
            variable,
            clim_start,
            clim_end,
        )
        setio_anom = _monthly_anomaly(
            setio,
            variable,
            clim_start,
            clim_end,
        )
        indices["IOD"] = _three_month_mean(
            wtio_anom - setio_anom,
            "IOD",
        )

    if include_siod:
        wsio = _spatial_mean(
            dat,
            (-27, -37),
            (55, 75),
        )
        esio = _spatial_mean(
            dat,
            (-18, -28),
            (95, 115),
        )
        wsio_anom = _monthly_anomaly(
            wsio,
            variable,
            clim_start,
            clim_end,
        )
        esio_anom = _monthly_anomaly(
            esio,
            variable,
            clim_start,
            clim_end,
        )
        indices["SIOD"] = _three_month_mean(
            wsio_anom - esio_anom,
            "SIOD",
        )

    if include_soi:
        indices["SOI"] = _read_soi(
            soi_file,
            data_start,
            data_end,
        )

    if not indices:
        raise ValueError("At least one climate driver must be selected.")

    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.linewidth"] = 1.2

    fig, ax1 = plt.subplots(
        figsize=(14, 6),
        dpi=300,
    )
    ax2 = ax1.twinx() if "SOI" in indices else None

    ax1.set_ylim(
        -2.5,
        3.0,
    )

    if ax2 is not None:
        ax2.set_ylim(
            -30,
            30,
        )

    shading_key = str(shading_index).strip().upper()

    if include_enso_shading and shading_key in indices:
        _shade_enso_categories(
            ax1,
            indices[shading_key],
        )

    ax1.axhline(
        0.5,
        color="#CC6666",
        linestyle=":",
        linewidth=1,
        zorder=1,
    )
    ax1.axhline(
        -0.5,
        color="#6699CC",
        linestyle=":",
        linewidth=1,
        zorder=1,
    )
    ax1.axhline(
        0.0,
        color="#555555",
        linestyle="-",
        linewidth=0.8,
        zorder=1,
    )

    ax1.set_xlim(
        pd.Timestamp(plot_start),
        pd.Timestamp(plot_end),
    )
    ax1.xaxis.set_major_locator(
        mdates.YearLocator(3)
    )
    ax1.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y")
    )
    ax1.xaxis.set_minor_locator(
        mdates.YearLocator(1)
    )
    ax1.tick_params(
        axis="x",
        which="major",
        length=5,
    )
    ax1.tick_params(
        axis="x",
        which="minor",
        length=2,
    )

    line_styles = {
        "RONI": {
            "color": "#004488",
            "linewidth": 2.0,
            "linestyle": "-",
            "label": "RONI",
            "zorder": 4,
        },
        "ONI": {
            "color": "#BB5566",
            "linewidth": 2.0,
            "linestyle": "--",
            "label": "ONI",
            "zorder": 4,
        },
        "IOD": {
            "color": "#117733",
            "linewidth": 1.4,
            "linestyle": "-.",
            "label": "IOD (DMI)",
            "zorder": 3,
        },
        "SIOD": {
            "color": "#AA4499",
            "linewidth": 1.4,
            "linestyle": ":",
            "label": "SIOD Index",
            "zorder": 3,
        },
    }

    line_handles = []

    for key in [
        "RONI",
        "ONI",
        "IOD",
        "SIOD",
    ]:
        if key not in indices:
            continue

        style = line_styles[key]
        line, = ax1.plot(
            indices[key]["time"],
            indices[key].values,
            **style,
        )
        line_handles.append(line)

    if "SOI" in indices:
        line, = ax2.plot(
            indices["SOI"]["time"],
            indices["SOI"].values,
            linewidth=1.5,
            color="#E69F00",
            linestyle="-",
            label="SOI",
            zorder=3,
        )
        line_handles.append(line)

    ax1.grid(
        True,
        which="both",
        linestyle="--",
        linewidth=0.5,
        color="gray",
        alpha=0.3,
        zorder=2,
    )
    ax1.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        labelsize=10,
    )

    if ax2 is not None:
        ax2.tick_params(
            axis="y",
            which="both",
            direction="in",
            labelsize=10,
        )
        ax2.set_ylabel(
            text["soi_axis"],
            fontsize=12,
            labelpad=12,
            rotation=270,
        )

    ax1.set_ylabel(
        text["temp_axis"],
        fontsize=12,
        labelpad=8,
    )
    ax1.set_xlabel(
        text["x_axis"],
        fontsize=12,
        labelpad=8,
    )

    selected_names = ", ".join(indices.keys())
    set_scientific_title(
        ax1,
        f"{text['title']}: {selected_names}",
        fontsize=13,
        pad=14,
    )

    handles = line_handles

    if include_enso_shading and shading_key in indices:
        handles = handles + _enso_legend_patches()

    ax1.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            -0.22,
        ),
        ncol=4,
        fontsize=8,
        frameon=True,
        edgecolor="black",
        framealpha=1,
        columnspacing=1.0,
        handlelength=2.2,
    )

    plt.tight_layout()

    if output_file is None:
        output_file = "ocean_indices_teleconnection_plot.png"

    plt.savefig(
        output_file,
        bbox_inches="tight",
        dpi=300,
    )

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"{text['saved']}: {output_file}")

    return {
        "figure": fig,
        "axis": ax1,
        "soi_axis": ax2,
        "indices": indices,
        "output_file": output_file,
    }


def plot_sst_map_variability(
    sst_file,
    target_start="2025-10-01",
    target_end="2026-04-01",
    clim_start="1991-01-01",
    clim_end="2020-12-01",
    months_of_interest=None,
    variable="sst",
    domain_extent=None,
    central_longitude=140,
    boxes=None,
    selected_boxes=None,
    language="fr",
    output_file=None,
    show=True,
):
    """
    Plot mean SST contours and SST anomaly colors for a target period.

    Climate index boxes can be controlled with ``selected_boxes``. Use
    ``selected_boxes=None`` to draw all default boxes, or pass a list such as
    ``["Nino 3.4", "WTIO (IOD)", "SETIO (IOD)"]``.
    """

    language = _validate_language(language)
    text = TRANSLATIONS[language]
    apply_scientific_plot_style()

    if months_of_interest is None:
        start_ts = pd.Timestamp(target_start)
        end_ts = pd.Timestamp(target_end)
        months_of_interest = sorted(
            pd.date_range(
                start_ts,
                end_ts,
                freq="MS",
            ).month.unique(),
            key=lambda month: (
                month < start_ts.month,
                month,
            ),
        )

    if domain_extent is None:
        domain_extent = [
            0,
            120,
            -60,
            40,
        ]

    if boxes is None:
        boxes = {
            "Nino 3.4": {
                "coords": (
                    190,
                    -5,
                    50,
                    10,
                ),
                "color": "#000000",
                "style": "-",
            },
            "WTIO (IOD)": {
                "coords": (
                    50,
                    -10,
                    20,
                    20,
                ),
                "color": "#117733",
                "style": "--",
            },
            "SETIO (IOD)": {
                "coords": (
                    90,
                    -10,
                    20,
                    10,
                ),
                "color": "#117733",
                "style": "--",
            },
            "WSIO (SIOD)": {
                "coords": (
                    55,
                    -37,
                    20,
                    10,
                ),
                "color": "#AA4499",
                "style": "-.",
            },
            "ESIO (SIOD)": {
                "coords": (
                    95,
                    -28,
                    20,
                    10,
                ),
                "color": "#AA4499",
                "style": "-.",
            },
        }

    if selected_boxes is None:
        selected_boxes = list(boxes.keys())
    elif isinstance(selected_boxes, str):
        selected_boxes = [selected_boxes]

    unknown_boxes = set(selected_boxes).difference(boxes)

    if unknown_boxes:
        raise ValueError(
            "Unknown climate box(es): "
            f"{', '.join(sorted(unknown_boxes))}. "
            f"Choose from: {', '.join(sorted(boxes))}."
        )

    ds = normalize_lat_lon(xr.open_dataset(sst_file))

    if "time_bnds" in ds.variables:
        ds = ds.drop_vars("time_bnds")

    variable = resolve_variable(ds, "sst", variable)

    ds = ds.assign_coords(
        lon=ds["lon"] % 360
    ).sortby(
        "lon"
    )

    target_ds = ds.sel(
        time=slice(
            target_start,
            target_end,
        )
    )

    if len(target_ds["time"]) == 0:
        raise ValueError("No SST data found for the selected target period.")

    mean_sst = target_ds[variable].mean(
        dim="time"
    )

    clim_base = ds.sel(
        time=slice(
            clim_start,
            clim_end,
        )
    )
    clim_matched_months = clim_base.sel(
        time=clim_base["time.month"].isin(
            months_of_interest
        )
    )

    if len(clim_matched_months["time"]) == 0:
        raise ValueError(
            "No climatology data found for selected months and period."
        )

    clim_mean = clim_matched_months[variable].mean(
        dim="time"
    )
    anomaly_sst = mean_sst - clim_mean

    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.linewidth"] = 1.0

    fig = plt.figure(
        figsize=(14, 7),
        dpi=300,
    )
    projection = ccrs.PlateCarree(
        central_longitude=central_longitude,
    )
    ax = plt.axes(
        projection=projection,
    )

    ax.set_extent(
        domain_extent,
        crs=ccrs.PlateCarree(),
    )
    ax.add_feature(
        cfeature.LAND,
        facecolor="#eaeaea",
        zorder=2,
    )
    ax.add_feature(
        cfeature.COASTLINE,
        linewidth=0.6,
        edgecolor="#333333",
        zorder=2,
    )

    lon_coords = anomaly_sst["lon"]
    lat_coords = anomaly_sst["lat"]

    v_limit = 2.0
    clevs_anom = np.linspace(
        -v_limit,
        v_limit,
        21,
    )
    mesh = ax.contourf(
        lon_coords,
        lat_coords,
        anomaly_sst,
        levels=clevs_anom,
        cmap="RdYlBu_r",
        extend="both",
        transform=ccrs.PlateCarree(),
        zorder=1,
    )

    clevs_sst = np.arange(
        10,
        32,
        2,
    )
    contours = ax.contour(
        lon_coords,
        lat_coords,
        mean_sst,
        levels=clevs_sst,
        colors="#222222",
        linewidths=0.8,
        transform=ccrs.PlateCarree(),
        zorder=1,
    )
    ax.clabel(
        contours,
        inline=True,
        fmt="%1.0f",
        fontsize=8,
        colors="#222222",
    )

    for name in selected_boxes:
        info = boxes[name]
        lon_min, lat_min, width, height = info["coords"]

        rect = patches.Rectangle(
            (
                lon_min,
                lat_min,
            ),
            width,
            height,
            linewidth=1.5,
            edgecolor=info["color"],
            facecolor="none",
            linestyle=info["style"],
            transform=ccrs.PlateCarree(),
            zorder=3,
        )
        ax.add_patch(rect)
        ax.text(
            lon_min + 1,
            lat_min + height + 1,
            name,
            color=info["color"],
            fontsize=8,
            weight="bold",
            transform=ccrs.PlateCarree(),
            zorder=4,
            bbox={
                "facecolor": "white",
                "alpha": 0.7,
                "edgecolor": "none",
                "pad": 1,
            },
        )

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle=":",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    style_cartopy_gridlines(gl)

    cbar = plt.colorbar(
        mesh,
        orientation="horizontal",
        pad=0.07,
        shrink=0.7,
        aspect=40,
    )
    cbar.set_label(
        f"{text['sst_map_colorbar']} {clim_start[:4]}-{clim_end[:4]}",
        fontsize=11,
    )
    cbar.ax.tick_params(
        labelsize=9,
    )

    set_scientific_title(
        ax,
        (
            f"{target_start[:7]} - {target_end[:7]}\n"
            f"{text['sst_map_title']}"
        ),
        fontsize=13,
        pad=15,
    )

    plt.tight_layout()

    if output_file is None:
        output_file = (
            "sst_anomaly_contours_with_boxes_"
            f"{target_start[:7]}_{target_end[:7]}_{language}.png"
        )

    plt.savefig(
        output_file,
        bbox_inches="tight",
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
        "mean_sst": mean_sst,
        "anomaly_sst": anomaly_sst,
        "selected_boxes": selected_boxes,
        "output_file": output_file,
    }
