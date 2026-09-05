"""Large-scale climate parameter monitoring."""

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

from matplotlib.colors import TwoSlopeNorm

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


TRANSLATIONS = {
    "fr": {
        "colorbar": (
            "Anomalie de la Temperature de la Surface de la Mer "
            "(TSM en deg C)"
        ),
        "title": "TSM (contour) et anomalie de TSM (couleur)",
        "mslp_colorbar": "Anomalie de MSLP (hPa)",
        "mslp_title": "MSLP (contour) et anomalie de MSLP (couleur)",
        "saved": "Graphique cree et enregistre",
    },
    "en": {
        "colorbar": "Sea surface temperature anomaly (SST in deg C)",
        "title": "SST (contours) and SST anomaly (colors)",
        "mslp_colorbar": "MSLP anomaly (hPa)",
        "mslp_title": "MSLP (contours) and MSLP anomaly (colors)",
        "saved": "Plot created and saved",
    },
    "mg": {
        "colorbar": "Fironan'ny hafanan'ny ranomasina (deg C)",
        "title": "TSM (tsipika) sy fironan'ny TSM (loko)",
        "mslp_colorbar": "Fironan'ny MSLP (hPa)",
        "mslp_title": "MSLP (tsipika) sy fironan'ny MSLP (loko)",
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


def _select_time(data, time):
    if time is None:
        return data.isel(time=-1).squeeze()

    return data.sel(time=time).squeeze()


def plot_map_latest_sst(
    sst_file,
    anomaly_file,
    time=None,
    domain_box=None,
    variable="sst",
    language="fr",
    output_file=None,
    show=True,
):
    """
    Plot SST anomaly as filled contours and SST values as line contours.

    Parameters
    ----------
    sst_file : str
        NetCDF file containing absolute SST values.

    anomaly_file : str
        NetCDF file containing SST anomaly values.

    time : str, optional
        Time selector passed to xarray, for example ``"2026-07"``. If omitted,
        the latest time step is used.

    domain_box : list or tuple, optional
        Map extent as ``[lon_min, lon_max, lat_min, lat_max]``. Defaults to
        ``[20, 300, -40, 20]``.

    variable : str
        Variable name in both NetCDF files. Defaults to ``sst``.
    """

    language = _validate_language(language)
    text = TRANSLATIONS[language]
    set_plot_defaults()

    if domain_box is None:
        domain_box = [
            20,
            300,
            -40,
            20,
        ]

    if len(domain_box) != 4:
        raise ValueError(
            "domain_box must be [lon_min, lon_max, lat_min, lat_max]."
        )

    ds_sst = normalize_lat_lon(xr.open_dataset(sst_file))
    ds_anom = normalize_lat_lon(xr.open_dataset(anomaly_file))
    sst_variable = resolve_variable(ds_sst, "sst", variable)
    anom_variable = resolve_variable(ds_anom, "sst", variable)

    sst = _select_time(
        ds_sst[sst_variable],
        time,
    )
    anom = _select_time(
        ds_anom[anom_variable],
        time,
    )

    proj_map = ccrs.PlateCarree(
        central_longitude=180,
    )
    data_crs = ccrs.PlateCarree()

    fig = plt.figure(
        figsize=(11, 8.5),
    )
    ax = plt.axes(
        projection=proj_map,
    )

    ax.set_extent(
        domain_box,
        crs=data_crs,
    )

    ax.add_feature(
        cfeature.OCEAN,
        facecolor="#edf7ff",
    )
    ax.add_feature(
        cfeature.LAND,
        facecolor="#f4f4f4",
        edgecolor="none",
    )
    ax.coastlines(
        resolution="110m",
        color="#404040",
        linewidth=0.8,
    )
    ax.add_feature(
        cfeature.BORDERS,
        linestyle=":",
        color="#666666",
        linewidth=0.5,
    )

    clevs_anom = np.array([
        -2.0,
        -1.5,
        -1.0,
        -0.5,
        -0.2,
        0.2,
        0.5,
        1.0,
        1.5,
        2.0,
    ])
    cmap = plt.cm.RdBu_r
    norm = TwoSlopeNorm(
        vcenter=0.0,
        vmin=-2.0,
        vmax=2.0,
    )

    cf = ax.contourf(
        anom["lon"],
        anom["lat"],
        anom,
        levels=clevs_anom,
        cmap=cmap,
        norm=norm,
        extend="both",
        transform=data_crs,
    )

    clevs_sst = np.arange(
        10,
        36,
        2,
    )
    cs = ax.contour(
        sst["lon"],
        sst["lat"],
        sst,
        levels=clevs_sst,
        colors="#111111",
        linewidths=0.9,
        alpha=0.8,
        transform=data_crs,
    )

    ax.clabel(
        cs,
        fmt="%d",
        fontsize=8,
        inline=True,
        inline_spacing=4,
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
        cf,
        orientation="horizontal",
        pad=0.08,
        aspect=40,
        shrink=0.85,
        drawedges=False,
    )
    cbar.set_label(
        text["colorbar"],
        fontsize=11,
        labelpad=10,
    )
    cbar.ax.tick_params(
        labelsize=10,
    )
    cbar.set_ticks(
        clevs_anom,
    )

    if time is None:
        selected_time = sst["time"].values if "time" in sst.coords else "latest"
    else:
        selected_time = time

    set_title(
        ax,
        f"{text['title']} - {selected_time}",
        fontsize=12,
    )

    plt.tight_layout()

    if output_file is None:
        time_part = str(selected_time).replace(":", "").replace(" ", "_")
        output_file = f"SST_Anom_Value_{time_part}_{language}.png"

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

    return None


def plot_map_latest_mslp(
    mslp_file,
    anomaly_file,
    time=None,
    domain_box=None,
    variable="slp",
    language="fr",
    output_file=None,
    show=True,
):
    """
    Plot MSLP anomaly as filled contours and MSLP values as line contours.

    Parameters
    ----------
    mslp_file : str
        NetCDF file containing absolute mean sea level pressure values.

    anomaly_file : str
        NetCDF file containing MSLP anomaly values.

    time : str, optional
        Time selector passed to xarray, for example ``"2026-02"``. If omitted,
        the latest time step is used.

    domain_box : list or tuple, optional
        Map extent as ``[lon_min, lon_max, lat_min, lat_max]``. Defaults to
        ``[20, 90, -40, 10]``.

    variable : str
        Variable name in both NetCDF files. Defaults to ``slp``.
    """

    language = _validate_language(language)
    text = TRANSLATIONS[language]
    set_plot_defaults()

    if domain_box is None:
        domain_box = [
            20,
            90,
            -40,
            10,
        ]

    if len(domain_box) != 4:
        raise ValueError(
            "domain_box must be [lon_min, lon_max, lat_min, lat_max]."
        )

    ds_mslp = normalize_lat_lon(xr.open_dataset(mslp_file))
    ds_anom = normalize_lat_lon(xr.open_dataset(anomaly_file))
    mslp_variable = resolve_variable(ds_mslp, "mslp", variable)
    anom_variable = resolve_variable(ds_anom, "mslp", variable)

    mslp = _select_time(
        ds_mslp[mslp_variable],
        time,
    )
    anom = _select_time(
        ds_anom[anom_variable],
        time,
    )

    data_crs = ccrs.PlateCarree()

    fig = plt.figure(
        figsize=(10, 7),
    )
    ax = plt.axes(
        projection=data_crs,
    )

    ax.set_extent(
        domain_box,
        crs=data_crs,
    )

    ax.add_feature(
        cfeature.OCEAN,
        facecolor="#f3f7fb",
    )
    ax.add_feature(
        cfeature.LAND,
        facecolor="#eeeeee",
        edgecolor="none",
    )
    ax.coastlines(
        resolution="110m",
        color="#404040",
        linewidth=0.8,
    )
    ax.add_feature(
        cfeature.BORDERS,
        linestyle=":",
        color="#666666",
        linewidth=0.5,
    )

    clevs_anom = np.arange(
        -5,
        6,
        1,
    )
    cmap = plt.cm.PuOr_r
    norm = TwoSlopeNorm(
        vcenter=0.0,
        vmin=-5.0,
        vmax=5.0,
    )

    cf = ax.contourf(
        anom["lon"],
        anom["lat"],
        anom,
        levels=clevs_anom,
        cmap=cmap,
        norm=norm,
        extend="both",
        transform=data_crs,
    )

    clevs_mslp = np.arange(
        1000,
        1040,
        5,
    )
    cs = ax.contour(
        mslp["lon"],
        mslp["lat"],
        mslp,
        levels=clevs_mslp,
        colors="#202020",
        linewidths=1,
        transform=data_crs,
    )

    ax.clabel(
        cs,
        fmt="%d",
        fontsize=8,
        inline=True,
        inline_spacing=4,
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
        cf,
        orientation="horizontal",
        pad=0.05,
        aspect=50,
        shrink=0.8,
    )
    cbar.set_label(
        text["mslp_colorbar"],
        fontsize=11,
        labelpad=10,
    )
    cbar.ax.tick_params(
        labelsize=10,
    )
    cbar.set_ticks(
        clevs_anom,
    )

    if time is None:
        selected_time = (
            mslp["time"].values
            if "time" in mslp.coords
            else "latest"
        )
    else:
        selected_time = time

    set_title(
        ax,
        f"{text['mslp_title']} - {selected_time}",
        fontsize=12,
    )

    plt.tight_layout()

    if output_file is None:
        time_part = str(selected_time).replace(":", "").replace(" ", "_")
        output_file = f"MSLP_Anom_Value_{time_part}_{language}.png"

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

    return None
