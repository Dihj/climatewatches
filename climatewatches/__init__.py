"""Climate Watches plotting and data preparation tools."""

__version__ = "0.1.0"

_EXPORTS = {
    "plot_precip_interannual_variability": (
        "intervariability",
        "plot_precip_interannual_variability",
    ),
    "plot_temp_interannual_variability": (
        "intervariability",
        "plot_temp_interannual_variability",
    ),
    "read_stn_data": ("intervariability", "read_stn_data"),
    "assign_season_year": ("intervariability", "assign_season_year"),
    "plot_precip_map": ("spatialmap", "plot_precip_map"),
    "plot_precip_anomaly": ("spatialmap", "plot_precip_anomaly"),
    "plot_temp_map": ("spatialmap", "plot_temp_map"),
    "plot_temp_anomaly": ("spatialmap", "plot_temp_anomaly"),
    "plot_precip_climatology_map": (
        "climatology",
        "plot_precip_climatology_map",
    ),
    "plot_temperature_climatology_map": (
        "climatology",
        "plot_temperature_climatology_map",
    ),
    "plot_climatology_spatial_mean": (
        "climatology",
        "plot_climatology_spatial_mean",
    ),
    "plot_diagram_ombro_one_point": (
        "climatology",
        "plot_diagram_ombro_one_point",
    ),
    "plot_composite_map": ("climatology", "plot_composite_map"),
    "plot_stn_temp_monitoring": (
        "monitoring_stn",
        "plot_stn_temp_monitoring",
    ),
    "plot_stn_precip_monitoring": (
        "monitoring_stn",
        "plot_stn_precip_monitoring",
    ),
    "plot_stn_monthly_monitoring": (
        "monitoring_stn",
        "plot_stn_monthly_monitoring",
    ),
    "plot_monthly_interannual_variability": (
        "monitoring_stn",
        "plot_monthly_interannual_variability",
    ),
    "plot_track_tc": ("monitor_TC", "plot_track_tc"),
    "temporal_evolution_tc": ("monitor_TC", "temporal_evolution_tc"),
    "plot_all_tc_season": ("monitor_TC", "plot_all_tc_season"),
    "plot_map_latest_sst": (
        "monitoringLargeScaleParam",
        "plot_map_latest_sst",
    ),
    "plot_map_latest_mslp": (
        "monitoringLargeScaleParam",
        "plot_map_latest_mslp",
    ),
    "plot_climate_indices": (
        "monitoring_climate_driver",
        "plot_climate_indices",
    ),
    "plot_sst_map_variability": (
        "monitoring_climate_driver",
        "plot_sst_map_variability",
    ),
    "blend_stn_cdt_data": ("preparation_data", "blend_stn_cdt_data"),
    "download_sst_mean_data": ("preparation_data", "download_sst_mean_data"),
    "normalize_lat_lon": ("preparation_data", "normalize_lat_lon"),
    "resolve_variable": ("preparation_data", "resolve_variable"),
    "is_precip_variable": ("preparation_data", "is_precip_variable"),
    "apply_scientific_plot_style": (
        "preparation_data",
        "apply_scientific_plot_style",
    ),
    "set_scientific_title": ("preparation_data", "set_scientific_title"),
    "style_scientific_grid": ("preparation_data", "style_scientific_grid"),
    "style_cartopy_gridlines": (
        "preparation_data",
        "style_cartopy_gridlines",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, object_name = _EXPORTS[name]
    module = __import__(
        f"{__name__}.{module_name}",
        fromlist=[object_name],
    )
    value = getattr(module, object_name)
    globals()[name] = value
    return value
