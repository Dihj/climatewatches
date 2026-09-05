"""ClimateWatches is a Python toolkit for climate bulletin monitoring and visualization."""

__version__ = "1.0.1"

_EXPORTS = {
    "plot_precip_interannual_variability": (
        "interannual_variability",
        "plot_precip_interannual_variability",
    ),
    "plot_temp_interannual_variability": (
        "interannual_variability",
        "plot_temp_interannual_variability",
    ),
    "read_stn_data": ("interannual_variability", "read_stn_data"),
    "assign_season_year": ("interannual_variability", "assign_season_year"),
    "plot_precip_map": ("spatial_map", "plot_precip_map"),
    "plot_precip_anomaly": ("spatial_map", "plot_precip_anomaly"),
    "plot_temp_map": ("spatial_map", "plot_temp_map"),
    "plot_temp_anomaly": ("spatial_map", "plot_temp_anomaly"),
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
    "plot_track_tc": ("monitor_tc", "plot_track_tc"),
    "temporal_evolution_tc": ("monitor_tc", "temporal_evolution_tc"),
    "plot_all_tc_season": ("monitor_tc", "plot_all_tc_season"),
    "plot_map_latest_sst": (
        "monitoring_large_scale_param",
        "plot_map_latest_sst",
    ),
    "plot_map_latest_mslp": (
        "monitoring_large_scale_param",
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
    "set_plot_defaults": ("preparation_data", "set_plot_defaults"),
    "set_title": ("preparation_data", "set_title"),
    "set_grid": ("preparation_data", "set_grid"),
    "set_map_grid": ("preparation_data", "set_map_grid"),
    "annual_trend": ("annual_analysis", "annual_trend"),
    "disaster_impact": ("disaster_analysis", "disaster_impact"),
    "plot_disaster_impact": ("plot_disaster_impact", "plot_disaster_impact"),
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
