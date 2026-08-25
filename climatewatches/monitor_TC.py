import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.dates as mdates

from geopy.distance import geodesic
from mpl_toolkits.basemap import Basemap


TRANSLATIONS = {
    "fr": {
        "title": "Trajectoire du cyclone tropical",
        "local_time": "Heure locale UTC",
        "outside_basin": "Hors bassin SI",
        "pt": "PT (< 50 km/h)",
        "dt": "DT (51-62 km/h)",
        "ttm": "TTM (63-88 km/h)",
        "ftt": "FTT (89-117 km/h)",
        "ct": "CT (118-165 km/h)",
        "cti": "CTI (166-212 km/h)",
        "ctti": "CTTI (> 212 km/h)",
        "date_axis": "Date (UTC",
        "wind_axis": "Vent et rafale (km/h)",
        "gust_envelope": "Enveloppe des rafales",
        "sustained_wind": "Vent max. (10 min)",
        "translation_speed_axis": "Vitesse de deplacement (km/h)",
        "translation_speed": "Vitesse de deplacement",
        "temporal_title": "Evolution temporelle du cyclone tropical",
        "temporal_subtitle": (
            "Intensite du vent vs. vitesse de deplacement"
        ),
        "season_title": "Cyclones tropicaux influencant Madagascar",
        "saved": "Graphique cree et enregistre",
    },
    "en": {
        "title": "Track of tropical cyclone",
        "local_time": "Local Time UTC",
        "outside_basin": "Outside SI basin",
        "pt": "PT (< 50 km/h)",
        "dt": "DT (51-62 km/h)",
        "ttm": "TTM (63-88 km/h)",
        "ftt": "FTT (89-117 km/h)",
        "ct": "CT (118-165 km/h)",
        "cti": "CTI (166-212 km/h)",
        "ctti": "CTTI (> 212 km/h)",
        "date_axis": "Date (UTC",
        "wind_axis": "Wind and gusts (km/h)",
        "gust_envelope": "Gust envelope",
        "sustained_wind": "Max wind (10 min)",
        "translation_speed_axis": "Translation speed (km/h)",
        "translation_speed": "Translation speed",
        "temporal_title": "Temporal evolution of tropical cyclone",
        "temporal_subtitle": "Wind intensity vs. translation speed",
        "season_title": "Tropical cyclones influencing Madagascar",
        "saved": "Plot created and saved",
    },
    "mg": {
        "title": "Lalan'ny rivodoza tropikaly",
        "local_time": "Ora eto an-toerana UTC",
        "outside_basin": "Ivelan'ny faritra SI",
        "pt": "PT (< 50 km/h)",
        "dt": "DT (51-62 km/h)",
        "ttm": "TTM (63-88 km/h)",
        "ftt": "FTT (89-117 km/h)",
        "ct": "CT (118-165 km/h)",
        "cti": "CTI (166-212 km/h)",
        "ctti": "CTTI (> 212 km/h)",
        "date_axis": "Daty (UTC",
        "wind_axis": "Rivotra sy tafiotra (km/h)",
        "gust_envelope": "Faritra tafiotra",
        "sustained_wind": "Rivotra ambony indrindra (10 min)",
        "translation_speed_axis": "Hafainganam-pandeha (km/h)",
        "translation_speed": "Hafainganam-pandeha",
        "temporal_title": "Fiovan'ny rivodoza tropikaly",
        "temporal_subtitle": (
            "Herin'ny rivotra sy hafainganam-pandeha"
        ),
        "season_title": "Rivodoza tropikaly misy fiantraikany amin'i Madagasikara",
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


def _intensity_color(wind_speed):
    if pd.isna(wind_speed):
        return "grey"

    try:
        wind_speed = float(wind_speed)
    except (TypeError, ValueError):
        return "grey"

    if wind_speed < 50:
        return "green"
    if wind_speed <= 62:
        return "yellow"
    if wind_speed <= 88:
        return "orange"
    if wind_speed <= 117:
        return "black"
    if wind_speed <= 165:
        return "red"
    if wind_speed <= 212:
        return "purple"

    return "brown"


def _is_inside_bounds(lon, lat, bounds):
    return (
        bounds["lon_min"] <= lon <= bounds["lon_max"]
        and bounds["lat_min"] <= lat <= bounds["lat_max"]
    )


def _load_tc_data(file, timezone_offset):
    df = pd.read_csv(
        file,
        low_memory=False,
    )

    required_columns = {
        "NAME",
        "LON",
        "LAT",
        "Date",
        "Vent max km/h",
    }
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required cyclone column(s): "
            f"{', '.join(sorted(missing_columns))}"
        )

    df = df.copy()
    df["LON"] = pd.to_numeric(
        df["LON"],
        errors="coerce",
    )
    df["LAT"] = pd.to_numeric(
        df["LAT"],
        errors="coerce",
    )
    df["WIND_INTENSITY_KMH"] = pd.to_numeric(
        df["Vent max km/h"],
        errors="coerce",
    )
    df["DATE"] = pd.to_datetime(
        df["Date"],
        format="%d/%m/%Y %H%M",
        errors="coerce",
    )
    df["ISO_TIME"] = df["DATE"] + pd.Timedelta(
        hours=timezone_offset,
    )

    return df.dropna(
        subset=[
            "NAME",
            "LON",
            "LAT",
            "ISO_TIME",
        ]
    )


def plot_track_tc(
    file,
    start_date,
    end_date,
    language="fr",
    cyclone_names=None,
    timezone_offset=3,
    map_extent=None,
    basin_bounds=None,
    output_file=None,
    title=None,
    show=True,
):
    """
    Plot tropical cyclone tracks from a CSV file.

    Expected columns are NAME, LON, LAT, Date, and Vent max km/h. Dates are
    parsed with the format ``%d/%m/%Y %H%M`` and converted to local time using
    ``timezone_offset`` hours.
    """

    language = _validate_language(language)
    text = TRANSLATIONS[language]

    if map_extent is None:
        map_extent = {
            "lat_min": -25,
            "lat_max": -10,
            "lon_min": 40,
            "lon_max": 55,
        }

    if basin_bounds is None:
        basin_bounds = {
            "lon_min": 30,
            "lon_max": 100,
            "lat_min": -40,
            "lat_max": 0,
        }

    df = _load_tc_data(
        file,
        timezone_offset,
    )

    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    filtered_df = df[
        (df["ISO_TIME"] >= start_date)
        & (df["ISO_TIME"] <= end_date)
    ].copy()

    if cyclone_names is not None:
        if isinstance(cyclone_names, str):
            cyclone_names = [cyclone_names]

        filtered_df = filtered_df[
            filtered_df["NAME"].isin(cyclone_names)
        ].copy()

    if filtered_df.empty:
        raise ValueError("No cyclone track data found for the selected period.")

    fig = plt.figure(
        figsize=(12, 10),
    )
    ax = fig.add_subplot(111)

    basemap = Basemap(
        projection="cyl",
        llcrnrlat=map_extent["lat_min"],
        urcrnrlat=map_extent["lat_max"],
        llcrnrlon=map_extent["lon_min"],
        urcrnrlon=map_extent["lon_max"],
        resolution="i",
        area_thresh=1000,
        ax=ax,
    )

    basemap.shadedrelief()
    basemap.drawparallels(
        range(-40, 1, 5),
        labels=[1, 0, 0, 0],
        linewidth=0.2,
    )
    basemap.drawmeridians(
        range(30, 101, 5),
        labels=[0, 0, 0, 1],
        linewidth=0.2,
    )

    label_positions = []
    plotted_cyclones = []

    for cyclone in filtered_df["NAME"].dropna().unique():
        cyclone_df = filtered_df[
            filtered_df["NAME"] == cyclone
        ].sort_values(
            "ISO_TIME"
        )

        if cyclone_df.empty:
            continue

        lats = cyclone_df["LAT"].values
        lons = cyclone_df["LON"].values
        winds = cyclone_df["WIND_INTENSITY_KMH"].values

        enters_basin = any(
            _is_inside_bounds(lon, lat, basin_bounds)
            for lon, lat in zip(lons, lats)
        )

        if not enters_basin:
            continue

        starts_outside_basin = not _is_inside_bounds(
            lons[0],
            lats[0],
            basin_bounds,
        )
        line_style = "--" if starts_outside_basin else "-"

        for index in range(len(lats) - 1):
            if _is_inside_bounds(
                lons[index],
                lats[index],
                basin_bounds,
            ):
                basemap.plot(
                    [lons[index], lons[index + 1]],
                    [lats[index], lats[index + 1]],
                    color=_intensity_color(winds[index]),
                    linestyle=line_style,
                    linewidth=2,
                    latlon=True,
                )
                basemap.scatter(
                    lons[index],
                    lats[index],
                    color=_intensity_color(winds[index]),
                    s=15,
                    latlon=True,
                    zorder=3,
                )

        for _, row in cyclone_df.iterrows():
            if row["ISO_TIME"].hour in [3, 15]:
                x = row["LON"]
                y = row["LAT"]

                if _is_inside_bounds(x, y, basin_bounds):
                    time_str = row["ISO_TIME"].strftime("%d/%m %H:%M")
                    ax.text(
                        x + 0.2,
                        y + 0.2,
                        time_str,
                        fontsize=7,
                        fontweight="bold",
                        bbox={
                            "facecolor": "white",
                            "alpha": 0.7,
                            "edgecolor": "none",
                            "pad": 1,
                        },
                        zorder=5,
                    )

        if starts_outside_basin:
            entry_point_index = next(
                (
                    index
                    for index, (lon, lat) in enumerate(zip(lons, lats))
                    if _is_inside_bounds(lon, lat, basin_bounds)
                ),
                0,
            )
            x_label = lons[entry_point_index]
            y_label = lats[entry_point_index]
        else:
            x_label = lons[0]
            y_label = lats[0]

        offset = 0.5
        while any(
            abs(x_label - x) < offset and abs(y_label - y) < offset
            for x, y in label_positions
        ):
            x_label += offset
            y_label += offset

        ax.text(
            x_label,
            y_label,
            cyclone,
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="bottom",
            color="red" if starts_outside_basin else "black",
            bbox={
                "facecolor": "white",
                "alpha": 0.8,
                "edgecolor": "black",
            },
            zorder=6,
        )
        label_positions.append(
            (
                x_label,
                y_label,
            )
        )
        plotted_cyclones.append(cyclone)

    if not plotted_cyclones:
        raise ValueError("No cyclone tracks enter the selected basin bounds.")

    legend_elements = [
        mlines.Line2D([], [], color="green", label=text["pt"]),
        mlines.Line2D([], [], color="yellow", label=text["dt"]),
        mlines.Line2D([], [], color="orange", label=text["ttm"]),
        mlines.Line2D([], [], color="black", label=text["ftt"]),
        mlines.Line2D([], [], color="red", label=text["ct"]),
        mlines.Line2D([], [], color="purple", label=text["cti"]),
        mlines.Line2D([], [], color="brown", label=text["ctti"]),
        mlines.Line2D(
            [],
            [],
            color="red",
            linestyle="--",
            label=text["outside_basin"],
        ),
    ]

    ax.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=9,
    )

    if title is None:
        cyclone_title = ", ".join(plotted_cyclones)
        title = (
            f"{text['title']} {cyclone_title} "
            f"({text['local_time']}+{timezone_offset})"
        )

    ax.set_title(
        title,
        fontsize=14,
    )

    if output_file is None:
        cyclone_part = "_".join(
            str(name).replace(" ", "_")
            for name in plotted_cyclones
        )
        output_file = f"{cyclone_part}_track.png"

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
        "basemap": basemap,
        "data": filtered_df,
        "plotted_cyclones": plotted_cyclones,
        "output_file": output_file,
    }


def temporal_evolution_tc(
    file,
    start_date,
    end_date,
    language="fr",
    cyclone_name=None,
    timezone_offset=3,
    gust_factor=1.25,
    output_file=None,
    title=None,
    show=True,
):
    """
    Plot temporal evolution of cyclone intensity and translation speed.

    Expected columns are NAME, LON, LAT, Date, and Vent max km/h. Gusts are
    estimated as sustained wind multiplied by ``gust_factor``.
    """

    language = _validate_language(language)
    text = TRANSLATIONS[language]

    df = _load_tc_data(
        file,
        timezone_offset,
    )

    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    df = df[
        (df["ISO_TIME"] >= start_date)
        & (df["ISO_TIME"] <= end_date)
    ].copy()

    if cyclone_name is not None:
        df = df[df["NAME"] == cyclone_name].copy()

    if df.empty:
        raise ValueError("No cyclone data found for the selected period.")

    if cyclone_name is None:
        cyclone_name = df["NAME"].iloc[0]
        df = df[df["NAME"] == cyclone_name].copy()

    df = df.sort_values("ISO_TIME")
    df["GUSTS"] = df["WIND_INTENSITY_KMH"] * gust_factor

    speeds = [0]

    for index in range(1, len(df)):
        previous = df.iloc[index - 1]
        current = df.iloc[index]
        previous_position = (
            previous["LAT"],
            previous["LON"],
        )
        current_position = (
            current["LAT"],
            current["LON"],
        )

        distance = geodesic(
            previous_position,
            current_position,
        ).km
        time_diff = (
            current["ISO_TIME"] - previous["ISO_TIME"]
        ).total_seconds() / 3600

        speeds.append(
            distance / time_diff
            if time_diff > 0
            else 0
        )

    df["TRANS_SPEED"] = speeds

    plt.rcParams["font.family"] = "serif"

    fig, ax1 = plt.subplots(
        figsize=(13, 7),
        dpi=120,
    )

    ax1.set_xlabel(
        f"{text['date_axis']}+{timezone_offset})",
        fontsize=12,
        fontweight="bold",
        labelpad=15,
    )
    ax1.set_ylabel(
        text["wind_axis"],
        fontsize=12,
        fontweight="bold",
        color="darkred",
    )

    ax1.fill_between(
        df["ISO_TIME"],
        df["WIND_INTENSITY_KMH"],
        df["GUSTS"],
        color="red",
        alpha=0.12,
        label=text["gust_envelope"],
    )

    line1, = ax1.plot(
        df["ISO_TIME"],
        df["WIND_INTENSITY_KMH"],
        color="darkred",
        marker="s",
        markersize=5,
        linewidth=2.5,
        label=text["sustained_wind"],
    )

    thresholds = {
        "TTM": 63,
        "FTT": 89,
        "CT": 118,
        "CTI": 166,
    }

    first_time = df["ISO_TIME"].iloc[0]

    for label, value in thresholds.items():
        ax1.axhline(
            y=value,
            color="black",
            linestyle="--",
            linewidth=0.8,
            alpha=0.3,
        )
        ax1.text(
            first_time,
            value + 1.5,
            f" {label}",
            fontsize=8,
            color="black",
            alpha=0.6,
        )

    ax2 = ax1.twinx()
    ax2.set_ylabel(
        text["translation_speed_axis"],
        fontsize=12,
        fontweight="bold",
        color="navy",
    )
    line2, = ax2.plot(
        df["ISO_TIME"],
        df["TRANS_SPEED"],
        color="navy",
        linestyle="-",
        marker="o",
        markersize=4,
        linewidth=1.2,
        alpha=0.5,
        label=text["translation_speed"],
    )

    lines = [
        line1,
        line2,
    ]
    labels = [
        line.get_label()
        for line in lines
    ]
    ax1.legend(
        lines,
        labels,
        loc="upper left",
        frameon=True,
        fontsize=10,
        shadow=False,
    )

    ax1.xaxis.set_major_locator(
        mdates.DayLocator()
    )
    ax1.xaxis.set_major_formatter(
        mdates.DateFormatter("%d %b\n%Y")
    )
    ax1.xaxis.set_minor_locator(
        mdates.HourLocator(
            byhour=[
                6,
                12,
                18,
            ]
        )
    )

    ax1.grid(
        True,
        which="major",
        linestyle=":",
        alpha=0.4,
    )

    wind_max = df["GUSTS"].max()
    speed_max = df["TRANS_SPEED"].max()
    ax1.set_ylim(
        0,
        wind_max + 30
        if pd.notna(wind_max)
        else 30,
    )
    ax2.set_ylim(
        0,
        speed_max + 10
        if pd.notna(speed_max)
        else 10,
    )

    if title is None:
        title = (
            f"{text['temporal_title']} {cyclone_name}\n"
            f"{text['temporal_subtitle']}"
        )

    ax1.set_title(
        title,
        fontsize=14,
        fontweight="bold",
        pad=25,
    )

    plt.tight_layout()

    if output_file is None:
        cyclone_part = str(cyclone_name).replace(" ", "_")
        output_file = f"Cyclone_Evolution_{cyclone_part}.pdf"

    plt.savefig(
        output_file,
        bbox_inches="tight",
    )

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"{text['saved']}: {output_file}")

    return {
        "figure": fig,
        "axis": ax1,
        "speed_axis": ax2,
        "data": df,
        "cyclone_name": cyclone_name,
        "output_file": output_file,
    }


def plot_all_tc_season(
    file,
    start_date,
    end_date,
    language="fr",
    timezone_offset=0,
    map_extent=None,
    basin_bounds=None,
    output_file=None,
    title=None,
    show=True,
):
    """
    Plot all tropical cyclone tracks for a selected season or period.

    Expected columns are NAME, LON, LAT, Date, and Vent max km/h. The default
    map and basin bounds are configured for the southwestern Indian Ocean near
    Madagascar.
    """

    language = _validate_language(language)
    text = TRANSLATIONS[language]

    if map_extent is None:
        map_extent = {
            "lat_min": -40,
            "lat_max": -5,
            "lon_min": 30,
            "lon_max": 70,
        }

    if basin_bounds is None:
        basin_bounds = {
            "lon_min": 30,
            "lon_max": 70,
            "lat_min": -40,
            "lat_max": -7,
        }

    df = _load_tc_data(
        file,
        timezone_offset,
    )

    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    filtered_df = df[
        (df["ISO_TIME"] >= start_date)
        & (df["ISO_TIME"] <= end_date)
    ].copy()

    if filtered_df.empty:
        raise ValueError("No cyclone track data found for the selected period.")

    fig = plt.figure(
        figsize=(12, 10),
    )
    ax = fig.add_subplot(111)

    basemap = Basemap(
        projection="cyl",
        llcrnrlat=map_extent["lat_min"],
        urcrnrlat=map_extent["lat_max"],
        llcrnrlon=map_extent["lon_min"],
        urcrnrlon=map_extent["lon_max"],
        resolution="i",
        area_thresh=1000,
        ax=ax,
    )

    basemap.shadedrelief()
    basemap.drawparallels(
        range(-40, -6, 10),
        labels=[1, 0, 0, 0],
        linewidth=0.2,
    )
    basemap.drawmeridians(
        range(30, 71, 10),
        labels=[0, 0, 0, 1],
        linewidth=0.2,
    )

    label_positions = []
    plotted_cyclones = []

    for cyclone in filtered_df["NAME"].dropna().unique():
        cyclone_df = filtered_df[
            filtered_df["NAME"] == cyclone
        ].sort_values(
            "ISO_TIME"
        )

        if cyclone_df.empty:
            continue

        lats = cyclone_df["LAT"].values
        lons = cyclone_df["LON"].values
        winds = cyclone_df["WIND_INTENSITY_KMH"].values

        enters_basin = any(
            _is_inside_bounds(lon, lat, basin_bounds)
            for lon, lat in zip(lons, lats)
        )

        if not enters_basin:
            continue

        starts_outside_basin = not _is_inside_bounds(
            lons[0],
            lats[0],
            basin_bounds,
        )
        line_style = "--" if starts_outside_basin else "-"

        for index in range(len(lats) - 1):
            if _is_inside_bounds(
                lons[index],
                lats[index],
                basin_bounds,
            ):
                basemap.plot(
                    [lons[index], lons[index + 1]],
                    [lats[index], lats[index + 1]],
                    color=_intensity_color(winds[index]),
                    linestyle=line_style,
                    linewidth=2,
                    latlon=True,
                )
                basemap.scatter(
                    lons[index],
                    lats[index],
                    color=_intensity_color(winds[index]),
                    s=10,
                    latlon=True,
                    zorder=3,
                )

        if starts_outside_basin:
            entry_point_index = next(
                (
                    index
                    for index, (lon, lat) in enumerate(zip(lons, lats))
                    if _is_inside_bounds(lon, lat, basin_bounds)
                ),
                0,
            )
            x_label = lons[entry_point_index]
            y_label = lats[entry_point_index]
        else:
            x_label = lons[0]
            y_label = lats[0]

        offset = 0.5
        while any(
            abs(x_label - x) < offset and abs(y_label - y) < offset
            for x, y in label_positions
        ):
            x_label += offset
            y_label += offset

        ax.text(
            x_label,
            y_label,
            cyclone,
            fontsize=8,
            ha="left",
            va="bottom",
            color="red" if starts_outside_basin else "black",
            bbox={
                "facecolor": "white",
                "alpha": 0.5,
                "edgecolor": "none",
            },
            zorder=6,
        )
        label_positions.append(
            (
                x_label,
                y_label,
            )
        )
        plotted_cyclones.append(cyclone)

    if not plotted_cyclones:
        raise ValueError("No cyclone tracks enter the selected basin bounds.")

    legend_elements = [
        mlines.Line2D([], [], color="green", label=text["pt"]),
        mlines.Line2D([], [], color="yellow", label=text["dt"]),
        mlines.Line2D([], [], color="orange", label=text["ttm"]),
        mlines.Line2D([], [], color="black", label=text["ftt"]),
        mlines.Line2D([], [], color="red", label=text["ct"]),
        mlines.Line2D([], [], color="purple", label=text["cti"]),
        mlines.Line2D([], [], color="brown", label=text["ctti"]),
    ]

    ax.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=10,
    )

    if title is None:
        title = (
            f"{text['season_title']} "
            f"({start_date.strftime('%d-%b-%Y')} - "
            f"{end_date.strftime('%d-%b-%Y')})"
        )

    ax.set_title(
        title,
        fontsize=14,
    )

    if output_file is None:
        output_file = (
            f"TC_Mdg_{start_date:%Y%m%d}_{end_date:%Y%m%d}.png"
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
        "basemap": basemap,
        "data": filtered_df,
        "plotted_cyclones": plotted_cyclones,
        "output_file": output_file,
    }
