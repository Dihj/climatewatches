# Python script to monitor station data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec

from scipy.stats import zscore

try:
    from .preparation_data import (
        set_grid,
        set_plot_defaults,
        set_title,
    )
except ImportError:
    from preparation_data import (
        set_grid,
        set_plot_defaults,
        set_title,
    )


TRANSLATIONS = {
    "fr": {
        "title": (
            "Variation journaliere des temperatures minimales et maximales "
            "comparee a la climatologie"
        ),
        "x_axis": "Jour de l'annee",
        "y_axis": "Temperature (deg C)",
        "tmin_climatology": "Tmin climatologie",
        "tmax_climatology": "Tmax climatologie",
        "tmin_mean_climatology": "Tmin climatologie moyenne",
        "tmax_mean_climatology": "Tmax climatologie moyenne",
        "tmin_recent": "Tmin saison",
        "tmax_recent": "Tmax saison",
        "tmin_current": "Tmin saison",
        "tmax_current": "Tmax saison",
        "precip_title": (
            "Variation des precipitations journalieres et cumulees "
            "comparee a la climatologie"
        ),
        "precip_cumulative_axis": "Precipitations cumulees (mm)",
        "precip_daily_axis": "Precipitations journalieres (mm)",
        "precip_climatology_range": "Plage",
        "precip_climatology": "Climatologie",
        "precip_season": "Saison",
        "precip_daily_current": "Precipitations journalieres",
        "monthly_title": "Temperatures et precipitations",
        "monthly_x_axis": "Jours",
        "monthly_temperature_axis": "Temperature (deg C)",
        "monthly_precip_axis": "Precipitations (mm)",
        "monthly_tmin_climatology": "Tmin climatologie",
        "monthly_tmax_climatology": "Tmax climatologie",
        "monthly_precip_climatology": "Climatologie de precipitations",
        "monthly_precip": "Precipitations",
        "no_month_data": "Aucune donnee trouvee",
        "no_month_climatology": (
            "Aucune donnee de climatologie trouvee pour 1991-2020 "
            "pour ce mois."
        ),
        "interannual_title": (
            "Variation interannuelle des precipitations "
            "(Juillet N - Juin N+1)"
        ),
        "interannual_ylabel": "Precipitations (mm)",
        "interannual_anomaly_ylabel": "Z-Score\n(Annuelle)",
        "interannual_legend_title": "Mois",
        "saved": "Graphique cree et enregistre",
    },
    "en": {
        "title": (
            "Daily minimum and maximum temperature variation compared "
            "with climatology"
        ),
        "x_axis": "Day of year",
        "y_axis": "Temperature (deg C)",
        "tmin_climatology": "Tmin climatology",
        "tmax_climatology": "Tmax climatology",
        "tmin_mean_climatology": "Mean Tmin climatology",
        "tmax_mean_climatology": "Mean Tmax climatology",
        "tmin_recent": "Tmin season",
        "tmax_recent": "Tmax season",
        "tmin_current": "Tmin season",
        "tmax_current": "Tmax season",
        "precip_title": (
            "Daily and cumulative rainfall variation compared "
            "with climatology"
        ),
        "precip_cumulative_axis": "Cumulative rainfall (mm)",
        "precip_daily_axis": "Daily rainfall (mm)",
        "precip_climatology_range": "Range",
        "precip_climatology": "Climatology",
        "precip_season": "Season",
        "precip_daily_current": "Daily rainfall",
        "monthly_title": "Temperature and rainfall",
        "monthly_x_axis": "Days",
        "monthly_temperature_axis": "Temperature (deg C)",
        "monthly_precip_axis": "Rainfall (mm)",
        "monthly_tmin_climatology": "Tmin climatology",
        "monthly_tmax_climatology": "Tmax climatology",
        "monthly_precip_climatology": "Rainfall climatology",
        "monthly_precip": "Rainfall",
        "no_month_data": "No data found",
        "no_month_climatology": (
            "No climatology data found for 1991-2020 for this month."
        ),
        "interannual_title": (
            "Interannual rainfall variability "
            "(July N - June N+1)"
        ),
        "interannual_ylabel": "Rainfall (mm)",
        "interannual_anomaly_ylabel": "Z-Score\n(Annual)",
        "interannual_legend_title": "Months",
        "saved": "Plot created and saved",
    },
    "mg": {
        "title": (
            "Fiovan'ny maripana ambany sy ambony isan'andro "
            "ampitahaina amin'ny mahazatra"
        ),
        "x_axis": "Andro ao anatin'ny taona",
        "y_axis": "Maripana (deg C)",
        "tmin_climatology": "Tmin mahazatra (Climatologie)",
        "tmax_climatology": "Tmax mahazatra (Climatologie)",
        "tmin_mean_climatology": "Salanisa Tmin mahazatra",
        "tmax_mean_climatology": "Salanisa Tmax mahazatra",
        "tmin_recent": "Tmin vanim-potoana",
        "tmax_recent": "Tmax vanim-potoana",
        "tmin_current": "Tmin vanim-potoana",
        "tmax_current": "Tmax vanim-potoana",
        "precip_title": (
            "Fiovan'ny rotsakorana isan'andro "
            "ampitahaina amin'ny mahazatra"
        ),
        "precip_cumulative_axis": "Rotsakorana (mm)",
        "precip_daily_axis": "Rotsakorana isan'andro (mm)",
        "precip_climatology_range": "Taha ambany/ambony indrindra",
        "precip_climatology": "Mahazatra",
        "precip_season": "Vanim-potoana",
        "precip_daily_current": "Rotsakorana isan'andro",
        "monthly_title": "Maripana sy rotsakorana",
        "monthly_x_axis": "Andro",
        "monthly_temperature_axis": "Maripana (deg C)",
        "monthly_precip_axis": "Rotsakorana (mm)",
        "monthly_tmin_climatology": "Tmin mahazatra",
        "monthly_tmax_climatology": "Tmax mahazatra",
        "monthly_precip_climatology": "Rotsakorana mahazatra",
        "monthly_precip": "Rotsakorana",
        "no_month_data": "Tsy misy angona hita",
        "no_month_climatology": (
            "Tsy misy antontan'isa hita ho an'ny 1991-2020 "
            "amin'io volana io."
        ),
        "interannual_title": (
            "Fiovan'ny rotsakorana isan-taona "
            "(Jolay N - Jona N+1)"
        ),
        "interannual_ylabel": "Rotsakorana (mm)",
        "interannual_anomaly_ylabel": "Z-Score\n(Isan-taona)",
        "interannual_legend_title": "Volana",
        "saved": "Vita sy voatahiry ny kisarisary",
    },
}


MISSING_VALUES = [-999, -99.9, -99, "NA", "N/A", "", "missing"]


def _get_translation(language):
    """Normalize a language code and return its station-plot translations."""

    language = str(language).strip().lower()
    if language not in TRANSLATIONS:
        raise ValueError(
            f"Unsupported language {language!r}. "
            "Choose 'fr', 'en', 'mg'."
        )

    return language, TRANSLATIONS[language]


def _read_station_data(file, required_columns, *, parse_dates=False):
    """Read station data and enforce the columns required by a plot."""

    kwargs = {"na_values": MISSING_VALUES}
    if parse_dates:
        kwargs["parse_dates"] = ["Date"]

    data = pd.read_csv(file, **kwargs)
    missing_columns = set(required_columns).difference(data.columns)
    if missing_columns:
        raise ValueError(
            "Missing required station column(s): "
            f"{', '.join(sorted(missing_columns))}"
        )

    return data


def plot_stn_temp_monitoring(
    file,
    language="fr",
    clim_years=(1991, 2020),
    season_start_month=7,
    season_year=None,
    include_recent=False,
    output_file=None,
    show=True,
):
    """
    Plot daily Tmin/Tmax monitoring from a station CSV file.

    The station file must contain Date, Tmin, and Tmax columns. Seasons are
    defined from July 1 to June 30 by default. If ``season_year`` is provided,
    that year is used as the season start year; for example, ``2024`` covers
    July 2024 through June 2025 when ``season_start_month=7``. If omitted, the
    latest season in the data is used.
    """

    language, text = _get_translation(language)
    set_plot_defaults()
    df = _read_station_data(
        file,
        {"Date", "Tmin", "Tmax"},
        parse_dates=True,
    )

    df = df.copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    df["Season_Year"] = np.where(
        df["Month"] >= season_start_month,
        df["Year"],
        df["Year"] - 1,
    )

    last_date = df["Date"].max()

    if pd.isna(last_date):
        raise ValueError("No valid Date values found in station file.")

    if season_year is None:
        if last_date.month < season_start_month:
            current_season = last_date.year - 1
        else:
            current_season = last_date.year
    else:
        current_season = int(season_year)

    daily_tmin_seasonal = {}
    daily_tmax_seasonal = {}

    for year in sorted(df["Season_Year"].dropna().unique()):
        year = int(year)
        start_date = pd.to_datetime(
            f"{year}-{season_start_month:02d}-01"
        )
        end_date = start_date + pd.DateOffset(years=1) - pd.Timedelta(days=1)

        mask = (
            (df["Date"] >= start_date)
            & (df["Date"] <= end_date)
        )

        data_season = df.loc[mask].copy()
        data_season = data_season.set_index("Date").sort_index()

        daily_tmin = data_season["Tmin"].resample("D").mean()
        daily_tmax = data_season["Tmax"].resample("D").mean()

        day_index = (daily_tmin.index - start_date).days
        daily_tmin.index = day_index
        daily_tmax.index = day_index

        daily_tmin_seasonal[year] = daily_tmin
        daily_tmax_seasonal[year] = daily_tmax

    all_years = sorted(daily_tmin_seasonal.keys())
    climatology_years = [
        year
        for year in all_years
        if clim_years[0] <= year <= clim_years[1]
    ]
    recent_years = [
        year
        for year in all_years
        if year < current_season
    ][-4:]

    if current_season not in daily_tmin_seasonal:
        raise ValueError(
            f"No data found for current season {current_season}-"
            f"{current_season + 1}."
        )

    if not climatology_years:
        raise ValueError(
            "No climatology seasons found for "
            f"{clim_years[0]}-{clim_years[1]}."
        )

    climatology_tmin_matrix = pd.DataFrame({
        year: daily_tmin_seasonal[year]
        for year in climatology_years
    })
    climatology_tmax_matrix = pd.DataFrame({
        year: daily_tmax_seasonal[year]
        for year in climatology_years
    })

    min_tmin_clim = climatology_tmin_matrix.min(axis=1)
    max_tmin_clim = climatology_tmin_matrix.max(axis=1)
    min_tmax_clim = climatology_tmax_matrix.min(axis=1)
    max_tmax_clim = climatology_tmax_matrix.max(axis=1)

    mean_tmin_clim = climatology_tmin_matrix.mean(axis=1)
    mean_tmax_clim = climatology_tmax_matrix.mean(axis=1)

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.fill_between(
        min_tmin_clim.index,
        min_tmin_clim.values,
        max_tmin_clim.values,
        color="lightblue",
        alpha=0.5,
        label=(
            f"{text['tmin_climatology']} "
            f"{clim_years[0]}-{clim_years[1]}"
        ),
    )

    ax.fill_between(
        min_tmax_clim.index,
        min_tmax_clim.values,
        max_tmax_clim.values,
        color="lightcoral",
        alpha=0.5,
        label=(
            f"{text['tmax_climatology']} "
            f"{clim_years[0]}-{clim_years[1]}"
        ),
    )

    ax.plot(
        mean_tmin_clim.index,
        mean_tmin_clim.values,
        "b--",
        label=text["tmin_mean_climatology"],
    )
    ax.plot(
        mean_tmax_clim.index,
        mean_tmax_clim.values,
        "r--",
        label=text["tmax_mean_climatology"],
    )

    if include_recent and recent_years:
        colors_tmin = cm.Blues(
            np.linspace(0.4, 1, len(recent_years))
        )
        colors_tmax = cm.Reds(
            np.linspace(0.4, 1, len(recent_years))
        )

        for index, year in enumerate(recent_years):
            ax.plot(
                daily_tmin_seasonal[year].index,
                daily_tmin_seasonal[year],
                color=colors_tmin[index],
                linewidth=1.2,
                alpha=0.8,
                label=f"{text['tmin_recent']} {year}-{year + 1}",
            )
            ax.plot(
                daily_tmax_seasonal[year].index,
                daily_tmax_seasonal[year],
                color=colors_tmax[index],
                linewidth=1.2,
                alpha=0.8,
                label=f"{text['tmax_recent']} {year}-{year + 1}",
            )

    ax.plot(
        daily_tmin_seasonal[current_season].index,
        daily_tmin_seasonal[current_season],
        "b-",
        linewidth=2.5,
        label=(
            f"{text['tmin_current']} "
            f"{current_season}-{current_season + 1}"
        ),
    )
    ax.plot(
        daily_tmax_seasonal[current_season].index,
        daily_tmax_seasonal[current_season],
        "r-",
        linewidth=2.5,
        label=(
            f"{text['tmax_current']} "
            f"{current_season}-{current_season + 1}"
        ),
    )

    set_title(
        ax,
        f"{text['title']} {clim_years[0]}-{clim_years[1]}",
        fontsize=16,
    )
    ax.set_xlabel(text["x_axis"], fontsize=12)
    ax.set_ylabel(text["y_axis"], fontsize=12)
    ax.set_xlim(0, 365)
    ax.set_xticks(range(0, 366, 31))

    tick_start = pd.to_datetime(
        f"2020-{season_start_month:02d}-01"
    )
    ax.set_xticklabels(
        [
            (tick_start + pd.Timedelta(days=day)).strftime("%b")
            for day in range(0, 366, 31)
        ],
        rotation=45,
    )

    ax.legend(loc="lower center", fontsize=10)
    set_grid(ax)
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
    )
    fig.subplots_adjust(
        right=0.82,
        hspace=0.4,
    )

    if output_file is None:
        output_file = (
            "Seasonal_Temperature_Mean_Tmin_Tmax_"
            f"{current_season}-{current_season + 1}_{language}.png"
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

    return None


def plot_stn_precip_monitoring(
    file,
    language="fr",
    clim_years=(1991, 2020),
    season_start_month=7,
    season_year=None,
    include_recent=True,
    output_file=None,
    show=True,
):
    """
    Plot daily and cumulative rainfall monitoring from a station CSV file.

    The station file must contain Date and Rainfall columns. Seasons are
    defined from July 1 to June 30 by default. If ``season_year`` is provided,
    that year is used as the season start year; for example, ``2024`` covers
    July 2024 through June 2025 when ``season_start_month=7``. If omitted, the
    latest season in the data is used.
    """

    language, text = _get_translation(language)
    set_plot_defaults()
    df = _read_station_data(
        file,
        {"Date", "Rainfall"},
        parse_dates=True,
    )

    df = df.copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    df["Season_Year"] = np.where(
        df["Month"] >= season_start_month,
        df["Year"],
        df["Year"] - 1,
    )

    last_date = df["Date"].max()

    if pd.isna(last_date):
        raise ValueError("No valid Date values found in station file.")

    if season_year is None:
        if last_date.month < season_start_month:
            current_season = last_date.year - 1
        else:
            current_season = last_date.year
    else:
        current_season = int(season_year)

    accumulated_rainfall = {}
    daily_rainfall_seasonal = {}

    for year in sorted(df["Season_Year"].dropna().unique()):
        year = int(year)
        start_date = pd.to_datetime(
            f"{year}-{season_start_month:02d}-01"
        )
        end_date = start_date + pd.DateOffset(years=1) - pd.Timedelta(days=1)

        mask = (
            (df["Date"] >= start_date)
            & (df["Date"] <= end_date)
        )

        data_season = df.loc[mask].copy()
        data_season = data_season.set_index("Date").sort_index()

        daily_rain = data_season["Rainfall"].resample("D").sum()
        day_index = (daily_rain.index - start_date).days

        accumulated = daily_rain.cumsum()
        accumulated.index = day_index
        daily_rain.index = day_index

        accumulated_rainfall[year] = accumulated
        daily_rainfall_seasonal[year] = daily_rain

    all_years = sorted(accumulated_rainfall.keys())
    climatology_years = [
        year
        for year in all_years
        if clim_years[0] <= year <= clim_years[1]
    ]
    recent_years = [
        year
        for year in all_years
        if year < current_season
    ][-4:]

    if current_season not in accumulated_rainfall:
        raise ValueError(
            f"No data found for current season {current_season}-"
            f"{current_season + 1}."
        )

    if not climatology_years:
        raise ValueError(
            "No climatology seasons found for "
            f"{clim_years[0]}-{clim_years[1]}."
        )

    climatology_matrix = pd.DataFrame({
        year: accumulated_rainfall[year]
        for year in climatology_years
    })
    min_clim = climatology_matrix.min(axis=1)
    max_clim = climatology_matrix.max(axis=1)

    daily_matrix = pd.DataFrame({
        year: daily_rainfall_seasonal[year]
        for year in climatology_years
    })
    daily_mean = daily_matrix.mean(axis=1)
    rolling_5d = daily_mean.rolling(
        window=5,
        center=True,
        min_periods=1,
    ).mean()
    climatology_cumsum = rolling_5d.cumsum()

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.fill_between(
        min_clim.index,
        min_clim.values,
        max_clim.values,
        color="lightgrey",
        alpha=0.5,
        label=(
            f"{clim_years[0]}-{clim_years[1]} "
            f"{text['precip_climatology_range']}"
        ),
    )

    ax1.plot(
        climatology_cumsum.index,
        climatology_cumsum.values,
        color="black",
        label=(
            f"{text['precip_climatology']} "
            f"{clim_years[0]}-{clim_years[1]}"
        ),
        linewidth=2,
    )

    if include_recent and recent_years:
        colors = cm.viridis(
            np.linspace(0, 1, len(recent_years))
        )

        for index, year in enumerate(recent_years):
            ax1.plot(
                accumulated_rainfall[year].index,
                accumulated_rainfall[year],
                label=f"{year}-{year + 1}",
                color=colors[index],
            )

    ax1.plot(
        accumulated_rainfall[current_season].index,
        accumulated_rainfall[current_season],
        color="red",
        label=(
            f"{text['precip_season']} "
            f"{current_season}-{current_season + 1}"
        ),
        linewidth=2,
    )

    ax2 = ax1.twinx()
    daily_current = daily_rainfall_seasonal.get(current_season)

    if daily_current is not None:
        ax2.bar(
            daily_current.index,
            daily_current.values,
            width=1.0,
            color="blue",
            alpha=0.4,
            label=(
                f"{text['precip_daily_current']} "
                f"{current_season}-{current_season + 1}"
            ),
        )
        ax2.set_ylabel(
            text["precip_daily_axis"],
            fontsize=12,
        )

        if daily_current.isnull().all():
            daily_ylim = 1
        else:
            daily_ylim = np.nanmax(daily_current) * 1.2
            if daily_ylim <= 0 or np.isnan(daily_ylim):
                daily_ylim = 1

        ax2.set_ylim(0, daily_ylim)

    set_title(
        ax1,
        f"{text['precip_title']} {clim_years[0]}-{clim_years[1]}",
        fontsize=16,
    )
    ax1.set_ylabel(
        text["precip_cumulative_axis"],
        fontsize=12,
    )
    ax1.set_xlim(0, 365)
    ax1.set_xticks(range(0, 366, 31))

    tick_start = pd.to_datetime(
        f"2020-{season_start_month:02d}-01"
    )
    ax1.set_xticklabels(
        [
            (tick_start + pd.Timedelta(days=day)).strftime("%b")
            for day in range(0, 366, 31)
        ],
        rotation=45,
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
    )

    set_grid(ax1)
    ax1.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
    )
    ax2.tick_params(
        axis="y",
        which="both",
        direction="in",
    )
    fig.subplots_adjust(
        right=0.82,
        hspace=0.4,
    )

    if output_file is None:
        output_file = (
            "Seasonal_Rainfall_Cumulative_And_Daily_"
            f"{current_season}-{current_season + 1}_{language}.png"
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

    return None


def plot_stn_monthly_monitoring(
    file,
    selected_month,
    selected_year,
    language="fr",
    clim_years=(1991, 2020),
    output_file=None,
    show=True,
):
    """
    Plot monthly Tmin, Tmax, and rainfall monitoring from station data.

    The station file must contain Date, Tmin, Tmax, and Rainfall columns.
    """

    language, text = _get_translation(language)
    set_plot_defaults()
    selected_month = int(selected_month)
    selected_year = int(selected_year)
    station_data = _read_station_data(
        file,
        {"Date", "Tmax", "Tmin", "Rainfall"},
        parse_dates=True,
    )

    station_data = station_data.copy()

    station_data_selected = station_data[
        (
            station_data["Date"].dt.month == selected_month
        )
        & (
            station_data["Date"].dt.year == selected_year
        )
    ].copy()

    if station_data_selected.empty:
        message = (
            f"{text['no_month_data']} for "
            f"{selected_month:02d}/{selected_year}."
        )
        print(message)
        return None

    climatology_period = station_data[
        (
            station_data["Date"].dt.year >= clim_years[0]
        )
        & (
            station_data["Date"].dt.year <= clim_years[1]
        )
        & (
            station_data["Date"].dt.month == selected_month
        )
    ].copy()

    if climatology_period.empty:
        print(text["no_month_climatology"])
        return None

    station_data_selected = station_data_selected.sort_values("Date")
    climatology_grouped = climatology_period.groupby(
        climatology_period["Date"].dt.day
    )

    tmax_clim_min = climatology_grouped["Tmax"].min()
    tmax_clim_max = climatology_grouped["Tmax"].max()
    tmin_clim_min = climatology_grouped["Tmin"].min()
    tmin_clim_max = climatology_grouped["Tmin"].max()
    rainfall_daily_clim = climatology_grouped["Rainfall"].mean()

    selected_days = station_data_selected["Date"].dt.day
    rainfall_clim_aligned = rainfall_daily_clim.reindex(
        selected_days
    ).values

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.fill_between(
        station_data_selected["Date"],
        tmin_clim_min.reindex(selected_days).values,
        tmin_clim_max.reindex(selected_days).values,
        color="blue",
        alpha=0.1,
        label=text["monthly_tmin_climatology"],
    )
    ax1.fill_between(
        station_data_selected["Date"],
        tmax_clim_min.reindex(selected_days).values,
        tmax_clim_max.reindex(selected_days).values,
        color="red",
        alpha=0.1,
        label=text["monthly_tmax_climatology"],
    )

    ax1.plot(
        station_data_selected["Date"],
        station_data_selected["Tmin"],
        label="Tmin",
        color="blue",
        linewidth=2,
    )
    ax1.plot(
        station_data_selected["Date"],
        station_data_selected["Tmax"],
        label="Tmax",
        color="red",
        linewidth=2,
    )

    ax1.set_xlabel(text["monthly_x_axis"])
    ax1.set_ylabel(
        text["monthly_temperature_axis"],
        color="tab:blue",
    )
    ax1.tick_params(
        axis="y",
        labelcolor="tab:blue",
    )

    ax2 = ax1.twinx()
    ax2.bar(
        station_data_selected["Date"],
        rainfall_clim_aligned,
        color="grey",
        alpha=0.3,
        label=text["monthly_precip_climatology"],
    )
    ax2.bar(
        station_data_selected["Date"],
        station_data_selected["Rainfall"],
        color="tab:green",
        alpha=0.6,
        label=text["monthly_precip"],
    )
    ax2.set_ylabel(
        text["monthly_precip_axis"],
        color="tab:green",
    )
    ax2.tick_params(
        axis="y",
        labelcolor="tab:green",
    )

    month_name = station_data_selected["Date"].dt.strftime("%B").iloc[0]

    set_title(
        ax1,
        f"{text['monthly_title']} - {month_name} {selected_year}",
        fontsize=14,
    )

    ax1.xaxis.set_major_formatter(
        mdates.DateFormatter("%d")
    )
    ax1.xaxis.set_major_locator(
        mdates.DayLocator(interval=5)
    )
    set_grid(ax1)
    ax1.grid(
        False,
        which="minor",
    )

    fig.tight_layout()
    plt.xticks(rotation=45)

    lines_labels = [
        axis.get_legend_handles_labels()
        for axis in [ax1, ax2]
    ]
    lines, labels = [
        sum(label_group, [])
        for label_group in zip(*lines_labels)
    ]
    ax1.legend(
        lines,
        labels,
        loc="upper left",
    )

    if output_file is None:
        output_file = (
            "Monthly_Temperature_Rainfall_"
            f"{selected_month:02d}_{selected_year}_{language}.png"
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

    return None


def plot_monthly_interannual_variability(
    file,
    zone="Rainfall",
    season_name="ONDJFMA",
    season_dict=None,
    language="fr",
    output_file=None,
    show=True,
):
    """
    Plot July-June monthly stacked totals with a seasonal z-score panel.

    By default this function analyzes rainfall and computes the standardized
    anomaly over the ONDJFMA rainy season.
    """

    language, text = _get_translation(language)

    if season_dict is None:
        season_dict = {
            "ONDJFMA": [
                10,
                11,
                12,
                1,
                2,
                3,
                4,
            ],
        }

    set_plot_defaults()
    season_name = str(season_name).strip().upper()
    df = _read_station_data(
        file,
        {"Date", zone},
    )

    if season_name not in season_dict:
        raise ValueError(
            f"Unknown season {season_name!r}. "
            f"Choose one of: {', '.join(sorted(season_dict))}."
        )

    df = df.copy()
    df["Date"] = pd.to_datetime(
        df["Date"],
        format="%Y%m%d",
        errors="coerce",
    )
    df = df.dropna(
        subset=[
            "Date",
        ]
    )
    df = df.set_index("Date").sort_index()

    month_cycle = [
        7,
        8,
        9,
        10,
        11,
        12,
        1,
        2,
        3,
        4,
        5,
        6,
    ]
    month_names = [
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
    ]

    colors_map = {
        10: "skyblue",
        11: "dodgerblue",
        12: "blue",
        1: "navy",
        2: "darkviolet",
        3: "turquoise",
        4: "mediumspringgreen",
    }
    month_colors = [
        "lightgray"
        if month in [
            5,
            6,
            7,
            8,
            9,
        ]
        else colors_map.get(month, "lightgray")
        for month in month_cycle
    ]

    seasonal_data = []
    years_all = sorted(
        df.index.year.unique()
    )

    for year in years_all:
        if (year + 1) not in years_all:
            continue

        label = f"{year}-{year + 1}"
        monthly_sums = []

        for month in month_cycle:
            target_year = year if month >= 7 else year + 1
            value = df[
                (
                    df.index.year == target_year
                )
                & (
                    df.index.month == month
                )
            ][zone].sum()
            monthly_sums.append(
                float(value)
            )

        seasonal_data.append(
            (
                label,
                monthly_sums,
            )
        )

    if not seasonal_data:
        raise ValueError(
            "No complete July-June seasons found in station data."
        )

    fig = plt.figure(
        figsize=(12, 8),
    )
    grid = gridspec.GridSpec(
        2,
        1,
        height_ratios=[
            3,
            1,
        ],
        hspace=0.4,
    )

    ax1 = fig.add_subplot(
        grid[0]
    )
    ax2 = fig.add_subplot(
        grid[1]
    )

    xpos = np.arange(
        len(seasonal_data)
    )
    labels = [
        item[0]
        for item in seasonal_data
    ]

    for index, (_, monthly_sums) in enumerate(seasonal_data):
        bottom = 0.0

        for month_index, value in enumerate(monthly_sums):
            ax1.bar(
                xpos[index],
                value,
                bottom=bottom,
                color=month_colors[month_index],
                label=(
                    month_names[month_index]
                    if index == 0
                    else "_nolegend_"
                ),
                edgecolor="black",
                linewidth=0.5,
            )
            bottom += value

    ax1.set_ylabel(
        text["interannual_ylabel"],
        fontsize=10,
        fontweight="bold",
    )
    set_title(
        ax1,
        text["interannual_title"],
        fontsize=12,
    )
    set_grid(ax1)

    selected_months = season_dict[season_name]
    indices = [
        month_cycle.index(month)
        for month in selected_months
    ]
    seasonal_totals = [
        sum(
            monthly_sums[index]
            for index in indices
        )
        for _, monthly_sums in seasonal_data
    ]

    if len(seasonal_totals) > 1:
        anomalies = zscore(
            seasonal_totals,
            nan_policy="omit",
        )
        colors = [
            "#084594"
            if value >= 0
            else "#99000d"
            for value in anomalies
        ]

        ax2.bar(
            xpos,
            anomalies,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
        )
        ax2.axhline(
            0,
            color="black",
            linewidth=1,
        )
        ax2.set_ylabel(
            text["interannual_anomaly_ylabel"],
            fontsize=10,
        )
        set_grid(ax2)

    for axis in [
        ax1,
        ax2,
    ]:
        axis.set_xticks(
            xpos
        )
        axis.set_xticklabels(
            labels,
            rotation=90,
            fontsize=9,
        )
        axis.tick_params(
            direction="in",
            top=True,
            right=True,
        )

    handles, _ = ax1.get_legend_handles_labels()
    ax1.legend(
        handles=handles,
        title=text["interannual_legend_title"],
        bbox_to_anchor=(
            1.02,
            0.85,
        ),
        loc="upper left",
        frameon=True,
    )

    fig.subplots_adjust(
        right=0.82,
        hspace=0.4,
    )

    if output_file is None:
        output_file = (
            "Variation_interannual_"
            f"{zone}_{season_name}_{language}.png"
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

    return None
