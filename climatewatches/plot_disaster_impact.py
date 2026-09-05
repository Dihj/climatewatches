from pathlib import Path
from calendar import monthrange
from io import BytesIO
import hashlib
import textwrap
import urllib.request

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.image as mpimg
from matplotlib.offsetbox import AnnotationBbox, OffsetImage


DEFAULT_MISSING_VALUES = [
    -999, -99.9, -99, "NA", "N/A", "", "missing", "Missing", 
    "MISSING", "nan", "NaN", "NULL", None,
]

DEFAULT_HAZARD_ICONS = {
    "drought": "DR",
    "storm": "TC",
    "tropical cyclone": "TC",
    "cyclone": "TC",
    "flood": "FL",
    "flood (general)": "FL",
    "riverine flood": "FL",
    "flash flood": "FF",
    "coastal flood": "CF",
    "epidemic": "EP",
    "viral disease": "VD",
    "bacterial disease": "BD",
    "infestation": "IN",
    "locust infestation": "LI",
    "mass movement (dry)": "MM",
    "rockfall (dry)": "RF",
}

DEFAULT_HAZARD_FORMATS = {
    "drought": {"marker": "*", "color": "#ca8a04"},
    "storm": {"marker": "o", "color": "#2563eb"},
    "tropical cyclone": {"marker": "o", "color": "#2563eb"},
    "cyclone": {"marker": "o", "color": "#2563eb"},
    "flood": {"marker": "v", "color": "#0284c7"},
    "flood (general)": {"marker": "v", "color": "#0284c7"},
    "riverine flood": {"marker": "v", "color": "#0369a1"},
    "flash flood": {"marker": "v", "color": "#0e7490"},
    "coastal flood": {"marker": "v", "color": "#0891b2"},
    "epidemic": {"marker": "P", "color": "#be123c"},
    "viral disease": {"marker": "P", "color": "#be123c"},
    "bacterial disease": {"marker": "P", "color": "#be123c"},
    "infestation": {"marker": "X", "color": "#65a30d"},
    "locust infestation": {"marker": "X", "color": "#65a30d"},
    "mass movement (dry)": {"marker": "^", "color": "#78716c"},
    "rockfall (dry)": {"marker": "^", "color": "#78716c"},
}

TWEMOJI_BASE_URL = "https://cdn.jsdelivr.net/gh/jdecked/twemoji@latest/assets/72x72"

DEFAULT_HAZARD_ICON_URLS = {
    "drought": f"{TWEMOJI_BASE_URL}/2600.png",
    "storm": f"{TWEMOJI_BASE_URL}/1f300.png",
    "tropical cyclone": f"{TWEMOJI_BASE_URL}/1f300.png",
    "cyclone": f"{TWEMOJI_BASE_URL}/1f300.png",
    "flood": f"{TWEMOJI_BASE_URL}/1f327.png",
    "flood (general)": f"{TWEMOJI_BASE_URL}/1f327.png",
    "riverine flood": f"{TWEMOJI_BASE_URL}/1f30a.png",
    "flash flood": f"{TWEMOJI_BASE_URL}/1f30a.png",
    "coastal flood": f"{TWEMOJI_BASE_URL}/1f30a.png",
    "epidemic": f"{TWEMOJI_BASE_URL}/1f9a0.png",
    "viral disease": f"{TWEMOJI_BASE_URL}/1f9a0.png",
    "bacterial disease": f"{TWEMOJI_BASE_URL}/1f9a0.png",
    "infestation": f"{TWEMOJI_BASE_URL}/1f997.png",
    "locust infestation": f"{TWEMOJI_BASE_URL}/1f997.png",
    "mass movement (dry)": f"{TWEMOJI_BASE_URL}/1faa8.png",
    "rockfall (dry)": f"{TWEMOJI_BASE_URL}/1faa8.png",
}


def _resolve_data_path(path):
    path = Path(path)
    if path.exists():
        return path
    repo_path = Path(__file__).resolve().parents[1] / path
    if repo_path.exists():
        return repo_path
    return path


def _event_date(row):
    year = pd.to_numeric(row.get("Start Year"), errors="coerce")
    if pd.isna(year):
        return pd.NaT

    month = pd.to_numeric(row.get("Start Month"), errors="coerce")
    day = pd.to_numeric(row.get("Start Day"), errors="coerce")

    month = 7 if pd.isna(month) else int(month)
    day = 15 if pd.isna(day) else int(day)
    month = min(max(month, 1), 12)
    day = min(max(day, 1), monthrange(int(year), month)[1])

    return pd.Timestamp(year=int(year), month=month, day=day)


def _hazard_icon(row, hazard_icons):
    for column in ("Disaster Subtype", "Disaster Type", "Disaster Subgroup"):
        value = row.get(column)
        if pd.isna(value):
            continue
        icon = hazard_icons.get(str(value).strip().lower())
        if icon:
            return icon
    return "!"


def _hazard_icon_url(row, icon_urls):
    for column in ("Disaster Subtype", "Disaster Type", "Disaster Subgroup"):
        value = row.get(column)
        if pd.isna(value):
            continue
        icon_url = icon_urls.get(str(value).strip().lower())
        if icon_url:
            return icon_url
    return None


def _hazard_label(row):
    for column in ("Event Name", "Disaster Subtype", "Disaster Type"):
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            return str(value).strip()
    return "Disaster"


def _wrap_label(value, width):
    return "\n".join(
        textwrap.wrap(
            str(value),
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def _hazard_format(row):
    for column in ("Disaster Subtype", "Disaster Type", "Disaster Subgroup"):
        value = row.get(column)
        if pd.isna(value):
            continue
        settings = DEFAULT_HAZARD_FORMATS.get(str(value).strip().lower())
        if settings:
            return settings
    return {"marker": "D", "color": "#7f1d1d"}


def _as_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


def _load_icon_image(url, cache_dir, timeout):
    if not url:
        return None

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(url.split("?", 1)[0]).suffix or ".png"
    cache_name = hashlib.sha256(url.encode("utf-8")).hexdigest() + suffix
    cache_path = cache_dir / cache_name

    try:
        if not cache_path.exists():
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "climatewatches/plot-disaster-impact"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                cache_path.write_bytes(response.read())

        with cache_path.open("rb") as image_file:
            return mpimg.imread(BytesIO(image_file.read()), format="png")
    except Exception:
        return None


def plot_disaster_impact(
    disaster_file="./data/public_emdat_data_Madagascar.csv",
    crop_file="./data/faostat_data.csv",
    item="Rice",
    element="Production",
    disaster_groups=("Natural",),
    disaster_subgroups=("Meteorological", "Hydrological", "Climatological"),
    start_year=None,
    end_year=None,
    max_events_per_year=1,
    label_wrap_width=13,
    use_online_icons=True,
    hazard_icons=None,
    hazard_icon_urls=None,
    icon_cache_dir=None,
    icon_zoom=0.15,
    icon_timeout=10,
    title=None,
    output_file=None,
    show=True,
):
    """Plot crop production vs disasters with tight hazard icon offsets and 5-year step labels."""
    disaster_file = _resolve_data_path(disaster_file)
    crop_file = _resolve_data_path(crop_file)
    hazard_icons = {
        **DEFAULT_HAZARD_ICONS,
        **{str(k).strip().lower(): v for k, v in (hazard_icons or {}).items()},
    }
    hazard_icon_urls = {
        **DEFAULT_HAZARD_ICON_URLS,
        **{str(k).strip().lower(): v for k, v in (hazard_icon_urls or {}).items()},
    }
    icon_cache_dir = Path(icon_cache_dir or "/tmp/climatewatches_icon_cache")
    icon_cache = {}

    df_disasters = pd.read_csv(disaster_file, na_values=DEFAULT_MISSING_VALUES)
    df_crops = pd.read_csv(crop_file, na_values=DEFAULT_MISSING_VALUES)

    # Clean & filter data
    df_crops["Year"] = pd.to_numeric(df_crops["Year"], errors="coerce")
    df_crops["Value"] = pd.to_numeric(df_crops["Value"], errors="coerce")
    df_disasters["Start Year"] = pd.to_numeric(df_disasters["Start Year"], errors="coerce")
    df_disasters["Total Affected"] = pd.to_numeric(df_disasters["Total Affected"], errors="coerce")
    df_disasters["Total Damage ('000 US$)"] = pd.to_numeric(df_disasters["Total Damage ('000 US$)"].fillna(0), errors="coerce")

    crop_mask = (
        df_crops["Item"].astype(str).str.contains(item, case=False, na=False)
        & df_crops["Element"].astype(str).str.contains(element, case=False, na=False)
    )
    crop_df = df_crops[crop_mask].dropna(subset=["Year"]).copy()
    crop_df["Year"] = crop_df["Year"].astype(int)
    rice_annual = crop_df.groupby("Year", as_index=False)["Value"].sum()

    disaster_mask = pd.Series(True, index=df_disasters.index)
    if disaster_groups:
        disaster_mask &= df_disasters["Disaster Group"].isin(_as_list(disaster_groups))
    if disaster_subgroups:
        disaster_mask &= df_disasters["Disaster Subgroup"].isin(_as_list(disaster_subgroups))

    disasters = df_disasters[disaster_mask].dropna(subset=["Start Year"]).copy()
    disasters["Start Year"] = disasters["Start Year"].astype(int)

    inferred_start = max(rice_annual["Year"].min(), disasters["Start Year"].min())
    inferred_end = min(rice_annual["Year"].max(), disasters["Start Year"].max())
    start_year = inferred_start if start_year is None else int(start_year)
    end_year = inferred_end if end_year is None else int(end_year)

    rice_annual = rice_annual[(rice_annual["Year"] >= start_year) & (rice_annual["Year"] <= end_year)].copy()
    disasters = disasters[(disasters["Start Year"] >= start_year) & (disasters["Start Year"] <= end_year)].copy()

    disaster_counts = (
        disasters.groupby("Start Year", as_index=False)["DisNo."]
        .count()
        .rename(columns={"Start Year": "Year", "DisNo.": "Disaster_Count"})
    )

    merged_df = pd.merge(rice_annual, disaster_counts, on="Year", how="left").fillna(0)
    merged_df["Plot_Date"] = pd.to_datetime(merged_df["Year"].astype(str) + "-07-01", format="%Y-%m-%d")

    disasters["Event_Date"] = disasters.apply(_event_date, axis=1)
    disasters = disasters.dropna(subset=["Event_Date"])
    disasters["Impact_Score"] = disasters["Total Affected"].fillna(0) + disasters["Total Damage ('000 US$)"].fillna(0)
    disasters = disasters.sort_values(["Start Year", "Impact_Score"], ascending=[True, False])

    if max_events_per_year is None:
        annotated_events = disasters.copy()
    else:
        annotated_events = disasters.groupby("Start Year", group_keys=False).head(int(max_events_per_year))
    annotated_events = annotated_events.sort_values("Event_Date").copy()

    # Moderate y-limit extension (compact vertical space)
    y_max_prod = merged_df["Value"].max() * 1.25 if merged_df["Value"].max() > 0 else 100
    plot_title = title or f"Madagascar: {item} {element.lower()} vs. hydrometeorological disasters ({start_year}-{end_year})"

    fig, ax1 = plt.subplots(figsize=(16, 8))

    # Production Line Plot
    color_line = "#15803d"
    ax1.plot(
        merged_df["Plot_Date"],
        merged_df["Value"],
        color=color_line,
        marker="o",
        linewidth=2.5,
        label=f"{item} {element.lower()}",
        zorder=3,
    )
    ax1.set_ylabel(f"{item} {element.lower()}", color=color_line, fontweight="bold", fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_line)
    ax1.set_ylim(0, y_max_prod)

    # 1. ONLY SHOW PRODUCTION VALUES EVERY 5 YEARS
    for _, row in merged_df.iterrows():
        yr = int(row["Year"])
        if yr % 5 == 0 and row["Value"] > 0:  # Filters for 2000, 2005, 2010, 2015, 2020, etc.
            ax1.annotate(
                f"{int(row['Value']):,}",
                (row["Plot_Date"], row["Value"]),
                textcoords="offset points",
                xytext=(0, -14),
                ha="center",
                fontsize=8,
                color=color_line,
                fontweight="bold",
                zorder=4,
            )

    # Disaster Count Bars
    ax2 = ax1.twinx()
    color_bars = "#1e3a8a"
    bars = ax2.bar(
        merged_df["Plot_Date"],
        merged_df["Disaster_Count"],
        color=color_bars,
        width=220,
        alpha=0.85,
        label="Disaster count",
        zorder=1,
    )
    ax2.set_ylabel("Number of hydrometeorological disasters", color=color_bars, fontweight="bold", fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_bars)

    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax2.annotate(
                f"{int(h)}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color_bars,
                fontweight="bold",
            )

    # 2. SMALL TIGHT OFFSETS FOR HAZARD ICONS & TEXT
    # Slightly lower heights: level 0.08 above line, alternating 0.15 for consecutive events
    level_heights = [0.08, 0.16]

    for i, (_, row) in enumerate(annotated_events.iterrows()):
        event_date = row["Event_Date"]
        year = int(row["Start Year"])
        prod_val = merged_df.loc[merged_df["Year"] == year, "Value"]
        if prod_val.empty:
            continue

        y_pos = float(prod_val.iloc[0])
        
        # Calculate tight icon height offset
        level_offset = level_heights[i % len(level_heights)]
        icon_y = y_pos + (y_max_prod * level_offset)
        
        icon_url = _hazard_icon_url(row, hazard_icon_urls) if use_online_icons else None
        icon_image = icon_cache.get(icon_url)
        if icon_url and icon_url not in icon_cache:
            icon_image = _load_icon_image(icon_url, icon_cache_dir, icon_timeout)
            icon_cache[icon_url] = icon_image

        event_name = _hazard_label(row)
        wrapped_name = _wrap_label(event_name, label_wrap_width)
        label_text = f"{year}\n{wrapped_name}"

        # Short dashed stem line pointing to icon base
        ax1.plot(
            [event_date, event_date],
            [y_pos, icon_y - (y_max_prod * 0.015)],
            color="#94a3b8",
            linestyle=":",
            linewidth=1.2,
            zorder=2,
        )

        # Draw icon
        if icon_image is None:
            hazard_format = _hazard_format(row)
            ax1.scatter(
                [event_date],
                [icon_y],
                marker=hazard_format["marker"],
                s=90,
                color=hazard_format["color"],
                edgecolor="white",
                zorder=10,
            )
        else:
            image_box = OffsetImage(icon_image, zoom=icon_zoom)
            icon_artist = AnnotationBbox(
                image_box,
                (event_date, icon_y),
                frameon=False,
                box_alignment=(0.5, 0.5),
                zorder=10,
            )
            ax1.add_artist(icon_artist)

        # Label directly above the icon with tight spacing
        ax1.text(
            event_date,
            icon_y + (y_max_prod * 0.025),
            label_text,
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
            color="#7f1d1d",
            linespacing=1.0,
            zorder=11,
        )

    # Date formatting & layout setup
    x_min = pd.Timestamp(year=start_year - 1, month=7, day=1)
    x_max = pd.Timestamp(year=end_year + 1, month=7, day=1)
    ax1.set_xlim(x_min, x_max)
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

    plt.title(plot_title, fontsize=13, fontweight="bold", pad=25, color="#1e293b")
    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    if output_file is not None:
        print(f"Plot created and saved: {output_file}")

    return None
