"""Disaster impact analysis."""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


GROUPING_COLUMNS = {
    "subgroup": "Disaster Subgroup",
    "disaster subgroup": "Disaster Subgroup",
    "type": "Disaster Type",
    "disaster type": "Disaster Type",
    "subtype": "Disaster Subtype",
    "disaster subtype": "Disaster Subtype",
}

DEFAULT_MISSING_VALUES = [
    -999,
    -99.9,
    -99,
    "NA",
    "N/A",
    "",
    "missing",
    "Missing",
    "MISSING",
    "nan",
    "NaN",
    "NULL",
    None,
]


def _country_from_file(file):
    stem = Path(file).stem
    parts = stem.split("_")
    return parts[-1].replace("-", " ").title() if parts else "Dataset"


def _resolve_grouping_column(analysis_by):
    key = str(analysis_by).strip().lower()
    if key not in GROUPING_COLUMNS:
        valid_options = ", ".join(sorted(GROUPING_COLUMNS))
        raise ValueError(f"analysis_by must be one of: {valid_options}")
    return GROUPING_COLUMNS[key]


def disaster_impact(
    file="./data/public_emdat_data_Madagascar.csv",
    analysis_by="type",
    disasters=None,
    start_year=2000,
    end_year=2025,
    title=None,
    country=None,
    output_file=None,
    show=True,
):
    """
    Plot disaster impacts by subgroup, type, or subtype using EMDAT data.

    Parameters
    ----------
    file : str or path-like, default "./data/public_emdat_data_Madagascar.csv"
        Path to the disaster CSV file.
    analysis_by : {"subgroup", "type", "subtype"}, default "type"
        Disaster level used for aggregation.
    disasters : str or list-like, optional
        One or more disaster names to keep from the selected analysis level.
        For example, use ``analysis_by="type", disasters="Storm"``.
    start_year, end_year : int, optional
        Start-year range to include. If omitted, all available years are used.
    title : str, optional
        Main plot title. If omitted, it is built from the filtered dataset.
    country : str, optional
        Country or dataset label used in the automatic title.
    output_file : str or path-like or None, default None
        Path where the plot is saved. Use None to skip saving.
    show : bool, default True
        If True, display the plot.

    Returns
    -------
    pandas.DataFrame
        Aggregated values used for the plot.
    """
    group_col = _resolve_grouping_column(analysis_by)
    df = pd.read_csv(file, na_values=DEFAULT_MISSING_VALUES)

    required_columns = {
        group_col,
        "Start Year",
        "End Year",
        "DisNo.",
        "Total Affected",
        "Total Damage ('000 US$)",
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required column(s): {missing_text}")

    df = df.copy()
    df["Start Year"] = pd.to_numeric(df["Start Year"], errors="coerce")
    df["End Year"] = pd.to_numeric(df["End Year"], errors="coerce")
    df["Total Affected"] = pd.to_numeric(df["Total Affected"], errors="coerce")
    df["Total Damage ('000 US$)"] = pd.to_numeric(
        df["Total Damage ('000 US$)"],
        errors="coerce",
    )
    df[group_col] = df[group_col].fillna("Unknown").astype(str).str.strip()
    df = df.dropna(subset=["Start Year"])
    df["End Year"] = df["End Year"].fillna(df["Start Year"])
    df["Start Year"] = df["Start Year"].astype(int)
    df["End Year"] = df["End Year"].astype(int)

    if start_year is not None:
        df = df[df["Start Year"] >= int(start_year)]
    if end_year is not None:
        df = df[df["Start Year"] <= int(end_year)]

    if disasters is not None:
        if isinstance(disasters, str):
            selected_disasters = {disasters.strip().lower()}
        else:
            selected_disasters = {
                str(disaster).strip().lower() for disaster in disasters
            }
        df = df[df[group_col].str.lower().isin(selected_disasters)]

    if df.empty:
        raise ValueError("No disaster records match the selected filters.")

    year_min = int(df[["Start Year", "End Year"]].min().min())
    year_max = int(df[["Start Year", "End Year"]].max().max())
    group_label = group_col.replace("Disaster ", "").lower()
    dataset_label = country if country is not None else _country_from_file(file)
    plot_title = title or (
        f"{dataset_label}: Impact by disaster {group_label} "
        f"({year_min}-{year_max})"
    )

    grouped = (
        df.groupby(group_col, dropna=False)
        .agg(
            Economic_Damage=(
                "Total Damage ('000 US$)",
                lambda values: values.sum(min_count=1) / 1000,
            ),
            Affected_Persons=(
                "Total Affected",
                lambda values: values.sum(min_count=1) / 1000,
            ),
            Event_Count=("DisNo.", "count"),
        )
        .reset_index()
    )

    grouped = grouped.sort_values(
        by=["Economic_Damage", "Affected_Persons", "Event_Count"],
        ascending=[True, True, True],
        na_position="first",
    )

    fig_height = max(5, 0.45 * len(grouped))
    fig, axes = plt.subplots(1, 3, figsize=(15, fig_height), sharey=True)
    fig.suptitle(
        plot_title,
        fontsize=15,
        fontweight="bold",
        y=0.98,
        color="#1e293b",
    )

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linestyle="-", alpha=0.25, color="#cbd5e1")
        ax.tick_params(colors="#475569")

    axes[0].barh(
        grouped[group_col],
        grouped["Economic_Damage"],
        color="#1e6091",
        height=0.65,
    )
    axes[0].set_title("Economic damage\n(USD Millions)", color="#1e293b", fontsize=11, pad=12)
    axes[0].set_xlabel("", fontweight="bold")

    axes[1].barh(
        grouped[group_col],
        grouped["Affected_Persons"],
        color="#e0a944",
        height=0.65,
    )
    axes[1].set_title("Affected persons\n(thousands)", color="#1e293b", fontsize=11, pad=12)
    axes[1].set_xlabel("", fontweight="bold")

    axes[2].barh(
        grouped[group_col],
        grouped["Event_Count"],
        color="#1d3557",
        height=0.65,
    )
    axes[2].set_title("Number of events", color="#1e293b", fontsize=11, pad=12)
    axes[2].set_xlabel("", fontweight="bold")

    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return grouped
