"""Annual climate analysis."""

import pandas as pd
import matplotlib.pyplot as plt


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


def annual_trend(
    file,
    date_col="Date",
    temp_col="Tmean",
    rainfall_col="Rainfall",
    date_format="%Y%m%d",
    missing_values=-99.9,
    smooth=True,
    smooth_year=5,
    title="Climate trends",
    output_file="annual_valueTrend.png",
    show=True,
):
    """
    Plot annual mean temperature and annual total rainfall trends.

    Parameters
    ----------
    file : str or path-like
        Path to the station CSV file.
    date_col : str, default "Date"
        Date column name.
    temp_col : str, default "Tmean"
        Mean temperature column name.
    rainfall_col : str, default "Rainfall"
        Rainfall column name.
    date_format : str or None, default "%Y%m%d"
        Date format used by ``pandas.to_datetime``. Use None to infer dates.
    missing_values : list-like or scalar, optional
        Extra missing-value markers. Common markers are handled by default.
    smooth : bool, default True
        If True, add rolling average lines.
    smooth_year : int, default 5
        Rolling average window size in years.
    title : str, default "Climate trends"
        Main title displayed above the plot.
    output_file : str or path-like or None, default "annual_valueTrend.png"
        Path where the plot is saved. Use None to skip saving.
    show : bool, default True
        If True, display the plot.

    Returns
    -------
    pandas.DataFrame
        Annual values used for the plot.
    """
    if smooth_year < 1:
        raise ValueError("smooth_year must be at least 1.")

    na_values = list(DEFAULT_MISSING_VALUES)
    if missing_values is not None:
        if isinstance(missing_values, (list, tuple, set)):
            na_values.extend(missing_values)
        else:
            na_values.append(missing_values)

    df = pd.read_csv(file, na_values=na_values)

    required_columns = {date_col, temp_col, rainfall_col}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required column(s): {missing_text}")

    df = df[[date_col, temp_col, rainfall_col]].copy()
    df[date_col] = pd.to_datetime(
        df[date_col].astype("string"),
        format=date_format,
        errors="coerce",
    )
    df[temp_col] = pd.to_numeric(df[temp_col], errors="coerce")
    df[rainfall_col] = pd.to_numeric(df[rainfall_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    if df.empty:
        raise ValueError("No valid dates found after removing missing values.")

    df["Year"] = df[date_col].dt.year

    annual_df = (
        df.groupby("Year", as_index=False)
        .agg(
            Mean_Temp=(temp_col, "mean"),
            Total_Rainfall=(rainfall_col, lambda values: values.sum(min_count=1)),
        )
        .sort_values("Year")
        .reset_index(drop=True)
    )

    if annual_df[["Mean_Temp", "Total_Rainfall"]].isna().all().all():
        raise ValueError("No valid temperature or rainfall values found.")

    if smooth:
        annual_df["Temp_Rolling_Avg"] = (
            annual_df["Mean_Temp"].rolling(window=smooth_year, min_periods=1).mean()
        )
        annual_df["Rain_Rolling_Avg"] = (
            annual_df["Total_Rainfall"].rolling(window=smooth_year, min_periods=1).mean()
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    fig.suptitle(
        title,
        fontsize=16,
        fontweight="bold",
        y=0.98,
        color="#1e293b",
    )

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="-", alpha=0.3, color="#cbd5e1")
        ax.tick_params(colors="#475569")

    axes[0].plot(
        annual_df["Year"],
        annual_df["Mean_Temp"],
        color="#93c5fd",
        linewidth=1,
        alpha=0.75,
        label="Annual mean",
    )
    if smooth:
        axes[0].plot(
            annual_df["Year"],
            annual_df["Temp_Rolling_Avg"],
            color="#1e3a8a",
            linewidth=2.5,
            label=f"{smooth_year}-yr rolling avg",
        )

    axes[0].set_title("Mean annual temperature (deg C)", color="#1e293b", pad=12)
    axes[0].set_ylabel("deg C", fontweight="bold", color="#475569")
    axes[0].legend(frameon=False, loc="upper right")

    axes[1].plot(
        annual_df["Year"],
        annual_df["Total_Rainfall"],
        color="#fde68a",
        linewidth=1,
        alpha=0.85,
        label="Annual total",
    )
    if smooth:
        axes[1].plot(
            annual_df["Year"],
            annual_df["Rain_Rolling_Avg"],
            color="#78350f",
            linewidth=2.5,
            label=f"{smooth_year}-yr rolling avg",
        )

    axes[1].set_title("Total annual precipitation (mm)", color="#1e293b", pad=12)
    axes[1].set_ylabel("mm", fontweight="bold", color="#475569")
    axes[1].legend(frameon=False, loc="upper left")

    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file, dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return annual_df
