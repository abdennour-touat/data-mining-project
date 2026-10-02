"""Cleaning and plotting helpers for the COVID-19 temporal dataset (Dataset 2)."""

import matplotlib.pyplot as plt
import pandas as pd

from .preprocessing import quartile

COUNT_COLUMNS = ["case count", "test count", "positive tests"]
RATE_COLUMNS = ["case rate", "test rate", "positivity rate"]
NUMERIC_COLUMNS = COUNT_COLUMNS + RATE_COLUMNS
DATE_COLUMNS = ["Start date", "end date"]
FULL_DATE_FORMAT = "%m/%d/%Y"


def _period_year_ranges(raw: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Min/max ``time_period`` observed per year, from rows with a full date.

    Rows whose date lacks a year (e.g. ``5-Apr``) are assigned a year by
    checking which range their ``time_period`` falls into. The partial year 2019
    is excluded because it only contains a handful of periods.
    """
    parsed = pd.to_datetime(raw[date_col], format=FULL_DATE_FORMAT, errors="coerce")
    ranges = raw.groupby(parsed.dt.year)["time_period"].agg(["min", "max"])
    return ranges.drop(index=2019, errors="ignore")


def fix_dates(raw: pd.DataFrame, date_col: str) -> pd.Series:
    """Parse a date column, inferring the year of ``d-Mon`` entries.

    Args:
        raw: Raw COVID-19 dataframe.
        date_col: Name of the date column to repair.

    Returns:
        A datetime Series (``NaT`` where the date cannot be recovered).
    """
    ranges = _period_year_ranges(raw, date_col)
    result = pd.to_datetime(raw[date_col], format=FULL_DATE_FORMAT, errors="coerce")

    for idx in result.index[result.isna()]:
        text = raw.at[idx, date_col]
        if not isinstance(text, str) or not text.strip():
            continue
        period = raw.at[idx, "time_period"]
        for year, row in ranges.iterrows():
            if row["min"] <= period <= row["max"]:
                result.at[idx] = pd.to_datetime(
                    f"{text.strip()}-{int(year)}", format="%d-%b-%Y", errors="coerce"
                )
    return result


def fill_missing_with_mean(df: pd.DataFrame, columns) -> pd.DataFrame:
    """Replace missing values of ``columns`` with the column mean."""
    for col in columns:
        df[col] = df[col].fillna(df[col].mean())
    return df


def replace_outliers_with_mean(df: pd.DataFrame, columns) -> pd.DataFrame:
    """Replace values outside the 1.5 * IQR fences with the column mean."""
    for col in columns:
        _, q1, _, q3, _ = quartile(df[col])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mean = df[col].mean()
        outliers = (df[col] < lower) | (df[col] > upper)
        df.loc[outliers, col] = mean
    return df


def clean_covid_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline: repair dates, impute gaps, treat outliers."""
    df = raw.copy(deep=True)
    for col in DATE_COLUMNS:
        df[col] = fix_dates(raw, col)
    df = fill_missing_with_mean(df, NUMERIC_COLUMNS)
    df = replace_outliers_with_mean(df, NUMERIC_COLUMNS)
    return df


# --------------------------------------------------------------------------- #
# Aggregations
# --------------------------------------------------------------------------- #
_FREQ_LABELS = {"W": "Weekly", "M": "Monthly", "Y": "Yearly"}


def aggregate_by_period(data: pd.DataFrame, freq: str, zcta=None) -> pd.DataFrame:
    """Sum tests/cases per period (``"W"``, ``"M"`` or ``"Y"``), optionally for one ZCTA."""
    df = data.copy()
    if zcta is not None:
        df = df[df["zcta"] == zcta]
    dates = pd.to_datetime(df["Start date"], errors="coerce")
    df["Start date"] = dates.dt.to_period(freq).dt.to_timestamp()
    return (
        df.groupby("Start date")[COUNT_COLUMNS].sum().reset_index()
    )


def top_zones(data: pd.DataFrame, n: int = 5) -> pd.Series:
    """The ``n`` ZCTAs with the most positive tests."""
    return data.groupby("zcta")["positive tests"].sum().nlargest(n)


# --------------------------------------------------------------------------- #
# Plots (all return a matplotlib Figure)
# --------------------------------------------------------------------------- #
def plot_by_zcta(data: pd.DataFrame, agg: str = "mean") -> plt.Figure:
    """Bar chart of positive tests vs. case count per ZCTA."""
    pt = data.groupby("zcta")["positive tests"].agg(agg)
    cc = data.groupby("zcta")["case count"].agg(agg)
    fig, ax = plt.subplots()
    pt.plot(kind="bar", ax=ax, color="blue", width=0.4, position=1)
    cc.plot(kind="bar", ax=ax, color="red", width=0.4, position=0)
    ax.set_xlabel("zcta")
    ax.set_ylabel(f"{agg.capitalize()} value")
    ax.set_title(f"{agg.capitalize()} Positive Tests and Case Count by ZCTA")
    ax.legend(["Positive Tests", "Case Count"])
    return fig


def plot_time_series(data: pd.DataFrame, freq: str, zcta=None) -> plt.Figure:
    """Line chart of tests, cases and positives aggregated per period."""
    agg = aggregate_by_period(data, freq, zcta)
    label = _FREQ_LABELS[freq]
    fig, ax = plt.subplots(figsize=(10, 6))
    for col, name in zip(COUNT_COLUMNS, ["Case Counts", "Test Counts", "Positive Tests"]):
        ax.plot(agg["Start date"], agg[col], label=name, marker="o")
    ax.set_xticks(agg["Start date"])
    ax.set_xticklabels(
        agg["Start date"].dt.strftime("%Y-%m-%d"), rotation=45, ha="right"
    )
    ax.set_xlabel(label.replace("ly", ""))
    ax.set_ylabel("Counts")
    ax.set_title(f"{label} Positive Tests and Case Counts")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_positive_by_year_and_zcta(data: pd.DataFrame) -> plt.Figure:
    """Stacked bar chart of positive tests per year, split by ZCTA."""
    df = data.copy()
    df["year"] = pd.to_datetime(df["Start date"], errors="coerce").dt.year
    pivot = df.pivot_table(
        values="positive tests", index="year", columns="zcta", aggfunc="sum"
    )
    fig, ax = plt.subplots()
    pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_xlabel("Year")
    ax.set_ylabel("Positive Cases")
    ax.set_title("Positive Cases by Year and ZCTA")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    return fig


def plot_tests_by_population(data: pd.DataFrame) -> plt.Figure:
    """Bar chart of total test count per population size."""
    tests = data.groupby("population")["test count"].sum()
    fig, ax = plt.subplots()
    tests.plot(kind="bar", ax=ax, color="blue", width=0.4)
    ax.set_xlabel("population")
    ax.set_ylabel("Test Count")
    ax.set_title("Test Count by Population")
    return fig
