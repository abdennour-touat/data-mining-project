"""Loading and cleaning of the three raw datasets."""

import pandas as pd

from . import config
from .temporal import clean_covid_data

CLIMATE_NUMERIC_COLUMNS = ["Temperature", "Humidity", "Rainfall"]


def load_fertility(path=config.FERTILITY_RAW) -> pd.DataFrame:
    """Dataset 1: soil properties and fertility class."""
    return pd.read_csv(path)


def load_covid(path=config.COVID_RAW) -> pd.DataFrame:
    """Dataset 2 (raw): COVID-19 cases per ZIP code tabulation area."""
    return pd.read_csv(path)


def load_climate(path=config.CLIMATE_RAW) -> pd.DataFrame:
    """Dataset 3: climate, soil and crop. Decimal commas are converted to floats
    and missing numeric values are replaced by the column mean."""
    df = pd.read_csv(path)
    for col in CLIMATE_NUMERIC_COLUMNS:
        df[col] = df[col].astype(str).str.replace(",", ".").astype(float)
        df[col] = df[col].fillna(df[col].mean())
    return df


def build_clean_covid(raw_path=config.COVID_RAW, out_path=config.COVID_CLEAN):
    """Clean Dataset 2 and write the result to ``out_path``."""
    clean = clean_covid_data(load_covid(raw_path))
    clean.to_csv(out_path, index=False)
    return clean


def build_clean_climate(raw_path=config.CLIMATE_RAW, out_path=config.CLIMATE_CLEAN):
    """Clean Dataset 3 and write the result to ``out_path``."""
    clean = load_climate(raw_path)
    clean.to_csv(out_path, index=False)
    return clean
