"""Regenerate the cleaned COVID-19 (Dataset 2) and climate (Dataset 3) CSVs.

Usage:  python scripts/build_datasets.py
"""

import _bootstrap  # noqa: F401
from datamining import config
from datamining.datasets import build_clean_climate, build_clean_covid


def main():
    covid = build_clean_covid()
    print(f"COVID-19: {len(covid)} rows -> {config.COVID_CLEAN}")
    climate = build_clean_climate()
    print(f"Climate:  {len(climate)} rows -> {config.CLIMATE_CLEAN}")


if __name__ == "__main__":
    main()
