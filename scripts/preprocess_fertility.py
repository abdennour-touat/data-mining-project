"""Run the full preprocessing pipeline on the fertility dataset (Dataset 1).

Steps: missing-value imputation (decision tree), outlier replacement (linear
regression), horizontal/vertical reduction, then min-max and z-score scaling.

Usage:  python scripts/preprocess_fertility.py
"""

import _bootstrap  # noqa: F401
from datamining.datasets import load_fertility
from datamining.preprocessing import (
    filter_data,
    min_max,
    reduce_data_horizontal,
    reduce_data_vertical,
    replace_aberrant_data,
    replace_missing_values_all,
    seperate_missing_values,
    z_score,
)


def main():
    raw = load_fertility()
    fertility = raw["Fertility"]

    missing = seperate_missing_values(raw)
    print("Missing values per column:", {k: len(v) for k, v in missing.items() if len(v)})
    imputed = replace_missing_values_all(raw.copy(), fertility, missing)

    features = filter_data(imputed).drop(columns=["Fertility"])
    replaced, aberrant = replace_aberrant_data(features, fertility)
    print("Outliers replaced per column:", {k: len(v) for k, v in aberrant.items()})

    reduced = reduce_data_horizontal(replaced.copy())
    reduced, dropped = reduce_data_vertical(reduced)
    print("Redundant columns removed:", dropped)

    print("\nMin-max normalised (head):")
    print(min_max(0, 1, reduced.copy()).head())
    print("\nZ-score normalised (head):")
    print(z_score(reduced.copy()).head())


if __name__ == "__main__":
    main()
