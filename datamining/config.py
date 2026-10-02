"""Project-wide paths."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# Raw datasets
FERTILITY_RAW = RAW_DIR / "Dataset1.csv"
COVID_RAW = RAW_DIR / "Dataset2.csv"
CLIMATE_RAW = RAW_DIR / "Dataset3.csv"

# Processed datasets
FERTILITY_NORMALIZED = PROCESSED_DIR / "Dataset11.csv"
FERTILITY_DISCRETIZED = PROCESSED_DIR / "discretized.csv"
FERTILITY_CLUSTERING = PROCESSED_DIR / "new_dataset.csv"
COVID_CLEAN = PROCESSED_DIR / "Dataset2_clean.csv"
CLIMATE_CLEAN = PROCESSED_DIR / "Dataset3-filtered.csv"
