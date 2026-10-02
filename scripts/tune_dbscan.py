"""Grid-search DBSCAN's ``eps`` / ``min_samples`` by silhouette score.

Usage:  python scripts/tune_dbscan.py
"""

import _bootstrap  # noqa: F401
import pandas as pd
from sklearn.metrics import silhouette_score

from datamining import config
from datamining import unsupervised as unsup

EPS_VALUES = [e / 100 for e in range(1, 100)]
MIN_SAMPLES_VALUES = range(1, 20)


def tune(features: pd.DataFrame):
    best_score, best_params = 0.0, None
    for eps in EPS_VALUES:
        for min_samples in MIN_SAMPLES_VALUES:
            db = unsup.DBSCAN(eps=eps, min_samples=min_samples)
            db.fit(features)
            try:
                score = silhouette_score(features, db.labels_)
            except ValueError:  # fewer than 2 clusters
                continue
            if score > best_score:
                best_score, best_params = score, (eps, min_samples)
                print(f"eps={eps} min_samples={min_samples} silhouette={score:.4f}")
    return best_score, best_params


def main():
    data = pd.read_csv(config.FERTILITY_CLUSTERING).round(3)
    features = pd.DataFrame(unsup.data_to_data_2d(data))
    score, params = tune(features)
    print(f"\nBest: eps/min_samples={params} silhouette={score:.4f}")


if __name__ == "__main__":
    main()
