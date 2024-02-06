import utilities.unsupervised as unsup
from sklearn.metrics import silhouette_score
import pandas as pd

dataset = pd.read_csv("new_dataset.csv")
dataset = dataset.round(3)
reduced = unsup.data_to_data_2d(dataset)
dataset = pd.DataFrame(reduced)


def bench(dataset):
    best = 0
    best_eps = 0
    best_min_samples = 0
    for eps in range(1, 100):
        eps = eps / 100
        for min_samples in range(1, 20):
            min_samples = min_samples
            dbscan = unsup.DBSCAN(eps=eps, min_samples=min_samples)
            dbscan.fit(dataset)
            labels = dbscan.labels_
            try:
                silhouette_score(dataset, labels)
                if silhouette_score(dataset, labels) > best:
                    best = silhouette_score(dataset, labels)
                    best_eps = eps
                    best_min_samples = min_samples
                    print(
                        "best_eps: ",
                        best_eps,
                        " best_min_samples: ",
                        best_min_samples,
                        " best_silhouette_score: ",
                        best,
                    )
                # else:
                #     print(
                #         "eps: ",
                #         eps,
                #         " min_samples: ",
                #         min_samples,
                #         " silhouette_score: ",
                #         silhouette_score(dataset, labels),
                #     )
            except:
                continue
                # print(
                #     "eps: ",
                #     eps,
                #     " min_samples: ",
                #     min_samples,
                #     " silhouette_score: ",
                #     "error",
                # )
    print("best_eps: ", best_eps, " best_min_samples: ", best_min_samples)


bench(dataset)
