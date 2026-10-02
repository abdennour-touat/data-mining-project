"""Streamlit dashboard for the data mining project.

Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from datamining import config, supervised as sup, temporal, unsupervised as unsup
from datamining.association_rules import AssociationRules
from datamining.preprocessing import (
    boxPlots,
    centralTrends,
    correlation_matrix,
    filter_data,
    min_max,
    reduce_data_horizontal,
    reduce_data_vertical,
    replace_aberrant_data,
    replace_missing_values_all,
    seperate_missing_values,
    z_score,
)

SECTIONS = [
    "Preprocessing static data",
    "Temporal data analysis",
    "Association rules",
    "Supervised learning",
    "Unsupervised learning",
]
DISTANCES = ["Euclidean", "Manhattan", "Cosine", "Minkowski"]

# Distance functions expected by each module (the clustering code works on
# row-wise arrays, the KNN code on pairs of vectors).
KNN_DISTANCES = {
    "Euclidean": (sup.euclidean, {}),
    "Manhattan": (sup.manhattan, {}),
    "Cosine": (sup.cosine, {}),
    "Minkowski": (sup.minkowski, {"p": 3}),
}
CLUSTER_DISTANCES = {
    "Euclidean": unsup.euclidean2,
    "Manhattan": unsup.manhattan2,
    "Cosine": unsup.cosine2,
    "Minkowski": unsup.minkowski2,
}


@st.cache_data
def read_csv(path) -> pd.DataFrame:
    return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# Section: static data preprocessing
# --------------------------------------------------------------------------- #
def show_preprocessing():
    st.title("Preprocessing")
    st.write("Dataset 1")
    dataset = read_csv(config.FERTILITY_RAW).copy()
    st.write(dataset.head())

    def filtered():
        return filter_data(dataset.copy())

    if st.sidebar.button("Central trends"):
        st.write("### Dataset after filtration")
        df = filtered()
        st.write(df)
        st.write("### Central trends")
        st.write(centralTrends(df))

    if st.sidebar.button("Boxplot"):
        df = filtered()
        bp, outliers = boxPlots(df.copy())
        st.write("### Outliers")
        st.write(outliers)
        st.write("### Boxplot")
        for col in bp.columns:
            st.plotly_chart(px.box(df, y=col, title=f"Boxplot for {col}"))

    if st.sidebar.button("Histogram"):
        df = filtered()
        for col in df.columns:
            st.plotly_chart(px.histogram(df, x=col, title=f"Histogram for {col}"))

    if st.sidebar.button("Correlation Matrix"):
        df = filtered()
        corr_matrix = df.corr()
        st.title("Correlation Matrix")
        st.plotly_chart(
            px.imshow(
                corr_matrix,
                labels=dict(color="Correlation"),
                x=corr_matrix.columns,
                y=corr_matrix.index,
            )
        )
        _, max_corr, min_corr = correlation_matrix(corr_matrix)
        for title, pairs in (("Maximum", max_corr), ("Minimum", min_corr)):
            st.write(f"### {title} correlation")
            for x, y in pairs:
                st.plotly_chart(px.scatter(df, x=x, y=y))

    if st.sidebar.button("Missing values"):
        missing = seperate_missing_values(dataset)
        st.write("### Missing values")
        st.write(pd.DataFrame.from_dict(missing, orient="index"))
        st.write("### Replace missing values")
        st.write(replace_missing_values_all(dataset.copy(), dataset["Fertility"], missing))

    if st.sidebar.button("Aberrant data"):
        df = filtered()
        fertility = df.pop("Fertility")
        replaced, aberrant = replace_aberrant_data(df, fertility)
        st.write("### Aberrant data")
        st.write(pd.DataFrame.from_dict(aberrant, orient="index"))
        st.write("### Replaced data (linear regression)")
        st.write(replaced)

    if st.sidebar.button("Horizontal reduction"):
        st.write(reduce_data_horizontal(filtered()))

    if st.sidebar.button("Vertical reduction"):
        df, removed = reduce_data_vertical(filtered())
        st.write("### Removed columns")
        st.write(removed)
        st.write("### New dataset")
        st.write(df)

    new_min = st.sidebar.slider("New min", 0, 5, 0)
    new_max = st.sidebar.slider("New max", 0, 10, 1)
    if st.sidebar.button("Normalization"):
        features = filtered().drop(columns=["Fertility"])
        for title, fn in (
            ("min-max", lambda d: min_max(new_min, new_max, d)),
            ("z-score", z_score),
        ):
            df = fn(features.copy())
            df["Fertility"] = dataset["Fertility"]
            st.write(f"### Normalized dataset ({title})")
            st.write(df)


# --------------------------------------------------------------------------- #
# Section: temporal analysis
# --------------------------------------------------------------------------- #
def show_temporal():
    st.title("Temporal data analysis")
    data = read_csv(config.COVID_CLEAN).copy()
    st.write(data.head())

    if st.sidebar.button("Mean positive tests and case counts by ZCTA"):
        st.pyplot(temporal.plot_by_zcta(data, "mean"))
    if st.sidebar.button("Positive tests and case counts by week"):
        st.pyplot(temporal.plot_time_series(data, "W"))
    if st.sidebar.button("Positive tests and case counts by month"):
        st.pyplot(temporal.plot_time_series(data, "M"))
    if st.sidebar.button("Positive tests and case counts by year"):
        st.pyplot(temporal.plot_time_series(data, "Y"))
    if st.sidebar.button("Positive cases distribution by year and ZCTA"):
        st.pyplot(temporal.plot_positive_by_year_and_zcta(data))
    if st.sidebar.button("Test count by population"):
        st.pyplot(temporal.plot_tests_by_population(data))
    if st.sidebar.button("Top 5 impacted zones"):
        st.write(temporal.top_zones(data))


# --------------------------------------------------------------------------- #
# Section: association rules
# --------------------------------------------------------------------------- #
def show_association_rules():
    st.title("Association rules")
    base = read_csv(config.CLIMATE_CLEAN).copy()
    st.write(base.head())
    k = st.sidebar.slider("Number of classes", 0, 10, 0)
    supp_min = st.sidebar.slider("Min support", 0.1, 1.0, 0.0)
    conf_min = st.sidebar.slider("Min confidence / threshold", 0.001, 0.5, 0.001)

    def run(discretizer: str, rule_getter: str):
        dataset = base.copy()
        ar = AssociationRules(dataset)
        discretize = getattr(ar, discretizer)
        for col, prefix in (("Temperature", "Temp"), ("Humidity", "Hum"), ("Rainfall", "Rain")):
            dataset[col] = discretize(dataset, col, prefix, k)
        st.write("### Dataset after discretization")
        st.write(dataset)
        ar.setDataset(dataset)
        itemsets = ar.appriori(suppmin=supp_min)
        st.write("### Frequent itemsets")
        st.write(pd.DataFrame.from_dict(itemsets))
        rules = getattr(ar, rule_getter)(itemsets, confmin=conf_min)
        st.write("### Association rules")
        rows = [
            {"Itemset": itemset, "Rule": f"{lhs} -> {rhs}"}
            for itemset, itemset_rules in rules.items()
            for lhs, rhs in itemset_rules
        ]
        st.write(pd.DataFrame(rows, columns=["Itemset", "Rule"]))

    if st.sidebar.button("Equal frequency"):
        run("equal_frequency", "get_best_rules")
    if st.sidebar.button("Equal width"):
        run("equal_width", "get_best_rules")
    if st.sidebar.button("Strong rules (lift)"):
        run("equal_width", "get_best_rules_lift")
    if st.sidebar.button("Strong rules (cosine)"):
        run("equal_width", "get_best_rules_cosine")


# --------------------------------------------------------------------------- #
# Section: supervised learning
# --------------------------------------------------------------------------- #
def report(title, dataset, pred):
    st.write(f"### {title}")
    st.write(dataset.head())
    st.write("### Confusion matrix")
    matrix, metrics = sup.confusion_matrix1(pred["Fertility"], pred["Predicted"])
    st.write(matrix)
    st.write("### Metrics")
    st.write(metrics)


def show_supervised():
    st.title("Supervised learning")
    dataset = read_csv(config.FERTILITY_NORMALIZED)
    discrete = read_csv(config.FERTILITY_DISCRETIZED)
    train_ratio = st.sidebar.slider("Train ratio", 0.1, 1.0, 0.8)
    k = st.sidebar.slider("Number of neighbors", 1, 15, 0)
    distance = st.sidebar.selectbox("Distance function", DISTANCES)

    if st.sidebar.button("KNN"):
        train_X, train_Y, test_X, test_Y = sup.split_data(dataset, train_ratio)
        knn = sup.KNN(train_X, train_Y, test_X, test_Y, k)
        fn, kwargs = KNN_DISTANCES[distance]
        report("KNN", dataset, knn.fit(fn, **kwargs))

    pure_threshold = st.sidebar.slider("Pure threshold", 0.1, 1.0, 0.0)
    if st.sidebar.button("Decision Tree (discrete)"):
        train_X, train_Y, test_X, test_Y = sup.split_data(discrete, train_ratio)
        tree = sup.DecisionTree(pure_threshold=pure_threshold)
        tree.fit(train_X, train_Y)
        report("Decision Tree (discrete)", discrete, tree.predict_all(test_X, test_Y))

    min_split = st.sidebar.slider("Min split", 1, 10, 2)
    max_depth = st.sidebar.slider("Max depth", 1, 20, 10)
    alg = st.sidebar.selectbox("Algorithm", ["entropy", "gini"])
    if st.sidebar.button("Decision Tree (continuous)"):
        train_X, train_Y, test_X, test_Y = sup.split_data(dataset, train_ratio)
        tree = sup.DecisionTreeC(min_split=min_split, max_depth=max_depth, alg=alg)
        tree.fit(train_X, train_Y)
        report("Decision Tree (continuous)", dataset, tree.predict_all(test_X, test_Y))

    n_estimators = st.sidebar.slider("N estimators", 1, 100, 10)
    max_features = st.sidebar.slider("Max features", 1, 20, 10)
    rf_threshold = st.sidebar.slider("Threshold", 0.1, 1.0, 0.0)
    if st.sidebar.button("Random forest (discrete)"):
        train_X, train_Y, test_X, test_Y = sup.split_data(discrete, train_ratio)
        forest = sup.RandomForest(
            n_estimators=n_estimators,
            max_features=max_features,
            pure_threshold=rf_threshold,
        )
        forest.fit(train_X, train_Y)
        report("Random forest (discrete)", discrete, forest.predict_all(test_X, test_Y))
    if st.sidebar.button("Random forest (continuous)"):
        train_X, train_Y, test_X, test_Y = sup.split_data(dataset, train_ratio)
        forest = sup.RandomForestC(n_estimators=n_estimators, max_features=max_features)
        forest.fit(train_X, train_Y)
        report("Random forest (continuous)", dataset, forest.predict_all(test_X, test_Y))


# --------------------------------------------------------------------------- #
# Section: unsupervised learning
# --------------------------------------------------------------------------- #
def show_unsupervised():
    st.title("Unsupervised learning")
    k = st.sidebar.slider("Number of clusters", 1, 15, 0)
    randomize = st.sidebar.checkbox("Random")
    distance = st.sidebar.selectbox("Distance function", DISTANCES)
    distance_fn = CLUSTER_DISTANCES[distance]

    def prepare():
        raw = read_csv(config.FERTILITY_CLUSTERING).round(3)
        st.write(raw.head())
        return pd.DataFrame(unsup.data_to_data_2d(raw))

    if st.sidebar.button("Kmeans"):
        st.write("### Kmeans")
        features = prepare()
        kmeans = unsup.Kmeans(k, features, randomize)
        kmeans.cluster(distance_fn)
        st.write("### Clusters")
        reduced = PCA(n_components=2).fit_transform(features)
        fig = px.scatter(
            x=reduced[:, 0], y=reduced[:, 1], color=kmeans.labels_.astype(int)
        )
        for cx, cy in (c[:2] for c in kmeans.centroids):
            fig.add_scatter(
                x=np.array(cx),
                y=np.array(cy),
                mode="markers",
                marker=dict(color="red", size=10),
                name="Centroids",
            )
        st.plotly_chart(fig)
        st.write("### Silhouette score")
        st.write(silhouette_score(features, kmeans.labels_))

    eps = st.sidebar.slider("Epsilon", 0.1, 1.0, 0.0)
    min_samples = st.sidebar.slider("Min samples", 1, 100, 1)
    if st.sidebar.button("DBSCAN"):
        st.title("DBSCAN")
        features = prepare()
        db = unsup.DBSCAN(eps=eps, min_samples=min_samples, distFN=distance_fn)
        db.fit(features)
        reduced = PCA(n_components=2).fit_transform(features)
        st.plotly_chart(
            px.scatter(x=reduced[:, 0], y=reduced[:, 1], color=db.labels_.astype(int))
        )
        st.write("### Silhouette score")
        st.write(silhouette_score(features, db.labels_))


def main_ui():
    st.sidebar.header("Functionalities")
    selection = st.sidebar.selectbox("Select an option", SECTIONS)
    {
        SECTIONS[0]: show_preprocessing,
        SECTIONS[1]: show_temporal,
        SECTIONS[2]: show_association_rules,
        SECTIONS[3]: show_supervised,
        SECTIONS[4]: show_unsupervised,
    }[selection]()


if __name__ == "__main__":
    main_ui()
