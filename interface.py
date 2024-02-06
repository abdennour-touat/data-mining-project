import streamlit as st
import pandas as pd
from utilities.preprocessing import *
from utilities.associationRules import *
from utilities.supervised import *
from utilities.unsupervised import *
import plotly.express as px
from sklearn.metrics import silhouette_score


def main_ui():
    st.sidebar.header("Functionalities ")
    selections = st.sidebar.selectbox(
        "Select an option",
        [
            "Preprocessing static data",
            "Temporal data analysis",
            "Association rules",
            "Suprvised learning",
            "unsupervisesd learning",
        ],
        on_change=None,
    )

    if selections == "Preprocessing static data":
        st.title("Preprocessing")
        st.write("Dataset 1")
        # upload the dataset and show the head
        dataset = pd.read_csv("Dataset1.csv")
        st.write(dataset.head())
        if st.sidebar.button("Central trends"):
            try:
                dataset = filter_data(dataset)
                st.write("### dataset after filtration")
                st.write(dataset)
                result = centralTrends(dataset)
                st.write("### central trends")
                st.write(result)
            except Exception as e:
                st.write(e)
        if st.sidebar.button("Boxplot"):
            dataset = filter_data(dataset)
            bp, outliers = boxPlots(dataset)
            st.write("### outliers")
            st.write(outliers)
            st.write("### Boxplot")
            for col in bp.columns:
                fig = px.box(dataset, y=col, title=f"Boxplot for {col}")
                st.plotly_chart(fig)
            # st.show()
        if st.sidebar.button("Histogram"):
            df = filter_data(dataset)
            for col in df.columns:
                fig = px.histogram(df, x=col, title=f"Histogram for {col}")
                st.plotly_chart(fig)
        if st.sidebar.button("Correlation Matrix"):
            df = filter_data(dataset)
            # Create a correlation matrix
            correlationMatrix = df.corr()
            # Create a Streamlit app
            st.title("Correlation Matrix")
            # Plot the correlation matrix using Plotly Express
            fig = px.imshow(
                correlationMatrix,
                labels=dict(color="Correlation"),
                x=correlationMatrix.columns,
                y=correlationMatrix.index,
            )
            st.plotly_chart(fig)
            st.write("### maximum and minimum correlation")
            corr, max_corr, min_corr = correlation_matrix(correlationMatrix)
            st.write("### maximum correlation")
            for key in max_corr.keys():
                x, y = key
                fig = px.scatter(df, x=x, y=y)
                st.plotly_chart(fig)
            st.write("### minimum correlation")
            for key in min_corr.keys():
                x, y = key
                fig = px.scatter(df, x=x, y=y)
                st.plotly_chart(fig)
        if st.sidebar.button("Missing values"):
            missing_data = seperate_missing_values(dataset)
            st.write("### missing values")
            df = pd.DataFrame.from_dict(missing_data, orient="index")
            st.write(df)
            st.write("### replace missing values")
            c_data = replace_missing_values_all(
                dataset, dataset["Fertility"], missing_data
            )
            st.write(c_data)
        if st.sidebar.button("Aberrant data"):
            df = filter_data(dataset)
            fertility = df["Fertility"]
            df = df.drop(columns=["Fertility"])
            replaced_data, abberant_data = replace_aberrant_data(df, fertility)
            st.write("### Abberant data")
            # convert the aberrant data to dataframe, the lengths of the columns are not equal
            abberant_data = pd.DataFrame.from_dict(abberant_data, orient="index")
            st.write(abberant_data)
            st.write("### Replaced data (using the Linear regression model)")
            st.write(replaced_data)
        if st.sidebar.button("Horizontal reduction"):
            df = filter_data(dataset)
            df = reduce_data_horizontal(df)
            st.write(df)
        if st.sidebar.button("Vertical reduction"):
            df = filter_data(dataset)
            df, corr_cols = reduce_data_vertical(df)
            st.write("### Removed columns")
            st.write(corr_cols)
            st.write("### New dataset")
            st.write(df)
        new_min = st.sidebar.slider("New min", 0, 5, 0)
        new_max = st.sidebar.slider("New max", 0, 10, 1)
        if st.sidebar.button("Normalization"):
            df = filter_data(dataset)
            df = min_max(new_min, new_max, df.drop(columns=["Fertility"]))
            # add the fertility column
            df["Fertility"] = dataset["Fertility"]
            st.write("### Normalized dataset (min max)")
            st.write(df)
            df = filter_data(dataset)
            df = z_score(df.drop(columns=["Fertility"]))
            # add the fertility column
            df["Fertility"] = dataset["Fertility"]
            st.write("### Normalized dataset (z score)")
            st.write(df)

    elif selections == "Association rules":
        st.title("Association rules")
        dataset = pd.read_csv("Dataset3-filtered.csv")
        st.write(dataset.head())
        k = st.sidebar.slider("Number of classes", 0, 10, 0)
        supp_min = st.sidebar.slider("Min support", 0.1, 1.0, 0.0)
        conf_min = st.sidebar.slider("Min Confidence", 0.001, 0.5, 0.001)
        if st.sidebar.button("Equal frequency"):
            ar = AssociationRules(dataset)
            dataset["Temperature"] = ar.equal_frequency(
                dataset, "Temperature", "Temp", k
            )
            dataset["Humidity"] = ar.equal_frequency(dataset, "Humidity", "Hum", k)
            dataset["Rainfall"] = ar.equal_frequency(dataset, "Rainfall", "Rain", k)
            st.write("### Dataset after discretization")
            st.write(dataset)
            ar.setDataset(dataset)
            l = ar.appriori(suppmin=supp_min)
            # convert to dataframe, the lengths aren't the same
            ldf = pd.DataFrame.from_dict(l)
            st.write("### Frequent motifs")
            st.write(ldf)
            br = ar.get_best_rules(l, confmin=conf_min)
            st.write("### Association rules")
            df = pd.DataFrame(columns=["Key", "Value"])
            for key, value in br.items():
                for tuple_value in value:
                    df = pd.concat(
                        [
                            df,
                            pd.DataFrame(
                                {
                                    "Key": [key],
                                    "Value": f"{tuple_value[0]}->{tuple_value[1]}",
                                }
                            ),
                        ],
                        ignore_index=True,
                    )
            st.write(df)

        if st.sidebar.button("Equal width"):
            ar = AssociationRules(dataset)
            dataset["Temperature"] = ar.equal_width(dataset, "Temperature", "Temp", k)
            dataset["Humidity"] = ar.equal_width(dataset, "Humidity", "Hum", k)
            dataset["Rainfall"] = ar.equal_width(dataset, "Rainfall", "Rain", k)
            st.write("### Dataset after discretization")
            st.write(dataset)
            ar.setDataset(dataset)
            l = ar.appriori(suppmin=supp_min)
            # convert to dataframe, the lengths aren't the same
            ldf = pd.DataFrame.from_dict(l)
            st.write("### Frequent motifs")
            st.write(ldf)
            br = ar.get_best_rules(l, confmin=conf_min)
            st.write("### Association rules")
            df = pd.DataFrame(columns=["Key", "Value"])
            for key, value in br.items():
                for tuple_value in value:
                    df = pd.concat(
                        [
                            df,
                            pd.DataFrame(
                                {
                                    "Key": [key],
                                    "Value": f"{tuple_value[0]}->{tuple_value[1]}",
                                }
                            ),
                        ],
                        ignore_index=True,
                    )

            st.write(df)
        if st.sidebar.button("Strong association rules with lift method"):
            ar = AssociationRules(dataset)
            dataset["Temperature"] = ar.equal_width(dataset, "Temperature", "Temp", k)
            dataset["Humidity"] = ar.equal_width(dataset, "Humidity", "Hum", k)
            dataset["Rainfall"] = ar.equal_width(dataset, "Rainfall", "Rain", k)
            st.write("### Dataset after discretization")
            st.write(dataset)
            ar.setDataset(dataset)
            l = ar.appriori(suppmin=supp_min)
            # convert to dataframe, the lengths aren't the same
            ldf = pd.DataFrame.from_dict(l)
            st.write("### Frequent motifs")
            st.write(ldf)
            br = ar.get_best_rules_lift(l, confmin=conf_min)
            st.write("### Association rules")
            df = pd.DataFrame(columns=["Key", "Value"])
            for key, value in br.items():
                for tuple_value in value:
                    df = pd.concat(
                        [
                            df,
                            pd.DataFrame(
                                {
                                    "Key": [key],
                                    "Value": f"{tuple_value[0]}->{tuple_value[1]}",
                                }
                            ),
                        ],
                        ignore_index=True,
                    )

            st.write(df)
        if st.sidebar.button("Strong association rules with cosine method"):
            ar = AssociationRules(dataset)
            dataset["Temperature"] = ar.equal_width(dataset, "Temperature", "Temp", k)
            dataset["Humidity"] = ar.equal_width(dataset, "Humidity", "Hum", k)
            dataset["Rainfall"] = ar.equal_width(dataset, "Rainfall", "Rain", k)
            st.write("### Dataset after discretization")
            st.write(dataset)
            ar.setDataset(dataset)
            l = ar.appriori(suppmin=supp_min)
            # convert to dataframe, the lengths aren't the same
            ldf = pd.DataFrame.from_dict(l)
            st.write("### Frequent motifs")
            st.write(ldf)
            br = ar.get_best_rules_cosine(l, confmin=conf_min)
            st.write("### Association rules")
            df = pd.DataFrame(columns=["Key", "Value"])
            for key, value in br.items():
                for tuple_value in value:
                    df = pd.concat(
                        [
                            df,
                            pd.DataFrame(
                                {
                                    "Key": [key],
                                    "Value": f"{tuple_value[0]}->{tuple_value[1]}",
                                }
                            ),
                        ],
                        ignore_index=True,
                    )

            st.write(df)

    elif selections == "Temporal data analysis":
        st.title("Temporal data analysis")
        data = pd.read_csv("Dataset2_clean.csv")
        st.write(data.head())
        if st.sidebar.button("Mean positive tests adn case counts by ZCTA"):
            pr = data.groupby("zcta")["positive tests"].mean()
            cc = data.groupby("zcta")["case count"].mean()
            # Create a new figure and a set of subplots
            fig, ax = plt.subplots()

            # Plot the 'positive tests' means
            pr.plot(kind="bar", ax=ax, color="blue", width=0.4, position=1)

            # Plot the 'case count' means
            cc.plot(kind="bar", ax=ax, color="red", width=0.4, position=0)

            # Set the x-label and y-label
            ax.set_xlabel("zcta")
            ax.set_ylabel("Mean Value")

            # Set the title
            ax.set_title("Mean Positive Tests and Case Count by ZCTA")

            # Show the legend
            ax.legend(["Positive Tests", "Case Count"])

            # Show the plot
            st.pyplot(fig)

        if st.sidebar.button("Mean positive tests adn case counts by week"):
            data["Start date"] = pd.to_datetime(data["Start date"], errors="coerce")

            # Convert 'Date' to weekly frequency
            data["Start date"] = data["Start date"].dt.to_period("W").dt.to_timestamp()
            # Group by 'Week' and aggregate sum
            weekly_data = (
                data.groupby("Start date")
                .agg(
                    {"positive tests": "sum", "case count": "sum", "test count": "sum"}
                )
                .reset_index()
            )

            # Plotting
            fig, ax = plt.subplots(figsize=(10, 6))

            # Line plot for Positive Tests
            ax.plot(
                weekly_data["Start date"],
                weekly_data["positive tests"],
                label="Positive Tests",
                marker="o",
            )

            # Line plot for Case Counts
            ax.plot(
                weekly_data["Start date"],
                weekly_data["case count"],
                label="Case Counts",
                marker="o",
            )
            ax.plot(
                weekly_data["Start date"],
                weekly_data["test count"],
                label="Test Counts",
                marker="o",
            )

            # X-axis labels and title
            ax.set_xticks(weekly_data["Start date"])
            ax.set_xticklabels(
                weekly_data["Start date"].dt.strftime("%Y-%m-%d"),
                rotation=45,
                ha="right",
            )
            ax.set_xlabel("Week")
            ax.set_ylabel("Counts")
            ax.set_title("Weekly Positive Tests and Case Counts")

            # Legend
            ax.legend()

            # Show the plot
            plt.tight_layout()
            st.pyplot(fig)
        if st.sidebar.button("Mean positive tests adn case counts by month"):
            data["Start date"] = pd.to_datetime(data["Start date"], errors="coerce")

            # Convert 'Date' to monthly frequency
            data["Start date"] = data["Start date"].dt.to_period("M").dt.to_timestamp()
            # Group by 'Month' and aggregate sum
            monthly_data = (
                data.groupby("Start date")
                .agg(
                    {"positive tests": "sum", "case count": "sum", "test count": "sum"}
                )
                .reset_index()
            )

            # Plotting
            fig, ax = plt.subplots(figsize=(10, 6))

            # Line plot for Positive Tests
            ax.plot(
                monthly_data["Start date"],
                monthly_data["positive tests"],
                label="Positive Tests",
                marker="o",
            )

            # Line plot for Case Counts
            ax.plot(
                monthly_data["Start date"],
                monthly_data["case count"],
                label="Case Counts",
                marker="o",
            )
            ax.plot(
                monthly_data["Start date"],
                monthly_data["test count"],
                label="Test Counts",
                marker="o",
            )

            # X-axis labels and title
            ax.set_xticks(monthly_data["Start date"])
            ax.set_xticklabels(
                monthly_data["Start date"].dt.strftime("%Y-%m-%d"),
                rotation=45,
                ha="right",
            )
            ax.set_xlabel("Month")
            ax.set_ylabel("Counts")
            ax.set_title("Monthly Positive Tests and Case Counts")

            # Legend
            ax.legend()

            # Show the plot
            plt.tight_layout()
            st.pyplot(fig)
        if st.sidebar.button("Mean positive tests and case counts by year"):
            data["Start date"] = pd.to_datetime(data["Start date"], errors="coerce")

            # Convert 'Date' to yearly frequency
            data["Start date"] = data["Start date"].dt.to_period("Y").dt.to_timestamp()
            # Group by 'Year' and aggregate sum
            yearly_data = (
                data.groupby("Start date")
                .agg(
                    {"positive tests": "sum", "case count": "sum", "test count": "sum"}
                )
                .reset_index()
            )

            # Plotting
            fig, ax = plt.subplots(figsize=(10, 6))

            # Line plot for Positive Tests
            ax.plot(
                yearly_data["Start date"],
                yearly_data["positive tests"],
                label="Positive Tests",
                marker="o",
            )

            # Line plot for Case Counts
            ax.plot(
                yearly_data["Start date"],
                yearly_data["case count"],
                label="Case Counts",
                marker="o",
            )
            ax.plot(
                yearly_data["Start date"],
                yearly_data["test count"],
                label="Test Counts",
                marker="o",
            )

            # X-axis labels and title
            ax.set_xticks(yearly_data["Start date"])
            ax.set_xticklabels(
                yearly_data["Start date"].dt.strftime("%Y-%m-%d"),
                rotation=45,
                ha="right",
            )
            ax.set_xlabel("Year")
            ax.set_ylabel("Counts")
            ax.set_title("Yearly Positive Tests and Case Counts")

            # Legend
            ax.legend()

            # Show the plot
            plt.tight_layout()
            st.pyplot(fig)
        if st.sidebar.button("Positive cases distrubution by year and ZCTA"):
            data["Start date"] = pd.to_datetime(data["Start date"], errors="coerce")
            data["Start date"] = data["Start date"].dt.to_period("Y").dt.to_timestamp()
            pivot = data.pivot_table(
                values="positive tests",
                index="Start date",
                columns="zcta",
                aggfunc="sum",
            )
            # stacked bar chart
            pivot.plot(kind="bar", stacked=True)
            fig, ax = plt.subplots()
            # stacked bar chart
            pivot.plot(kind="bar", stacked=True, ax=ax)
            ax.set_xlabel("Year")
            ax.set_ylabel("Positive Cases")
            ax.set_title("Positive Cases by Year and ZCTA")
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
            plt.tight_layout()
            st.pyplot(fig)
        if st.sidebar.button("Test count by population"):
            tests = data.groupby("population")["test count"].sum()
            # Create a new figure and a set of subplots
            fig, ax = plt.subplots()
            tests.plot(kind="bar", ax=ax, color="blue", width=0.4, position=1)
            ax.set_xlabel("population")
            ax.set_ylabel("Test Count")
            ax.set_title("Test Count by Population")
            ax.legend(["Test Count"])
            st.pyplot(fig)
        if st.sidebar.button("Top 5 impacted zones"):
            res = data.groupby("zcta")["positive tests"].sum()
            res.sort_values(ascending=False, inplace=True)
            st.write(res[:5])
    elif selections == "Suprvised learning":
        st.title("Suprvised learning")
        dataset = pd.read_csv("Dataset11.csv")
        datasetdis = pd.read_csv("discretized.csv")
        test_ration = st.sidebar.slider("Test ration", 0.1, 1.0, 0.8)
        k = st.sidebar.slider("Number of neighbors", 1, 15, 0)
        distanceFn = st.sidebar.selectbox(
            "Distance function", ["Euclidean", "Manhattan", "Cosine", "Minkowsky"]
        )
        if st.sidebar.button("KNN"):
            st.write("### KNN")
            st.write(dataset.head())
            df_train_X, df_test_X, df_train_y, df_test_y = split_data(
                dataset, test_ration
            )
            knn = KNN(df_train_X, df_test_X, df_train_y, df_test_y, k)
            if distanceFn == "Euclidean":
                pred = knn.fit(euclidean)
            elif distanceFn == "Manhattan":
                pred = knn.fit(manhattan)
            elif distanceFn == "Cosine":
                pred = knn.ft(cosine)
            elif distanceFn == "Minkowsky":
                pred = knn.fit(minkowski, p=3)
            st.write("### Confusion matrix")
            res, metrics = confusion_matrix1(pred["Fertility"], pred["Predicted"])
            st.write(res)
            st.write("### Metrics")
            st.write(metrics)
        pure_threshold = st.sidebar.slider("Pure threshold", 0.1, 1.0, 0.0)
        if st.sidebar.button("Decision Tree (discrete)"):
            st.write("### Decision Tree (discrete)")
            st.write(datasetdis.head())
            df_train_X, df_train_Y, df_test_X, df_test_Y = split_data(
                datasetdis, test_ration
            )
            dt = DecisionTree(pure_threshold=pure_threshold)
            dt.fit(df_train_X, df_train_Y)
            pred = dt.predict_all(df_test_X, df_test_Y)
            st.write("### Confusion matrix")
            res, metrics = confusion_matrix1(pred["Fertility"], pred["Predicted"])
            st.write(res)
            st.write("### Metrics")
            st.write(metrics)
        min_split = st.sidebar.slider("Min split", 1, 10, 2)
        max_depth = st.sidebar.slider("Max depth", 1, 20, 10)
        alg = st.sidebar.selectbox("Algorithm", ["entropy", "gini"])
        if st.sidebar.button("Decision Tree (continuous)"):
            st.write("### Decision Tree (continuous)")
            st.write(dataset.head())
            df_train_X, df_train_Y, df_test_X, df_test_Y = split_data(
                dataset, test_ration
            )
            dt = DecisionTreeC(
                min_split=min_split,
                max_depth=max_depth,
                alg=alg,
            )
            dt.fit(df_train_X, df_train_Y)
            pred = dt.predict_all(df_test_X, df_test_Y)
            st.write("### Confusion matrix")
            res, metrics = confusion_matrix1(pred["Fertility"], pred["Predicted"])
            st.write(res)
            st.write("### Metrics")
            st.write(metrics)
        n_estimators = st.sidebar.slider("N estimators", 1, 100, 10)
        max_features = st.sidebar.slider("Max features", 1, 20, 10)
        pure_threshold = st.sidebar.slider("Threshold", 0.1, 1.0, 0.0)
        if st.sidebar.button("Random forest (discrete)"):
            st.write("### Random forest (discrete)")
            st.write(datasetdis.head())
            df_train_X, df_train_Y, df_test_X, df_test_Y = split_data(
                datasetdis, test_ration
            )
            rf = RandomForest(
                n_estimators=n_estimators,
                max_features=max_features,
                pure_threshold=pure_threshold,
            )
            rf.fit(df_train_X, df_train_Y)
            pred = rf.predict_all(df_test_X, df_test_Y)
            st.write("### Confusion matrix")
            res, metrics = confusion_matrix1(pred["Fertility"], pred["Predicted"])
            st.write(res)
            st.write("### Metrics")
            st.write(metrics)
        if st.sidebar.button("Random forest (continuous)"):
            st.write("### Random forest (continuous)")
            st.write(dataset.head())
            df_train_X, df_train_Y, df_test_X, df_test_Y = split_data(
                dataset, test_ration
            )
            rf = RandomForestC(
                n_estimators=n_estimators,
                max_features=max_features,
            )
            rf.fit(df_train_X, df_train_Y)
            pred = rf.predict_all(df_test_X, df_test_Y)
            st.write("### Confusion matrix")
            res, metrics = confusion_matrix1(pred["Fertility"], pred["Predicted"])
            st.write(res)
            st.write("### Metrics")
            st.write(metrics)
    elif selections == "unsupervisesd learning":
        st.title("unsupervisesd learning")
        k = st.sidebar.slider("Number of clusters", 1, 15, 0)
        rnd = st.sidebar.checkbox("Random")
        dataset = pd.read_csv("new_dataset.csv")
        distanceFn = st.sidebar.selectbox(
            "Distance function", ["Euclidean", "Manhattan", "Cosine", "Minkowsky"]
        )
        if st.sidebar.button("Kmeans"):
            st.write("### Kmeans")
            # dataset = dataset.drop("Fertility", axis=1)
            dataset = dataset.round(3)
            st.write(dataset.head())
            reduced = data_to_data_2d(dataset)
            dataset = pd.DataFrame(reduced)
            # joind the test and train
            kmeans = Kmeans(k, dataset, rnd)
            if distanceFn == "Euclidean":
                kmeans.cluster(euclidean2)
            elif distanceFn == "Manhattan":
                kmeans.cluster(manhattan2)
            elif distanceFn == "Cosine":
                kmeans.cluster(cosine2)
            elif distanceFn == "Minkowsky":
                kmeans.cluster(minkowski2)
            st.write("### Clusters")
            # scatter plot
            reduced = PCA(n_components=2).fit_transform(dataset)
            fig = px.scatter(
                x=reduced[:, 0], y=reduced[:, 1], color=kmeans.labels_.astype(int)
            )
            # print the centroids points in different color
            for centroids in kmeans.centroids:
                print("centroids", centroids[0], centroids[1])
                fig.add_scatter(
                    x=np.array(centroids[0]),
                    y=np.array(centroids[1]),
                    mode="markers",
                    marker=dict(color="red", size=10),
                    name="Centroids",
                )

            st.plotly_chart(fig)
            st.write("### silhouette score")
            score = silhouette_score(dataset, kmeans.labels_)
            st.write(score)
        eps = st.sidebar.slider("Epesilon ", 0.1, 1.0, 0.0)
        min_samples = st.sidebar.slider("Min samples", 1, 100, 1)
        if st.sidebar.button("DBSCAN"):
            st.title("DBSCAN")
            dataset = dataset.round(3)
            st.write(dataset.head())
            reduced = data_to_data_2d(dataset)
            dataset = pd.DataFrame(reduced)
            if distanceFn == "Euclidean":
                db = DBSCAN(eps=eps, min_samples=min_samples, distFN=euclidean2)
            elif distanceFn == "Manhattan":
                db = DBSCAN(eps=eps, min_samples=min_samples, distFN=manhattan2)
            elif distanceFn == "Cosine":
                db = DBSCAN(eps=eps, min_samples=min_samples, distFN=cosine2)
            elif distanceFn == "Minkowsky":
                db = DBSCAN(eps=eps, min_samples=min_samples, distFN=minkowski2)
            db.fit(dataset)
            reduced = PCA(n_components=2).fit_transform(dataset)
            fig = px.scatter(
                x=reduced[:, 0], y=reduced[:, 1], color=db.labels_.astype(int)
            )
            st.plotly_chart(fig)
            st.write("### silhouette score")
            score = silhouette_score(dataset, db.labels_)
            st.write(score)


if __name__ == "__main__":
    main_ui()
