import pandas as pd
import numpy as np
from collections import Counter
from random import sample
import random

# confusion matrix
from sklearn.metrics import confusion_matrix

# import knn


def split_data(df, test_ratio):
    """
    Split the given DataFrame into training and testing sets based on the specified test ratio.

    Parameters:
    df (pandas.DataFrame): The DataFrame to be split.
    test_ratio (float): The ratio of the data to be used for testing.

    Returns:
    df_train_X (pandas.DataFrame): The training set features.
    df_train_Y (pandas.Series): The training set target variable.
    df_test_X (pandas.DataFrame): The testing set features.
    df_test_Y (pandas.Series): The testing set target variable.
    """
    df_fert = df[df["Fertility"] == 1.0]
    df_no_fert = df[df["Fertility"] == 0.0]
    df_plus_fert = df[df["Fertility"] == 2.0]
    df_fert = df_fert.sample(frac=1).reset_index(drop=True)
    df_no_fert = df_no_fert.sample(frac=1).reset_index(drop=True)
    df_plus_fert = df_plus_fert.sample(frac=1).reset_index(drop=True)

    df_fert_train = df_fert.iloc[: int(len(df_fert) * test_ratio)]
    df_fert_test = df_fert.iloc[int(len(df_fert) * test_ratio) :]
    df_no_fert_train = df_no_fert.iloc[: int(len(df_no_fert) * test_ratio)]
    df_no_fert_test = df_no_fert.iloc[int(len(df_no_fert) * test_ratio) :]
    df_plus_fert_train = df_plus_fert.iloc[: int(len(df_plus_fert) * test_ratio)]
    df_plus_fert_test = df_plus_fert.iloc[int(len(df_plus_fert) * test_ratio) :]

    df_train = pd.concat([df_fert_train, df_no_fert_train, df_plus_fert_train])
    df_train = df_train.sample(frac=1).reset_index(drop=True)

    df_test = pd.concat([df_fert_test, df_no_fert_test, df_plus_fert_test])
    df_test = df_test.sample(frac=1).reset_index(drop=True)
    df_train_Y = df_train["Fertility"]
    df_train_X = df_train.drop("Fertility", axis=1)
    df_test_Y = df_test["Fertility"]
    df_test_X = df_test.drop("Fertility", axis=1)
    return df_train_X, df_train_Y, df_test_X, df_test_Y


def manhattan(A, B):
    """
    Calculates the Manhattan distance between two vectors A and B.

    Parameters:
    A (array-like): First vector.
    B (array-like): Second vector.

    Returns:
    float: The Manhattan distance between A and B.
    """
    return np.sum(np.abs(A - B))


def euclidean(A, B):
    """
    Calculate the Euclidean distance between two vectors A and B.

    Parameters:
    A (numpy.ndarray): The first vector.
    B (numpy.ndarray): The second vector.

    Returns:
    float: The Euclidean distance between A and B.
    """
    return np.sqrt(np.sum(np.square(A - B)))


def cosine(A, B):
    """
    Calculate the cosine similarity between two vectors A and B.

    Parameters:
    A (array-like): The first vector.
    B (array-like): The second vector.

    Returns:
    float: The cosine similarity between A and B.
    """
    return 1 - np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))


def minkowski(A, B, p):
    """
    Calculates the Minkowski distance between two vectors A and B.

    Parameters:
    A (array-like): The first vector.
    B (array-like): The second vector.
    p (float): The order of the Minkowski distance.

    Returns:
    float: The Minkowski distance between A and B.
    """
    return np.sum(np.power(np.abs(A - B), p)) ** (1 / p)


def hamming(A, B):
    """
    Calculates the Hamming distance between two arrays A and B.

    Parameters:
    A (ndarray): First array.
    B (ndarray): Second array.

    Returns:
    int: The Hamming distance between A and B.
    """
    return np.sum(A != B)


class KNN:
    """
    K-Nearest Neighbors (KNN) classifier.

    Parameters:
    - train_X (DataFrame): Training data features.
    - train_Y (Series): Training data labels.
    - test_X (DataFrame): Test data features.
    - test_Y (Series): Test data labels.
    - k (int): Number of nearest neighbors to consider (default: 3).
    """

    def __init__(self, train_X, train_Y, test_X, test_Y, k=3) -> None:
        self.train_X = train_X
        self.test_X = test_X
        self.train_Y = train_Y
        self.test_Y = test_Y
        self.k = k

    def sort(self, instance, distanceFN, p=None):
        """
        Sorts the training instances based on their distance to the given instance.

        Parameters:
        - instance: The instance for which distances are calculated.
        - distanceFN: The distance function to use.
        - p: The power parameter for Minkowski distance (default: None).

        Returns:
        - distances: A list of tuples containing the distances and corresponding labels.
        """
        distances = []
        for x, y in zip(self.train_X.itertuples(index=False), self.train_Y):
            if p is None:
                distances.append((distanceFN(x, instance), y))
            else:
                distances.append((distanceFN(x, instance, p), y))
        distances.sort(key=lambda x: x[0])
        return distances

    def predict(self, instance, distanceFN=euclidean, p=None):
        """
        Predicts the label for the given instance using KNN algorithm.

        Parameters:
        - instance: The instance to predict the label for.
        - distanceFN: The distance function to use (default: euclidean).
        - p: The power parameter for Minkowski distance (default: None).

        Returns:
        - predicted_label: The predicted label for the instance.
        """
        distances = self.sort(instance, distanceFN, p)
        neighbors = distances[: self.k]
        neighbors = [i[1] for i in neighbors]
        return max(neighbors, key=neighbors.count)

    def fit(self, distanceFN, p=None):
        """
        Fits the KNN model and returns the predicted labels for the test data.

        Parameters:
        - distanceFN: The distance function to use.
        - p: The power parameter for Minkowski distance (default: None).

        Returns:
        - results: A DataFrame containing the test data labels and predicted labels.
        """
        results = []
        for i in range(len(self.test_X)):
            results.append(self.predict(self.test_X.iloc[i], distanceFN, p))
        return pd.DataFrame({"Fertility": self.test_Y, "Predicted": results})


class Node:
    """
    Represents a node in a decision tree.

    Attributes:
        label (str): The label assigned to the node.
        attr (str): The attribute associated with the node.
        childs (list): The list of child nodes.
    """

    def __init__(self, label, attr, childs):
        self.label = label
        self.childs = childs
        self.attr = attr


class DecisionTree:
    """
    A class representing a decision tree classifier.

    Parameters:
    - pure_threshold: The purity threshold used to determine when to stop splitting the tree.

    Methods:
    - entropie(Y): Calculate the entropy of a target variable.
    - gain(S, Sv): Calculate the information gain of a split.
    - fit(X, Y): Build the decision tree using the given features and target variable.
    - create_tree(X, Y, attr=None): Recursively create the decision tree.
    - predict(tree, x): Predict the class label for a single instance.
    - predict_all(X, Y): Predict the class labels for multiple instances.

    Attributes:
    - root: The root node of the decision tree.
    - pure_threshold: The purity threshold used to determine when to stop splitting the tree.
    - max_attrs: A dictionary storing the maximum number of unique values for each attribute.
    """

    def __init__(self, pure_threshold) -> None:
        self.root = None
        self.pure_threshold = pure_threshold

    def entropie(self, Y):
        """
        Calculate the entropy of a given target variable.

        Parameters:
        - Y: The target variable (array-like).

        Returns:
        - entropy: The entropy value (float).
        """
        entropy = 0
        for i in Y.unique():
            p = len(Y[Y == i]) / len(Y)
            entropy += -p * np.log2(p)
        return entropy

    def gain(self, S, Sv):
        """
        Calculates the information gain of a given attribute.

        Parameters:
        S (pandas.Series): The target variable.
        Sv (pandas.Series): The attribute values.

        Returns:
        float: The information gain.
        """
        gain = 0
        for i in Sv.unique():
            gain += (len(Sv[Sv == i]) / len(S)) * self.entropie(S[Sv == i])
        return self.entropie(S) - gain

    def fit(self, X, Y):
        """
        Fits the model to the training data.

        Parameters:
        - X: The input features.
        - Y: The target variable.

        Returns:
        None
        """
        self.max_attrs = {}
        for x in X.columns:
            self.max_attrs[x] = len(X[x].unique())
        self.root = self.create_tree(X, Y)

    def create_tree(self, X, Y, attr=None):
        """
        Creates a decision tree based on the given dataset.

        Args:
            X (pandas.DataFrame): The feature matrix.
            Y (pandas.Series): The target variable.
            attr (str, optional): The attribute associated with the current node. Defaults to None.

        Returns:
            Node: The root node of the decision tree.
        """
        if len(Y.unique()) == 1:
            return Node(Y.unique()[0], attr, [])
        if len(X.columns) == 0:
            return Node(Y.mode()[0], attr, [])

        gains = []
        for i in X.columns:
            gains.append([i, self.gain(X[i], Y)])
        # get the attribute with the max gain
        max_gain = max(gains, key=lambda x: x[1])
        # check if the max gain is above the purity_threhold
        root = Node(max_gain[0], attr, [])
        attrs = X[max_gain[0]].unique()
        if len(attrs) < self.max_attrs[max_gain[0]]:
            return Node(Y.mode()[0], attr, [])
        if max_gain[1] >= self.pure_threshold:
            # return the mode for every attribute
            for i in attrs:
                X_i = X[X[max_gain[0]] == i]
                if len(X_i[max_gain[0]].unique()) < self.max_attrs[max_gain[0]]:
                    root.childs.append(Node(Y[X[max_gain[0]] == i].mode()[0], i, []))
                else:
                    root.childs.append(Node(Y.mode()[0], i, []))

            return root
        for i in attrs:
            X_i = X[X[max_gain[0]] == i]
            X_i = X_i.drop(max_gain[0], axis=1)
            # remove the attribute from the subset
            root.childs.append(self.create_tree(X_i, Y[X[max_gain[0]] == i], i))
        return root

    def predict(self, tree, x):
        """
        Predicts the label for a given input using the decision tree.

        Parameters:
        tree (TreeNode): The root node of the decision tree.
        x (list): The input features.

        Returns:
        The predicted label for the input.
        """
        node = tree
        while len(node.childs) > 0:
            if type(node.label) == float:
                return node.label
            for i in node.childs:
                if i.attr == x[node.label]:
                    node = i
                    break
        return node.label

    def predict_all(self, X, Y):
        """
        Predicts the target variable for all instances in the input data.

        Parameters:
            X (pandas.DataFrame): The input data containing the features.
            Y (list): The true values of the target variable.

        Returns:
            pandas.DataFrame: A DataFrame containing the true values and the predicted values of the target variable.
        """
        Y_pred = []
        for i in range(len(X)):
            Y_pred.append(self.predict(self.root, X.iloc[i]))
        return pd.DataFrame({"Fertility": Y, "Predicted": Y_pred})


class NodeC:
    """
    Represents a node in a decision tree.

    Attributes:
        label (str): The label associated with the node.
        threshold (float): The threshold value used for splitting the data.
        left (NodeC): The left child node.
        right (NodeC): The right child node.
        info_gain (float): The information gain achieved by splitting the data at this node.
        value (float): The predicted value associated with the node.
    """

    def __init__(
        self,
        label=None,
        threshold=None,
        left=None,
        right=None,
        info_gain=None,
        value=None,
    ):
        self.label = label
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value

    # Rest of the code...


class DecisionTreeC:
    """
    DecisionTreeC is a class that represents a decision tree classifier.

    Parameters:
    - min_split (int): The minimum number of instances required to split a node. Default is 2.
    - max_depth (int): The maximum depth of the decision tree. Default is 3.
    - alg (str): The algorithm used to calculate information gain. Can be "entropy" or "gini". Default is "entropy".
    - thrsh (int): The maximum number of unique thresholds to consider for each feature. Default is 10.
    """

    def __init__(self, min_split=2, max_depth=3, alg="entropy", thrsh=10):
        self.root = None
        self.min_split = min_split
        self.max_depth = max_depth
        self.alg = alg
        self.thrsh = thrsh

    def create_tree(self, X, Y, curr_depth=0):
        num_inst = len(Y)
        num_col = X.columns

        if num_inst >= self.min_split and curr_depth <= self.max_depth:
            # find the best split
            best_split = self.best_split(X, Y, num_col)
            # check if best split is not empty
            if best_split != {}:
                if best_split["info_gain"] > 0:
                    left = self.create_tree(
                        best_split["left_X"], best_split["left_Y"], curr_depth + 1
                    )
                    right = self.create_tree(
                        best_split["right_X"], best_split["right_Y"], curr_depth + 1
                    )
                    return NodeC(
                        best_split["label"],
                        best_split["threshold"],
                        left,
                        right,
                        best_split["info_gain"],
                    )
        leaf = self.get_leaf_value(Y)
        return NodeC(value=leaf)

    def best_split(self, X, Y, num_col):
        best_split = {}
        max_info_gain = -float("inf")
        for col in num_col:
            threshs = X[col].unique()
            # keep the number of threshs equal to self.thrsh
            if len(threshs) > self.thrsh:
                threshs = set(random.sample(list(threshs), self.thrsh))
            for thresh in threshs:
                left_X, right_X, left_Y, right_Y = self.split(X, Y, col, thresh)
                if len(left_Y) > 0 and len(right_Y) > 0:
                    info_gain = self.info_gain(Y, left_Y, right_Y, self.alg)
                    if info_gain > max_info_gain:
                        max_info_gain = info_gain
                        best_split["label"] = col
                        best_split["threshold"] = thresh
                        best_split["left_X"] = left_X
                        best_split["right_X"] = right_X
                        best_split["left_Y"] = left_Y
                        best_split["right_Y"] = right_Y
                        best_split["info_gain"] = info_gain
        return best_split

    def split(self, X, Y, col, threshold):
        left_X = X[X[col] <= threshold]
        right_X = X[X[col] > threshold]
        left_Y = Y[X[col] <= threshold]
        right_Y = Y[X[col] > threshold]
        return left_X, right_X, left_Y, right_Y

    def info_gain(self, Y, left_Y, right_Y, alg="entropy"):
        w_l = len(left_Y) / (len(left_Y) + len(right_Y))
        w_r = len(right_Y) / (len(left_Y) + len(right_Y))

        if alg == "entropy":
            gain = (
                self.entropy(Y)
                - w_l * self.entropy(left_Y)
                - w_r * self.entropy(right_Y)
            )
        elif alg == "gini":
            gain = self.gini(Y) - w_l * self.gini(left_Y) - w_r * self.gini(right_Y)
        return gain

    def get_leaf_value(self, Y):
        Y = list(Y)
        return max(Y, key=Y.count)

    def entropy(self, Y):
        entropy = 0
        for i in Y.unique():
            p = len(Y[Y == i]) / len(Y)
            entropy += -p * np.log2(p)
        return entropy

    def gini(self, Y):
        gini = 1
        for i in Y.unique():
            p = len(Y[Y == i]) / len(Y)
            gini -= p**2
        return gini

    def fit(self, X, Y):
        self.root = self.create_tree(X, Y)

    def predict(self, x):
        node = self.root
        while node.value == None:
            if x[node.label] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict_all(self, X, Y):
        Y_pred = []
        for i in range(len(X)):
            Y_pred.append(self.predict(X.iloc[i]))
        return pd.DataFrame({"Fertility": Y, "Predicted": Y_pred})


class RandomForest:
    def __init__(self, n_estimators=10, max_features=None, pure_threshold=0.7):
        """
        Initializes a RandomForest object.

        Parameters:
        - n_estimators (int): The number of decision trees in the random forest. Default is 10.
        - max_features (int or None): The maximum number of features to consider when splitting a node.
          If None, all features will be considered. Default is None.
        - pure_threshold (float): The purity threshold for a leaf node. Default is 0.7.
        """
        self.n_estimators = n_estimators
        self.estimators = []
        self.max_features = max_features
        self.pure_threshold = pure_threshold

    def fit(self, X, Y):
        """
        Fits the random forest to the training data.

        Parameters:
        - X (pandas.DataFrame): The input features.
        - Y (pandas.Series): The target variable.
        """
        self.estimators = []
        for _ in range(self.n_estimators):
            estimator = DecisionTree(self.pure_threshold)
            bootstrap_indices = np.random.choice(len(X), size=len(X), replace=True)
            bootstrap_X = X.iloc[bootstrap_indices]
            bootstrap_Y = Y.iloc[bootstrap_indices]
            # Select a subset of attributes
            if self.max_features is not None:
                selected_features = sample(list(X.columns), self.max_features)
                bootstrap_X = bootstrap_X[selected_features]

            estimator.fit(bootstrap_X, bootstrap_Y)
            self.estimators.append(estimator)

    def predict(self, X):
        """
        Predicts the target variable for the input features.

        Parameters:
        - X (pandas.DataFrame): The input features.

        Returns:
        - float: The predicted target variable.
        """
        predictions = []
        for estimator in self.estimators:
            predictions.append(estimator.predict(estimator.root, X))
        predictions = np.array(predictions)
        mcv = np.apply_along_axis(
            lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=predictions
        )
        return float(mcv)

    def predict_all(self, X, Y):
        """
        Predicts the target variable for all instances in the input features.

        Parameters:
        - X (pandas.DataFrame): The input features.
        - Y (pandas.Series): The actual target variable.

        Returns:
        - pandas.DataFrame: A DataFrame containing the actual and predicted target variables.
        """
        Y_pred = []
        for i in range(len(X)):
            Y_pred.append(self.predict(X.iloc[i]))
        return pd.DataFrame({"Fertility": Y, "Predicted": Y_pred})


class RandomForestC:
    """
    Random Forest Classifier.

    Parameters:
    - n_estimators (int): The number of decision trees in the random forest. Default is 10.
    - max_features (int or None): The maximum number of features to consider when looking for the best split.
      If None, all features will be considered. Default is None.
    """

    def __init__(self, n_estimators=10, max_features=None):
        self.n_estimators = n_estimators
        self.estimators = []
        self.max_features = max_features

    def fit(self, X, Y):
        """
        Fit the random forest to the training data.

        Parameters:
        - X (pandas.DataFrame): The input features.
        - Y (pandas.Series): The target variable.
        """
        self.estimators = []
        for _ in range(self.n_estimators):
            min_split = np.random.randint(2, 5)
            max_depth = np.random.randint(2, 5)
            alg = np.random.choice(["entropy", "gini"])
            thrsh = np.random.randint(5, 10)
            estimator = DecisionTreeC(
                min_split=min_split, max_depth=max_depth, alg=alg, thrsh=thrsh
            )
            bootstrap_indices = np.random.choice(len(X), size=len(X), replace=True)
            bootstrap_X = X.iloc[bootstrap_indices]
            bootstrap_Y = Y.iloc[bootstrap_indices]
            # Select a subset of attributes
            if self.max_features is not None:
                selected_features = sample(list(X.columns), self.max_features)
                bootstrap_X = bootstrap_X[selected_features]

            estimator.fit(bootstrap_X, bootstrap_Y)
            self.estimators.append(estimator)

    def predict(self, X):
        """
        Predict the class label for each sample in X.

        Parameters:
        - X (pandas.DataFrame): The input features.

        Returns:
        - float: The predicted class label.
        """
        predictions = []
        for estimator in self.estimators:
            predictions.append(estimator.predict(X))
        predictions = np.array(predictions)
        mcv = np.apply_along_axis(
            lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=predictions
        )
        return float(mcv)

    def predict_all(self, X, Y):
        """
        Predict the class labels for all samples in X and compare them with the true labels Y.

        Parameters:
        - X (pandas.DataFrame): The input features.
        - Y (pandas.Series): The true class labels.

        Returns:
        - pandas.DataFrame: A DataFrame containing the true labels and the predicted labels.
        """
        Y_pred = []
        for i in range(len(X)):
            Y_pred.append(self.predict(X.iloc[i]))
        return pd.DataFrame({"Fertility": Y, "Predicted": Y_pred})


def confusion_matrix1(Y, Y_pred):
    """
    Calculate the confusion matrix and various performance metrics based on the predicted and actual labels.

    Args:
        Y (array-like): The actual labels.
        Y_pred (array-like): The predicted labels.

    Returns:
        tuple: A tuple containing two DataFrames:
            - The confusion matrix, with rows representing the actual labels and columns representing the predicted labels.
            - A DataFrame containing various performance metrics, including specificity, precision, recall, F-score, and accuracy.
    """
    cm = confusion_matrix(Y, Y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]

    specifity = TN / (TN + FP)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f_score = (2 * precision * recall) / (precision + recall)
    exactitude = (TP + TN) / (TP + TN + FP + FN)
    metrics_df_global = {
        "Specificity": specifity,
        "Precision": precision,
        "Recall": recall,
        "F-score": f_score,
        "Accuracy": exactitude,
    }
    return pd.DataFrame(
        cm,
        index=["NO Fertility", "Fertility", "Plus Fertility"],
        columns=["NO Fertility", "Fertility", "Plus Fertility"],
    ), pd.DataFrame(metrics_df_global, index=["Global"])
