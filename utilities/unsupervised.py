import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA


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


def cosine2(B, A):
    # calculate for A as an array and B as a single vector
    nom = np.sum(A * B, axis=1)
    denom = np.sqrt(np.sum(np.power(A, 2), axis=1)) * np.sqrt(np.sum(np.power(B, 2)))
    return 1 - (nom / denom)


def manhattan2(B, A):
    return np.sum(np.abs(A - B), axis=1)


def euclidean2(A, B):
    return np.sqrt(np.sum((A - B) ** 2, axis=1))


def minkowski2(A, B, p=3):
    return np.sum(np.abs(A - B) ** p, axis=1) ** (1 / p)


class DBSCAN:
    """
    Density-Based Spatial Clustering of Applications with Noise (DBSCAN) algorithm.

    Parameters:
    - eps: float, optional (default=0.7)
        The maximum distance between two samples for them to be considered as neighbors.
    - min_samples: int, optional (default=15)
        The minimum number of samples in a neighborhood for a point to be considered as a core point.
    - distFN: function, optional (default=euclidean2)
        The distance function used to calculate the distance between instances.

    Attributes:
    - eps: float
        The maximum distance between two samples for them to be considered as neighbors.
    - min_samples: int
        The minimum number of samples in a neighborhood for a point to be considered as a core point.
    - visited: set
        A set to keep track of visited points during clustering.
    - distFN: function
        The distance function used to calculate the distance between instances.
    - labels_: ndarray, shape (n_samples,)
        Cluster labels for each point in the dataset.

    Methods:
    - cluster(X: pd.DataFrame) -> ndarray:
        Perform DBSCAN clustering on the input dataset and return the cluster labels.
    - expand_cluster(X: pd.DataFrame, indices, c):
        Recursive function to expand a cluster by finding its neighbors and assigning them to the cluster.
    - sort_eps(X, instance, eps) -> ndarray:
        Sort the instances based on their distance to a given instance within a specified epsilon distance.

    """

    def __init__(self, eps=0.7, min_samples=15, distFN=euclidean2) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.visited = set()
        self.distFN = distFN
        self.labels_ = None

    def cluster(self, X: pd.DataFrame):
        c = 0
        self.labels_ = np.zeros(len(X), dtype=int)

        # loop over all the points, recursively find the neighbors and assign them to a cluster
        for i in range(len(X)):
            if i in self.visited:
                continue
            self.visited.add(i)
            neighbors_i = self.sort_eps(X, X.iloc[i])
            if len(neighbors_i) < self.min_samples:
                self.labels_[i] = 0
            else:
                c += 1
                self.labels_[i] = c
                self.expand_cluster(X, neighbors_i, c)
        return self.labels_

    def expand_cluster(self, X: pd.DataFrame, indices, c):
        # recursive
        for i, n in enumerate(indices):
            if n in self.visited:
                if self.labels_[n] == 0:
                    self.labels_[n] = c
                continue
            self.visited.add(n)
            self.labels_[n] = c
            neighbors_n = self.sort_eps(X, X.iloc[n])
            if len(neighbors_n) >= self.min_samples:
                self.expand_cluster(X, neighbors_n, c)

    def sort_eps(self, X, instance):
        distances = self.distFN(X.values, np.array(instance))
        mask = (distances <= self.eps) & (X.index != instance.name)
        return X[mask].index

    def fit(self, X):
        self.cluster(X)
        return self.labels_


class Kmeans:
    """
    K-means clustering algorithm implementation.

    Parameters:
    - k (int): The number of clusters.
    - dataset (pandas.DataFrame): The input dataset.
    - rnd (bool): Whether to initialize centroids randomly or not.

    Attributes:
    - k (int): The number of clusters.
    - dataset (pandas.DataFrame): The input dataset.
    - rnd (bool): Whether to initialize centroids randomly or not.
    - labels_ (numpy.ndarray): The cluster labels assigned to each data point.

    Methods:
    - generate_cluster(distanceFn, centroids=None): Generates clusters based on the given distance function and centroids.
    - cluster(distanceFn, max_iter=100): Performs the clustering process using the given distance function and maximum number of iterations.
    """

    def __init__(self, k, dataset, rnd=False) -> None:
        self.k = k
        self.dataset = dataset
        self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)
        if rnd:
            self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)
        else:
            self.dataset = self.dataset.sort_values(by=list(self.dataset.columns))
        self.rnd = rnd
        self.labels_ = None
        self.centroids = None

    def generate_cluster(self, distanceFn, centroids=None):
        """
        Generates clusters based on the given distance function and centroids.

        Parameters:
        - distanceFn (function): The distance function to calculate the distance between data points and centroids.
        - centroids (list): The initial centroids. If None, new centroids will be generated.

        Returns:
        - distances (dict): A dictionary containing the distances between data points and centroids for each cluster.
        - centroids (list): The final centroids.
        """
        if centroids is None:
            if not self.rnd:
                # get k number of points evenly spaced using the linspace function
                centroids = self.dataset.iloc[
                    np.linspace(0, len(self.dataset) - 1, self.k, dtype=int)
                ]
            else:
                # get k number of points randomly
                centroids = self.dataset.sample(n=self.k)
        centroids = np.array(centroids)
        distances = np.zeros((self.k, len(self.dataset)))
        for c in range(self.k):
            distances[c] = distanceFn(self.dataset.values, centroids[c].reshape(1, -1))
            # round the distances to 2 decimal places
            distances[c] = np.round(distances[c], 2)

        distances = distances.T
        closest_centroids = np.argmin(distances, axis=1)
        self.labels_ = closest_centroids
        return distances, centroids

    def cluster(self, distanceFn, max_iter=100):
        """
        Performs the clustering process using the given distance function and maximum number of iterations.

        Parameters:
        - distanceFn (function): The distance function to calculate the distance between data points and centroids.
        - max_iter (int): The maximum number of iterations for the clustering process.

        Returns:
        - distances (dict): A dictionary containing the distances between data points and centroids for each cluster.
        - centroids (list): The final centroids.
        """
        self.labels_ = np.zeros(len(self.dataset), dtype=int)
        distances, centroids = self.generate_cluster(distanceFn)
        # calculate new centroids
        repeat = True
        nb_iter = 0
        while repeat:
            centroids_old = centroids.copy()
            # caluclate the mean of each cluster
            for c in range(self.k):
                cluster = self.dataset[self.labels_ == c]
                centroids[c] = np.mean(cluster, axis=0)
            # calculate the distances between the data points and the new centroids
            distances, centroids = self.generate_cluster(distanceFn, centroids)
            nb_iter += 1
            if np.array_equal(centroids_old, centroids) or nb_iter > max_iter:
                repeat = False
        print(f"Number of iterations: {nb_iter}")
        print(f"repeat: {repeat}")
        self.centroids = centroids
        return distances, centroids


def data_to_data_2d(data):
    # Extract the features from your dataset
    features = data.iloc[:, :].values  # Exclude the target variable

    # Standardize the features (optional but recommended for PCA)
    features_standardized = StandardScaler().fit_transform(features)

    # Apply PCA to reduce the features to 2 components
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features_standardized)

    # Create a new DataFrame with the reduced features
    data_2d = pd.DataFrame(data=features_2d, columns=["PC1", "PC2"])
    return data_2d
