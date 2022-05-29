import numpy as np
import pandas as pd

class KMeans:
    """
        Class to handle KMeans Clustering

        Parameters
        ----------
        X: pd.DataFrame or np.ndarray
            Data set to cluster on. Will be converted
            to np.ndarray for faster computation.

        k : int
            Number of k clusters.
       """

    def __init__(
        self,
        X : pd.DataFrame or np.ndarray,
        n_clusters : int
    ):
        if type(X) != np.ndarray:
            self.X = X.to_numpy()
            self.features = list(X.columns)
        else:
            self.X = X
            self.features = []

        self.n_clusters = n_clusters

        # Matrix of n x p dimensions
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]

        # Vector with SSE and initial centroids
        self.sse = np.zeros([self.n_clusters, 1])
        self._centroids = np.zeros([self.n_clusters, self.p])

    def fit(self):
        """
        Fits clustering algorithm.
        1. Assigns initial random clusters.
        2. Then iterate the following until no further change in centroids:
            - Calculate new centroids (Matrix of k row vectors with p feature
            means)
            - Assign points to its closest cluster

        Parameters
        ----------
        self

        Returns
        -------
        self

        """
        self._init_clusters()
        self.n_iter = 0
        # We need copy for numpy matrices, otherwise objects will be modified
        prev_centroids = self._centroids.copy()

        print(f"Fitting KMeans with {self.n_clusters} K")
        while 1:
            new_centroids = self._update_centroids(prev_centroids)
            self._update_clusters(new_centroids)
            self.n_iter += 1

            if not np.array_equal(new_centroids, prev_centroids):
                prev_centroids = new_centroids.copy()
            else:
                self._centroids = new_centroids
                self._get_total_sse()
                print(f"Converged with {self.n_iter} iterations")
                break
        return self

    def _init_clusters(self):
        """
        Initiates random clusters.
        The algorithm generally chooses random centroids
        based on different things.

        In this case, it's easier to assign random clusters.
        However, this may make convergence slower.

        Parameters
        ----------
        self

        Returns
        -------
        self

        """
        self._clusters = np.random.randint(0, self.n_clusters, self.n)
        return self

    def _update_centroids(self, centroids):
        """
        Calculate centroids for each cluster.

        Parameters
        ----------
        centroids : np.ndarray
            matrix with K row vector means, and m dimensions.

        Returns
        -------
        centroids: np.ndarray
            Updated centroids using existing clusters
        """
        centroids = centroids.copy()
        # For each cluster
        for k in range(self.n_clusters):
            # Get index of this cluster in the data matrix
            k_index = np.where(self._clusters == k)
            # Filter array and calculate mean vector
            centroids[k] = self.X[k_index].mean(axis = 0)
        return centroids

    def _update_clusters(self, centroids):
        """
        Calculates distance to all centroids,
        and re-assigns to closest cluster accordingly.

        Parameters
        ----------
        centroids : np.ndarray
        """
        for i in range(self.n):
            row = self.X[i]
            distances = []
            # Calculate all distances and append
            for k in range(self.n_clusters):
                distances.append(squared_euclidean(row, centroids[k]))
            # Take minimum distance
            min_distance = min(distances)
            cluster = distances.index(min_distance)

            # Re-assign if not the same
            if self._clusters[i] != cluster:
                self._clusters[i] = cluster
        return self

    def _get_total_sse(self):
        """
        Calculates total SSE

        Parameters
        ----------
        self

        Returns
        -------
        self
        """
        self.total_sse = 0
        for k in range(self.n_clusters):
            self.total_sse += self._get_sse(k)
        return self

    def _get_sse(self, k):
        """
        Calculates Squared Standard Error for a given cluster K

        This is the sum of squared euclidean distance between all points
        belonging to cluster k and its current cluster centroid.

        Parameters
        ----------
        k : int
            Number of K clusters

        Returns
        -------
        w_sse : float
            Within cluster SSE
        """
        # Get K vector mean
        k_centroid = self._centroids[k]
        # Filter matrix by observations in cluster k
        filtered_matrix = self.X[np.where(self._clusters == k)]
        w_sse = 0
        for row in filtered_matrix:
            w_sse += squared_euclidean(row, k_centroid)
        return w_sse

def squared_euclidean(a, b):
    """
        Calculates Squared Euclidean Distance between vectors a and b.
        This will be used to compare points in the sample.
    """
    sse = 0
    for i, j in zip(a, b):
        sse += (i - j)**2
    return sse
