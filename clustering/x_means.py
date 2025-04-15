# Based on the paper: https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=c9a69742be991cc17c9a961da7791f6a5b7c2c3b
# apparently an efficient way of optimizing k for kmeans 

import pandas as pd
import numpy as np
from sktime.clustering.k_means import TimeSeriesKMeans
   

class TimeSeriesXMeans():
    def __init__(self, metric='dtw', k_max=42, random_seed=42):
        self.metric = metric
        self.k_max = k_max
        self.random_seed = random_seed
        
    def fit(self, X, y=None):
        clusterer = TimeSeriesKMeans(2, metric=self.metric, random_state=self.random_seed)
        clusterer.fit(X)
        k = 3
        while k <= self.k_max:
            init = self._try_splits(X, clusterer)
            if len(clusterer.cluster_centers_) == len(init):
                return clusterer, k
            else:
                k = len(init)
                clusterer = TimeSeriesKMeans(k, init, self.metric, random_state=self.random_seed)
                clusterer.fit(X)
        return clusterer, k


    def _try_splits(self, X, clusterer, tol=256):
        new_centers = []
        for k in range(clusterer.n_clusters):
            # compute cluster bic for 1 center
            subset_labels = clusterer.labels_[clusterer.labels_ == k]
            subset_labels[:]=0
            if len(subset_labels) < 2:
                new_centers.append(clusterer.cluster_centers_[k])
                continue
            subset = np.asarray(X[clusterer.labels_ == k, ...])
            bic_1 = self._heavy_bic(subset, subset_labels, 1)
            # compute cluster bic for 2 center using 2means
            k2_means = TimeSeriesKMeans(2, metric=self.metric, random_state=self.random_seed)
            k2_means.fit(subset)
            bic_2 = self._heavy_bic(subset, k2_means.labels_, 2)
            if (bic_2-bic_1)>tol:
                new_centers.append(k2_means.cluster_centers_[0])
                new_centers.append(k2_means.cluster_centers_[1])
            else:
                new_centers.append(clusterer.cluster_centers_[k])
        return np.asarray(new_centers)


    def _heavy_bic(self, X, labels, k): ## all vibe coded lol
        n, d, _ = X.shape
        R = n
        M = d
        K = k

        # Count points per cluster and compute centroids
        centroids = []
        Rk = []
        for cluster in range(K):
            members = X[labels == cluster]
            if len(members) == 0:
                centroids.append(np.zeros(d))
                Rk.append(0)
            else:
                centroids.append(np.mean(members, axis=0))
                Rk.append(len(members))
        centroids = np.array(centroids)

        # Estimate shared variance (spherical Gaussian)
        total_sq_error = 0
        for i in range(n):
            cluster = labels[i]
            total_sq_error += np.sum((X[i] - centroids[cluster]) ** 2)
        sigma_sq = total_sq_error / (R - K + 1e-10)  # avoid div-by-zero

        # Compute log-likelihood according to the paper
        log_likelihood = 0
        for cluster in range(K):
            Nk = Rk[cluster]
            if Nk == 0:
                continue
            term1 = -Nk * M / 2 * np.log(2 * np.pi * sigma_sq)
            term2 = -1 / (2 * sigma_sq) * np.sum((X[labels == cluster] - centroids[cluster]) ** 2)
            term3 = Nk * np.log(Nk / R + 1e-10)  # add epsilon to avoid log(0)
            log_likelihood += term1 + term2 + term3

        # Number of free parameters: K-1 (cluster weights) + K*M (centroids) + 1 (shared variance)
        p_j = (K - 1) + K * M + 1

        bic = log_likelihood - (p_j / 2) * np.log(R)
        return bic
