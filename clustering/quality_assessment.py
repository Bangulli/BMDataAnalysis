import pandas as pd
import numpy as np

def aic(model, X, y): # adapted from stepmix source code
    return -2 * model.score(X, y) * X.shape[0] + 2 * model.n_parameters

def bic(model, X, y): # adapted from stepmix source code
    return 2 * model.score(X, y) * X.shape[0] + model.n_parameters * np.log(X.shape[0])

def approx_compute_aic_bic(X, labels, k):
    n, d = X.shape
    p = k * d

    # Compute total within-cluster sum of squares
    RSS = 0
    for cluster in range(k):
        members = X[labels == cluster]
        if len(members) == 0:
            continue
        centroid = np.mean(members, axis=0)
        RSS += np.sum((members - centroid) ** 2)

    aic = n * np.log(RSS / n) + 2 * p
    bic = n * np.log(RSS / n) + p * np.log(n)

    return aic, bic

def approx_bic(X, labels, k):
    n, d = X.shape
    p = k * d

    # Compute total within-cluster sum of squares
    RSS = 0
    for cluster in range(k):
        members = X[labels == cluster]
        if len(members) == 0:
            continue
        centroid = np.mean(members, axis=0)
        RSS += np.sum((members - centroid) ** 2)

    bic = n * np.log(RSS / n) + p * np.log(n)

    return bic


def approx_aic(X, labels, k):
    n, d = X.shape
    p = k * d

    # Compute total within-cluster sum of squares
    RSS = 0
    for cluster in range(k):
        members = X[labels == cluster]
        if len(members) == 0:
            continue
        centroid = np.mean(members, axis=0)
        RSS += np.sum((members - centroid) ** 2)

    aic = n * np.log(RSS / n) + 2 * p

    return aic

def good_approx_bic_aic(X, labels, k): ## all vibe coded lol
        X = np.asarray(X)
        n, d = X.shape
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
        aic = log_likelihood - 2 * p_j
        return bic, aic