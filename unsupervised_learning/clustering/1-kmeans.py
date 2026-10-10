#!/usr/bin/env python3
"""
Module for performing K-means clustering.
"""

import numpy as np

initialize = __import__('0-initialize').initialize


def kmeans(X, k, iterations=1000):
    """
    Perform K-means clustering on a dataset.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        k (int): Number of clusters.
        iterations (int): Maximum number of iterations.

    Returns:
        tuple: (C, clss), where C contains the final centroids
        and clss contains the cluster index for each data point.
        Returns (None, None) if the inputs are invalid.
    """
    # Validate the input arguments.
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    if X.shape[0] == 0 or X.shape[1] == 0:
        return None, None

    if not np.issubdtype(X.dtype, np.number):
        return None, None

    if not np.all(np.isfinite(X)):
        return None, None

    if type(k) is not int or k <= 0:
        return None, None

    if type(iterations) is not int or iterations <= 0:
        return None, None

    # Initialize the centroids.
    C = initialize(X, k)

    if C is None:
        return None, None

    # Find the feature-wise bounds for empty-cluster reinitialization.
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    # Repeat assignment and centroid updates.
    for _ in range(iterations):
        # Save the current centroids for the convergence check.
        C_prev = C.copy()

        # Calculate distances from every point to every centroid.
        distances = np.linalg.norm(
            X[:, np.newaxis, :] - C,
            axis=2
        )

        # Assign each point to its nearest centroid.
        clss = np.argmin(distances, axis=1)

        # Update each centroid.
        for j in range(k):
            points = X[clss == j]

            if points.shape[0] == 0:
                # Reinitialize a centroid if its cluster is empty.
                C[j] = np.random.uniform(min_vals, max_vals)
            else:
                # Move the centroid to the mean of its assigned points.
                C[j] = np.mean(points, axis=0)

        # Stop if none of the centroids changed.
        if np.array_equal(C, C_prev):
            break

    # Reassign points using the final centroids.
    distances = np.linalg.norm(
        X[:, np.newaxis, :] - C,
        axis=2
    )
    clss = np.argmin(distances, axis=1)

    return C, clss
