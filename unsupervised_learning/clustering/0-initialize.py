#!/usr/bin/env python3
"""
Module for initializing centroids for the K-means clustering algorithm.
"""
import numpy as np


def initialize(X, k):
   """
    Initialize K-means centroids using random sampling.

    The centroids are sampled from a uniform distribution bounded by
    the minimum and maximum values of each feature in the dataset.

    Args:
        X (numpy.ndarray): Input data of shape (n, d), where n is the
            number of data points and d is the number of features.
        k (int): Number of centroids to initialize. Must be positive.

    Returns:
        numpy.ndarray: Randomly initialized centroids of shape (k, d).
        None: If the input data is invalid or k is not a positive integer.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    centroids = np.random.uniform(
        min_vals,
        max_vals,
        size=(k, X.shape[1])
    )

    return centroids
