
#!/usr/bin/env python3
import numpy as np


def initialize(X, k):
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

