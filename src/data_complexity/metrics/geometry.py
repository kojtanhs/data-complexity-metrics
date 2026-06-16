"""
Module: geometry.py
Description: Computes geometric and neighborhood-based metrics to analyze 
             the topological shape and boundary complexity of datasets.
"""

import numpy as np
from scipy.spatial.distance import cdist


def neighborhood_frontier_ratio(X: np.ndarray, y: np.ndarray) -> float:
    """
    Computes the fraction of instances whose nearest neighbor belongs to the opposite class.
    This serves as a geometric proxy for boundary roughness (N1-style complexity).
    
    A value close to 1.0 means the classes are highly interleaved or mixed at a 
    local geometric level, implying maximum topological complexity.

    Parameters:
    -----------
    X : np.ndarray
        Standardized feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target binary labels array of shape (n_samples,).

    Returns:
    --------
    float
        The ratio of boundary instances over total instances (between 0.0 and 1.0).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    
    n_samples = X.shape[0]
    if n_samples < 2:
        raise ValueError("Dataset must contain at least 2 instances to compute neighborhood metrics.")

    # 1. Compute pairwise Euclidean distance matrix (Efficient Scipy implementation)
    # distance_matrix[i, j] is the distance between sample i and sample j
    distance_matrix = cdist(X, X, metric='euclidean')

    # 2. Mask the diagonal to ignore self-distance (which is always 0.0)
    np.fill_diagonal(distance_matrix, np.inf)

    # 3. Find the index of the nearest neighbor for each sample
    nearest_neighbor_indices = np.argmin(distance_matrix, axis=1)

    # 4. Extract the labels of those nearest neighbors
    nearest_neighbor_labels = y[nearest_neighbor_indices]

    # 5. Count how many samples have a nearest neighbor from the opposite class
    boundary_points_mask = (y != nearest_neighbor_labels)
    boundary_points_count = np.sum(boundary_points_mask)

    return float(boundary_points_count / n_samples)
