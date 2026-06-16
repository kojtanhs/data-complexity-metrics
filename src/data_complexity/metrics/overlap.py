"""
Module: overlap.py
Description: Computes class overlap metrics based on data geometry and statistical 
             discriminant ratios to assess ML dataset complexity.
"""

import numpy as np
import pandas as pd


def fisher_discriminant_ratio(X: np.ndarray, y: np.ndarray) -> float:
    """
    Computes the Maximum Fisher's Discriminant Ratio (F1) for a binary classification dataset.
    
    A higher F1 value implies that at least one feature can linearly separate 
    the classes easily, meaning lower complexity.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target labels array of shape (n_samples,) containing binary classes (0 and 1).

    Returns:
    --------
    float
        The maximum Fisher's Discriminant Ratio across all features.
    """
    # Defensive programming: Ensure inputs are numpy arrays
    X = np.asarray(X)
    y = np.asarray(y)
    
    if len(X) == 0:
        raise ValueError("Input feature matrix X cannot be empty.")
        
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("F1 metric is strictly designed for binary classification tasks.")

    class_1_mask = (y == classes[0])
    class_2_mask = (y == classes[1])

    f1_ratios = []

    # Iterate over each feature column (vectorized math per column)
    for col_idx in range(X.shape[1]):
        feat_c1 = X[class_1_mask, col_idx]
        feat_c2 = X[class_2_mask, col_idx]

        mean_c1, mean_c2 = np.mean(feat_c1), np.mean(feat_c2)
        var_c1, var_c2 = np.var(feat_c1, ddof=1), np.var(feat_c2, ddof=1)

        numerator = (mean_c1 - mean_c2) ** 2
        denominator = var_c1 + var_c2

        # Handle edge case: Avoid division by zero if variances are both 0
        if denominator == 0:
            f1_ratios.append(0.0)
        else:
            f1_ratio = numerator / denominator
            f1_ratios.append(f1_ratio)

    # Return the maximum separation power found among features (F1 Metric)
    return float(np.max(f1_ratios))
