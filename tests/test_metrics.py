"""
Unit tests for data complexity metrics modules.
Executed using pytest framework.
"""

import numpy as np
import pytest
from data_complexity.metrics.overlap import fisher_discriminant_ratio


def test_fisher_discriminant_ratio_perfect_separation():
    """
    Test F1 metric with two perfectly distinct and linearly separable clusters.
    Expects a high, clean separation ratio.
    """
    # Feature 1 separates classes perfectly; Feature 2 is noise
    X = np.array([
        [1.0, 5.0],
        [1.2, 5.2],
        [1.1, 4.8],
        [10.0, 5.1],
        [10.2, 4.9],
        [10.1, 5.0]
    ])
    y = np.array([0, 0, 0, 1, 1, 1])

    f1_result = fisher_discriminant_ratio(X, y)
    
    # Assert F1 is a valid positive float and significantly high
    assert isinstance(f1_result, float)
    assert f1_result > 10.0


def test_fisher_discriminant_ratio_zero_variance():
    """
    Edge case: Features have zero variance within classes.
    Validates defensive division-by-zero handling.
    """
    X = np.array([
        [2.0, 3.0],
        [2.0, 3.0],
        [2.0, 3.0],
        [2.0, 3.0]
    ])
    y = np.array([0, 0, 1, 1])

    f1_result = fisher_discriminant_ratio(X, y)
    
    # The denominator (var_c1 + var_c2) will be 0. Result should catch this and be 0.0
    assert f1_result == 0.0


def test_fisher_discriminant_ratio_invalid_multiclass():
    """
    Exception testing: Passing 3 unique classes instead of a binary problem.
    Expects a strict ValueError exception.
    """
    X = np.random.rand(9, 2)
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    with pytest.raises(ValueError, match="strictly designed for binary classification"):
        fisher_discriminant_ratio(X, y)


from data_complexity.metrics.geometry import neighborhood_frontier_ratio

# ... (Tus funciones de prueba anteriores se quedan intactas)

def test_neighborhood_frontier_ratio_interleaved():
    """
    Test the neighborhood geometric metric with highly interleaved data.
    If points alternate classes in space, the frontier ratio should be 1.0.
    """
    # 1D line where classes alternate exactly at every step
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 1, 0, 1])

    # For point at 2.0 (class 1), neighbors are 1.0 (class 0) and 3.0 (class 0). Enemy is nearest!
    ratio = neighborhood_frontier_ratio(X, y)
    
    # Every single point has an enemy as its closest neighbor
    assert ratio == 1.0
