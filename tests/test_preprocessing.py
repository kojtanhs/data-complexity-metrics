"""
Unit tests for the DataPreprocessor component.
"""

import numpy as np
import pandas as pd
import pytest
from data_complexity.preprocessing import DataPreprocessor


def test_preprocessor_imputation_and_scaling():
    """
    Validates that missing values are correctly imputed with the median 
    and features are standardized (mean ~ 0).
    """
    # Create dummy dataset with a NaN value and different magnitudes
    data = {
        'feature_1': [10.0, 20.0, np.nan, 40.0, 50.0],
        'feature_2': [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
        'credit_risk': [0, 0, 1, 1, 1]
    }
    df = pd.DataFrame(data)
    
    preprocessor = DataPreprocessor(target_column='credit_risk', scale_features=True)
    X, y = preprocessor.fit_transform(df)

    # 1. Assert shapes match expected matrix dimensions (5 samples, 2 features)
    assert X.shape == (5, 2)
    assert len(y) == 5
    
    # 2. Check that NaN in feature_1 was imputed with the median (30.0)
    # The mean of [10, 20, 30, 40, 50] is 30.0, standard deviation is ~15.81
    # Standardized value for the center (30) must be exactly 0
    assert np.allclose(np.mean(X, axis=0), 0.0, atol=1e-7)
    assert np.allclose(np.std(X, axis=0, ddof=1), 1.0, atol=1e-7)


def test_preprocessor_missing_target_exception():
    """
    Validates that a KeyError is raised if the target column is missing.
    """
    df = pd.DataFrame({'feat_1': [1, 2], 'feat_2': [3, 4]})
    preprocessor = DataPreprocessor(target_column='non_existent_column')
    
    with pytest.raises(KeyError):
        preprocessor.fit_transform(df)
