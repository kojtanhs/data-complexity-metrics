"""
Module: preprocessing.py
Description: Robust pipeline component to clean, impute, and standardize 
             data matrices before complexity evaluation.
"""

import numpy as np
import pandas as pd


class DataPreprocessor:
    """
    Handles defensive data cleaning and scaling for matrix pipelines.
    """
    def __init__(self, target_column: str, scale_features: bool = True):
        """
        Parameters:
        -----------
        target_column : str
            The name of the label/class column in the source DataFrame.
        scale_features : bool, optional
            Whether to apply standard normalization (Z-score) to features. Defaults to True.
        """
        self.target_column = target_column
        self.scale_features = scale_features
        self.means = None
        self.stds = None
        self.medians = None

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Fits the preprocessor to the DataFrame states and transforms the data 
        into analytical Numpy arrays.

        Parameters:
        -----------
        df : pd.DataFrame
            The raw input dataset containing both features and the target.

        Returns:
        --------
        tuple[np.ndarray, np.ndarray]
            X (feature matrix standardized and cleaned) and y (target labels).
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
            
        if self.target_column not in df.columns:
            raise KeyError(f"Target column '{self.target_column}' not found in DataFrame.")

        # Separate features and target
        X_raw = df.drop(columns=[self.target_column]).copy()
        y_raw = df[self.target_column].to_numpy()

        # Handle Missing Values (Imputation using column medians)
        # Select only numeric columns for mathematical operations
        numeric_cols = X_raw.select_dtypes(include=[np.number]).columns
        self.medians = X_raw[numeric_cols].median()
        X_raw[numeric_cols] = X_raw[numeric_cols].fillna(self.medians)

        # Convert features to numpy array
        X = X_raw[numeric_cols].to_numpy(dtype=np.float64)

        # Feature Scaling (Standard Normalization: Mean=0, Std=1)
        if self.scale_features:
            self.means = np.mean(X, axis=0)
            self.stds = np.std(X, axis=0, ddof=1)
            
            # Handle edge case: Avoid division by zero if a feature column is constant
            self.stds[self.stds == 0.0] = 1.0
            
            X = (X - self.means) / self.stds

        return X, y_raw
