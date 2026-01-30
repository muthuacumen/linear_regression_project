"""
Preprocessing Module
====================

Responsibilities:
- Handle missing values
- Feature scaling (normalization/standardization)
- Train/test split (temporal for time-series)
- Data validation before training

Functions:
----------
- handle_missing_values(df, strategy) -> pd.DataFrame
- scale_features(X, method) -> Tuple[np.ndarray, scaler]
- temporal_train_test_split(df, ratio) -> Tuple[df, df]
- prepare_regression_data(df, feature, target) -> Tuple[X, y]
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class ScalerParams:
    """Parameters for feature scaling."""
    method: str  # 'minmax' or 'standard'
    mean: float = 0.0
    std: float = 1.0
    min_val: float = 0.0
    max_val: float = 1.0


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'forward_fill',
    columns: Optional[list] = None
) -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Parameters:
    -----------
    df : pd.DataFrame
        Input data
    strategy : str
        Strategy for handling missing values:
        - 'forward_fill': Use previous valid value (best for time-series)
        - 'backward_fill': Use next valid value
        - 'mean': Fill with column mean
        - 'median': Fill with column median
        - 'drop': Remove rows with missing values
    columns : list, optional
        Specific columns to process (default: all numeric)

    Returns:
    --------
    pd.DataFrame : Data with missing values handled

    Notes:
    ------
    For time-series data, forward_fill is preferred as it:
    - Preserves temporal order
    - Doesn't use future information
    - Simulates real-world sensor behavior
    """
    result = df.copy()

    if columns is None:
        columns = result.select_dtypes(include=[np.number]).columns.tolist()

    before_missing = result[columns].isnull().sum().sum()

    if strategy == 'forward_fill':
        result[columns] = result[columns].ffill()
        # Fill remaining NaN at the start with backward fill
        result[columns] = result[columns].bfill()
    elif strategy == 'backward_fill':
        result[columns] = result[columns].bfill()
    elif strategy == 'mean':
        for col in columns:
            result[col] = result[col].fillna(result[col].mean())
    elif strategy == 'median':
        for col in columns:
            result[col] = result[col].fillna(result[col].median())
    elif strategy == 'drop':
        result = result.dropna(subset=columns)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    after_missing = result[columns].isnull().sum().sum()
    print(f"Missing values: {before_missing} -> {after_missing} (strategy: {strategy})")

    return result


def scale_features(
    X: np.ndarray,
    method: str = 'minmax',
    fit: bool = True,
    scaler_params: Optional[ScalerParams] = None
) -> Tuple[np.ndarray, ScalerParams]:
    """
    Scale features using normalization or standardization.

    Parameters:
    -----------
    X : np.ndarray
        Feature array (n_samples,) or (n_samples, 1)
    method : str
        'minmax': Scale to [0, 1] range (Normalization)
        'standard': Zero mean, unit variance (Standardization)
    fit : bool
        If True, compute scaling parameters from X
        If False, use provided scaler_params
    scaler_params : ScalerParams, optional
        Pre-computed scaling parameters (for transform only)

    Returns:
    --------
    Tuple[np.ndarray, ScalerParams] : Scaled data and parameters

    Notes:
    ------
    For linear regression:
    - MinMax is preferred when data has known bounds
    - Standard is preferred for gradient descent optimization
    - For this workshop, we use MinMax for interpretability
    """
    X = np.asarray(X).flatten()

    if fit:
        params = ScalerParams(method=method)

        if method == 'minmax':
            params.min_val = X.min()
            params.max_val = X.max()
            X_scaled = (X - params.min_val) / (params.max_val - params.min_val + 1e-8)
        elif method == 'standard':
            params.mean = X.mean()
            params.std = X.std()
            X_scaled = (X - params.mean) / (params.std + 1e-8)
        else:
            raise ValueError(f"Unknown method: {method}")
    else:
        if scaler_params is None:
            raise ValueError("scaler_params required when fit=False")
        params = scaler_params

        if method == 'minmax':
            X_scaled = (X - params.min_val) / (params.max_val - params.min_val + 1e-8)
        elif method == 'standard':
            X_scaled = (X - params.mean) / (params.std + 1e-8)

    return X_scaled, params


def inverse_scale(
    X_scaled: np.ndarray,
    scaler_params: ScalerParams
) -> np.ndarray:
    """
    Inverse transform scaled data back to original scale.

    Parameters:
    -----------
    X_scaled : np.ndarray
        Scaled data
    scaler_params : ScalerParams
        Scaling parameters used for transformation

    Returns:
    --------
    np.ndarray : Data in original scale
    """
    X_scaled = np.asarray(X_scaled).flatten()

    if scaler_params.method == 'minmax':
        X_original = X_scaled * (scaler_params.max_val - scaler_params.min_val) + scaler_params.min_val
    elif scaler_params.method == 'standard':
        X_original = X_scaled * scaler_params.std + scaler_params.mean
    else:
        raise ValueError(f"Unknown method: {scaler_params.method}")

    return X_original


def temporal_train_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    time_column: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally (not randomly) for time-series.

    Parameters:
    -----------
    df : pd.DataFrame
        Time-series data
    train_ratio : float
        Fraction of data for training (default 0.8)
    time_column : str, optional
        Name of time column (auto-detected if None)

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame] : (train_df, test_df)

    Notes:
    ------
    Why temporal split for time-series:
    - Random split causes data leakage (future -> past)
    - Temporal split mimics real deployment
    - First N% for training, last (100-N)% for testing
    - Preserves temporal order for trend detection
    """
    # Sort by time
    if time_column is None:
        time_column = 'Time' if 'Time' in df.columns else 'timestamp'

    if time_column in df.columns:
        df = df.sort_values(time_column).reset_index(drop=True)

    # Split at train_ratio point
    split_idx = int(len(df) * train_ratio)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"Temporal split: {len(train_df):,} train, {len(test_df):,} test ({train_ratio:.0%}/{1-train_ratio:.0%})")

    return train_df, test_df


def prepare_regression_data(
    df: pd.DataFrame,
    feature_column: str,
    target_column: str,
    scale_features: bool = True,
    scale_method: str = 'minmax'
) -> Dict[str, Any]:
    """
    Prepare data for univariate linear regression.

    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed data
    feature_column : str
        Name of X (independent variable)
    target_column : str
        Name of y (dependent variable)
    scale_features : bool
        Whether to scale features
    scale_method : str
        Scaling method ('minmax' or 'standard')

    Returns:
    --------
    dict : {
        'X': np.ndarray,
        'y': np.ndarray,
        'X_scaler': ScalerParams,
        'y_scaler': ScalerParams,
        'n_samples': int
    }
    """
    # Extract feature and target
    X = df[feature_column].values.astype(float)
    y = df[target_column].values.astype(float)

    # Remove NaN
    mask = ~(np.isnan(X) | np.isnan(y))
    X = X[mask]
    y = y[mask]

    result = {
        'X_raw': X.copy(),
        'y_raw': y.copy(),
        'n_samples': len(X),
        'feature_name': feature_column,
        'target_name': target_column
    }

    if scale_features:
        X, X_scaler = scale_features_func(X, method=scale_method)
        y, y_scaler = scale_features_func(y, method=scale_method)
        result['X'] = X
        result['y'] = y
        result['X_scaler'] = X_scaler
        result['y_scaler'] = y_scaler
    else:
        result['X'] = X
        result['y'] = y
        result['X_scaler'] = None
        result['y_scaler'] = None

    print(f"Prepared {result['n_samples']:,} samples: X={feature_column}, y={target_column}")

    return result


# Alias to avoid naming conflict
scale_features_func = scale_features


def get_preprocessing_summary(
    original_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Dict:
    """
    Generate summary of preprocessing steps.

    Returns:
    --------
    dict : Summary statistics
    """
    return {
        'original_records': len(original_df),
        'processed_records': len(processed_df),
        'records_removed': len(original_df) - len(processed_df),
        'train_records': len(train_df),
        'test_records': len(test_df),
        'train_ratio': len(train_df) / len(processed_df),
        'test_ratio': len(test_df) / len(processed_df)
    }


# =============================================================================
# MAIN - Test module independently
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("PREPROCESSING MODULE TEST")
    print("=" * 60)

    # Create test data with missing values
    np.random.seed(42)
    test_data = pd.DataFrame({
        'Time': pd.date_range('2024-01-01', periods=100, freq='1min'),
        'Axis #1': np.random.uniform(5, 15, 100),
        'Axis #2': np.random.uniform(4, 12, 100),
    })

    # Introduce missing values
    test_data.loc[10:15, 'Axis #1'] = np.nan
    test_data.loc[50:55, 'Axis #2'] = np.nan

    print(f"\nBefore preprocessing: {test_data.isnull().sum().sum()} missing values")

    # Handle missing values
    cleaned = handle_missing_values(test_data, strategy='forward_fill')

    # Scale features
    X = cleaned['Axis #1'].values
    X_scaled, scaler = scale_features_func(X, method='minmax')
    print(f"Scaling: min={X_scaled.min():.4f}, max={X_scaled.max():.4f}")

    # Temporal split
    train, test = temporal_train_test_split(cleaned, train_ratio=0.8)

    # Summary
    summary = get_preprocessing_summary(test_data, cleaned, train, test)
    print(f"\nSummary: {summary}")

    print("\n[OK] Preprocessing module test passed")
