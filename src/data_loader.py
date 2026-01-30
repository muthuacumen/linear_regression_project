"""
Data Loader Module
==================

Responsibilities:
- Load robot sensor data from CSV or database
- Validate data schema
- Handle multiple robot datasets
- Provide data summary statistics

Functions:
----------
- load_csv(file_path) -> pd.DataFrame
- load_from_db(db_config) -> pd.DataFrame
- validate_schema(df) -> bool
- get_data_summary(df) -> dict
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime
import os


def load_csv(file_path: str, robot_id: Optional[str] = None) -> pd.DataFrame:
    """
    Load robot sensor data from CSV file.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    robot_id : str, optional
        Filter for specific robot ID

    Returns:
    --------
    pd.DataFrame : Loaded and validated data

    Raises:
    -------
    FileNotFoundError : If file doesn't exist
    ValueError : If schema validation fails
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)

    # Parse timestamp column
    time_col = 'Time' if 'Time' in df.columns else 'timestamp'
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])

    # Filter by robot_id if specified
    if robot_id and 'robot_id' in df.columns:
        df = df[df['robot_id'] == robot_id].copy()
        print(f"Filtered to robot: {robot_id}")

    print(f"Loaded {len(df):,} records")
    return df


def load_from_db(db_config: Dict) -> pd.DataFrame:
    """
    Load robot sensor data from PostgreSQL database.

    Parameters:
    -----------
    db_config : dict
        Database connection parameters:
        - host, database, user, password, port

    Returns:
    --------
    pd.DataFrame : Data from database
    """
    import psycopg2

    conn_string = (
        f"host={db_config['host']} "
        f"dbname={db_config['database']} "
        f"user={db_config['user']} "
        f"password={db_config['password']} "
        f"port={db_config.get('port', 5432)}"
    )

    print("Connecting to database...")
    conn = psycopg2.connect(conn_string)

    query = "SELECT * FROM robot_data ORDER BY timestamp"
    df = pd.read_sql(query, conn)

    conn.close()
    print(f"Loaded {len(df):,} records from database")

    return df


def validate_schema(df: pd.DataFrame) -> Tuple[bool, list]:
    """
    Validate that DataFrame has required columns.

    Parameters:
    -----------
    df : pd.DataFrame
        Data to validate

    Returns:
    --------
    Tuple[bool, list] : (is_valid, list of issues)
    """
    issues = []

    # Check for time column
    if 'Time' not in df.columns and 'timestamp' not in df.columns:
        issues.append("Missing time column (Time or timestamp)")

    # Check for at least one axis column
    axis_cols = [c for c in df.columns if 'Axis' in c or 'axis_' in c]
    if len(axis_cols) == 0:
        issues.append("No axis columns found")

    # Check for empty DataFrame
    if len(df) == 0:
        issues.append("DataFrame is empty")

    is_valid = len(issues) == 0

    if is_valid:
        print("Schema validation: PASSED")
    else:
        print(f"Schema validation: FAILED - {len(issues)} issues")
        for issue in issues:
            print(f"  - {issue}")

    return is_valid, issues


def get_data_summary(df: pd.DataFrame) -> Dict:
    """
    Generate summary statistics for the dataset.

    Parameters:
    -----------
    df : pd.DataFrame
        Robot sensor data

    Returns:
    --------
    dict : Summary statistics
    """
    # Identify axis columns
    axis_cols = [c for c in df.columns if 'Axis' in c or 'axis_' in c]

    # Time column
    time_col = 'Time' if 'Time' in df.columns else 'timestamp'

    summary = {
        'record_count': len(df),
        'column_count': len(df.columns),
        'axis_columns': axis_cols,
        'axis_count': len(axis_cols),
        'missing_values': df.isnull().sum().to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
    }

    # Time range
    if time_col in df.columns:
        summary['time_range'] = {
            'start': str(df[time_col].min()),
            'end': str(df[time_col].max()),
            'duration': str(df[time_col].max() - df[time_col].min())
        }

    # Robot IDs
    if 'robot_id' in df.columns:
        summary['robots'] = df['robot_id'].unique().tolist()
        summary['records_per_robot'] = df['robot_id'].value_counts().to_dict()

    # Axis statistics
    if axis_cols:
        summary['axis_stats'] = {}
        for col in axis_cols[:8]:  # First 8 axes
            data = df[col].dropna()
            if len(data) > 0:
                summary['axis_stats'][col] = {
                    'mean': round(data.mean(), 4),
                    'std': round(data.std(), 4),
                    'min': round(data.min(), 4),
                    'max': round(data.max(), 4)
                }

    return summary


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived features for regression.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw robot sensor data

    Returns:
    --------
    pd.DataFrame : Data with computed features
    """
    result = df.copy()

    # Time column
    time_col = 'Time' if 'Time' in result.columns else 'timestamp'

    # Compute elapsed time in hours
    if time_col in result.columns:
        result[time_col] = pd.to_datetime(result[time_col])
        start_time = result[time_col].min()
        result['elapsed_hours'] = (result[time_col] - start_time).dt.total_seconds() / 3600

    # Identify axis columns
    axis_cols = [c for c in result.columns if 'Axis' in c or 'axis_' in c]

    # Compute mean current across axes
    if axis_cols:
        result['mean_current'] = result[axis_cols].mean(axis=1)
        result['max_current'] = result[axis_cols].max(axis=1)
        result['current_std'] = result[axis_cols].std(axis=1)

    # Rolling statistics (1-hour window, ~60 samples)
    if 'mean_current' in result.columns:
        result['rolling_mean_1h'] = result['mean_current'].rolling(
            window=60, min_periods=1
        ).mean()
        result['rolling_std_1h'] = result['mean_current'].rolling(
            window=60, min_periods=1
        ).std().fillna(0)

    print(f"Computed {len(result.columns) - len(df.columns)} new features")
    return result


# =============================================================================
# MAIN - Test module independently
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("DATA LOADER MODULE TEST")
    print("=" * 60)

    # Test with sample data
    test_data = pd.DataFrame({
        'Time': pd.date_range('2024-01-01', periods=100, freq='1min'),
        'Axis #1': np.random.uniform(5, 15, 100),
        'Axis #2': np.random.uniform(4, 12, 100),
        'Axis #3': np.random.uniform(6, 18, 100),
        'robot_id': 'Robot_A'
    })

    # Validate schema
    is_valid, issues = validate_schema(test_data)

    # Get summary
    summary = get_data_summary(test_data)
    print(f"\nSummary: {summary['record_count']} records, {summary['axis_count']} axes")

    # Compute features
    featured_data = compute_features(test_data)
    print(f"New columns: {[c for c in featured_data.columns if c not in test_data.columns]}")

    print("\n[OK] Data loader module test passed")
