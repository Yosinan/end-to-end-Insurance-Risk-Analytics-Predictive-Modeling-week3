"""
Data processing utilities for insurance risk analytics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime


def load_and_validate_data(file_path: str) -> pd.DataFrame:
    """
    Load and perform initial validation on insurance data.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded and validated dataframe
    """
    df = pd.read_csv(file_path)
    
    # Convert TransactionMonth to datetime if it exists
    if 'TransactionMonth' in df.columns:
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
    
    return df


def get_data_summary(df: pd.DataFrame) -> Dict:
    """
    Generate comprehensive data summary.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    dict
        Dictionary containing summary statistics
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object', 'category']).columns),
        'datetime_columns': list(df.select_dtypes(include=['datetime64']).columns),
    }
    
    return summary


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for missing values in the dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with missing value statistics
    """
    missing = pd.DataFrame({
        'column': df.columns,
        'missing_count': df.isnull().sum().values,
        'missing_percentage': (df.isnull().sum() / len(df) * 100).values,
        'dtype': df.dtypes.values
    })
    
    missing = missing[missing['missing_count'] > 0].sort_values('missing_percentage', ascending=False)
    
    return missing


def detect_outliers_iqr(df: pd.DataFrame, column: str, factor: float = 1.5) -> pd.DataFrame:
    """
    Detect outliers using IQR method.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Column name to check for outliers
    factor : float
        IQR factor (default 1.5)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with outlier information
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    return {
        'outlier_count': len(outliers),
        'outlier_percentage': len(outliers) / len(df) * 100,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'outliers': outliers
    }


def calculate_descriptive_stats(df: pd.DataFrame, numeric_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculate descriptive statistics for numeric columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    numeric_cols : list, optional
        List of numeric columns to analyze. If None, uses all numeric columns.
        
    Returns:
    --------
    pd.DataFrame
        Descriptive statistics
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    stats = df[numeric_cols].describe()
    
    # Add additional statistics
    additional_stats = pd.DataFrame({
        'skewness': df[numeric_cols].skew(),
        'kurtosis': df[numeric_cols].kurtosis(),
        'coefficient_of_variation': df[numeric_cols].std() / df[numeric_cols].mean()
    })
    
    stats = pd.concat([stats, additional_stats.T])
    
    return stats

