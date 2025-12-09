"""
Data processing utilities for insurance risk analytics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import logging

from logger import setup_logger

# Set up logger
logger = setup_logger('data_processing')

# Expected columns for insurance data
EXPECTED_COLUMNS = {
    'policy': ['UnderwrittenCoverID', 'PolicyID', 'TransactionMonth'],
    'client': ['IsVATRegistered', 'Citizenship', 'LegalType', 'Title', 'Language', 
               'Bank', 'AccountType', 'MaritalStatus', 'Gender'],
    'location': ['Country', 'Province', 'PostalCode', 'MainCrestaZone', 'SubCrestaZone'],
    'vehicle': ['ItemType', 'Mmcode', 'VehicleType', 'RegistrationYear', 'Make', 'Model',
                'Cylinders', 'Cubiccapacity', 'Kilowatts', 'Bodytype', 'NumberOfDoors',
                'VehicleIntroDate', 'CustomValueEstimate', 'AlarmImmobiliser', 'TrackingDevice',
                'CapitalOutstanding', 'NewVehicle', 'WrittenOff', 'Rebuilt', 'Converted',
                'CrossBorder', 'NumberOfVehiclesInFleet'],
    'plan': ['SumInsured', 'TermFrequency', 'CalculatedPremiumPerTerm', 'ExcessSelected',
             'CoverCategory', 'CoverType', 'CoverGroup', 'Section', 'Product',
             'StatutoryClass', 'StatutoryRiskType'],
    'financial': ['TotalPremium', 'TotalClaims']
}

REQUIRED_COLUMNS = ['PolicyID', 'TotalPremium', 'TotalClaims']  # Minimum required columns


def load_and_validate_data(file_path: str, 
                           validate_columns: bool = True,
                           required_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load and perform initial validation on insurance data with comprehensive error handling.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    validate_columns : bool
        Whether to validate expected columns
    required_cols : list, optional
        List of required columns. If None, uses default REQUIRED_COLUMNS.
        
    Returns:
    --------
    pd.DataFrame
        Loaded and validated dataframe
        
    Raises:
    -------
    FileNotFoundError
        If the file doesn't exist
    ValueError
        If required columns are missing or data validation fails
    pd.errors.EmptyDataError
        If the file is empty
    """
    if required_cols is None:
        required_cols = REQUIRED_COLUMNS
    
    file_path_obj = Path(file_path)
    
    # Check if file exists
    if not file_path_obj.exists():
        error_msg = f"Data file not found: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Check if file is readable
    if not file_path_obj.is_file():
        error_msg = f"Path is not a file: {file_path}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"Loading data from: {file_path}")
    
    try:
        # Load CSV with error handling
        df = pd.read_csv(file_path, low_memory=False)
        logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
    except pd.errors.EmptyDataError as e:
        error_msg = f"Data file is empty: {file_path}"
        logger.error(error_msg)
        raise pd.errors.EmptyDataError(error_msg) from e
    except pd.errors.ParserError as e:
        error_msg = f"Error parsing CSV file {file_path}: {str(e)}"
        logger.error(error_msg)
        raise pd.errors.ParserError(error_msg) from e
    except Exception as e:
        error_msg = f"Unexpected error loading data file {file_path}: {str(e)}"
        logger.error(error_msg)
        raise
    
    # Check if dataframe is empty
    if df.empty:
        error_msg = "Loaded dataframe is empty"
        logger.warning(error_msg)
        raise ValueError(error_msg)
    
    # Validate required columns
    if validate_columns:
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            error_msg = f"Missing required columns: {missing_cols}"
            logger.error(error_msg)
            logger.info(f"Available columns: {list(df.columns)}")
            raise ValueError(error_msg)
        logger.info(f"All required columns present: {required_cols}")
    
    # Convert TransactionMonth to datetime if it exists
    if 'TransactionMonth' in df.columns:
        try:
            df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
            invalid_dates = df['TransactionMonth'].isna().sum()
            if invalid_dates > 0:
                logger.warning(f"Found {invalid_dates} invalid dates in TransactionMonth column")
        except Exception as e:
            logger.warning(f"Error converting TransactionMonth to datetime: {str(e)}")
    
    # Validate data types for key numeric columns
    numeric_cols = ['TotalPremium', 'TotalClaims']
    for col in numeric_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                invalid_values = df[col].isna().sum()
                if invalid_values > 0:
                    logger.warning(f"Found {invalid_values} invalid numeric values in {col}")
            except Exception as e:
                logger.warning(f"Error converting {col} to numeric: {str(e)}")
    
    logger.info("Data loading and validation completed successfully")
    return df


def get_data_summary(df: pd.DataFrame) -> Dict:
    """
    Generate comprehensive data summary with error handling.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    dict
        Dictionary containing summary statistics
        
    Raises:
    -------
    ValueError
        If dataframe is empty or invalid
    """
    try:
        if df is None or df.empty:
            error_msg = "Cannot generate summary for empty dataframe"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object', 'category']).columns),
            'datetime_columns': list(df.select_dtypes(include=['datetime64']).columns),
        }
        
        logger.debug(f"Generated summary for dataframe with shape {df.shape}")
        return summary
    except Exception as e:
        error_msg = f"Error generating data summary: {str(e)}"
        logger.error(error_msg)
        raise


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for missing values in the dataframe with error handling.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with missing value statistics
        
    Raises:
    -------
    ValueError
        If dataframe is empty or invalid
    """
    try:
        if df is None or df.empty:
            error_msg = "Cannot check missing values for empty dataframe"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        missing = pd.DataFrame({
            'column': df.columns,
            'missing_count': df.isnull().sum().values,
            'missing_percentage': (df.isnull().sum() / len(df) * 100).values,
            'dtype': df.dtypes.values
        })
        
        missing = missing[missing['missing_count'] > 0].sort_values('missing_percentage', ascending=False)
        
        if len(missing) > 0:
            logger.warning(f"Found {len(missing)} columns with missing values")
        else:
            logger.info("No missing values found in dataframe")
        
        return missing
    except Exception as e:
        error_msg = f"Error checking missing values: {str(e)}"
        logger.error(error_msg)
        raise


def detect_outliers_iqr(df: pd.DataFrame, column: str, factor: float = 1.5) -> Dict:
    """
    Detect outliers using IQR method with error handling.
    
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
    dict
        Dictionary with outlier information
        
    Raises:
    -------
    ValueError
        If column doesn't exist or is not numeric
    """
    try:
        if df is None or df.empty:
            error_msg = "Cannot detect outliers for empty dataframe"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if column not in df.columns:
            error_msg = f"Column '{column}' not found in dataframe"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if not pd.api.types.is_numeric_dtype(df[column]):
            error_msg = f"Column '{column}' is not numeric"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Remove NaN values for calculation
        col_data = df[column].dropna()
        if len(col_data) == 0:
            error_msg = f"Column '{column}' contains only NaN values"
            logger.warning(error_msg)
            return {
                'outlier_count': 0,
                'outlier_percentage': 0,
                'lower_bound': None,
                'upper_bound': None,
                'outliers': pd.DataFrame()
            }
        
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR == 0:
            logger.warning(f"IQR is 0 for column '{column}', no outliers detected")
            return {
                'outlier_count': 0,
                'outlier_percentage': 0,
                'lower_bound': Q1,
                'upper_bound': Q3,
                'outliers': pd.DataFrame()
            }
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        
        result = {
            'outlier_count': len(outliers),
            'outlier_percentage': len(outliers) / len(df) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outliers': outliers
        }
        
        logger.info(f"Detected {result['outlier_count']} outliers ({result['outlier_percentage']:.2f}%) in column '{column}'")
        return result
    except Exception as e:
        error_msg = f"Error detecting outliers in column '{column}': {str(e)}"
        logger.error(error_msg)
        raise


def calculate_descriptive_stats(df: pd.DataFrame, numeric_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculate descriptive statistics for numeric columns with error handling.
    
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
        
    Raises:
    -------
    ValueError
        If dataframe is empty or no numeric columns found
    """
    try:
        if df is None or df.empty:
            error_msg = "Cannot calculate statistics for empty dataframe"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            error_msg = "No numeric columns found in dataframe"
            logger.warning(error_msg)
            return pd.DataFrame()
        
        # Filter to only existing columns
        numeric_cols = [col for col in numeric_cols if col in df.columns]
        if not numeric_cols:
            error_msg = "None of the specified numeric columns exist in dataframe"
            logger.warning(error_msg)
            return pd.DataFrame()
        
        stats = df[numeric_cols].describe()
        
        # Add additional statistics with error handling
        try:
            additional_stats = pd.DataFrame({
                'skewness': df[numeric_cols].skew(),
                'kurtosis': df[numeric_cols].kurtosis(),
                'coefficient_of_variation': df[numeric_cols].std() / df[numeric_cols].mean()
            })
            stats = pd.concat([stats, additional_stats.T])
        except Exception as e:
            logger.warning(f"Error calculating additional statistics: {str(e)}")
        
        logger.debug(f"Calculated descriptive statistics for {len(numeric_cols)} numeric columns")
        return stats
    except Exception as e:
        error_msg = f"Error calculating descriptive statistics: {str(e)}"
        logger.error(error_msg)
        raise

