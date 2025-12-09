"""
Data validation utilities for insurance risk analytics.
Validates expected columns, data types, and data quality.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from logger import setup_logger

logger = setup_logger('data_validation')

# Expected columns from the project specification
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

REQUIRED_COLUMNS = ['PolicyID', 'TotalPremium', 'TotalClaims']

# Expected data types for key columns
EXPECTED_DTYPES = {
    'PolicyID': 'object',
    'TotalPremium': 'float64',
    'TotalClaims': 'float64',
    'TransactionMonth': 'datetime64[ns]',
    'Province': 'object',
    'PostalCode': 'object',
    'Gender': 'object',
    'VehicleType': 'object',
    'Make': 'object'
}


def validate_columns(df: pd.DataFrame, 
                    required_cols: Optional[List[str]] = None,
                    expected_cols: Optional[Dict[str, List[str]]] = None) -> Tuple[bool, List[str]]:
    """
    Validate that required and expected columns are present.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    required_cols : list, optional
        List of required columns. If None, uses default REQUIRED_COLUMNS.
    expected_cols : dict, optional
        Dictionary of expected columns by category. If None, uses default EXPECTED_COLUMNS.
        
    Returns:
    --------
    tuple
        (is_valid, missing_columns)
    """
    if required_cols is None:
        required_cols = REQUIRED_COLUMNS
    if expected_cols is None:
        expected_cols = EXPECTED_COLUMNS
    
    missing_required = [col for col in required_cols if col not in df.columns]
    
    if missing_required:
        logger.error(f"Missing required columns: {missing_required}")
        return False, missing_required
    
    logger.info(f"All required columns present: {required_cols}")
    return True, []


def validate_data_types(df: pd.DataFrame, 
                       expected_dtypes: Optional[Dict[str, str]] = None) -> Tuple[bool, Dict[str, str]]:
    """
    Validate data types for key columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    expected_dtypes : dict, optional
        Dictionary mapping column names to expected dtypes
        
    Returns:
    --------
    tuple
        (is_valid, type_mismatches)
    """
    if expected_dtypes is None:
        expected_dtypes = EXPECTED_DTYPES
    
    type_mismatches = {}
    
    for col, expected_dtype in expected_dtypes.items():
        if col not in df.columns:
            continue
        
        actual_dtype = str(df[col].dtype)
        
        # Flexible type checking
        if expected_dtype == 'float64' and not pd.api.types.is_numeric_dtype(df[col]):
            type_mismatches[col] = f"Expected numeric, got {actual_dtype}"
        elif expected_dtype == 'object' and not pd.api.types.is_object_dtype(df[col]):
            type_mismatches[col] = f"Expected object, got {actual_dtype}"
        elif expected_dtype == 'datetime64[ns]' and not pd.api.types.is_datetime64_any_dtype(df[col]):
            type_mismatches[col] = f"Expected datetime, got {actual_dtype}"
    
    if type_mismatches:
        logger.warning(f"Data type mismatches found: {type_mismatches}")
        return False, type_mismatches
    
    logger.info("Data types validated successfully")
    return True, {}


def validate_data_quality(df: pd.DataFrame) -> Dict:
    """
    Perform comprehensive data quality validation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    dict
        Dictionary with validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'summary': {}
    }
    
    try:
        # Check if dataframe is empty
        if df.empty:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Dataframe is empty")
            return validation_results
        
        validation_results['summary']['total_rows'] = len(df)
        validation_results['summary']['total_columns'] = len(df.columns)
        
        # Validate required columns
        is_valid, missing_cols = validate_columns(df)
        if not is_valid:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_cols}")
        
        # Validate data types
        is_valid_types, type_mismatches = validate_data_types(df)
        if not is_valid_types:
            validation_results['warnings'].append(f"Data type mismatches: {type_mismatches}")
        
        # Check for negative values in financial columns
        if 'TotalPremium' in df.columns:
            negative_premiums = (df['TotalPremium'] < 0).sum()
            if negative_premiums > 0:
                validation_results['warnings'].append(f"Found {negative_premiums} negative premium values")
        
        if 'TotalClaims' in df.columns:
            negative_claims = (df['TotalClaims'] < 0).sum()
            if negative_claims > 0:
                validation_results['warnings'].append(f"Found {negative_claims} negative claim values")
        
        # Check for duplicate PolicyIDs
        if 'PolicyID' in df.columns:
            duplicates = df['PolicyID'].duplicated().sum()
            if duplicates > 0:
                validation_results['warnings'].append(f"Found {duplicates} duplicate PolicyIDs")
        
        logger.info(f"Data quality validation completed. Valid: {validation_results['is_valid']}")
        return validation_results
        
    except Exception as e:
        error_msg = f"Error during data quality validation: {str(e)}"
        logger.error(error_msg)
        validation_results['is_valid'] = False
        validation_results['errors'].append(error_msg)
        return validation_results

