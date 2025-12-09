"""
Utility functions for the insurance risk analytics project.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load insurance data from CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataframe
    """
    return pd.read_csv(file_path)


def calculate_loss_ratio(claims: pd.Series, premiums: pd.Series) -> float:
    """
    Calculate loss ratio: TotalClaims / TotalPremium
    
    Parameters:
    -----------
    claims : pd.Series
        Total claims amounts
    premiums : pd.Series
        Total premium amounts
        
    Returns:
    --------
    float
        Loss ratio
    """
    total_claims = claims.sum()
    total_premiums = premiums.sum()
    
    if total_premiums == 0:
        return np.nan
    
    return total_claims / total_premiums


def calculate_claim_frequency(df: pd.DataFrame, claims_col: str = 'TotalClaims') -> float:
    """
    Calculate claim frequency: proportion of policies with at least one claim.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Insurance data
    claims_col : str
        Name of the claims column
        
    Returns:
    --------
    float
        Claim frequency (0 to 1)
    """
    policies_with_claims = (df[claims_col] > 0).sum()
    total_policies = len(df)
    
    if total_policies == 0:
        return np.nan
    
    return policies_with_claims / total_policies


def calculate_claim_severity(df: pd.DataFrame, claims_col: str = 'TotalClaims') -> float:
    """
    Calculate claim severity: average claim amount given a claim occurred.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Insurance data
    claims_col : str
        Name of the claims column
        
    Returns:
    --------
    float
        Average claim severity
    """
    claims_only = df[df[claims_col] > 0][claims_col]
    
    if len(claims_only) == 0:
        return np.nan
    
    return claims_only.mean()


def calculate_margin(premiums: pd.Series, claims: pd.Series) -> pd.Series:
    """
    Calculate margin: TotalPremium - TotalClaims
    
    Parameters:
    -----------
    premiums : pd.Series
        Total premium amounts
    claims : pd.Series
        Total claims amounts
        
    Returns:
    --------
    pd.Series
        Margin values
    """
    return premiums - claims

