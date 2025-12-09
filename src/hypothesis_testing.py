"""
Hypothesis testing functions for insurance risk analytics.
Implements statistical tests for A/B testing scenarios.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

from utils import calculate_claim_frequency, calculate_claim_severity, calculate_margin


def prepare_groups(df: pd.DataFrame, group_col: str, group_a_value: str, 
                   group_b_value: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare two groups for hypothesis testing.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    group_col : str
        Column name to group by
    group_a_value : str
        Value for Group A (control)
    group_b_value : str
        Value for Group B (test)
        
    Returns:
    --------
    tuple
        (Group A dataframe, Group B dataframe)
    """
    group_a = df[df[group_col] == group_a_value].copy()
    group_b = df[df[group_col] == group_b_value].copy()
    
    return group_a, group_b


def test_claim_frequency_difference(group_a: pd.DataFrame, group_b: pd.DataFrame,
                                   claims_col: str = 'TotalClaims',
                                   alpha: float = 0.05) -> Dict:
    """
    Test for difference in claim frequency between two groups using chi-squared test.
    
    Parameters:
    -----------
    group_a : pd.DataFrame
        Group A data
    group_b : pd.DataFrame
        Group B data
    claims_col : str
        Name of claims column
    alpha : float
        Significance level
        
    Returns:
    --------
    dict
        Test results including p-value, statistic, and conclusion
    """
    # Calculate claim frequencies
    freq_a = calculate_claim_frequency(group_a, claims_col)
    freq_b = calculate_claim_frequency(group_b, claims_col)
    
    # Create contingency table
    claims_a = (group_a[claims_col] > 0).sum()
    no_claims_a = (group_a[claims_col] == 0).sum()
    claims_b = (group_b[claims_col] > 0).sum()
    no_claims_b = (group_b[claims_col] == 0).sum()
    
    contingency_table = np.array([
        [claims_a, no_claims_a],
        [claims_b, no_claims_b]
    ])
    
    # Perform chi-squared test
    chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
    
    # Calculate proportions
    n_a = len(group_a)
    n_b = len(group_b)
    prop_a = claims_a / n_a if n_a > 0 else 0
    prop_b = claims_b / n_b if n_b > 0 else 0
    
    # Conclusion
    reject_null = p_value < alpha
    conclusion = "Reject H₀" if reject_null else "Fail to reject H₀"
    
    return {
        'test_name': 'Chi-squared test for claim frequency',
        'group_a_frequency': freq_a,
        'group_b_frequency': freq_b,
        'frequency_difference': freq_b - freq_a,
        'frequency_difference_pct': ((freq_b - freq_a) / freq_a * 100) if freq_a > 0 else 0,
        'chi2_statistic': chi2,
        'p_value': p_value,
        'alpha': alpha,
        'reject_null': reject_null,
        'conclusion': conclusion,
        'group_a_size': n_a,
        'group_b_size': n_b,
        'contingency_table': contingency_table
    }


def test_claim_severity_difference(group_a: pd.DataFrame, group_b: pd.DataFrame,
                                 claims_col: str = 'TotalClaims',
                                 alpha: float = 0.05) -> Dict:
    """
    Test for difference in claim severity between two groups using t-test or Mann-Whitney U test.
    
    Parameters:
    -----------
    group_a : pd.DataFrame
        Group A data
    group_b : pd.DataFrame
        Group B data
    claims_col : str
        Name of claims column
    alpha : float
        Significance level
        
    Returns:
    --------
    dict
        Test results including p-value, statistic, and conclusion
    """
    # Get claims only
    claims_a = group_a[group_a[claims_col] > 0][claims_col]
    claims_b = group_b[group_b[claims_col] > 0][claims_col]
    
    if len(claims_a) == 0 or len(claims_b) == 0:
        return {
            'test_name': 'Claim severity test',
            'error': 'Insufficient data: one or both groups have no claims',
            'group_a_size': len(claims_a),
            'group_b_size': len(claims_b)
        }
    
    # Calculate severities
    severity_a = calculate_claim_severity(group_a, claims_col)
    severity_b = calculate_claim_severity(group_b, claims_col)
    
    # Check normality (Shapiro-Wilk test for smaller samples)
    use_parametric = True
    if len(claims_a) <= 5000 and len(claims_b) <= 5000:
        _, p_norm_a = stats.shapiro(claims_a.sample(min(5000, len(claims_a))))
        _, p_norm_b = stats.shapiro(claims_b.sample(min(5000, len(claims_b))))
        use_parametric = p_norm_a > 0.05 and p_norm_b > 0.05
    
    # Perform appropriate test
    if use_parametric:
        # T-test
        statistic, p_value = stats.ttest_ind(claims_a, claims_b, equal_var=False)
        test_name = 'Welch\'s t-test for claim severity'
    else:
        # Mann-Whitney U test (non-parametric)
        statistic, p_value = stats.mannwhitneyu(claims_a, claims_b, alternative='two-sided')
        test_name = 'Mann-Whitney U test for claim severity'
    
    # Conclusion
    reject_null = p_value < alpha
    conclusion = "Reject H₀" if reject_null else "Fail to reject H₀"
    
    return {
        'test_name': test_name,
        'group_a_severity': severity_a,
        'group_b_severity': severity_b,
        'severity_difference': severity_b - severity_a,
        'severity_difference_pct': ((severity_b - severity_a) / severity_a * 100) if severity_a > 0 else 0,
        'statistic': statistic,
        'p_value': p_value,
        'alpha': alpha,
        'reject_null': reject_null,
        'conclusion': conclusion,
        'group_a_size': len(claims_a),
        'group_b_size': len(claims_b),
        'use_parametric': use_parametric
    }


def test_margin_difference(group_a: pd.DataFrame, group_b: pd.DataFrame,
                          premium_col: str = 'TotalPremium',
                          claims_col: str = 'TotalClaims',
                          alpha: float = 0.05) -> Dict:
    """
    Test for difference in margin (profit) between two groups using t-test.
    
    Parameters:
    -----------
    group_a : pd.DataFrame
        Group A data
    group_b : pd.DataFrame
        Group B data
    premium_col : str
        Name of premium column
    claims_col : str
        Name of claims column
    alpha : float
        Significance level
        
    Returns:
    --------
    dict
        Test results including p-value, statistic, and conclusion
    """
    # Calculate margins
    margin_a = calculate_margin(group_a[premium_col], group_a[claims_col])
    margin_b = calculate_margin(group_b[premium_col], group_b[claims_col])
    
    # Calculate means
    mean_margin_a = margin_a.mean()
    mean_margin_b = margin_b.mean()
    
    # Perform t-test
    statistic, p_value = stats.ttest_ind(margin_a, margin_b, equal_var=False)
    
    # Conclusion
    reject_null = p_value < alpha
    conclusion = "Reject H₀" if reject_null else "Fail to reject H₀"
    
    return {
        'test_name': 'Welch\'s t-test for margin difference',
        'group_a_mean_margin': mean_margin_a,
        'group_b_mean_margin': mean_margin_b,
        'margin_difference': mean_margin_b - mean_margin_a,
        'margin_difference_pct': ((mean_margin_b - mean_margin_a) / abs(mean_margin_a) * 100) if mean_margin_a != 0 else 0,
        'statistic': statistic,
        'p_value': p_value,
        'alpha': alpha,
        'reject_null': reject_null,
        'conclusion': conclusion,
        'group_a_size': len(group_a),
        'group_b_size': len(group_b)
    }


def test_province_risk_differences(df: pd.DataFrame, 
                                  claims_col: str = 'TotalClaims',
                                  alpha: float = 0.05) -> Dict:
    """
    Test H₀: There are no risk differences across provinces.
    Uses ANOVA for claim severity and chi-squared for claim frequency.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    claims_col : str
        Name of claims column
    alpha : float
        Significance level
        
    Returns:
    --------
    dict
        Test results for both frequency and severity
    """
    if 'Province' not in df.columns:
        return {'error': 'Province column not found'}
    
    provinces = df['Province'].unique()
    if len(provinces) < 2:
        return {'error': 'Need at least 2 provinces for comparison'}
    
    # Test claim frequency (chi-squared)
    contingency_data = []
    for province in provinces:
        province_data = df[df['Province'] == province]
        claims_count = (province_data[claims_col] > 0).sum()
        no_claims_count = (province_data[claims_col] == 0).sum()
        contingency_data.append([claims_count, no_claims_count])
    
    contingency_table = np.array(contingency_data)
    chi2_freq, p_value_freq = stats.chi2_contingency(contingency_table)[:2]
    
    # Test claim severity (ANOVA or Kruskal-Wallis)
    severity_data = []
    for province in provinces:
        province_claims = df[(df['Province'] == province) & (df[claims_col] > 0)][claims_col]
        if len(province_claims) > 0:
            severity_data.append(province_claims.values)
    
    # Use Kruskal-Wallis (non-parametric) for robustness
    h_stat, p_value_sev = stats.kruskal(*severity_data)
    
    reject_freq = p_value_freq < alpha
    reject_sev = p_value_sev < alpha
    
    return {
        'test_name': 'Province risk differences',
        'null_hypothesis': 'H₀: There are no risk differences across provinces',
        'frequency_test': {
            'test_type': 'Chi-squared test',
            'statistic': chi2_freq,
            'p_value': p_value_freq,
            'reject_null': reject_freq,
            'conclusion': 'Reject H₀' if reject_freq else 'Fail to reject H₀'
        },
        'severity_test': {
            'test_type': 'Kruskal-Wallis test',
            'statistic': h_stat,
            'p_value': p_value_sev,
            'reject_null': reject_sev,
            'conclusion': 'Reject H₀' if reject_sev else 'Fail to reject H₀'
        },
        'overall_conclusion': 'Reject H₀' if (reject_freq or reject_sev) else 'Fail to reject H₀',
        'provinces_tested': len(provinces),
        'alpha': alpha
    }


def test_zipcode_risk_differences(df: pd.DataFrame,
                                  claims_col: str = 'TotalClaims',
                                  alpha: float = 0.05,
                                  min_samples_per_zipcode: int = 30) -> Dict:
    """
    Test H₀: There are no risk differences between zip codes.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    claims_col : str
        Name of claims column
    alpha : float
        Significance level
    min_samples_per_zipcode : int
        Minimum samples required per zipcode
        
    Returns:
    --------
    dict
        Test results
    """
    if 'PostalCode' not in df.columns:
        return {'error': 'PostalCode column not found'}
    
    # Filter zipcodes with sufficient samples
    zipcode_counts = df['PostalCode'].value_counts()
    valid_zipcodes = zipcode_counts[zipcode_counts >= min_samples_per_zipcode].index.tolist()
    
    if len(valid_zipcodes) < 2:
        return {'error': f'Need at least 2 zipcodes with {min_samples_per_zipcode}+ samples'}
    
    df_filtered = df[df['PostalCode'].isin(valid_zipcodes)]
    
    # Test claim frequency
    contingency_data = []
    for zipcode in valid_zipcodes[:20]:  # Limit to top 20 for computational efficiency
        zipcode_data = df_filtered[df_filtered['PostalCode'] == zipcode]
        claims_count = (zipcode_data[claims_col] > 0).sum()
        no_claims_count = (zipcode_data[claims_col] == 0).sum()
        contingency_data.append([claims_count, no_claims_count])
    
    contingency_table = np.array(contingency_data)
    chi2_freq, p_value_freq = stats.chi2_contingency(contingency_table)[:2]
    
    # Test claim severity
    severity_data = []
    for zipcode in valid_zipcodes[:20]:
        zipcode_claims = df_filtered[(df_filtered['PostalCode'] == zipcode) & 
                                    (df_filtered[claims_col] > 0)][claims_col]
        if len(zipcode_claims) > 0:
            severity_data.append(zipcode_claims.values)
    
    if len(severity_data) >= 2:
        h_stat, p_value_sev = stats.kruskal(*severity_data)
    else:
        h_stat, p_value_sev = None, None
    
    reject_freq = p_value_freq < alpha
    reject_sev = p_value_sev < alpha if p_value_sev is not None else False
    
    return {
        'test_name': 'Zipcode risk differences',
        'null_hypothesis': 'H₀: There are no risk differences between zip codes',
        'frequency_test': {
            'test_type': 'Chi-squared test',
            'statistic': chi2_freq,
            'p_value': p_value_freq,
            'reject_null': reject_freq,
            'conclusion': 'Reject H₀' if reject_freq else 'Fail to reject H₀'
        },
        'severity_test': {
            'test_type': 'Kruskal-Wallis test',
            'statistic': h_stat,
            'p_value': p_value_sev,
            'reject_null': reject_sev,
            'conclusion': 'Reject H₀' if reject_sev else 'Fail to reject H₀'
        },
        'overall_conclusion': 'Reject H₀' if (reject_freq or reject_sev) else 'Fail to reject H₀',
        'zipcodes_tested': len(valid_zipcodes[:20]),
        'total_valid_zipcodes': len(valid_zipcodes),
        'alpha': alpha
    }


def test_zipcode_margin_differences(df: pd.DataFrame,
                                   premium_col: str = 'TotalPremium',
                                   claims_col: str = 'TotalClaims',
                                   alpha: float = 0.05,
                                   min_samples_per_zipcode: int = 30) -> Dict:
    """
    Test H₀: There is no significant margin (profit) difference between zip codes.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    premium_col : str
        Name of premium column
    claims_col : str
        Name of claims column
    alpha : float
        Significance level
    min_samples_per_zipcode : int
        Minimum samples required per zipcode
        
    Returns:
    --------
    dict
        Test results
    """
    if 'PostalCode' not in df.columns:
        return {'error': 'PostalCode column not found'}
    
    # Filter zipcodes with sufficient samples
    zipcode_counts = df['PostalCode'].value_counts()
    valid_zipcodes = zipcode_counts[zipcode_counts >= min_samples_per_zipcode].index.tolist()
    
    if len(valid_zipcodes) < 2:
        return {'error': f'Need at least 2 zipcodes with {min_samples_per_zipcode}+ samples'}
    
    df_filtered = df[df['PostalCode'].isin(valid_zipcodes)]
    
    # Calculate margins by zipcode
    margin_data = []
    for zipcode in valid_zipcodes[:20]:  # Limit to top 20
        zipcode_data = df_filtered[df_filtered['PostalCode'] == zipcode]
        margins = calculate_margin(zipcode_data[premium_col], zipcode_data[claims_col])
        margin_data.append(margins.values)
    
    # Perform Kruskal-Wallis test
    h_stat, p_value = stats.kruskal(*margin_data)
    
    reject_null = p_value < alpha
    
    return {
        'test_name': 'Zipcode margin differences',
        'null_hypothesis': 'H₀: There is no significant margin (profit) difference between zip codes',
        'test_type': 'Kruskal-Wallis test',
        'statistic': h_stat,
        'p_value': p_value,
        'alpha': alpha,
        'reject_null': reject_null,
        'conclusion': 'Reject H₀' if reject_null else 'Fail to reject H₀',
        'zipcodes_tested': len(valid_zipcodes[:20]),
        'total_valid_zipcodes': len(valid_zipcodes)
    }


def test_gender_risk_differences(df: pd.DataFrame,
                                 claims_col: str = 'TotalClaims',
                                 alpha: float = 0.05) -> Dict:
    """
    Test H₀: There is no significant risk difference between Women and Men.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    claims_col : str
        Name of claims column
    alpha : float
        Significance level
        
    Returns:
    --------
    dict
        Test results for both frequency and severity
    """
    if 'Gender' not in df.columns:
        return {'error': 'Gender column not found'}
    
    # Prepare groups
    gender_values = df['Gender'].unique()
    if 'Female' in gender_values and 'Male' in gender_values:
        group_a = df[df['Gender'] == 'Female'].copy()
        group_b = df[df['Gender'] == 'Male'].copy()
    elif len(gender_values) >= 2:
        # Use first two gender values if Female/Male not found
        group_a = df[df['Gender'] == gender_values[0]].copy()
        group_b = df[df['Gender'] == gender_values[1]].copy()
    else:
        return {'error': 'Need at least 2 gender categories'}
    
    # Test claim frequency
    freq_result = test_claim_frequency_difference(group_a, group_b, claims_col, alpha)
    
    # Test claim severity
    sev_result = test_claim_severity_difference(group_a, group_b, claims_col, alpha)
    
    reject_freq = freq_result.get('reject_null', False)
    reject_sev = sev_result.get('reject_null', False)
    
    return {
        'test_name': 'Gender risk differences',
        'null_hypothesis': 'H₀: There is no significant risk difference between Women and Men',
        'frequency_test': freq_result,
        'severity_test': sev_result,
        'overall_conclusion': 'Reject H₀' if (reject_freq or reject_sev) else 'Fail to reject H₀',
        'alpha': alpha
    }

