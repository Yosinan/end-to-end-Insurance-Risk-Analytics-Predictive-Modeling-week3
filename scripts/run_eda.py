"""
Script to run Exploratory Data Analysis (EDA) for Task 1.
This script can be used to generate EDA reports programmatically.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from data_processing import load_and_validate_data, get_data_summary, check_missing_values
from utils import calculate_loss_ratio, calculate_claim_frequency, calculate_claim_severity

def main():
    """Main function to run EDA analysis."""
    # Update this path to your actual data file
    data_path = '../data/raw/insurance_data.csv'
    
    print("="*80)
    print("INSURANCE RISK ANALYTICS - EDA SCRIPT")
    print("="*80)
    
    try:
        # Load data
        print("\n1. Loading data...")
        df = load_and_validate_data(data_path)
        print(f"   ✓ Data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
        
        # Data summary
        print("\n2. Generating data summary...")
        summary = get_data_summary(df)
        print(f"   ✓ Numeric columns: {len(summary['numeric_columns'])}")
        print(f"   ✓ Categorical columns: {len(summary['categorical_columns'])}")
        
        # Missing values
        print("\n3. Checking data quality...")
        missing_df = check_missing_values(df)
        if len(missing_df) > 0:
            print(f"   ⚠ Found {len(missing_df)} columns with missing values")
        else:
            print("   ✓ No missing values found")
        
        # Key metrics
        if 'TotalPremium' in df.columns and 'TotalClaims' in df.columns:
            print("\n4. Calculating key business metrics...")
            loss_ratio = calculate_loss_ratio(df['TotalClaims'], df['TotalPremium'])
            claim_freq = calculate_claim_frequency(df)
            claim_sev = calculate_claim_severity(df)
            
            print(f"   ✓ Overall Loss Ratio: {loss_ratio:.4f}")
            print(f"   ✓ Claim Frequency: {claim_freq:.4f}")
            print(f"   ✓ Claim Severity: ZAR {claim_sev:,.2f}")
        
        print("\n" + "="*80)
        print("EDA script completed successfully!")
        print("For detailed analysis, run the Jupyter notebook: notebooks/task1_eda.ipynb")
        print("="*80)
        
    except FileNotFoundError:
        print(f"\n❌ Error: Data file not found at {data_path}")
        print("Please update the data_path variable with the correct path to your insurance data.")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()

