"""
Script to run all hypothesis tests for Task 3.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from data_processing import load_and_validate_data
from hypothesis_testing import (
    test_province_risk_differences,
    test_zipcode_risk_differences,
    test_zipcode_margin_differences,
    test_gender_risk_differences
)

def main():
    """Main function to run all hypothesis tests."""
    data_path = '../data/raw/insurance_data.csv'
    
    print("="*80)
    print("HYPOTHESIS TESTING - TASK 3")
    print("="*80)
    
    try:
        # Load data
        print("\n1. Loading data...")
        df = load_and_validate_data(data_path)
        print(f"   ✓ Data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
        
        # Test 1: Provinces
        if 'Province' in df.columns:
            print("\n2. Testing province risk differences...")
            province_results = test_province_risk_differences(df, alpha=0.05)
            print(f"   Conclusion: {province_results.get('overall_conclusion', 'N/A')}")
        
        # Test 2: Zipcodes (Risk)
        if 'PostalCode' in df.columns:
            print("\n3. Testing zipcode risk differences...")
            zipcode_results = test_zipcode_risk_differences(df, alpha=0.05, min_samples_per_zipcode=30)
            print(f"   Conclusion: {zipcode_results.get('overall_conclusion', 'N/A')}")
        
        # Test 3: Zipcodes (Margin)
        if 'PostalCode' in df.columns:
            print("\n4. Testing zipcode margin differences...")
            margin_results = test_zipcode_margin_differences(df, alpha=0.05, min_samples_per_zipcode=30)
            print(f"   Conclusion: {margin_results.get('conclusion', 'N/A')}")
        
        # Test 4: Gender
        if 'Gender' in df.columns:
            print("\n5. Testing gender risk differences...")
            gender_results = test_gender_risk_differences(df, alpha=0.05)
            print(f"   Conclusion: {gender_results.get('overall_conclusion', 'N/A')}")
        
        print("\n" + "="*80)
        print("Hypothesis testing completed!")
        print("For detailed analysis, run the Jupyter notebook: notebooks/task3_hypothesis_testing.ipynb")
        print("="*80)
        
    except FileNotFoundError:
        print(f"\n❌ Error: Data file not found at {data_path}")
        print("Please update the data_path variable with the correct path to your insurance data.")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()

