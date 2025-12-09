"""
Script to train all machine learning models for Task 4.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from data_processing import load_and_validate_data, engineer_features
from modeling import (
    prepare_features_for_modeling, encode_categorical_features,
    impute_missing_values, train_linear_regression, train_random_forest,
    train_xgboost, get_feature_importance
)
from sklearn.model_selection import train_test_split
from logger import setup_logger

logger = setup_logger('train_models')

def main():
    """Main function to train all models."""
    data_path = '../data/raw/insurance_data.csv'
    
    logger.info("="*80)
    logger.info("MACHINE LEARNING MODEL TRAINING - TASK 4")
    logger.info("="*80)
    
    try:
        # Load and prepare data
        logger.info("\n1. Loading and preparing data...")
        df = load_and_validate_data(data_path)
        df = engineer_features(df)
        logger.info(f"   ✓ Data prepared: {df.shape}")
        
        # Train premium prediction model
        logger.info("\n2. Training Premium Prediction Model...")
        target_premium = 'CalculatedPremiumPerTerm' if 'CalculatedPremiumPerTerm' in df.columns else 'TotalPremium'
        
        X_prem, y_prem, cat_cols_prem, num_cols_prem = prepare_features_for_modeling(
            df, target_col=target_premium,
            exclude_cols=['PolicyID', 'UnderwrittenCoverID', 'TotalClaims']
        )
        
        X_prem_encoded, _ = encode_categorical_features(X_prem, cat_cols_prem, method='onehot')
        X_prem_processed, _ = impute_missing_values(X_prem_encoded)
        
        X_train_prem, X_test_prem, y_train_prem, y_test_prem = train_test_split(
            X_prem_processed, y_prem, test_size=0.2, random_state=42
        )
        
        # Train models
        premium_models = {}
        premium_models['linear'] = train_linear_regression(
            X_train_prem, y_train_prem, X_test_prem, y_test_prem
        )
        premium_models['rf'] = train_random_forest(
            X_train_prem, y_train_prem, X_test_prem, y_test_prem,
            is_classification=False, n_estimators=100
        )
        premium_models['xgb'] = train_xgboost(
            X_train_prem, y_train_prem, X_test_prem, y_test_prem,
            is_classification=False, n_estimators=100
        )
        
        # Find best model
        best_prem = max(premium_models.items(), key=lambda x: x[1]['test_r2'])
        logger.info(f"\n   Best Premium Model: {best_prem[1]['model_name']}")
        logger.info(f"   Test R²: {best_prem[1]['test_r2']:.4f}")
        logger.info(f"   Test RMSE: {best_prem[1]['test_rmse']:.2f}")
        
        # Feature importance
        feature_importance = get_feature_importance(
            best_prem[1]['model'], X_train_prem.columns.tolist(), top_n=10
        )
        if not feature_importance.empty:
            logger.info("\n   Top 10 Features:")
            for idx, row in feature_importance.iterrows():
                logger.info(f"   {idx+1}. {row['feature']}: {row['importance']:.4f}")
        
        logger.info("\n" + "="*80)
        logger.info("Model training completed!")
        logger.info("For detailed analysis, run the Jupyter notebook: notebooks/task4_modeling.ipynb")
        logger.info("="*80)
        
    except FileNotFoundError:
        logger.error(f"Data file not found at {data_path}")
        logger.error("Please update the data_path variable with the correct path.")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()

