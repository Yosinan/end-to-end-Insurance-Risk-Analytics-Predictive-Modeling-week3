"""
Machine learning modeling functions for insurance risk analytics.
Includes feature engineering, model training, evaluation, and interpretability.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

from logger import setup_logger

logger = setup_logger('modeling')


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer new features from existing data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with engineered features
    """
    df = df.copy()
    
    try:
        # Age of vehicle (if RegistrationYear exists)
        if 'RegistrationYear' in df.columns:
            current_year = pd.Timestamp.now().year
            df['VehicleAge'] = current_year - pd.to_numeric(df['RegistrationYear'], errors='coerce')
            df['VehicleAge'] = df['VehicleAge'].clip(lower=0, upper=50)  # Cap at reasonable age
        
        # Claim indicator
        if 'TotalClaims' in df.columns:
            df['HasClaim'] = (df['TotalClaims'] > 0).astype(int)
        
        # Premium to Sum Insured ratio
        if 'TotalPremium' in df.columns and 'SumInsured' in df.columns:
            df['PremiumToSumInsuredRatio'] = df['TotalPremium'] / (df['SumInsured'] + 1)
        
        # Loss ratio per policy
        if 'TotalPremium' in df.columns and 'TotalClaims' in df.columns:
            df['LossRatio'] = df['TotalClaims'] / (df['TotalPremium'] + 1)
        
        # Vehicle value categories
        if 'CustomValueEstimate' in df.columns:
            df['VehicleValueCategory'] = pd.cut(
                df['CustomValueEstimate'],
                bins=[0, 50000, 150000, 300000, np.inf],
                labels=['Low', 'Medium', 'High', 'VeryHigh']
            )
        
        # Premium categories
        if 'TotalPremium' in df.columns:
            df['PremiumCategory'] = pd.qcut(
                df['TotalPremium'],
                q=4,
                labels=['Low', 'Medium', 'High', 'VeryHigh'],
                duplicates='drop'
            )
        
        logger.info("Feature engineering completed successfully")
        return df
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        raise


def prepare_features_for_modeling(df: pd.DataFrame,
                                 target_col: str,
                                 exclude_cols: Optional[List[str]] = None,
                                 categorical_cols: Optional[List[str]] = None,
                                 numeric_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Prepare features for modeling by selecting and encoding features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of target column
    exclude_cols : list, optional
        Columns to exclude from features
    categorical_cols : list, optional
        List of categorical columns (auto-detected if None)
    numeric_cols : list, optional
        List of numeric columns (auto-detected if None)
        
    Returns:
    --------
    tuple
        (X, y, categorical_features, numeric_features)
    """
    try:
        if exclude_cols is None:
            exclude_cols = [target_col, 'PolicyID', 'UnderwrittenCoverID']
        
        # Remove exclude columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Auto-detect column types if not provided
        if categorical_cols is None:
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if numeric_cols is None:
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter to only existing columns
        categorical_cols = [col for col in categorical_cols if col in X.columns]
        numeric_cols = [col for col in numeric_cols if col in X.columns]
        
        logger.info(f"Prepared {len(categorical_cols)} categorical and {len(numeric_cols)} numeric features")
        logger.info(f"Target: {target_col}, Shape: {X.shape}")
        
        return X, y, categorical_cols, numeric_cols
        
    except Exception as e:
        logger.error(f"Error preparing features: {str(e)}")
        raise


def encode_categorical_features(X: pd.DataFrame,
                               categorical_cols: List[str],
                               method: str = 'onehot',
                               handle_unknown: str = 'ignore') -> Tuple[pd.DataFrame, Any]:
    """
    Encode categorical features.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature dataframe
    categorical_cols : list
        List of categorical column names
    method : str
        Encoding method ('onehot' or 'label')
    handle_unknown : str
        How to handle unknown categories
        
    Returns:
    --------
    tuple
        (encoded_X, encoder)
    """
    try:
        X_encoded = X.copy()
        encoder = None
        
        if method == 'onehot':
            encoder = OneHotEncoder(sparse_output=False, handle_unknown=handle_unknown, drop='first')
            encoded_array = encoder.fit_transform(X[categorical_cols])
            encoded_df = pd.DataFrame(
                encoded_array,
                columns=encoder.get_feature_names_out(categorical_cols),
                index=X.index
            )
            X_encoded = pd.concat([X_encoded.drop(columns=categorical_cols), encoded_df], axis=1)
            
        elif method == 'label':
            encoder = {}
            for col in categorical_cols:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X[col].astype(str))
                encoder[col] = le
        
        logger.info(f"Encoded {len(categorical_cols)} categorical features using {method} encoding")
        return X_encoded, encoder
        
    except Exception as e:
        logger.error(f"Error encoding categorical features: {str(e)}")
        raise


def impute_missing_values(X: pd.DataFrame,
                         numeric_strategy: str = 'median',
                         categorical_strategy: str = 'most_frequent') -> Tuple[pd.DataFrame, Any]:
    """
    Impute missing values in features.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature dataframe
    numeric_strategy : str
        Strategy for numeric imputation
    categorical_strategy : str
        Strategy for categorical imputation
        
    Returns:
    --------
    tuple
        (imputed_X, imputer)
    """
    try:
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        X_imputed = X.copy()
        imputers = {}
        
        if numeric_cols:
            numeric_imputer = SimpleImputer(strategy=numeric_strategy)
            X_imputed[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])
            imputers['numeric'] = numeric_imputer
        
        if categorical_cols:
            categorical_imputer = SimpleImputer(strategy=categorical_strategy)
            X_imputed[categorical_cols] = categorical_imputer.fit_transform(X[categorical_cols])
            imputers['categorical'] = categorical_imputer
        
        logger.info(f"Imputed missing values for {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns")
        return X_imputed, imputers
        
    except Exception as e:
        logger.error(f"Error imputing missing values: {str(e)}")
        raise


def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series,
                           X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    """
    Train a linear regression model.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
        
    Returns:
    --------
    dict
        Model and evaluation metrics
    """
    try:
        logger.info("Training Linear Regression model...")
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        results = {
            'model': model,
            'model_name': 'Linear Regression',
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred
        }
        
        logger.info(f"Linear Regression - Test RMSE: {test_rmse:.2f}, Test R²: {test_r2:.4f}")
        return results
        
    except Exception as e:
        logger.error(f"Error training Linear Regression: {str(e)}")
        raise


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                       X_test: pd.DataFrame, y_test: pd.Series,
                       is_classification: bool = False,
                       n_estimators: int = 100,
                       max_depth: Optional[int] = None,
                       random_state: int = 42) -> Dict:
    """
    Train a Random Forest model.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    is_classification : bool
        Whether this is a classification task
    n_estimators : int
        Number of trees
    max_depth : int, optional
        Maximum tree depth
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    dict
        Model and evaluation metrics
    """
    try:
        model_type = "Classification" if is_classification else "Regression"
        logger.info(f"Training Random Forest {model_type} model...")
        
        if is_classification:
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )
        else:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        results = {
            'model': model,
            'model_name': f'Random Forest {model_type}',
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred
        }
        
        # Metrics
        if is_classification:
            results['train_accuracy'] = accuracy_score(y_train, y_train_pred)
            results['test_accuracy'] = accuracy_score(y_test, y_test_pred)
            results['train_precision'] = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
            results['test_precision'] = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
            results['train_recall'] = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)
            results['test_recall'] = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
            results['train_f1'] = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
            results['test_f1'] = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
            
            # ROC AUC if binary classification
            if len(np.unique(y_train)) == 2:
                y_train_proba = model.predict_proba(X_train)[:, 1]
                y_test_proba = model.predict_proba(X_test)[:, 1]
                results['train_roc_auc'] = roc_auc_score(y_train, y_train_proba)
                results['test_roc_auc'] = roc_auc_score(y_test, y_test_proba)
            
            logger.info(f"Random Forest {model_type} - Test Accuracy: {results['test_accuracy']:.4f}, Test F1: {results['test_f1']:.4f}")
        else:
            results['train_rmse'] = np.sqrt(mean_squared_error(y_train, y_train_pred))
            results['test_rmse'] = np.sqrt(mean_squared_error(y_test, y_test_pred))
            results['train_r2'] = r2_score(y_train, y_train_pred)
            results['test_r2'] = r2_score(y_test, y_test_pred)
            results['train_mae'] = mean_absolute_error(y_train, y_train_pred)
            results['test_mae'] = mean_absolute_error(y_test, y_test_pred)
            
            logger.info(f"Random Forest {model_type} - Test RMSE: {results['test_rmse']:.2f}, Test R²: {results['test_r2']:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error training Random Forest: {str(e)}")
        raise


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                 X_test: pd.DataFrame, y_test: pd.Series,
                 is_classification: bool = False,
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 random_state: int = 42) -> Dict:
    """
    Train an XGBoost model.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    is_classification : bool
        Whether this is a classification task
    n_estimators : int
        Number of boosting rounds
    max_depth : int
        Maximum tree depth
    learning_rate : float
        Learning rate
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    dict
        Model and evaluation metrics
    """
    try:
        model_type = "Classification" if is_classification else "Regression"
        logger.info(f"Training XGBoost {model_type} model...")
        
        if is_classification:
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                eval_metric='logloss',
                n_jobs=-1
            )
        else:
            model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                n_jobs=-1
            )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        results = {
            'model': model,
            'model_name': f'XGBoost {model_type}',
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred
        }
        
        # Metrics
        if is_classification:
            results['train_accuracy'] = accuracy_score(y_train, y_train_pred)
            results['test_accuracy'] = accuracy_score(y_test, y_test_pred)
            results['train_precision'] = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
            results['test_precision'] = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
            results['train_recall'] = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)
            results['test_recall'] = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
            results['train_f1'] = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
            results['test_f1'] = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
            
            if len(np.unique(y_train)) == 2:
                y_train_proba = model.predict_proba(X_train)[:, 1]
                y_test_proba = model.predict_proba(X_test)[:, 1]
                results['train_roc_auc'] = roc_auc_score(y_train, y_train_proba)
                results['test_roc_auc'] = roc_auc_score(y_test, y_test_proba)
            
            logger.info(f"XGBoost {model_type} - Test Accuracy: {results['test_accuracy']:.4f}, Test F1: {results['test_f1']:.4f}")
        else:
            results['train_rmse'] = np.sqrt(mean_squared_error(y_train, y_train_pred))
            results['test_rmse'] = np.sqrt(mean_squared_error(y_test, y_test_pred))
            results['train_r2'] = r2_score(y_train, y_train_pred)
            results['test_r2'] = r2_score(y_test, y_test_pred)
            results['train_mae'] = mean_absolute_error(y_train, y_train_pred)
            results['test_mae'] = mean_absolute_error(y_test, y_test_pred)
            
            logger.info(f"XGBoost {model_type} - Test RMSE: {results['test_rmse']:.2f}, Test R²: {results['test_r2']:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error training XGBoost: {str(e)}")
        raise


def get_feature_importance(model: Any, feature_names: List[str], top_n: int = 20) -> pd.DataFrame:
    """
    Extract feature importance from a trained model.
    
    Parameters:
    -----------
    model : Any
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    top_n : int
        Number of top features to return
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with feature importance
    """
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
            if importances.ndim > 1:
                importances = importances.mean(axis=0)
        else:
            logger.warning("Model does not have feature_importances_ or coef_ attribute")
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': feature_names[:len(importances)],
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
        
    except Exception as e:
        logger.error(f"Error extracting feature importance: {str(e)}")
        return pd.DataFrame()


def calculate_shap_values(model: Any, X_sample: pd.DataFrame, 
                         max_samples: int = 100) -> Optional[Dict]:
    """
    Calculate SHAP values for model interpretability.
    
    Parameters:
    -----------
    model : Any
        Trained model
    X_sample : pd.DataFrame
        Sample data for SHAP calculation
    max_samples : int
        Maximum number of samples to use
        
    Returns:
    --------
    dict, optional
        Dictionary with SHAP values and summary
    """
    try:
        import shap
        
        # Limit sample size for performance
        if len(X_sample) > max_samples:
            X_sample = X_sample.sample(max_samples, random_state=42)
        
        # Create SHAP explainer
        if isinstance(model, (xgb.XGBRegressor, xgb.XGBClassifier)):
            explainer = shap.TreeExplainer(model)
        elif isinstance(model, RandomForestRegressor) or isinstance(model, RandomForestClassifier):
            explainer = shap.TreeExplainer(model)
        else:
            # Use KernelExplainer for other models
            explainer = shap.KernelExplainer(model.predict, X_sample.sample(min(50, len(X_sample))))
        
        shap_values = explainer.shap_values(X_sample)
        
        # Handle multi-class output
        if isinstance(shap_values, list):
            shap_values = shap_values[0] if len(shap_values) > 0 else shap_values
        
        # Calculate mean absolute SHAP values for feature importance
        if isinstance(shap_values, np.ndarray):
            mean_shap = np.abs(shap_values).mean(axis=0)
            feature_importance = pd.DataFrame({
                'feature': X_sample.columns[:len(mean_shap)],
                'mean_abs_shap': mean_shap
            }).sort_values('mean_abs_shap', ascending=False)
        else:
            feature_importance = pd.DataFrame()
        
        return {
            'shap_values': shap_values,
            'feature_importance': feature_importance,
            'explainer': explainer,
            'X_sample': X_sample
        }
        
    except ImportError:
        logger.warning("SHAP not installed. Install with: pip install shap")
        return None
    except Exception as e:
        logger.error(f"Error calculating SHAP values: {str(e)}")
        return None

