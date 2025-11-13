"""
Data Preprocessing Utilities for Delivery Time Prediction

This module contains functions for data validation, cleaning, and preprocessing
of shipment data before model training or inference.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings


def validate_required_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that all required columns are present in the dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe to validate
    required_columns : list
        List of column names that must be present
        
    Returns:
    --------
    bool
        True if all required columns exist, raises ValueError otherwise
    """
    missing_cols = set(required_columns) - set(df.columns)
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True


def check_data_quality(df: pd.DataFrame, verbose: bool = True) -> Dict:
    """
    Perform comprehensive data quality checks on a dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe to check
    verbose : bool
        If True, print detailed quality report
        
    Returns:
    --------
    dict
        Dictionary with quality metrics (null counts, duplicates, etc.)
    """
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'null_counts': df.isnull().sum().to_dict(),
        'null_percentages': (df.isnull().sum() / len(df) * 100).to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'columns_with_nulls': df.columns[df.isnull().any()].tolist(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'datetime_columns': df.select_dtypes(include=['datetime']).columns.tolist()
    }
    
    if verbose:
        print("=" * 60)
        print("DATA QUALITY REPORT")
        print("=" * 60)
        print(f"Total Rows: {quality_report['total_rows']:,}")
        print(f"Total Columns: {quality_report['total_columns']}")
        print(f"Duplicate Rows: {quality_report['duplicate_rows']}")
        print(f"\nColumns with Nulls: {len(quality_report['columns_with_nulls'])}")
        
        if quality_report['columns_with_nulls']:
            print("\nNull Value Details:")
            for col in quality_report['columns_with_nulls']:
                pct = quality_report['null_percentages'][col]
                count = quality_report['null_counts'][col]
                print(f"  {col}: {count} ({pct:.2f}%)")
        
        print("\n" + "=" * 60)
    
    return quality_report


def handle_missing_values(df: pd.DataFrame, 
                         strategy: Dict[str, str] = None,
                         default_numeric: str = 'median',
                         default_categorical: str = 'mode') -> pd.DataFrame:
    """
    Handle missing values using specified strategies.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    strategy : dict, optional
        Dictionary mapping column names to fill strategies
        e.g., {'distance': 'mean', 'carrier_id': 'mode', 'notes': 'drop'}
    default_numeric : str
        Default strategy for numeric columns ('mean', 'median', 'zero', 'drop')
    default_categorical : str
        Default strategy for categorical columns ('mode', 'unknown', 'drop')
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with missing values handled
    """
    df = df.copy()
    strategy = strategy or {}
    
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
            
        # Use custom strategy if specified
        if col in strategy:
            method = strategy[col]
        # Otherwise use default based on dtype
        elif pd.api.types.is_numeric_dtype(df[col]):
            method = default_numeric
        else:
            method = default_categorical
        
        # Apply the strategy
        if method == 'mean':
            df[col].fillna(df[col].mean(), inplace=True)
        elif method == 'median':
            df[col].fillna(df[col].median(), inplace=True)
        elif method == 'mode':
            if not df[col].mode().empty:
                df[col].fillna(df[col].mode()[0], inplace=True)
        elif method == 'zero':
            df[col].fillna(0, inplace=True)
        elif method == 'unknown':
            df[col].fillna('Unknown', inplace=True)
        elif method == 'drop':
            df.dropna(subset=[col], inplace=True)
        else:
            warnings.warn(f"Unknown strategy '{method}' for column '{col}'")
    
    return df


def remove_outliers(df: pd.DataFrame, 
                    columns: List[str], 
                    method: str = 'iqr',
                    threshold: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers from specified numeric columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list
        List of numeric column names to check for outliers
    method : str
        Method to use: 'iqr' (Interquartile Range) or 'zscore'
    threshold : float
        For IQR: multiplier for IQR (default 1.5)
        For zscore: number of standard deviations (default 3.0)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with outliers removed
    """
    df = df.copy()
    original_len = len(df)
    
    for col in columns:
        if col not in df.columns:
            warnings.warn(f"Column '{col}' not found in dataframe")
            continue
            
        if not pd.api.types.is_numeric_dtype(df[col]):
            warnings.warn(f"Column '{col}' is not numeric, skipping outlier removal")
            continue
        
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            
        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df = df[z_scores <= threshold]
        
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'iqr' or 'zscore'")
    
    removed_count = original_len - len(df)
    print(f"Removed {removed_count} outlier rows ({removed_count/original_len*100:.2f}%)")
    
    return df


def encode_categorical_features(df: pd.DataFrame, 
                                columns: List[str], 
                                method: str = 'label',
                                drop_first: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """
    Encode categorical features for model training.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list
        List of categorical column names to encode
    method : str
        Encoding method: 'label', 'onehot', or 'target'
    drop_first : bool
        For one-hot encoding, whether to drop first category
        
    Returns:
    --------
    tuple
        (encoded_df, encoding_info_dict)
    """
    df = df.copy()
    encoding_info = {}
    
    for col in columns:
        if col not in df.columns:
            warnings.warn(f"Column '{col}' not found, skipping")
            continue
        
        if method == 'label':
            # Convert to category codes
            df[col] = df[col].astype('category')
            encoding_info[col] = {
                'method': 'label',
                'categories': df[col].cat.categories.tolist()
            }
            df[col] = df[col].cat.codes
            
        elif method == 'onehot':
            # One-hot encoding
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=drop_first)
            encoding_info[col] = {
                'method': 'onehot',
                'columns': dummies.columns.tolist()
            }
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[col])
        
        else:
            raise ValueError(f"Unknown encoding method '{method}'")
    
    return df, encoding_info


def normalize_numeric_features(df: pd.DataFrame, 
                               columns: List[str],
                               method: str = 'standard') -> Tuple[pd.DataFrame, Dict]:
    """
    Normalize or standardize numeric features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list
        List of numeric column names to normalize
    method : str
        Normalization method: 'standard' (z-score) or 'minmax'
        
    Returns:
    --------
    tuple
        (normalized_df, normalization_params_dict)
    """
    df = df.copy()
    norm_params = {}
    
    for col in columns:
        if col not in df.columns:
            warnings.warn(f"Column '{col}' not found, skipping")
            continue
        
        if method == 'standard':
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / std
            norm_params[col] = {'method': 'standard', 'mean': mean, 'std': std}
            
        elif method == 'minmax':
            min_val = df[col].min()
            max_val = df[col].max()
            df[col] = (df[col] - min_val) / (max_val - min_val)
            norm_params[col] = {'method': 'minmax', 'min': min_val, 'max': max_val}
        
        else:
            raise ValueError(f"Unknown normalization method '{method}'")
    
    return df, norm_params


def prepare_model_features(df: pd.DataFrame, 
                          feature_columns: List[str],
                          target_column: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Extract feature matrix and target variable for model training.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    feature_columns : list
        List of column names to use as features
    target_column : str, optional
        Name of target variable column
        
    Returns:
    --------
    tuple
        (X, y) where X is feature dataframe and y is target series (or None)
    """
    # Validate feature columns exist
    missing_features = set(feature_columns) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    
    X = df[feature_columns].copy()
    
    if target_column:
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        y = df[target_column].copy()
        return X, y
    
    return X, None


def clean_shipment_data(df: pd.DataFrame, 
                       remove_negatives: bool = True,
                       remove_future_dates: bool = True) -> pd.DataFrame:
    """
    Apply domain-specific cleaning rules for shipment data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Shipment dataframe
    remove_negatives : bool
        Remove rows with negative delivery days
    remove_future_dates : bool
        Remove rows with future ship dates
        
    Returns:
    --------
    pd.DataFrame
        Cleaned shipment dataframe
    """
    df = df.copy()
    original_len = len(df)
    
    # Remove negative delivery days
    if remove_negatives and 'delivery_days_actual' in df.columns:
        df = df[df['delivery_days_actual'] >= 0]
    
    # Remove future ship dates
    if remove_future_dates and 'ship_date' in df.columns:
        df['ship_date'] = pd.to_datetime(df['ship_date'])
        df = df[df['ship_date'] <= pd.Timestamp.now()]
    
    # Remove unrealistic delivery times (e.g., > 90 days)
    if 'delivery_days_actual' in df.columns:
        df = df[df['delivery_days_actual'] <= 90]
    
    removed = original_len - len(df)
    print(f"Cleaned {removed} invalid shipment records ({removed/original_len*100:.2f}%)")
    
    return df


def split_train_val_test(df: pd.DataFrame,
                         train_size: float = 0.7,
                         val_size: float = 0.15,
                         test_size: float = 0.15,
                         random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into train, validation, and test sets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    train_size : float
        Proportion for training set
    val_size : float
        Proportion for validation set
    test_size : float
        Proportion for test set
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (train_df, val_df, test_df)
    """
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.0")
    
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    print(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    return train_df, val_df, test_df
