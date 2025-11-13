"""
Feature Engineering Utilities for Delivery Time Prediction

This module contains reusable functions for creating features from raw shipping data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def create_temporal_features(df, date_column='ship_date', prefix='ship'):
    """
    Create temporal features from a date column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    date_column : str
        Name of the date column to extract features from
    prefix : str
        Prefix for the new feature columns
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added temporal features
    """
    df = df.copy()
    
    # Ensure datetime type
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Extract temporal features
    df[f'{prefix}_dayofweek'] = df[date_column].dt.dayofweek  # Monday=0, Sunday=6
    df[f'{prefix}_month'] = df[date_column].dt.month
    df[f'{prefix}_quarter'] = df[date_column].dt.quarter
    df[f'{prefix}_week_of_year'] = df[date_column].dt.isocalendar().week
    df[f'{prefix}_day'] = df[date_column].dt.day
    df[f'{prefix}_year'] = df[date_column].dt.year
    
    # Binary features
    df[f'is_weekend_{prefix}'] = df[f'{prefix}_dayofweek'].isin([5, 6]).astype(int)
    df[f'is_month_end_{prefix}'] = df[date_column].dt.is_month_end.astype(int)
    df[f'is_month_start_{prefix}'] = df[date_column].dt.is_month_start.astype(int)
    
    return df


def calculate_date_differences(df, date_col1, date_col2, output_col):
    """
    Calculate the difference in days between two date columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    date_col1 : str
        First date column (earlier date)
    date_col2 : str
        Second date column (later date)
    output_col : str
        Name for the output column
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added date difference column
    """
    df = df.copy()
    
    # Ensure datetime types
    df[date_col1] = pd.to_datetime(df[date_col1])
    df[date_col2] = pd.to_datetime(df[date_col2])
    
    # Calculate difference in days
    df[output_col] = (df[date_col2] - df[date_col1]).dt.days
    
    return df


def create_distance_bands(df, distance_column='distance_km', bins=None, labels=None):
    """
    Create distance band categories from continuous distance values.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    distance_column : str
        Name of the distance column
    bins : list, optional
        Custom bin edges. Default: [0, 100, 500, 1000, 2000, inf]
    labels : list, optional
        Custom labels for bins
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added distance_band column
    """
    df = df.copy()
    
    if bins is None:
        bins = [0, 100, 500, 1000, 2000, np.inf]
    
    if labels is None:
        labels = ['very_short', 'short', 'medium', 'long', 'very_long']
    
    df['distance_band'] = pd.cut(
        df[distance_column],
        bins=bins,
        labels=labels,
        include_lowest=True
    )
    
    return df


def create_region_features(df, origin_col='origin_city', dest_col='destination_city'):
    """
    Create region-based features from origin and destination.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    origin_col : str
        Origin location column
    dest_col : str
        Destination location column
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added region features
    """
    df = df.copy()
    
    # Create combined route feature
    df['route'] = df[origin_col].astype(str) + '_to_' + df[dest_col].astype(str)
    
    # Flag if origin equals destination (same city delivery)
    df['is_same_location'] = (df[origin_col] == df[dest_col]).astype(int)
    
    return df


def create_service_level_features(df, service_level_col='service_level'):
    """
    Create features based on service level.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    service_level_col : str
        Service level column name
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with service level binary features
    """
    df = df.copy()
    
    # Create binary flags for common service levels
    service_levels = ['standard', 'express', 'overnight', 'economy']
    
    for level in service_levels:
        df[f'is_{level}'] = (df[service_level_col].str.lower() == level).astype(int)
    
    return df


def create_weight_bands(df, weight_column='weight_kg', bins=None, labels=None):
    """
    Create weight band categories from continuous weight values.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    weight_column : str
        Name of the weight column
    bins : list, optional
        Custom bin edges
    labels : list, optional
        Custom labels for bins
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added weight_band column
    """
    df = df.copy()
    
    if bins is None:
        bins = [0, 5, 20, 50, 100, np.inf]
    
    if labels is None:
        labels = ['very_light', 'light', 'medium', 'heavy', 'very_heavy']
    
    df['weight_band'] = pd.cut(
        df[weight_column],
        bins=bins,
        labels=labels,
        include_lowest=True
    )
    
    return df


def create_interaction_features(df, categorical_cols):
    """
    Create interaction features between categorical columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    categorical_cols : list
        List of categorical column names to create interactions
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added interaction features
    """
    df = df.copy()
    
    # Create pairwise interactions
    for i in range(len(categorical_cols)):
        for j in range(i + 1, len(categorical_cols)):
            col1, col2 = categorical_cols[i], categorical_cols[j]
            interaction_name = f'{col1}_x_{col2}'
            df[interaction_name] = df[col1].astype(str) + '_' + df[col2].astype(str)
    
    return df


def create_all_features(df, 
                       ship_date_col='ship_date',
                       order_date_col='order_date',
                       delivery_date_col='delivery_date',
                       distance_col='distance_km',
                       weight_col='weight_kg'):
    """
    Apply all feature engineering steps in sequence.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with raw shipping data
    ship_date_col : str
        Shipment date column name
    order_date_col : str
        Order date column name
    delivery_date_col : str
        Delivery date column name (for training data only)
    distance_col : str
        Distance column name
    weight_col : str
        Weight column name
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with all engineered features
    """
    df = df.copy()
    
    print("🔧 Starting feature engineering pipeline...")
    
    # 1. Temporal features
    df = create_temporal_features(df, ship_date_col, prefix='ship')
    print("  ✅ Temporal features created")
    
    # 2. Date differences
    if order_date_col in df.columns:
        df = calculate_date_differences(df, order_date_col, ship_date_col, 'order_to_ship_days')
        print("  ✅ Order-to-ship days calculated")
    
    # 3. Target variable (for training data)
    if delivery_date_col in df.columns:
        df = calculate_date_differences(df, ship_date_col, delivery_date_col, 'delivery_days_actual')
        print("  ✅ Target variable (delivery_days_actual) created")
    
    # 4. Distance bands
    if distance_col in df.columns:
        df = create_distance_bands(df, distance_col)
        print("  ✅ Distance bands created")
    
    # 5. Weight bands
    if weight_col in df.columns:
        df = create_weight_bands(df, weight_col)
        print("  ✅ Weight bands created")
    
    # 6. Region features
    if 'origin_city' in df.columns and 'destination_city' in df.columns:
        df = create_region_features(df, 'origin_city', 'destination_city')
        print("  ✅ Region features created")
    
    # 7. Service level features
    if 'service_level' in df.columns:
        df = create_service_level_features(df, 'service_level')
        print("  ✅ Service level features created")
    
    print("✅ Feature engineering pipeline complete!")
    
    return df


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Utilities Module")
    print("Import this module to use feature engineering functions")
