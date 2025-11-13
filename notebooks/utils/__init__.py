"""
Utilities package for Delivery Time Prediction POC

This package contains reusable modules for:
- Feature engineering
- Data preprocessing
- Model evaluation and MLflow integration
"""

__version__ = "1.0.0"

# Import key functions for easy access
from .feature_engineering import (
    create_temporal_features,
    calculate_date_differences,
    create_distance_bands
)

from .preprocessing import (
    validate_required_columns,
    check_data_quality,
    handle_missing_values,
    clean_shipment_data
)

from .model_utils import (
    calculate_regression_metrics,
    evaluate_model,
    print_metrics,
    get_feature_importance
)

__all__ = [
    # Feature engineering
    'create_temporal_features',
    'calculate_date_differences',
    'create_distance_bands',
    
    # Preprocessing
    'validate_required_columns',
    'check_data_quality',
    'handle_missing_values',
    'clean_shipment_data',
    
    # Model utilities
    'calculate_regression_metrics',
    'evaluate_model',
    'print_metrics',
    'get_feature_importance',
]
