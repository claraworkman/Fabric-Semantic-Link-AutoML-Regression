"""
Model Training and Evaluation Utilities for Delivery Time Prediction

This module contains functions for model evaluation, metrics calculation,
MLflow integration, and model management.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score,
    mean_absolute_percentage_error
)
import json
import warnings


def calculate_regression_metrics(y_true: np.ndarray, 
                                 y_pred: np.ndarray,
                                 prefix: str = '') -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted values
    prefix : str
        Prefix to add to metric names (e.g., 'train_', 'test_')
        
    Returns:
    --------
    dict
        Dictionary of metric names and values
    """
    metrics = {}
    
    # Mean Absolute Error
    metrics[f'{prefix}mae'] = mean_absolute_error(y_true, y_pred)
    
    # Root Mean Squared Error
    metrics[f'{prefix}rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # R-squared Score
    metrics[f'{prefix}r2'] = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error
    # Handle division by zero
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            metrics[f'{prefix}mape'] = mean_absolute_percentage_error(y_true, y_pred) * 100
        except:
            metrics[f'{prefix}mape'] = np.nan
    
    # Median Absolute Error
    metrics[f'{prefix}median_ae'] = np.median(np.abs(y_true - y_pred))
    
    # Max Error
    metrics[f'{prefix}max_error'] = np.max(np.abs(y_true - y_pred))
    
    # Custom: Percentage within threshold (e.g., within 2 days)
    threshold = 2.0
    within_threshold = np.mean(np.abs(y_true - y_pred) <= threshold) * 100
    metrics[f'{prefix}pct_within_{int(threshold)}days'] = within_threshold
    
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Model Metrics"):
    """
    Pretty print model metrics.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of metric names and values
    title : str
        Title for the metrics display
    """
    print("=" * 60)
    print(f"{title:^60}")
    print("=" * 60)
    
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric_name:.<40} {value:>12.4f}")
        else:
            print(f"{metric_name:.<40} {value:>12}")
    
    print("=" * 60)


def evaluate_model(model: Any,
                  X_train: pd.DataFrame,
                  y_train: pd.Series,
                  X_test: pd.DataFrame,
                  y_test: pd.Series,
                  verbose: bool = True) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model on both training and test sets.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    X_train, y_train : DataFrame, Series
        Training data
    X_test, y_test : DataFrame, Series
        Test data
    verbose : bool
        Print metrics if True
        
    Returns:
    --------
    dict
        Dictionary with 'train' and 'test' metrics
    """
    # Training predictions
    y_train_pred = model.predict(X_train)
    train_metrics = calculate_regression_metrics(y_train, y_train_pred, prefix='train_')
    
    # Test predictions
    y_test_pred = model.predict(X_test)
    test_metrics = calculate_regression_metrics(y_test, y_test_pred, prefix='test_')
    
    if verbose:
        print_metrics(train_metrics, "Training Set Metrics")
        print_metrics(test_metrics, "Test Set Metrics")
    
    return {
        'train': train_metrics,
        'test': test_metrics
    }


def create_prediction_dataframe(y_true: pd.Series,
                               y_pred: np.ndarray,
                               X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Create a dataframe comparing actual vs predicted values.
    
    Parameters:
    -----------
    y_true : Series
        Actual values
    y_pred : array
        Predicted values
    X : DataFrame, optional
        Feature dataframe to include additional columns
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with actual, predicted, and error columns
    """
    result_df = pd.DataFrame({
        'actual': y_true.values,
        'predicted': y_pred,
        'error': y_true.values - y_pred,
        'abs_error': np.abs(y_true.values - y_pred),
        'pct_error': np.abs((y_true.values - y_pred) / y_true.values) * 100
    })
    
    if X is not None:
        result_df = pd.concat([X.reset_index(drop=True), result_df], axis=1)
    
    return result_df


def get_feature_importance(model: Any,
                          feature_names: List[str],
                          top_n: int = 20) -> pd.DataFrame:
    """
    Extract and rank feature importances from a trained model.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    top_n : int
        Number of top features to return
        
    Returns:
    --------
    pd.DataFrame
        Sorted dataframe with features and their importance scores
    """
    if not hasattr(model, 'feature_importances_'):
        raise AttributeError("Model does not have feature_importances_ attribute")
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    if top_n:
        importance_df = importance_df.head(top_n)
    
    return importance_df


def log_model_to_mlflow(model: Any,
                       model_name: str,
                       metrics: Dict[str, float],
                       params: Optional[Dict[str, Any]] = None,
                       artifacts: Optional[Dict[str, str]] = None,
                       tags: Optional[Dict[str, str]] = None) -> str:
    """
    Log model, metrics, parameters, and artifacts to MLflow.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    model_name : str
        Name for the model artifact
    metrics : dict
        Dictionary of metrics to log
    params : dict, optional
        Dictionary of parameters to log
    artifacts : dict, optional
        Dictionary mapping artifact names to file paths
    tags : dict, optional
        Dictionary of tags to add to the run
        
    Returns:
    --------
    str
        Run ID of the MLflow run
    """
    with mlflow.start_run() as run:
        
        # Log parameters
        if params:
            mlflow.log_params(params)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log tags
        if tags:
            mlflow.set_tags(tags)
        
        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name
        )
        
        # Log additional artifacts
        if artifacts:
            for artifact_name, file_path in artifacts.items():
                mlflow.log_artifact(file_path, artifact_path=artifact_name)
        
        run_id = run.info.run_id
        print(f"Model logged to MLflow with run_id: {run_id}")
        
    return run_id


def register_model(model_uri: str,
                  model_name: str,
                  description: Optional[str] = None,
                  tags: Optional[Dict[str, str]] = None) -> Any:
    """
    Register a model in MLflow Model Registry.
    
    Parameters:
    -----------
    model_uri : str
        URI of the model (e.g., 'runs:/<run_id>/model')
    model_name : str
        Name to register the model under
    description : str, optional
        Description of the model
    tags : dict, optional
        Tags to add to the registered model
        
    Returns:
    --------
    ModelVersion
        MLflow ModelVersion object
    """
    client = MlflowClient()
    
    # Register the model
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )
    
    # Add description if provided
    if description:
        client.update_model_version(
            name=model_name,
            version=model_version.version,
            description=description
        )
    
    # Add tags if provided
    if tags:
        for key, value in tags.items():
            client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key=key,
                value=value
            )
    
    print(f"Model registered: {model_name} (version {model_version.version})")
    
    return model_version


def load_model_from_registry(model_name: str,
                             version: Optional[str] = None,
                             stage: Optional[str] = None) -> Any:
    """
    Load a model from MLflow Model Registry.
    
    Parameters:
    -----------
    model_name : str
        Name of the registered model
    version : str, optional
        Specific version number to load
    stage : str, optional
        Stage to load from ('Staging', 'Production', 'Archived')
        
    Returns:
    --------
    model
        Loaded sklearn model
    """
    if version:
        model_uri = f"models:/{model_name}/{version}"
    elif stage:
        model_uri = f"models:/{model_name}/{stage}"
    else:
        # Load latest version
        model_uri = f"models:/{model_name}/latest"
    
    print(f"Loading model from: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    
    return model


def transition_model_stage(model_name: str,
                          version: str,
                          stage: str,
                          archive_existing: bool = True):
    """
    Transition a model to a different stage in the registry.
    
    Parameters:
    -----------
    model_name : str
        Name of the registered model
    version : str
        Version number to transition
    stage : str
        Target stage ('Staging', 'Production', 'Archived')
    archive_existing : bool
        Archive existing models in target stage
    """
    client = MlflowClient()
    
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=archive_existing
    )
    
    print(f"Model {model_name} v{version} transitioned to {stage}")


def compare_models(models_dict: Dict[str, Any],
                  X_test: pd.DataFrame,
                  y_test: pd.Series) -> pd.DataFrame:
    """
    Compare multiple models on the same test set.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary mapping model names to trained models
    X_test, y_test : DataFrame, Series
        Test data
        
    Returns:
    --------
    pd.DataFrame
        Comparison table with metrics for each model
    """
    results = []
    
    for model_name, model in models_dict.items():
        y_pred = model.predict(X_test)
        metrics = calculate_regression_metrics(y_test, y_pred)
        metrics['model'] = model_name
        results.append(metrics)
    
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.set_index('model')
    
    # Sort by MAE (lower is better)
    comparison_df = comparison_df.sort_values('mae')
    
    return comparison_df


def create_residual_analysis(y_true: np.ndarray,
                            y_pred: np.ndarray) -> pd.DataFrame:
    """
    Create residual analysis dataframe for model diagnostics.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with residual statistics
    """
    residuals = y_true - y_pred
    
    analysis = pd.DataFrame({
        'predicted': y_pred,
        'actual': y_true,
        'residual': residuals,
        'abs_residual': np.abs(residuals),
        'squared_residual': residuals ** 2
    })
    
    # Add bins for predicted values
    analysis['predicted_bin'] = pd.qcut(analysis['predicted'], q=10, duplicates='drop')
    
    return analysis


def save_model_metadata(model_name: str,
                       model_version: str,
                       metrics: Dict[str, float],
                       feature_list: List[str],
                       metadata: Dict[str, Any],
                       output_path: str):
    """
    Save comprehensive model metadata to JSON file.
    
    Parameters:
    -----------
    model_name : str
        Name of the model
    model_version : str
        Version of the model
    metrics : dict
        Model performance metrics
    feature_list : list
        List of features used
    metadata : dict
        Additional metadata
    output_path : str
        Path to save JSON file
    """
    model_info = {
        'model_name': model_name,
        'model_version': model_version,
        'metrics': metrics,
        'features': feature_list,
        'feature_count': len(feature_list),
        'metadata': metadata,
        'created_at': pd.Timestamp.now().isoformat()
    }
    
    with open(output_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"Model metadata saved to: {output_path}")


def detect_data_drift(reference_data: pd.DataFrame,
                     current_data: pd.DataFrame,
                     numeric_columns: List[str],
                     threshold: float = 0.1) -> Dict[str, Dict]:
    """
    Detect data drift by comparing distributions.
    
    Parameters:
    -----------
    reference_data : pd.DataFrame
        Reference (training) data
    current_data : pd.DataFrame
        Current (production) data
    numeric_columns : list
        Columns to check for drift
    threshold : float
        Threshold for drift detection (relative difference)
        
    Returns:
    --------
    dict
        Drift report for each column
    """
    drift_report = {}
    
    for col in numeric_columns:
        if col not in reference_data.columns or col not in current_data.columns:
            continue
        
        ref_mean = reference_data[col].mean()
        cur_mean = current_data[col].mean()
        ref_std = reference_data[col].std()
        cur_std = current_data[col].std()
        
        mean_drift = abs((cur_mean - ref_mean) / ref_mean) if ref_mean != 0 else 0
        std_drift = abs((cur_std - ref_std) / ref_std) if ref_std != 0 else 0
        
        drift_report[col] = {
            'reference_mean': ref_mean,
            'current_mean': cur_mean,
            'reference_std': ref_std,
            'current_std': cur_std,
            'mean_drift_pct': mean_drift * 100,
            'std_drift_pct': std_drift * 100,
            'drift_detected': mean_drift > threshold or std_drift > threshold
        }
    
    return drift_report


def generate_model_card(model_name: str,
                       model_type: str,
                       metrics: Dict[str, float],
                       features: List[str],
                       description: str,
                       use_case: str,
                       output_path: str):
    """
    Generate a model card documentation file.
    
    Parameters:
    -----------
    model_name : str
        Name of the model
    model_type : str
        Type of model (e.g., 'Random Forest', 'LightGBM')
    metrics : dict
        Performance metrics
    features : list
        List of features
    description : str
        Model description
    use_case : str
        Intended use case
    output_path : str
        Path to save model card markdown file
    """
    card_content = f"""# Model Card: {model_name}

## Model Details
- **Model Type**: {model_type}
- **Created**: {pd.Timestamp.now().strftime('%Y-%m-%d')}
- **Purpose**: {use_case}

## Description
{description}

## Performance Metrics
"""
    
    for metric, value in metrics.items():
        card_content += f"- **{metric}**: {value:.4f}\n"
    
    card_content += f"""
## Features ({len(features)})
"""
    
    for i, feature in enumerate(features, 1):
        card_content += f"{i}. {feature}\n"
    
    card_content += """
## Intended Use
This model predicts shipment delivery times for operational planning and customer communication.

## Limitations
- Model performance may degrade with data from new geographic regions
- Accuracy may vary during seasonal peaks or unusual events
- Requires retraining when business processes change

## Recommendations
- Monitor prediction accuracy on new data
- Retrain model monthly or when MAE increases significantly
- Validate predictions against actual outcomes regularly
"""
    
    with open(output_path, 'w') as f:
        f.write(card_content)
    
    print(f"Model card saved to: {output_path}")
