# MLflow Experiment Setup Guide

This document explains how to configure and use MLflow for tracking experiments in the Delivery Time Prediction POC.

## Overview

MLflow is used for:
- **Experiment tracking**: Log parameters, metrics, and artifacts
- **Model registry**: Version and manage trained models
- **Model deployment**: Load models for inference

## Microsoft Fabric MLflow Integration

Microsoft Fabric has **built-in MLflow tracking** that automatically logs experiments to your workspace.

### Automatic Configuration

In Fabric notebooks, MLflow is automatically configured:

```python
import mlflow

# MLflow is already connected to your Fabric workspace
# No additional configuration needed!
```

### Tracking URI

In Microsoft Fabric, MLflow tracking is automatically configured to use the workspace backend:

```python
import mlflow

# MLflow tracking is automatically configured in Fabric
# The tracking backend is integrated with your Fabric workspace
# You can verify the configuration:
print(f"Tracking URI: {mlflow.get_tracking_uri()}")

# Experiments and runs are stored in your Fabric workspace
experiment = mlflow.get_experiment_by_name("DeliveryTimePrediction")
if experiment:
    print(f"Experiment ID: {experiment.experiment_id}")
```

**Note**: The exact tracking URI format is managed by Fabric and may be workspace-specific. You don't need to configure it manually.

## Experiment Organization

### Create or Set Experiment

```python
import mlflow

# Set experiment name
mlflow.set_experiment("DeliveryTimePrediction")

# Or create explicitly
experiment = mlflow.get_experiment_by_name("DeliveryTimePrediction")
if experiment is None:
    mlflow.create_experiment("DeliveryTimePrediction")
```

### Recommended Experiment Names

- `DeliveryTimePrediction` - Main production experiments
- `DeliveryTimePrediction-Dev` - Development/testing
- `DeliveryTimePrediction-Baseline` - Baseline model benchmarks

## Logging Best Practices

### 1. Log Training Runs

```python
with mlflow.start_run(run_name="automl_training_v1") as run:
    
    # Log parameters
    mlflow.log_param("time_budget", 300)
    mlflow.log_param("estimator", "rf")
    mlflow.log_param("test_size", 0.2)
    
    # Train model
    automl.fit(X_train, y_train, **settings)
    
    # Log metrics
    mlflow.log_metric("train_mae", train_mae)
    mlflow.log_metric("test_mae", test_mae)
    mlflow.log_metric("r2_score", r2)
    
    # Log model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="POC-DeliveryTimeModel-AutoML-Safe"
    )
    
    # Log artifacts
    mlflow.log_artifact("config/automl_settings.json")
    
    run_id = run.info.run_id
    print(f"Run ID: {run_id}")
```

### 2. Add Tags for Organization

```python
mlflow.set_tags({
    "model_type": "AutoML",
    "framework": "FLAML",
    "data_source": "semantic_model",
    "purpose": "production",
    "created_by": "analytics_team"
})
```

### 3. Log Feature Information

```python
# Log feature names
mlflow.log_param("features", ",".join(feature_columns))
mlflow.log_param("num_features", len(feature_columns))

# Log feature importance (if available)
if hasattr(model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    })
    importance_df.to_csv('feature_importance.csv', index=False)
    mlflow.log_artifact('feature_importance.csv')
```

## Model Registry

### Register Model

```python
# During training
with mlflow.start_run():
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="POC-DeliveryTimeModel-AutoML-Safe"
    )

# Or after training
model_uri = f"runs:/{run_id}/model"
mlflow.register_model(
    model_uri=model_uri,
    name="POC-DeliveryTimeModel-AutoML-Safe"
)
```

### Version Management

Models are automatically versioned:
- Version 1: Initial baseline model
- Version 2: Retrained with more data
- Version 3: Hyperparameter tuning
- etc.

### Load Specific Version

```python
# Load by version number
model = mlflow.sklearn.load_model("models:/POC-DeliveryTimeModel-AutoML-Safe/2")

# Load latest version
model = mlflow.sklearn.load_model("models:/POC-DeliveryTimeModel-AutoML-Safe/latest")

# Load by stage (if using stages)
model = mlflow.sklearn.load_model("models:/POC-DeliveryTimeModel-AutoML-Safe/Production")
```

## Comparing Runs

### In Fabric UI

1. Navigate to your workspace
2. Go to **Experiments** section
3. Select `DeliveryTimePrediction` experiment
4. View all runs with metrics, parameters, and artifacts
5. Compare runs side-by-side

### Programmatically

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Search runs
experiment = mlflow.get_experiment_by_name("DeliveryTimePrediction")
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.test_mae ASC"],
    max_results=10
)

print(runs[['run_id', 'metrics.test_mae', 'metrics.r2_score']])
```

## Experiment Workflow

### 1. Initial Baseline

```python
mlflow.set_experiment("DeliveryTimePrediction")

with mlflow.start_run(run_name="baseline_simple"):
    mlflow.set_tag("stage", "baseline")
    # Train simple model
    # Log metrics
```

### 2. AutoML Search

```python
with mlflow.start_run(run_name="automl_search"):
    mlflow.set_tag("stage", "automl")
    mlflow.log_param("time_budget", 300)
    # Run AutoML
    # Log best model
```

### 3. Model Validation

```python
with mlflow.start_run(run_name="validation"):
    mlflow.set_tag("stage", "validation")
    # Test on holdout set
    # Log validation metrics
```

### 4. Production Registration

```python
# Register best model for production
best_run_id = "abc123"  # From comparison
model_uri = f"runs:/{best_run_id}/model"

mlflow.register_model(
    model_uri=model_uri,
    name="POC-DeliveryTimeModel-AutoML-Safe"
)
```

## Viewing Experiments in Fabric

### Access Experiment Runs

1. Open your Fabric workspace
2. Click on **Experiments** in the left navigation
3. Select your experiment name
4. View:
   - Run metrics and parameters
   - Model artifacts
   - Charts and visualizations
   - Model lineage

### Download Artifacts

```python
client = MlflowClient()
artifacts = client.list_artifacts(run_id)

for artifact in artifacts:
    client.download_artifacts(run_id, artifact.path, dst_path="./downloads")
```

## Best Practices Summary

✅ **Do:**
- Use descriptive run names
- Log all hyperparameters
- Log both training and test metrics
- Add tags for filtering
- Register models with clear names
- Document experiments

❌ **Don't:**
- Log sensitive data
- Use generic run names like "run1", "test"
- Forget to log the model artifact
- Mix dev and prod experiments

## Troubleshooting

### Issue: MLflow not configured

**Solution**: Ensure you're running in a Fabric notebook. MLflow is automatically configured in Fabric environments.

### Issue: Can't find registered model

**Solution**: Check the exact model name in the registry:
```python
client = MlflowClient()
models = client.search_registered_models()
for model in models:
    print(model.name)
```

### Issue: Run not appearing in UI

**Solution**: Ensure you're viewing the correct experiment. Refresh the Fabric UI.

## Additional Resources

- [Microsoft Fabric MLflow Documentation](https://learn.microsoft.com/fabric/data-science/mlflow-autologging)
- [MLflow Official Docs](https://mlflow.org/docs/latest/index.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
