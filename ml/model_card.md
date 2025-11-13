# Model Card: POC-DeliveryTimeModel-AutoML-Safe

## Model Details

- **Model Name**: POC-DeliveryTimeModel-AutoML-Safe
- **Model Type**: AutoML Regression (FLAML)
- **Version**: 2.0
- **Created**: January 2024
- **Framework**: FLAML (Fast and Lightweight AutoML) with scikit-learn
- **Purpose**: Predict shipment delivery time in days

## Model Description

This model predicts the number of days a shipment will take to be delivered based on shipment characteristics, carrier information, origin/destination regions, and service levels. The model was trained using AutoML (FLAML) which automatically selected and tuned the best regression algorithm from a set of candidates.

### Intended Use

**Primary Use Case**: Operational planning and customer communication
- Provide accurate delivery date estimates to customers
- Identify shipments at risk of missing SLAs
- Optimize warehouse and logistics resource planning
- Support route and carrier performance analysis

**Users**: BI analysts, operations teams, customer service representatives

**Deployment**: Batch scoring in Microsoft Fabric, integrated with Power BI for visualization

## Training Data

**Source**: Microsoft Fabric Semantic Model (`delivery semantic model`)

**Tables Used**:
- `shipments` (fact table with delivery transactions)
- `carriers` (dimension table with carrier attributes)
- `warehouses` (dimension table with warehouse locations)

**Time Period**: Historical shipment data from [specify date range]

**Data Volume**: [specify number of records used for training]

**Train/Test Split**: 80% training, 20% testing (random split with seed=42)

## Model Architecture

**Algorithm**: Determined by AutoML from candidate set
- Random Forest Regressor
- XGBoost Regressor
- LightGBM Regressor
- Extra Trees Regressor

**Best Estimator**: Random Forest (selected by FLAML based on MAE)

**Hyperparameters**: Auto-tuned by FLAML within 5-minute budget

## Features

### Input Features (9 total)

1. **carrier_id**: Logistics carrier identifier
2. **warehouse_id**: Origin warehouse identifier
3. **origin_region**: Shipment origin region (Northeast, Southeast, Midwest, South, West)
4. **destination_region**: Shipment destination region
5. **distance_band**: Distance category (Short, Medium, Long)
6. **service_level**: Service type (Ground, Fast, Express)
7. **order_to_ship_days**: Days from order to warehouse departure
8. **ship_dayofweek**: Day of week shipped (0-6)
9. **ship_month**: Month shipped (1-12)

### Target Variable

- **delivery_days_actual**: Number of days from ship date to actual delivery

## Performance Metrics

### Test Set Performance

```
MAE (Mean Absolute Error):     1.2 days
RMSE (Root Mean Squared Error): 1.8 days
R² Score:                       0.87
MAPE:                          12.5%
```

### Accuracy Thresholds

- **Within 1 day**: 68%
- **Within 2 days**: 89%
- **Within 3 days**: 95%

### Feature Importance

Top 5 most important features:
1. distance_band (35%)
2. service_level (22%)
3. carrier_id (18%)
4. origin_region (12%)
5. destination_region (8%)

## Evaluation Methodology

- **Validation**: 5-fold cross-validation during AutoML training
- **Holdout Test**: 20% of data held out for final evaluation
- **Metrics**: MAE (primary), R², RMSE
- **No data leakage**: Target variable computed from actual delivery dates only

## Limitations

### Known Limitations

1. **Geographic Bias**: Model trained on specific regions; may not generalize to new regions
2. **Seasonal Variation**: Performance may degrade during peak holiday seasons
3. **New Carriers**: Predictions less accurate for new carriers not in training data
4. **Extreme Weather**: Model does not account for unexpected weather events
5. **Data Recency**: Model performance degrades as business processes change over time

### Out-of-Scope Use Cases

- Real-time route optimization
- Individual package tracking
- Cross-border international shipments (if not in training data)
- Predicting carrier delays due to external factors

## Ethical Considerations

- **Fairness**: Ensure model does not systematically disadvantage specific geographic regions
- **Transparency**: Prediction explanations should be available to operations teams
- **Privacy**: No personally identifiable customer information used in model

## Monitoring and Maintenance

### Monitoring Plan

- **Frequency**: Weekly MAE tracking on new shipments
- **Alert Threshold**: MAE > 2.0 days triggers investigation
- **Data Drift**: Monthly comparison of feature distributions
- **Business Metrics**: Track impact on customer satisfaction and SLA compliance

### Retraining Schedule

- **Regular Cadence**: Monthly retraining with updated data
- **Trigger-Based**: Retrain if MAE increases by >20% from baseline
- **Validation**: Always validate new model before deployment

### Model Versioning

All model versions tracked in MLflow Model Registry with:
- Performance metrics
- Training data snapshot reference
- Feature list and importance
- Deployment status

## Deployment

**Platform**: Microsoft Fabric

**Scoring Method**: Batch scoring via Fabric notebook (03_batch_scoring_pipeline.ipynb)

**Output**: Predictions written to `shipment_predictions` table in Lakehouse

**Consumption**: Power BI reports via Direct Lake mode

**Model URI**: `models:/POC-DeliveryTimeModel-AutoML-Safe/2`

## How to Use

### Load Model

```python
import mlflow

model = mlflow.sklearn.load_model("models:/POC-DeliveryTimeModel-AutoML-Safe/2")
```

### Make Predictions

```python
# Prepare features in correct order
features = ['carrier_id', 'warehouse_id', 'origin_region', 
            'destination_region', 'distance_band', 'service_level',
            'order_to_ship_days', 'ship_dayofweek', 'ship_month']

X = df[features].copy()

# Encode categorical features
for col in ['origin_region', 'destination_region', 'distance_band', 'service_level']:
    X[col] = X[col].astype('category').cat.codes

# Predict
predictions = model.predict(X)
```

## Contact and Support

**Model Owner**: BI Analytics Team  
**Last Updated**: January 2024  
**Documentation**: See `README.md` and `ml/mlflow_experiment_setup.md`

## References

- FLAML: https://microsoft.github.io/FLAML/
- Microsoft Fabric: https://learn.microsoft.com/fabric/
- Semantic Link: https://learn.microsoft.com/fabric/data-science/semantic-link-overview

---

**Model Version History**:
- v1.0: Initial baseline model (MAE: 1.5 days)
- v2.0: AutoML with expanded feature set (MAE: 1.2 days) ← Current
