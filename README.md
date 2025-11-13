# 🚚 Delivery Time Prediction POC

**Predict shipment delivery dates using Microsoft Fabric, Semantic Link, AutoML, and Power BI**

---

## 📋 Overview

This Proof of Concept (POC) demonstrates a **complete end-to-end machine learning pipeline built entirely on Microsoft Fabric**, using AutoML, MLflow, Semantic Link, Lakehouse tables, and Power BI.

The solution predicts shipment delivery times and surfaces insights on carrier performance, warehouse efficiency, and prediction accuracy. This POC showcases how an organization can **operationalize AI using Fabric's unified analytics platform** without needing multiple disconnected tools.

### 🚀 What This POC Delivers

✅ **End-to-end ML workflow** - Training → Registry → Scoring → BI reporting  
✅ **High-accuracy prediction model** - AutoML with Random Forest/XGBoost  
✅ **Delivery time predictions** - For every shipment in your data  
✅ **Performance dashboards** - Interactive Power BI reports  
✅ **Fully repeatable pattern** - Scalable, Fabric-native architecture  

### Business Value

- **Proactive customer communication** - Accurate delivery date predictions
- **Resource optimization** - Better planning for warehouse and logistics
- **Identify delays early** - Flag shipments at risk of missing SLAs
- **Data-driven insights** - Understand key factors impacting delivery times
- **Unified platform** - All analytics workloads in one place (Fabric)

---

## 🎯 Why Semantic Link?

**Semantic Link** bridges Power BI semantic models and Python notebooks in Fabric, providing key advantages:

- **Single Source of Truth** - Use your existing Power BI semantic model; no data duplication needed
- **Leverage Existing Work** - Reuse semantic models and relationships already created by your BI team
- **Always Current** - Live data access means you're always working with the latest information
- **Simplified Code** - Pre-joined tables and business logic reduce complexity
- **Full Circle** - Train on BI data → Score predictions → Power BI reports (all in Fabric)

```python
import sempy.fabric as fabric

# Read tables directly from your Power BI semantic model
shipments = fabric.read_table("delivery semantic model", "shipments")
carriers = fabric.read_table("delivery semantic model", "carriers")
warehouses = fabric.read_table("delivery semantic model", "warehouses")
```

---

## 🏗️ Architecture

### Workflow Overview

```
┌─────────────────┐
│ Fabric Semantic │
│     Model       │ ◄─── Existing shipping data
│  (Shipping Data)│
└────────┬────────┘
         │
         │ Semantic Link
         ▼
┌─────────────────┐
│  Fabric Notebook│
│  Data Prep +    │
│  Feature Eng    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   AutoML        │
│   Training      │ ◄─── FLAML (LightGBM, XGBoost, RF)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  MLflow Model   │
│   Registry      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Batch Scoring   │
│   Pipeline      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Fabric Lakehouse│
│ (Delta Table)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Power BI      │
│    Report       │ ◄─── Direct Lake mode
└─────────────────┘
```

### How It Works

**Step 1: Data Preparation**
- Load shipments, carriers, and warehouses from your Power BI semantic model
- Feature engineering automatically creates prediction variables
- Data validation ensures quality

**Step 2: Model Training**
- AutoML finds the best model (Random Forest, XGBoost, or Extra Trees)
- Model registered in MLflow for tracking and versioning
- Training takes ~3 minutes

**Step 3: Generate Predictions**
- Score all shipments with predicted delivery times
- Write predictions back to Lakehouse
- Predictions appear automatically in Power BI

**Step 4: Power BI Dashboards**
- **Executive Overview** - Model performance and key metrics
- **Carrier Performance** - Identify best/worst carriers
- **Shipment Explorer** - Detailed shipment analysis with filters

---

## 📂 Repository Structure

```
delivery-time-prediction-poc/
│
├── 01_semantic_link_data_preparation.ipynb    # Load data from semantic model
├── 02_autoML_training_pipeline.ipynb          # AutoML training + MLflow registry
├── 03_batch_scoring_pipeline.ipynb            # Batch predictions
│
├── notebooks/                                  # Utility modules
│   └── utils/
│       ├── __init__.py
│       ├── preprocessing.py                    # Data validation, cleaning, encoding
│       ├── feature_engineering.py              # Feature creation
│       └── model_utils.py                      # Evaluation, MLflow helpers
│
├── powerbi/                                    # Power BI artifacts
│   ├── delivery semantic model.pbix            # Semantic model
│   └── dax/
│       ├── measures_basic.dax                  # Basic DAX measures
│       ├── measures_advanced.dax               # Advanced metrics
│       └── README.md                           # DAX documentation
│
├── data/                                       # Data documentation
│   └── schema/
│       ├── shipments_schema.json               # Shipments table schema
│       ├── carriers_schema.json                # Carriers table schema
│       └── warehouses_schema.json              # Warehouses table schema
│
├── ml/                                         # ML documentation
│   ├── models/                                 # Model artifacts (MLflow managed)
│   ├── feature_list.json                       # Feature catalog
│   ├── model_card.md                           # Model documentation
│   └── mlflow_experiment_setup.md              # MLflow setup guide
│
├── config/                                     # Configuration files
│   ├── environment.yml                         # Conda environment
│   ├── automl_settings.json                    # FLAML AutoML config
│   └── fabric_lakehouse_paths.yaml             # Fabric resource IDs
│
├── scripts/                                    # Setup and validation scripts
│   ├── setup_fabric_environment.py             # Environment validation
│   └── validate_semantic_model.py              # Schema validation
│
└── README.md                                   # This file
```

---

## 🚀 Getting Started

### Prerequisites

- Microsoft Fabric workspace with:
  - Lakehouse with shipments, carriers, warehouses tables
  - Power BI semantic model: `delivery semantic model`
  - MLflow experiment and model registry enabled
- Python 3.10+ (automatically available in Fabric notebooks)

### Setup Steps

#### 1. **Upload Notebooks to Fabric**

1. Navigate to your Fabric workspace (Data Engineering or Data Science experience)
2. Upload the three notebooks:
   - `01_semantic_link_data_preparation.ipynb`
   - `02_autoML_training_pipeline.ipynb`
   - `03_batch_scoring_pipeline.ipynb`
3. Attach notebooks to your default Lakehouse

#### 2. **Configure Fabric Resources**

Update `config/fabric_lakehouse_paths.yaml` with your Fabric workspace details:

```yaml
workspace_id: "your-workspace-id"
lakehouse_id: "your-lakehouse-id"
semantic_model_name: "delivery semantic model"
mlflow_experiment_name: "delivery-time-prediction"
model_registry_name: "POC-DeliveryTimeModel-AutoML-Safe"
```

#### 3. **Validate Semantic Model**

Run the validation script to ensure your semantic model has the required tables:

```python
# In a Fabric notebook cell
%run scripts/validate_semantic_model.py
```

This checks for:
- `shipments` table with required columns
- `carriers` table with required columns
- `warehouses` table with required columns

#### 4. **Run the ML Pipeline**

Execute notebooks **in order**:

1. **Data Preparation** - `01_semantic_link_data_preparation.ipynb`
2. **Model Training** - `02_autoML_training_pipeline.ipynb` (~3 minutes)
3. **Batch Scoring** - `03_batch_scoring_pipeline.ipynb`

#### 5. **Connect Power BI Report**

1. Open `powerbi/delivery semantic model.pbix`
2. Verify semantic model connection
3. Predictions appear automatically via Direct Lake mode
4. Publish to Power BI Service

---

## 🔄 Retraining the Model

To retrain with new data:

1. Refresh your Power BI semantic model
2. Run `02_autoML_training_pipeline.ipynb`
3. Run `03_batch_scoring_pipeline.ipynb` for updated predictions

**When to retrain:**
- Monthly (recommended)
- When adding new carriers or warehouses
- If prediction accuracy drops

---

## 📊 Power BI Dashboards

### Recommended Visuals

**Page 1 - Executive Overview**
- KPI cards: MAE, Avg Predicted Days, Avg Actual Days
- Line chart: Actual vs Predicted over time
- Error distribution histogram

**Page 2 - Carrier Performance**
- MAE by Carrier (bar chart)
- Prediction Bias by Carrier
- Shipment Count by Carrier

**Page 3 - Shipment Explorer**
- Detailed shipment table
- Filters: Carrier, Warehouse, Region, Service Level
- Prediction error scatter plot

### Key DAX Measures

```dax
AvgActualDeliveryDays = AVERAGE(shipments[delivery_days])
AvgPredictedDeliveryDays = AVERAGE(shipment_predictions[predicted_delivery_days])
```

See `powerbi/dax/` folder for complete measures.

---

## 📚 What's Included

### Notebooks
- **Data Preparation** - Load from semantic model, validate, engineer features
- **Training Pipeline** - AutoML training with MLflow tracking
- **Batch Scoring** - Generate predictions for all shipments

### Utilities (`notebooks/utils/`)
- **preprocessing.py** - Data validation and cleaning
- **feature_engineering.py** - Feature creation functions
- **model_utils.py** - Model evaluation and MLflow helpers

### Configuration (`config/`)
- **automl_settings.json** - AutoML configuration (180 sec, MAE metric)
- **environment.yml** - Python environment setup
- **fabric_lakehouse_paths.yaml** - Fabric resource IDs

### Documentation (`ml/`)
- **model_card.md** - Model documentation
- **feature_list.json** - Feature catalog (9 features)
- **mlflow_experiment_setup.md** - MLflow setup guide

### Data Schemas (`data/schema/`)
- JSON schemas for shipments, carriers, warehouses tables

---

## 📈 Model Performance

Expected performance:
- **MAE:** ~1.2 days (average error)
- **RMSE:** ~1.8 days
- **R² Score:** ~0.85

*Performance varies based on your data*

---

## 🔍 Troubleshooting

**Semantic Link connection fails**
- Verify semantic model name and workspace permissions

**MLflow model not found**
- Check model registry name in MLflow experiments

**Power BI shows no predictions**
- Verify `shipment_predictions` table exists in Lakehouse

---

## 📚 Resources

- [Semantic Link Documentation](https://learn.microsoft.com/fabric/data-science/semantic-link-overview)
- [FLAML AutoML](https://microsoft.github.io/FLAML/)
- [MLflow in Fabric](https://learn.microsoft.com/fabric/data-science/mlflow-overview)

---

## 🎯 Next Steps

Ideas for extending this POC:

- Add weather data or traffic patterns
- Deploy as real-time API endpoint
- Add SHAP values for prediction explanations
- Schedule automated retraining
- A/B test model versions

---

## 📄 License

This POC is provided as-is for demonstration purposes.

---

**Built with Microsoft Fabric, Semantic Link, FLAML AutoML, and Power BI** 🚀
