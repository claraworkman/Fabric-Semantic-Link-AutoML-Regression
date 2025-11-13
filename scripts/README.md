# Setup and Deployment Scripts

This directory contains scripts to help set up and deploy the Delivery Time Prediction POC in Microsoft Fabric.

## Scripts

### `setup_fabric_environment.py`
Validates Fabric environment configuration and checks prerequisites.

### `validate_semantic_model.py`
Checks that the semantic model has all required tables and columns.

### `deploy_notebooks.py`
Helper script to organize and prepare notebooks for Fabric deployment.

## Quick Start

### 1. Validate Environment

```python
python scripts/setup_fabric_environment.py
```

This checks:
- Python version
- Required packages
- Fabric workspace connectivity

### 2. Validate Semantic Model

```python
python scripts/validate_semantic_model.py --model "delivery semantic model"
```

This verifies:
- All required tables exist
- Required columns are present
- Relationships are configured

### 3. Deploy to Fabric

1. Upload notebooks to your Fabric workspace
2. Import notebooks in this order:
   - 01_semantic_link_data_preparation.ipynb
   - 02_autoML_training_pipeline.ipynb
   - 03_batch_scoring_pipeline.ipynb

## Deployment Checklist

- [ ] Fabric workspace created
- [ ] Lakehouse created
- [ ] Semantic model published with required tables
- [ ] Python 3.10+ environment available
- [ ] Required packages installed (see config/environment.yml)
- [ ] Workspace ID updated in config/fabric_lakehouse_paths.yaml
- [ ] Notebooks uploaded to Fabric
- [ ] Test run of 01_semantic_link_data_preparation.ipynb
- [ ] Test run of 02_autoML_training_pipeline.ipynb
- [ ] Model registered in MLflow
- [ ] Test run of 03_batch_scoring_pipeline.ipynb
- [ ] Predictions table created in Lakehouse
- [ ] Power BI report connected to Lakehouse
- [ ] DAX measures imported to semantic model

## Manual Setup Steps

### Create Fabric Lakehouse

1. Navigate to your Fabric workspace
2. Click "+ New" → "Lakehouse"
3. Name it "DeliveryPredictions"
4. Copy the Lakehouse ID from settings

### Configure Semantic Model

1. Ensure semantic model contains:
   - shipments table
   - carriers table
   - warehouses table
   - orders table (optional)

2. Configure relationships:
   - shipments → carriers (many-to-one on carrier_id)
   - shipments → warehouses (many-to-one on warehouse_id)

### Upload Notebooks

1. Go to Data Engineering experience
2. Import each .ipynb file from root directory
3. Attach notebooks to your Lakehouse
4. Run in sequence

### Deploy Power BI Report

1. Open `delivery semantic model.pbix`
2. Update data source to your Lakehouse
3. Refresh data
4. Publish to Power BI Service

## Troubleshooting

See `docs/TROUBLESHOOTING.md` for common issues and solutions.
