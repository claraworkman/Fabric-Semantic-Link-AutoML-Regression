# Configuration Files

This directory contains configuration files for the delivery time prediction POC.

## Files

### `environment.yml`
Conda environment specification with all required Python packages.

**Usage:**
```bash
conda env create -f environment.yml
conda activate delivery-prediction
```

### `automl_settings.json`
AutoML (FLAML) configuration settings for model training.

**Key Settings:**
- **time_budget**: 300 seconds (5 minutes)
- **metric**: MAE (Mean Absolute Error)
- **estimators**: Random Forest, XGBoost, LightGBM, Extra Trees
- **cross_validation**: 5-fold CV

**Usage in Notebook:**
```python
import json
with open('config/automl_settings.json', 'r') as f:
    settings = json.load(f)

automl.fit(X_train, y_train, **settings)
```

### `fabric_lakehouse_paths.yaml`
Microsoft Fabric workspace, lakehouse, and semantic model configuration.

**Required Updates:**
1. `workspace_id` - Your Fabric workspace ID
2. `lakehouse_id` - Your lakehouse ID
3. `semantic_model_id` - Your semantic model ID

**Finding Your IDs:**
- Navigate to Fabric workspace → Settings
- Copy IDs from URL or resource properties

**Usage in Notebook:**
```python
import yaml
with open('config/fabric_lakehouse_paths.yaml', 'r') as f:
    config = yaml.safe_load(f)

workspace_id = config['workspace_id']
lakehouse_name = config['lakehouse_name']
```

## Environment Variables

For sensitive information, consider using a `.env` file (not tracked in git):

```env
FABRIC_WORKSPACE_ID=your-workspace-id
FABRIC_LAKEHOUSE_ID=your-lakehouse-id
AZURE_TENANT_ID=your-tenant-id
```

Load with:
```python
from dotenv import load_dotenv
import os

load_dotenv()
workspace_id = os.getenv('FABRIC_WORKSPACE_ID')
```

## Production Considerations

For production deployments:
- Store sensitive IDs in Azure Key Vault
- Use managed identities for authentication
- Implement separate configs for dev/staging/prod environments
- Version control configuration changes
