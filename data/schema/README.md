# Data Schema Documentation

This directory contains JSON schema definitions for tables in the **delivery semantic model**.

## Overview

These schemas document the expected structure of tables accessed via Semantic Link from Power BI. They serve as:
- Documentation for data consumers
- Validation specifications
- Reference for data quality checks

## Schema Files

### `shipments_schema.json`
**Type**: Fact Table  
**Description**: Shipment transactions with delivery metrics

**Key Fields**:
- `shipment_id` (Primary Key)
- `delivery_days_actual` (TARGET VARIABLE for ML model)
- Foreign keys: `carrier_id`, `warehouse_id`, `order_id`

**Relationships**:
- Many-to-one with `carriers` (via `carrier_id`)
- Many-to-one with `warehouses` (via `warehouse_id`)
- Many-to-one with `orders` (via `order_id`)

### `carriers_schema.json`
**Type**: Dimension Table  
**Description**: Carrier/logistics provider attributes

**Key Fields**:
- `carrier_id` (Primary Key)
- `carrier_name`
- `speed_factor` (performance metric)

### `warehouses_schema.json`
**Type**: Dimension Table  
**Description**: Warehouse/fulfillment center attributes

**Key Fields**:
- `warehouse_id` (Primary Key)
- `origin_region` (geographic dimension)
- `operational_hours` (capacity indicator)

## Using These Schemas

### Validate Data in Python

```python
import json
import jsonschema
import pandas as pd

# Load schema
with open('data/schema/shipments_schema.json', 'r') as f:
    schema = json.load(f)

# Validate a record
record = {
    'shipment_id': 'SHP-001',
    'carrier_id': 'CAR-01',
    'warehouse_id': 'WH-01',
    'ship_date': '2024-01-15',
    'delivery_days_actual': 4
}

jsonschema.validate(instance=record, schema=schema)
```

### Check DataFrame Compliance

```python
def validate_dataframe_schema(df, required_columns):
    """Check if dataframe has all required columns"""
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return True

# Example usage
required_shipment_cols = [
    'shipment_id', 'carrier_id', 'warehouse_id', 
    'ship_date', 'delivery_days_actual'
]
validate_dataframe_schema(shipments_df, required_shipment_cols)
```

## Semantic Model Requirements

When setting up your Power BI semantic model, ensure:

1. **Table Names Match**: Use exact names (`shipments`, `carriers`, `warehouses`)
2. **Column Types**: Match data types specified in schemas
3. **Relationships**: Configure foreign key relationships in the model
4. **Required Fields**: Ensure no nulls in required columns

## Data Quality Expectations

- **No nulls** in required fields
- **Date formats**: ISO 8601 (YYYY-MM-DD)
- **Numeric ranges**: Within specified min/max bounds
- **Categorical values**: Must match enum lists where specified
- **delivery_days_actual**: Should be >= 0 and <= 90 days

## Updating Schemas

If your semantic model structure changes:
1. Update corresponding JSON schema file
2. Update validation logic in `notebooks/utils/preprocessing.py`
3. Re-test all notebooks
4. Update this README with changes
