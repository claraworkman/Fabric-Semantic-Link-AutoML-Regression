# Sample Data

This directory contains sample datasets for testing and development purposes.

## Files

### `shipments_sample.csv`
Sample shipment transaction data with 30 records.

**Columns:**
- `shipment_id`: Unique shipment identifier
- `order_id`: Associated order ID
- `carrier_id`: Foreign key to carriers table
- `warehouse_id`: Foreign key to warehouses table
- `origin_region`: Origin geographic region
- `destination_region`: Destination geographic region
- `distance_band`: Short/Medium/Long distance category
- `service_level`: Ground/Fast service type
- `order_date`: Date order was placed
- `ship_date`: Date shipment departed warehouse
- `delivery_date_expected`: Expected delivery date
- `delivery_date_actual`: Actual delivery date
- `order_to_ship_days`: Processing time in warehouse
- `ship_dayofweek`: Day of week shipped (0=Monday, 6=Sunday)
- `ship_month`: Month shipped (1-12)
- `delivery_days_actual`: Target variable - actual delivery time in days

### `carriers_sample.csv`
Sample carrier/logistics provider data.

**Columns:**
- `carrier_id`: Unique carrier identifier
- `carrier_name`: Carrier company name
- `carrier_type`: Type of carrier service
- `speed_factor`: Relative speed multiplier
- `reliability_score`: Quality score (1-5)
- `active`: Whether carrier is currently active

### `warehouses_sample.csv`
Sample warehouse/fulfillment center data.

**Columns:**
- `warehouse_id`: Unique warehouse identifier
- `warehouse_name`: Warehouse location name
- `origin_region`: Geographic region
- `capacity`: Storage capacity (units)
- `operational_hours`: Hours per day operational (16 or 24)
- `latitude`: Geographic latitude
- `longitude`: Geographic longitude

## Usage

### Load Sample Data in Python

```python
import pandas as pd

# Load shipments
shipments = pd.read_csv('data/sample/shipments_sample.csv', 
                        parse_dates=['order_date', 'ship_date', 
                                    'delivery_date_expected', 'delivery_date_actual'])

# Load carriers
carriers = pd.read_csv('data/sample/carriers_sample.csv')

# Load warehouses
warehouses = pd.read_csv('data/sample/warehouses_sample.csv')
```

### Load in Fabric Notebook

```python
# Upload CSVs to Fabric Lakehouse Files section, then:
shipments = spark.read.csv('Files/sample/shipments_sample.csv', header=True, inferSchema=True)
shipments_df = shipments.toPandas()
```

## Data Generation

This sample data was created to demonstrate the POC. For production:
- Replace with actual shipment history
- Ensure at least 6-12 months of historical data for training
- Include representative samples from all routes and carriers
- Validate data quality before training

## Privacy Note

This is synthetic test data. Do not commit actual customer or operational data to version control.
