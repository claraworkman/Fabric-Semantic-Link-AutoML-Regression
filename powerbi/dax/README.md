-- =====================================================
-- Power BI DAX Measures - README
-- =====================================================

This directory contains DAX measure definitions for the Delivery Time Prediction Power BI report.

## Files

### measures_basic.dax
Core metrics for model performance and delivery time analysis:
- Actual delivery metrics (average, min, max)
- Predicted delivery metrics
- Error metrics (MAE, RMSE, Mean Error, Median Error)
- Accuracy rates
- Business metrics (on-time delivery, SLA compliance)

### measures_advanced.dax
Advanced analytics measures:
- Time intelligence (trends, MoM changes)
- Carrier performance comparisons
- Route-level analysis
- Segmentation metrics
- Alert conditions

## Data Model Requirements

These measures expect the following tables in your semantic model:

### shipments
- shipment_id (key)
- delivery_days_actual (target variable)
- carrier_id
- warehouse_id
- origin_region
- destination_region
- ship_date
- etc.

### shipment_predictions
- shipment_id (key, links to shipments)
- predicted_delivery_days (model output)

### carriers
- carrier_id (key)
- carrier_name
- etc.

### warehouses
- warehouse_id (key)
- warehouse_name
- origin_region
- etc.

## Relationship Setup

Ensure these relationships exist in your model:
- shipments[shipment_id] → shipment_predictions[shipment_id] (1:1)
- shipments[carrier_id] → carriers[carrier_id] (Many:1)
- shipments[warehouse_id] → warehouses[warehouse_id] (Many:1)

## Importing Measures

### Option 1: Manual Copy-Paste
1. Open your Power BI Desktop file
2. Go to Model view
3. Select appropriate table for measures
4. Copy-paste measure definitions from DAX files

### Option 2: Using Tabular Editor (Recommended)
1. Install Tabular Editor 2 or 3
2. Connect to your semantic model
3. Import measures from DAX files
4. Save changes back to model

### Option 3: DAX Studio
1. Open DAX Studio
2. Connect to your model
3. Use scripts to create measures

## Key Metrics Explained

### MAE (Mean Absolute Error)
Average absolute difference between predicted and actual delivery days.
**Target**: < 2.0 days
**Lower is better**

### RMSE (Root Mean Squared Error)
Square root of average squared errors. Penalizes large errors more.
**Lower is better**

### R² Score
Proportion of variance explained by the model.
**Range**: 0 to 1
**Higher is better**
**Target**: > 0.80

### Accuracy Rate (Within 2 Days)
Percentage of predictions within 2 days of actual delivery.
**Target**: > 85%

### On-Time Delivery Rate
Percentage of deliveries meeting or beating expected date.
**Business KPI**

## Usage in Visuals

### KPI Cards
- [MAE]
- [RMSE]
- [R2Score]
- [AccuracyRate]

### Line Charts
- X-axis: ship_date
- Y-axis: [AvgActualDeliveryDays], [AvgPredictedDeliveryDays]

### Scatter Plot (Actual vs Predicted)
- X-axis: [AvgPredictedDeliveryDays]
- Y-axis: [AvgActualDeliveryDays]
- Diagonal reference line for perfect predictions

### Table (At-Risk Shipments)
- shipment_id
- predicted_delivery_days
- delivery_date_expected
- [DaysToSLA]
- [IsAtRisk]

## Customization

### Adjust Thresholds
Edit these values in measures as needed:
- Accuracy window: Currently 2 days
- SLA warning threshold: Currently 1 day
- At-risk definition

### Add New Metrics
When adding metrics:
1. Document purpose and calculation
2. Add to appropriate DAX file (basic vs advanced)
3. Update this README
4. Test on sample data

## Performance Tips

1. **Use Variables**: DAX variables (VAR) improve performance and readability
2. **Avoid Row Context**: Prefer calculated columns for row-level calculations
3. **Filter Early**: Apply filters before expensive calculations
4. **Test Performance**: Use Performance Analyzer in Power BI Desktop

## Troubleshooting

### Measure returns BLANK
- Check if shipment_predictions table exists
- Verify relationships are active
- Ensure predicted_delivery_days column exists

### Incorrect values
- Verify table relationships direction
- Check filter context
- Test with simple visuals first

### Slow performance
- Use DAX Studio to analyze query plans
- Consider adding calculated columns for frequently used logic
- Optimize relationship cardinality

## Additional Resources

- [DAX Guide](https://dax.guide/)
- [SQLBI DAX Patterns](https://www.daxpatterns.com/)
- [Microsoft DAX Reference](https://learn.microsoft.com/dax/)
