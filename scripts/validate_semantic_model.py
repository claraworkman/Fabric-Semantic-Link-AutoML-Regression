"""
Validate Semantic Model

This script checks that your Power BI semantic model has all required
tables and columns for the Delivery Time Prediction POC.

Usage:
    python scripts/validate_semantic_model.py
    
Note: This script should be run from a Fabric notebook environment
where sempy.fabric is available and configured.
"""

import sys
import json
from pathlib import Path

try:
    import sempy.fabric as fabric
    SEMPY_AVAILABLE = True
except ImportError:
    SEMPY_AVAILABLE = False
    print("⚠ semantic-link not available. This script should run in a Fabric environment.")


def load_schema_requirements():
    """Load expected schema from JSON files"""
    project_root = Path(__file__).parent.parent
    schema_dir = project_root / 'data' / 'schema'
    
    schemas = {}
    schema_files = {
        'shipments': 'shipments_schema.json',
        'carriers': 'carriers_schema.json',
        'warehouses': 'warehouses_schema.json'
    }
    
    for table_name, filename in schema_files.items():
        schema_path = schema_dir / filename
        if schema_path.exists():
            with open(schema_path, 'r') as f:
                schemas[table_name] = json.load(f)
        else:
            print(f"⚠ Schema file not found: {filename}")
    
    return schemas


def validate_table_exists(dataset_name, table_name):
    """Check if table exists in semantic model"""
    try:
        tables = fabric.list_tables(dataset_name)
        table_list = tables['Name'].tolist() if hasattr(tables, 'Name') else []
        
        if table_name in table_list:
            print(f"✓ Table '{table_name}' found")
            return True
        else:
            print(f"✗ Table '{table_name}' NOT found")
            return False
    except Exception as e:
        print(f"✗ Error checking table '{table_name}': {e}")
        return False


def validate_columns(dataset_name, table_name, required_columns):
    """Check if required columns exist in table"""
    try:
        columns_df = fabric.list_columns(dataset_name, table_name)
        actual_columns = columns_df['Column'].tolist() if hasattr(columns_df, 'Column') else []
        
        missing_columns = []
        for col in required_columns:
            if col in actual_columns:
                print(f"  ✓ Column '{col}' found")
            else:
                print(f"  ✗ Column '{col}' NOT found")
                missing_columns.append(col)
        
        return len(missing_columns) == 0, missing_columns
    
    except Exception as e:
        print(f"  ✗ Error checking columns: {e}")
        return False, required_columns


def validate_semantic_model(dataset_name):
    """Validate entire semantic model"""
    if not SEMPY_AVAILABLE:
        print("\n⚠ Cannot validate - semantic-link package not available")
        print("This validation should be run from a Fabric notebook.")
        return False
    
    print("=" * 60)
    print(f"Validating Semantic Model: {dataset_name}")
    print("=" * 60)
    
    # Load schema requirements
    schemas = load_schema_requirements()
    if not schemas:
        print("✗ Could not load schema requirements")
        return False
    
    all_valid = True
    
    # Validate each table
    for table_name, schema in schemas.items():
        print(f"\nValidating table: {table_name}")
        
        # Check table exists
        if not validate_table_exists(dataset_name, table_name):
            all_valid = False
            continue
        
        # Get required columns from schema
        required_columns = schema.get('required', [])
        if not required_columns:
            # If no required list, use all properties
            required_columns = list(schema.get('properties', {}).keys())
        
        # Validate columns
        columns_valid, missing = validate_columns(dataset_name, table_name, required_columns)
        if not columns_valid:
            all_valid = False
            print(f"  ⚠ Missing columns: {', '.join(missing)}")
    
    # Summary
    print("\n" + "=" * 60)
    if all_valid:
        print("✓ Semantic model validation PASSED")
        print("\nYour semantic model is ready for use!")
        print("\nNext steps:")
        print("1. Run notebook: 01_semantic_link_data_preparation.ipynb")
        print("2. Verify data loads correctly")
    else:
        print("✗ Semantic model validation FAILED")
        print("\nPlease fix the issues above in your semantic model.")
    
    print("=" * 60)
    
    return all_valid


def main():
    """Main validation entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Validate Power BI semantic model for Delivery Time Prediction POC'
    )
    parser.add_argument(
        '--model',
        default='delivery semantic model',
        help='Name of the semantic model (default: "delivery semantic model")'
    )
    
    args = parser.parse_args()
    
    success = validate_semantic_model(args.model)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
