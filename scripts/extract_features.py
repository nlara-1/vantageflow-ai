#!/usr/bin/env python3
"""
Feature extraction script for VantageFlow AI.

Extracts 30-40 features from transaction data in the database and joins with
labels to create a complete feature dataset for model training.

Usage:
    python scripts/extract_features.py
"""

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.engineer import FeatureEngineer


def main():
    """
    Main feature extraction pipeline.

    Steps:
    1. Connect to database
    2. Load all borrowers
    3. Extract features for each borrower
    4. Join with labels
    5. Save to CSV
    6. Print summary
    """
    print("=" * 80)
    print("VantageFlow AI - Feature Extraction Pipeline")
    print("=" * 80)

    # Configuration
    db_path = "data/vantageflow.db"
    output_dir = "data/output"
    output_file = f"{output_dir}/features.csv"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n✓ Output directory created: {output_dir}")

    # Check database exists
    if not os.path.exists(db_path):
        print(f"\n❌ Error: Database not found at {db_path}")
        print("Please run the data generation script first:")
        print("  python scripts/generate_data.py")
        sys.exit(1)

    print(f"✓ Database found: {db_path}")

    # Initialize feature engineer
    print("\n" + "-" * 80)
    print("Step 1: Initialize Feature Engineer")
    print("-" * 80)

    engineer = FeatureEngineer(db_path=db_path)
    print(f"✓ Feature engineer initialized")
    print(f"  - Total features to extract: {len(engineer.feature_names)}")

    # Display feature groups
    print("\nFeature groups:")
    print("  - Income features: 8")
    print("  - Spending features: 8")
    print("  - Financial health features: 7")
    print("  - Temporal features: 6")
    print("  - Category features: 5")
    print("  - Derived features: 5")
    print(f"  - Total: {len(engineer.feature_names)} features")

    # Load borrower IDs
    print("\n" + "-" * 80)
    print("Step 2: Load Borrowers")
    print("-" * 80)

    engine = create_engine(f"sqlite:///{db_path}")

    # Get unique borrower IDs from transactions
    query = "SELECT DISTINCT borrower_id FROM transactions ORDER BY borrower_id"
    borrower_ids = pd.read_sql_query(query, engine)["borrower_id"].tolist()

    print(f"✓ Found {len(borrower_ids):,} unique borrowers")

    # Extract features
    print("\n" + "-" * 80)
    print("Step 3: Extract Features")
    print("-" * 80)

    features_df = engineer.extract_all_features(
        borrower_ids=borrower_ids,
        verbose=True
    )

    print(f"\n✓ Feature extraction complete")
    print(f"  - Shape: {features_df.shape}")
    print(f"  - Features: {features_df.shape[1] - 1} (excluding borrower_id)")

    # Load labels
    print("\n" + "-" * 80)
    print("Step 4: Load and Join Labels")
    print("-" * 80)

    labels_query = """
    SELECT
        borrower_id,
        default_label,
        default_probability,
        overall_score
    FROM labels
    """

    labels_df = pd.read_sql_query(labels_query, engine)
    print(f"✓ Loaded labels for {len(labels_df):,} borrowers")

    # Join features with labels
    final_df = features_df.merge(
        labels_df,
        on="borrower_id",
        how="left"
    )

    print(f"✓ Merged features with labels")
    print(f"  - Final shape: {final_df.shape}")
    print(f"  - Columns: borrower_id + {len(engineer.feature_names)} features + labels")

    # Check for missing labels
    missing_labels = final_df["default_label"].isna().sum()
    if missing_labels > 0:
        print(f"  ⚠️  Warning: {missing_labels} borrowers missing labels")

    # Reorder columns: borrower_id, features, then labels
    feature_cols = ["borrower_id"] + engineer.feature_names
    label_cols = ["default_label", "default_probability", "overall_score"]

    # Only include label columns that exist
    label_cols = [col for col in label_cols if col in final_df.columns]

    final_df = final_df[feature_cols + label_cols]

    # Save to CSV
    print("\n" + "-" * 80)
    print("Step 5: Save Features")
    print("-" * 80)

    final_df.to_csv(output_file, index=False)
    print(f"✓ Features saved to: {output_file}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("FEATURE EXTRACTION SUMMARY")
    print("=" * 80)

    print(f"\nDataset Shape:")
    print(f"  - Total rows: {len(final_df):,}")
    print(f"  - Total columns: {len(final_df.columns)}")
    print(f"  - Feature columns: {len(engineer.feature_names)}")
    print(f"  - Label columns: {len(label_cols)}")

    print(f"\nLabel Distribution:")
    if "default_label" in final_df.columns:
        label_counts = final_df["default_label"].value_counts()
        total = len(final_df)

        # Handle boolean or int labels
        for label in sorted(label_counts.index):
            count = label_counts[label]
            pct = count / total * 100
            label_name = "Default" if label else "No Default"
            print(f"  - {label_name}: {count:,} ({pct:.1f}%)")

    print(f"\nMissing Values:")
    missing = final_df.isnull().sum()
    missing_features = missing[missing > 0]

    if len(missing_features) == 0:
        print("  ✓ No missing values detected")
    else:
        print(f"  Total features with missing values: {len(missing_features)}")
        print(f"\n  Top 10 features with missing values:")
        for feature, count in missing_features.nlargest(10).items():
            pct = count / len(final_df) * 100
            print(f"    - {feature}: {count:,} ({pct:.1f}%)")

    print(f"\nFeature Statistics (sample):")
    print("\n" + "  Income Features:")
    income_features = [f for f in engineer.feature_names if "income" in f][:3]
    for feature in income_features:
        if feature in final_df.columns:
            stats = final_df[feature].describe()
            print(f"    - {feature}:")
            print(f"      Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}, "
                  f"Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")

    print("\n" + "  Spending Features:")
    spending_features = [f for f in engineer.feature_names if "spending" in f][:3]
    for feature in spending_features:
        if feature in final_df.columns:
            stats = final_df[feature].describe()
            print(f"    - {feature}:")
            print(f"      Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}, "
                  f"Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")

    print("\n" + "  Financial Health:")
    health_features = [f for f in engineer.feature_names if "balance" in f or "overdraft" in f][:3]
    for feature in health_features:
        if feature in final_df.columns:
            stats = final_df[feature].describe()
            print(f"    - {feature}:")
            print(f"      Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}, "
                  f"Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")

    print("\n" + "=" * 80)
    print("✓ Feature extraction completed successfully!")
    print("=" * 80)

    print(f"\nNext steps:")
    print(f"  1. Review the feature distribution and missing values")
    print(f"  2. Train a model using the extracted features")
    print(f"  3. Evaluate model performance")

    print(f"\nOutput file: {output_file}")
    print(f"File size: {os.path.getsize(output_file) / 1024:.1f} KB")

    # Close database connection
    engine.dispose()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Feature extraction interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during feature extraction: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
