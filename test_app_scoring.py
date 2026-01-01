"""
Test script to verify SHAP explainability in the demo app.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import XGBoostModel class so joblib can unpickle it
from src.models.train import XGBoostModel

# Import from demo app
from src.demo.app import load_model_and_explainer, score_borrower, generate_sample_data

def test_shap_explainability():
    """Test SHAP explainability functionality."""
    print("=" * 70)
    print("Testing SHAP Explainability for Demo App")
    print("=" * 70)

    # Load model and explainer
    print("\n1. Loading model and explainer...")
    model, explainer, error = load_model_and_explainer()

    if error:
        print(f"   ❌ ERROR: {error}")
        return False

    if model is None:
        print("   ❌ ERROR: Model is None")
        return False

    if explainer is None:
        print("   ⚠️  WARNING: Explainer is None (SHAP not available)")
        print("   This means reason codes will not be generated")
        return False

    print("   ✓ Model loaded successfully")
    print("   ✓ SHAP explainer loaded successfully")

    # Generate sample transaction data
    print("\n2. Generating sample transaction data...")
    transactions_df = generate_sample_data()
    print(f"   ✓ Generated {len(transactions_df)} transactions")
    print(f"   ✓ Borrower ID: {transactions_df['borrower_id'].iloc[0]}")

    # Score borrower
    print("\n3. Scoring borrower with SHAP explanations...")

    # Temporarily modify score_borrower to print exceptions
    import streamlit as st
    original_warning = st.warning
    caught_warnings = []

    def capture_warning(msg):
        caught_warnings.append(msg)
        print(f"   ⚠️  Captured warning: {msg}")

    st.warning = capture_warning

    try:
        result = score_borrower(transactions_df, model, explainer)
        print(f"   ✓ Scoring successful")
        print(f"   ✓ Score: {result['score']}")
        print(f"   ✓ Risk Tier: {result['risk_tier']}")
        print(f"   ✓ Probability: {result['probability']:.4f}")
    except Exception as e:
        print(f"   ❌ ERROR during scoring: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        st.warning = original_warning

    # Check SHAP values
    print("\n4. Checking SHAP values...")
    if result['shap_values']:
        print(f"   ✓ SHAP values generated: {len(result['shap_values'])} features")
        # Show top 5 SHAP values
        sorted_shap = sorted(result['shap_values'].items(),
                            key=lambda x: abs(x[1]), reverse=True)[:5]
        print("\n   Top 5 SHAP contributions:")
        for feat, val in sorted_shap:
            direction = "↑ increases risk" if val > 0 else "↓ decreases risk"
            print(f"      {feat:30s}: {val:+.4f} {direction}")
    else:
        print("   ❌ ERROR: No SHAP values generated")
        return False

    # Check reason codes
    print("\n5. Checking reason codes...")
    if result['reason_codes']:
        print(f"   ✓ Reason codes generated: {len(result['reason_codes'])} codes")
        print("\n   Reason codes:")
        for i, rc in enumerate(result['reason_codes'], 1):
            impact_symbol = "✓" if rc['impact'] == 'positive' else "⚠"
            print(f"      {i}. [{impact_symbol}] {rc['code']} - {rc['name']}")
            print(f"         {rc['description']}")
            print(f"         Impact: {rc['impact']}, Magnitude: {rc['magnitude']}, "
                  f"Contribution: {rc['contribution']:.4f}")
            print()
    else:
        print("   ⚠️  WARNING: No reason codes generated")
        print("   This may be expected if no features meet thresholds")

    print("=" * 70)
    print("✓ SHAP Explainability Test PASSED")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = test_shap_explainability()
    sys.exit(0 if success else 1)
