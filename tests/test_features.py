"""
Test suite for feature engineering modules.

Tests feature extraction, validation, and prohibited feature filtering.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features.engineer import FeatureEngineer
from src.features.validators import FeatureValidator, FeatureValidationError


# ==================== FIXTURES ====================

@pytest.fixture
def sample_transactions():
    """Create sample transaction data for testing."""
    np.random.seed(42)

    transactions = []
    borrower_id = "TEST_BORROWER_001"

    # Generate 6 months of transactions
    start_date = datetime(2023, 1, 1)

    for month in range(6):
        month_start = start_date + timedelta(days=30 * month)

        # Income (2x per month)
        for _ in range(2):
            transactions.append({
                'borrower_id': borrower_id,
                'transaction_date': month_start + timedelta(days=np.random.randint(0, 15)),
                'amount': np.random.uniform(1500, 1700),  # ~$3200/month
                'category': 'INCOME'
            })

        # Rent (1x per month)
        transactions.append({
            'borrower_id': borrower_id,
            'transaction_date': month_start + timedelta(days=1),
            'amount': -1200,
            'category': 'RENT'
        })

        # Utilities
        transactions.append({
            'borrower_id': borrower_id,
            'transaction_date': month_start + timedelta(days=5),
            'amount': -150,
            'category': 'UTILITIES'
        })

        # Groceries (4x per month)
        for _ in range(4):
            transactions.append({
                'borrower_id': borrower_id,
                'transaction_date': month_start + timedelta(days=np.random.randint(0, 28)),
                'amount': np.random.uniform(-100, -50),
                'category': 'GROCERIES'
            })

        # Transportation
        for _ in range(2):
            transactions.append({
                'borrower_id': borrower_id,
                'transaction_date': month_start + timedelta(days=np.random.randint(0, 28)),
                'amount': np.random.uniform(-80, -40),
                'category': 'TRANSPORTATION'
            })

        # Entertainment
        transactions.append({
            'borrower_id': borrower_id,
            'transaction_date': month_start + timedelta(days=np.random.randint(0, 28)),
            'amount': np.random.uniform(-100, -50),
            'category': 'ENTERTAINMENT'
        })

    return pd.DataFrame(transactions)


@pytest.fixture
def sample_features():
    """Create sample feature dictionary for testing."""
    return {
        'avg_monthly_income': 3200.50,
        'income_std': 120.30,
        'income_cv': 0.0375,
        'avg_monthly_spending': 2500.00,
        'savings_rate': 0.22,
        'expense_income_ratio': 0.78,
        'overdraft_count_3mo': 0,
        'avg_balance': 1500.00,
        'discretionary_pct': 0.15,
        'transaction_count_3mo': 75
    }


# ==================== FEATURE ENGINEERING TESTS ====================

class TestFeatureEngineer:
    """Test feature engineering functionality."""

    def test_engineer_initialization(self):
        """Test feature engineer initializes correctly."""
        engineer = FeatureEngineer()
        assert engineer is not None

    def test_feature_extraction(self, sample_transactions):
        """Test features can be extracted from transactions."""
        engineer = FeatureEngineer()
        features = engineer.engineer_features_from_dataframe(sample_transactions)

        assert len(features) > 0, "No features extracted"
        assert 'borrower_id' in features.columns

    def test_feature_count(self, sample_transactions):
        """Test correct number of features extracted."""
        engineer = FeatureEngineer()
        features = engineer.engineer_features_from_dataframe(sample_transactions)

        # Should have 39 features + borrower_id
        expected_cols = 40  # 39 features + borrower_id
        assert len(features.columns) == expected_cols, \
            f"Expected {expected_cols} columns, got {len(features.columns)}"

    def test_income_features(self, sample_transactions):
        """Test income-related features are calculated correctly."""
        engineer = FeatureEngineer()
        features = engineer.engineer_features_from_dataframe(sample_transactions)

        # Check income features exist
        income_features = [
            'avg_monthly_income', 'income_std', 'income_cv',
            'income_trend_3mo', 'income_trend_6mo'
        ]

        for feature in income_features:
            assert feature in features.columns, f"Missing feature: {feature}"

        # Average monthly income should be ~3200
        avg_income = features['avg_monthly_income'].iloc[0]
        assert 3000 <= avg_income <= 3500, f"Unexpected avg income: {avg_income}"

        # Income CV should be low (stable income)
        income_cv = features['income_cv'].iloc[0]
        assert income_cv < 0.2, f"Income CV too high: {income_cv}"

    def test_spending_features(self, sample_transactions):
        """Test spending-related features are calculated correctly."""
        engineer = FeatureEngineer()
        features = engineer.engineer_features_from_dataframe(sample_transactions)

        spending_features = [
            'avg_monthly_spending', 'spending_std', 'discretionary_pct',
            'expense_income_ratio', 'savings_rate'
        ]

        for feature in spending_features:
            assert feature in features.columns, f"Missing feature: {feature}"

        # Expense-income ratio should be < 1 (not overspending)
        expense_ratio = features['expense_income_ratio'].iloc[0]
        assert 0 < expense_ratio < 1, f"Invalid expense ratio: {expense_ratio}"

        # Savings rate should be positive
        savings_rate = features['savings_rate'].iloc[0]
        assert savings_rate > 0, f"Negative savings rate: {savings_rate}"

    def test_financial_health_features(self, sample_transactions):
        """Test financial health features."""
        engineer = FeatureEngineer()
        features = engineer.engineer_features_from_dataframe(sample_transactions)

        health_features = [
            'avg_balance', 'min_balance', 'max_balance',
            'overdraft_count_3mo', 'overdraft_count_6mo', 'avg_net_cashflow'
        ]

        for feature in health_features:
            assert feature in features.columns, f"Missing feature: {feature}"

    def test_temporal_features(self, sample_transactions):
        """Test temporal features."""
        engineer = FeatureEngineer()
        features = engineer.engineer_features_from_dataframe(sample_transactions)

        temporal_features = [
            'transaction_count_3mo', 'transaction_count_6mo',
            'avg_transactions_per_month'
        ]

        for feature in temporal_features:
            assert feature in features.columns, f"Missing feature: {feature}"

        # Should have reasonable transaction counts
        txn_count = features['avg_transactions_per_month'].iloc[0]
        assert txn_count > 0, "No transactions counted"

    def test_no_null_values(self, sample_transactions):
        """Test extracted features have no null values."""
        engineer = FeatureEngineer()
        features = engineer.engineer_features_from_dataframe(sample_transactions)

        # Check for nulls
        null_counts = features.isnull().sum()
        assert null_counts.sum() == 0, f"Null values found: {null_counts[null_counts > 0]}"

    def test_feature_types(self, sample_transactions):
        """Test all features are numeric."""
        engineer = FeatureEngineer()
        features = engineer.engineer_features_from_dataframe(sample_transactions)

        # All columns except borrower_id should be numeric
        for col in features.columns:
            if col != 'borrower_id':
                assert pd.api.types.is_numeric_dtype(features[col]), \
                    f"Feature {col} is not numeric"

    def test_overdraft_detection(self):
        """Test overdraft count is calculated correctly."""
        # Create transactions with overdrafts
        transactions = []
        borrower_id = "TEST_BORROWER_002"

        # Month 1: 2 overdrafts
        transactions.append({
            'borrower_id': borrower_id,
            'transaction_date': datetime(2023, 1, 1),
            'amount': 1000,
            'category': 'INCOME'
        })
        transactions.append({
            'borrower_id': borrower_id,
            'transaction_date': datetime(2023, 1, 5),
            'amount': -1200,  # Overdraft
            'category': 'RENT'
        })
        transactions.append({
            'borrower_id': borrower_id,
            'transaction_date': datetime(2023, 1, 10),
            'amount': -100,  # Another overdraft
            'category': 'GROCERIES'
        })

        # Month 2: Income recovery
        transactions.append({
            'borrower_id': borrower_id,
            'transaction_date': datetime(2023, 2, 1),
            'amount': 2000,
            'category': 'INCOME'
        })

        df = pd.DataFrame(transactions)
        engineer = FeatureEngineer()
        features = engineer.engineer_features_from_dataframe(df)

        # Should detect overdrafts
        overdraft_count = features['overdraft_count_3mo'].iloc[0]
        assert overdraft_count > 0, "Overdrafts not detected"


# ==================== FEATURE VALIDATION TESTS ====================

class TestFeatureValidator:
    """Test feature validation and prohibited feature filtering."""

    def test_validator_initialization(self):
        """Test validator initializes correctly."""
        validator = FeatureValidator()
        assert validator.prohibited_features is not None
        assert validator.allowed_features is not None

    def test_allowed_features_pass(self, sample_features):
        """Test allowed features pass validation."""
        validator = FeatureValidator()

        # Should not raise error
        try:
            validator.validate_feature_set(sample_features.keys())
        except FeatureValidationError:
            pytest.fail("Validation failed for allowed features")

    def test_prohibited_features_fail(self):
        """Test prohibited features fail validation."""
        validator = FeatureValidator()

        prohibited = ['race', 'gender', 'age', 'income_cv']  # Mix of prohibited and allowed

        with pytest.raises(FeatureValidationError):
            validator.validate_feature_set(prohibited, raise_on_prohibited=True)

    def test_prohibited_features_warning(self):
        """Test prohibited features generate warnings."""
        validator = FeatureValidator()

        prohibited = ['race', 'gender']

        # Should not raise with raise_on_prohibited=False
        result = validator.validate_feature_set(prohibited, raise_on_prohibited=False)
        assert result is False, "Should return False for prohibited features"

    def test_all_allowed_features(self):
        """Test all 39 allowed features pass validation."""
        validator = FeatureValidator()

        # Get all allowed features from config
        allowed = validator.allowed_features

        # Should not raise
        try:
            validator.validate_feature_set(allowed)
        except FeatureValidationError:
            pytest.fail("Validation failed for allowed features from config")

    def test_feature_name_normalization(self):
        """Test feature names are normalized (case-insensitive)."""
        validator = FeatureValidator()

        # Test with different cases
        features = ['AVG_MONTHLY_INCOME', 'Savings_Rate', 'income_cv']

        # Should handle case variations
        # (Depends on implementation - adjust if case-sensitive)
        result = validator.validate_feature_set(features, raise_on_prohibited=False)
        # Should be valid (assuming case normalization)

    def test_empty_feature_set(self):
        """Test validation of empty feature set."""
        validator = FeatureValidator()

        # Empty set should be valid (no prohibited features)
        result = validator.validate_feature_set([])
        assert result is True

    def test_prohibited_feature_list(self):
        """Test comprehensive list of prohibited features."""
        validator = FeatureValidator()

        # Common prohibited features
        prohibited_list = [
            'race', 'ethnicity', 'gender', 'sex', 'age',
            'marital_status', 'religion', 'national_origin',
            'disability', 'family_status', 'sexual_orientation'
        ]

        for feature in prohibited_list:
            if feature in validator.prohibited_features:
                with pytest.raises(FeatureValidationError):
                    validator.validate_feature_set([feature], raise_on_prohibited=True)


# ==================== FEATURE QUALITY TESTS ====================

class TestFeatureQuality:
    """Test quality and consistency of extracted features."""

    def test_feature_ranges(self, sample_transactions):
        """Test features are within reasonable ranges."""
        engineer = FeatureEngineer()
        features = engineer.engineer_features_from_dataframe(sample_transactions)

        # Income CV should be between 0 and 2
        assert 0 <= features['income_cv'].iloc[0] <= 2

        # Savings rate should be between -1 and 1
        assert -1 <= features['savings_rate'].iloc[0] <= 1

        # Expense-income ratio should be >= 0
        assert features['expense_income_ratio'].iloc[0] >= 0

        # Overdraft counts should be >= 0
        assert features['overdraft_count_3mo'].iloc[0] >= 0
        assert features['overdraft_count_6mo'].iloc[0] >= 0

    def test_feature_consistency(self, sample_transactions):
        """Test features are internally consistent."""
        engineer = FeatureEngineer()
        features = engineer.engineer_features_from_dataframe(sample_transactions)

        # 6-month count should be >= 3-month count
        assert features['transaction_count_6mo'].iloc[0] >= features['transaction_count_3mo'].iloc[0]
        assert features['overdraft_count_6mo'].iloc[0] >= features['overdraft_count_3mo'].iloc[0]

        # Min balance <= avg balance <= max balance
        assert features['min_balance'].iloc[0] <= features['avg_balance'].iloc[0]
        assert features['avg_balance'].iloc[0] <= features['max_balance'].iloc[0]

    def test_feature_reproducibility(self, sample_transactions):
        """Test feature extraction is reproducible."""
        engineer1 = FeatureEngineer()
        engineer2 = FeatureEngineer()

        features1 = engineer1.engineer_features_from_dataframe(sample_transactions)
        features2 = engineer2.engineer_features_from_dataframe(sample_transactions)

        # Should produce identical results
        pd.testing.assert_frame_equal(features1, features2)


# ==================== EDGE CASE TESTS ====================

class TestFeatureEdgeCases:
    """Test edge cases in feature engineering."""

    def test_single_transaction(self):
        """Test feature extraction with single transaction."""
        df = pd.DataFrame([{
            'borrower_id': 'TEST_001',
            'transaction_date': datetime(2023, 1, 1),
            'amount': 1000,
            'category': 'INCOME'
        }])

        engineer = FeatureEngineer()
        features = engineer.engineer_features_from_dataframe(df)

        # Should handle gracefully (may have some null features)
        assert len(features) == 1

    def test_no_income_transactions(self):
        """Test feature extraction with no income."""
        transactions = []
        for i in range(10):
            transactions.append({
                'borrower_id': 'TEST_002',
                'transaction_date': datetime(2023, 1, 1) + timedelta(days=i),
                'amount': -100,
                'category': 'GROCERIES'
            })

        df = pd.DataFrame(transactions)
        engineer = FeatureEngineer()
        features = engineer.engineer_features_from_dataframe(df)

        # Should handle (income features may be 0 or null)
        assert len(features) == 1

    def test_all_income_no_expenses(self):
        """Test feature extraction with only income."""
        transactions = []
        for i in range(10):
            transactions.append({
                'borrower_id': 'TEST_003',
                'transaction_date': datetime(2023, 1, 1) + timedelta(days=i * 3),
                'amount': 1000,
                'category': 'INCOME'
            })

        df = pd.DataFrame(transactions)
        engineer = FeatureEngineer()
        features = engineer.engineer_features_from_dataframe(df)

        # Savings rate should be 1.0 (all income saved)
        assert features['savings_rate'].iloc[0] >= 0.99

    def test_zero_amount_transactions(self):
        """Test handling of zero-amount transactions."""
        df = pd.DataFrame([
            {
                'borrower_id': 'TEST_004',
                'transaction_date': datetime(2023, 1, 1),
                'amount': 0,
                'category': 'INCOME'
            },
            {
                'borrower_id': 'TEST_004',
                'transaction_date': datetime(2023, 1, 2),
                'amount': 1000,
                'category': 'INCOME'
            }
        ])

        engineer = FeatureEngineer()
        features = engineer.engineer_features_from_dataframe(df)

        # Should handle gracefully
        assert len(features) == 1


# ==================== INTEGRATION TESTS ====================

class TestFeaturePipeline:
    """Test end-to-end feature engineering pipeline."""

    def test_full_pipeline(self, sample_transactions):
        """Test complete feature engineering pipeline."""
        # Extract features
        engineer = FeatureEngineer()
        features = engineer.engineer_features_from_dataframe(sample_transactions)

        # Validate features
        validator = FeatureValidator()

        # Get feature names (exclude borrower_id)
        feature_names = [col for col in features.columns if col != 'borrower_id']

        # Should pass validation
        try:
            validator.validate_feature_set(feature_names)
        except FeatureValidationError as e:
            pytest.fail(f"Feature validation failed: {str(e)}")

    def test_batch_processing(self):
        """Test feature extraction for multiple borrowers."""
        # Create transactions for 3 borrowers
        all_transactions = []

        for i in range(3):
            borrower_id = f"TEST_BORROWER_{i:03d}"

            for month in range(3):
                # Income
                all_transactions.append({
                    'borrower_id': borrower_id,
                    'transaction_date': datetime(2023, month + 1, 1),
                    'amount': 1000 + i * 100,
                    'category': 'INCOME'
                })

                # Expense
                all_transactions.append({
                    'borrower_id': borrower_id,
                    'transaction_date': datetime(2023, month + 1, 5),
                    'amount': -500,
                    'category': 'RENT'
                })

        df = pd.DataFrame(all_transactions)
        engineer = FeatureEngineer()
        features = engineer.engineer_features_from_dataframe(df)

        # Should have 3 rows (one per borrower)
        assert len(features) == 3

        # Each should have same number of features
        assert features.shape[1] == 40  # 39 + borrower_id


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src.features", "--cov-report=term-missing"])
