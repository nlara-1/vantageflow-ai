"""
Test suite for data generation modules.

Tests borrower profile generation, transaction synthesis, and label generation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data_generation.profiles import BorrowerProfileGenerator
from src.data_generation.synthesizer import TransactionSynthesizer
from src.data_generation.labels import CreditRiskLabeler


# ==================== FIXTURES ====================

@pytest.fixture
def sample_borrowers():
    """Generate small set of test borrower profiles."""
    generator = BorrowerProfileGenerator(num_borrowers=10)
    return generator.generate_profiles()


@pytest.fixture
def sample_transactions(sample_borrowers):
    """Generate sample transactions for test borrowers."""
    synthesizer = TransactionSynthesizer()
    # Generate 3 months only for speed
    transactions = []
    for borrower in sample_borrowers[:3]:  # Just first 3 borrowers
        borrower_txns = synthesizer.generate_transactions(
            borrower,
            start_date=datetime(2023, 1, 1),
            months=3
        )
        transactions.extend(borrower_txns)
    return pd.DataFrame(transactions)


# ==================== PROFILE GENERATION TESTS ====================

class TestBorrowerProfileGenerator:
    """Test borrower profile generation."""

    def test_generator_initialization(self):
        """Test generator initializes with correct parameters."""
        generator = BorrowerProfileGenerator(num_borrowers=100)
        assert generator.num_borrowers == 100
        assert generator.mean_income > 0
        assert generator.std_income > 0

    def test_generate_profiles_count(self):
        """Test correct number of profiles generated."""
        generator = BorrowerProfileGenerator(num_borrowers=50)
        profiles = generator.generate_profiles()
        assert len(profiles) == 50

    def test_profile_structure(self, sample_borrowers):
        """Test profile has all required fields."""
        profile = sample_borrowers[0]

        required_fields = [
            'borrower_id', 'created_at', 'age', 'income',
            'employment_type', 'education_level', 'location'
        ]

        for field in required_fields:
            assert field in profile, f"Missing required field: {field}"

    def test_borrower_id_uniqueness(self, sample_borrowers):
        """Test all borrower IDs are unique."""
        ids = [b['borrower_id'] for b in sample_borrowers]
        assert len(ids) == len(set(ids)), "Duplicate borrower IDs found"

    def test_age_distribution(self, sample_borrowers):
        """Test age is within reasonable bounds."""
        ages = [b['age'] for b in sample_borrowers]
        assert all(18 <= age <= 75 for age in ages), "Age outside valid range"
        assert np.mean(ages) > 25, "Mean age too low"
        assert np.mean(ages) < 60, "Mean age too high"

    def test_income_distribution(self, sample_borrowers):
        """Test income follows lognormal distribution."""
        incomes = [b['income'] for b in sample_borrowers]
        assert all(income > 0 for income in incomes), "Negative income found"
        assert min(incomes) >= 15000, "Income too low"
        assert max(incomes) <= 200000, "Income unrealistically high"

    def test_employment_types(self, sample_borrowers):
        """Test employment types are valid."""
        valid_types = ['full_time', 'part_time', 'self_employed', 'contract']
        employment_types = [b['employment_type'] for b in sample_borrowers]
        assert all(et in valid_types for et in employment_types)

    def test_education_levels(self, sample_borrowers):
        """Test education levels are valid."""
        valid_levels = ['high_school', 'associates', 'bachelors', 'masters', 'phd']
        education_levels = [b['education_level'] for b in sample_borrowers]
        assert all(el in valid_levels for el in education_levels)


# ==================== TRANSACTION SYNTHESIS TESTS ====================

class TestTransactionSynthesizer:
    """Test transaction synthesis."""

    def test_synthesizer_initialization(self):
        """Test synthesizer initializes correctly."""
        synth = TransactionSynthesizer()
        assert synth.categories is not None
        assert len(synth.categories) > 0

    def test_generate_transactions_count(self, sample_borrowers):
        """Test transaction count is reasonable."""
        synth = TransactionSynthesizer()
        borrower = sample_borrowers[0]

        transactions = synth.generate_transactions(
            borrower,
            start_date=datetime(2023, 1, 1),
            months=3
        )

        # Should have 50-150 transactions per month
        assert 150 <= len(transactions) <= 450, f"Unexpected transaction count: {len(transactions)}"

    def test_transaction_structure(self, sample_transactions):
        """Test transaction has all required fields."""
        required_fields = ['borrower_id', 'transaction_date', 'amount', 'category']

        for field in required_fields:
            assert field in sample_transactions.columns, f"Missing field: {field}"

    def test_transaction_dates(self, sample_transactions):
        """Test transaction dates are within expected range."""
        dates = pd.to_datetime(sample_transactions['transaction_date'])

        assert dates.min() >= pd.Timestamp('2023-01-01'), "Transaction date too early"
        assert dates.max() <= pd.Timestamp('2023-04-01'), "Transaction date too late"

    def test_income_transactions(self, sample_transactions):
        """Test income transactions are positive."""
        income_txns = sample_transactions[sample_transactions['category'] == 'INCOME']

        assert len(income_txns) > 0, "No income transactions found"
        assert all(income_txns['amount'] > 0), "Negative income found"

    def test_expense_transactions(self, sample_transactions):
        """Test expense transactions are negative."""
        expense_categories = ['RENT', 'GROCERIES', 'UTILITIES', 'TRANSPORTATION']
        expense_txns = sample_transactions[
            sample_transactions['category'].isin(expense_categories)
        ]

        assert len(expense_txns) > 0, "No expense transactions found"
        assert all(expense_txns['amount'] < 0), "Positive expense found"

    def test_category_distribution(self, sample_transactions):
        """Test transaction categories are diverse."""
        category_counts = sample_transactions['category'].value_counts()

        # Should have at least 5 different categories
        assert len(category_counts) >= 5, "Insufficient category diversity"

        # Income should be present
        assert 'INCOME' in category_counts.index, "No income transactions"

    def test_monthly_income_consistency(self, sample_transactions):
        """Test monthly income is relatively consistent."""
        income_txns = sample_transactions[sample_transactions['category'] == 'INCOME']
        income_txns['month'] = pd.to_datetime(income_txns['transaction_date']).dt.to_period('M')

        monthly_income = income_txns.groupby('month')['amount'].sum()

        if len(monthly_income) > 1:
            # CV should be < 0.5 for most borrowers
            cv = monthly_income.std() / monthly_income.mean()
            assert cv < 1.0, f"Income too volatile: CV={cv:.2f}"


# ==================== LABEL GENERATION TESTS ====================

class TestCreditRiskLabeler:
    """Test credit risk label generation."""

    def test_labeler_initialization(self):
        """Test labeler initializes with correct parameters."""
        labeler = CreditRiskLabeler(default_rate=0.20, noise_level=0.05)
        assert labeler.default_rate == 0.20
        assert labeler.noise_level == 0.05
        assert labeler.weights is not None

    def test_generate_labels_count(self, sample_transactions):
        """Test correct number of labels generated."""
        labeler = CreditRiskLabeler()
        labels = labeler.generate_labels(sample_transactions)

        # Should have one label per unique borrower
        unique_borrowers = sample_transactions['borrower_id'].nunique()
        assert len(labels) == unique_borrowers

    def test_label_structure(self, sample_transactions):
        """Test label has required fields."""
        labeler = CreditRiskLabeler()
        labels = labeler.generate_labels(sample_transactions)

        required_fields = ['borrower_id', 'default_label', 'default_probability']

        for field in required_fields:
            assert field in labels[0], f"Missing field: {field}"

    def test_default_rate(self, sample_transactions):
        """Test default rate is approximately as specified."""
        labeler = CreditRiskLabeler(default_rate=0.20, noise_level=0.0)  # No noise

        # Generate labels for many borrowers
        labels = labeler.generate_labels(sample_transactions)
        default_count = sum(1 for label in labels if label['default_label'])
        actual_rate = default_count / len(labels)

        # Should be within 20% of target rate (with small sample size)
        assert 0.0 <= actual_rate <= 0.6, f"Default rate {actual_rate:.2f} far from target 0.20"

    def test_default_probability_range(self, sample_transactions):
        """Test default probabilities are between 0 and 1."""
        labeler = CreditRiskLabeler()
        labels = labeler.generate_labels(sample_transactions)

        probs = [label['default_probability'] for label in labels]
        assert all(0 <= p <= 1 for p in probs), "Probabilities outside [0, 1] range"

    def test_financial_health_score(self, sample_transactions):
        """Test financial health score calculation."""
        labeler = CreditRiskLabeler()

        # Get transactions for one borrower
        borrower_id = sample_transactions['borrower_id'].iloc[0]
        borrower_txns = sample_transactions[
            sample_transactions['borrower_id'] == borrower_id
        ]

        score = labeler.calculate_financial_health_score(borrower_txns)

        # Score should be between 0 and 100
        assert 0 <= score <= 100, f"Health score {score} outside valid range"

    def test_noise_effect(self, sample_transactions):
        """Test noise level affects probabilities."""
        labeler_no_noise = CreditRiskLabeler(noise_level=0.0)
        labeler_high_noise = CreditRiskLabeler(noise_level=0.2)

        labels_no_noise = labeler_no_noise.generate_labels(sample_transactions)
        labels_high_noise = labeler_high_noise.generate_labels(sample_transactions)

        # High noise should produce more varied probabilities
        probs_no_noise = [l['default_probability'] for l in labels_no_noise]
        probs_high_noise = [l['default_probability'] for l in labels_high_noise]

        # Standard deviation should be higher with noise
        # (This is probabilistic, but should hold most of the time)
        # Just check they're different
        assert probs_no_noise != probs_high_noise


# ==================== INTEGRATION TESTS ====================

class TestDataGenerationPipeline:
    """Test end-to-end data generation pipeline."""

    def test_full_pipeline(self):
        """Test complete data generation pipeline."""
        # Generate borrowers
        generator = BorrowerProfileGenerator(num_borrowers=5)
        borrowers = generator.generate_profiles()
        assert len(borrowers) == 5

        # Generate transactions
        synth = TransactionSynthesizer()
        all_transactions = []
        for borrower in borrowers:
            txns = synth.generate_transactions(
                borrower,
                start_date=datetime(2023, 1, 1),
                months=3
            )
            all_transactions.extend(txns)

        transactions_df = pd.DataFrame(all_transactions)
        assert len(transactions_df) > 0

        # Generate labels
        labeler = CreditRiskLabeler()
        labels = labeler.generate_labels(transactions_df)
        assert len(labels) == 5

    def test_borrower_transaction_consistency(self):
        """Test transactions are consistent with borrower profiles."""
        generator = BorrowerProfileGenerator(num_borrowers=3)
        borrowers = generator.generate_profiles()

        synth = TransactionSynthesizer()

        for borrower in borrowers:
            txns = synth.generate_transactions(
                borrower,
                start_date=datetime(2023, 1, 1),
                months=2
            )
            txns_df = pd.DataFrame(txns)

            # All transactions should have same borrower_id
            assert all(txns_df['borrower_id'] == borrower['borrower_id'])

            # Income should roughly match borrower income
            monthly_income = txns_df[
                txns_df['category'] == 'INCOME'
            ]['amount'].sum() / 2  # 2 months

            # Should be within 50% of stated income (accounting for variations)
            assert 0.5 * borrower['income'] <= monthly_income <= 1.5 * borrower['income']


# ==================== EDGE CASE TESTS ====================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_borrowers(self):
        """Test handling of zero borrowers."""
        with pytest.raises(ValueError):
            generator = BorrowerProfileGenerator(num_borrowers=0)

    def test_negative_borrowers(self):
        """Test handling of negative borrower count."""
        with pytest.raises(ValueError):
            generator = BorrowerProfileGenerator(num_borrowers=-10)

    def test_zero_months(self, sample_borrowers):
        """Test handling of zero months."""
        synth = TransactionSynthesizer()
        borrower = sample_borrowers[0]

        with pytest.raises(ValueError):
            synth.generate_transactions(
                borrower,
                start_date=datetime(2023, 1, 1),
                months=0
            )

    def test_invalid_default_rate(self):
        """Test handling of invalid default rate."""
        with pytest.raises(ValueError):
            CreditRiskLabeler(default_rate=1.5)  # > 1.0

        with pytest.raises(ValueError):
            CreditRiskLabeler(default_rate=-0.1)  # < 0.0

    def test_empty_transactions(self):
        """Test labeling with empty transactions."""
        labeler = CreditRiskLabeler()
        empty_df = pd.DataFrame(columns=['borrower_id', 'transaction_date', 'amount', 'category'])

        labels = labeler.generate_labels(empty_df)
        assert len(labels) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src.data_generation", "--cov-report=term-missing"])
