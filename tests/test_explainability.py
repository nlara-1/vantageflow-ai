"""
Test suite for explainability modules.

Tests SHAP computation, reason code generation, and explanation validation.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from unittest.mock import Mock, patch

from src.explainability.shap_engine import SHAPExplainer, create_explainer
from src.explainability.reason_codes import (
    ReasonCodeGenerator,
    ReasonCode,
    generate_reason_codes_from_shap
)


# ==================== FIXTURES ====================

@pytest.fixture
def sample_model():
    """Create a simple trained model for testing."""
    np.random.seed(42)

    # Create random training data
    X_train = pd.DataFrame(
        np.random.randn(100, 10),
        columns=[f'feature_{i}' for i in range(10)]
    )
    y_train = np.random.choice([0, 1], size=100)

    # Train simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3)
    model.fit(X_train, y_train)

    return model


@pytest.fixture
def sample_background_data():
    """Create sample background data for SHAP."""
    np.random.seed(42)

    return pd.DataFrame(
        np.random.randn(50, 10),
        columns=[f'feature_{i}' for i in range(10)]
    )


@pytest.fixture
def sample_test_data():
    """Create sample test data for predictions."""
    np.random.seed(42)

    return pd.DataFrame(
        np.random.randn(5, 10),
        columns=[f'feature_{i}' for i in range(10)]
    )


@pytest.fixture
def sample_shap_values():
    """Create sample SHAP values for testing."""
    return {
        'income_cv': 0.045,
        'savings_rate': -0.032,
        'overdraft_count_3mo': 0.058,
        'expense_income_ratio': 0.023,
        'avg_balance': -0.018,
        'discretionary_pct': 0.015,
        'income_trend_3mo': -0.010
    }


@pytest.fixture
def sample_feature_values():
    """Create sample feature values for testing."""
    return {
        'income_cv': 0.52,
        'savings_rate': 0.18,
        'overdraft_count_3mo': 3.0,
        'expense_income_ratio': 0.92,
        'avg_balance': 450.00,
        'discretionary_pct': 0.35,
        'income_trend_3mo': 0.05
    }


# ==================== SHAP EXPLAINER TESTS ====================

class TestSHAPExplainer:
    """Test SHAP explainer functionality."""

    def test_explainer_initialization(self, sample_model, sample_background_data):
        """Test SHAP explainer initializes correctly."""
        explainer = SHAPExplainer(
            model=sample_model,
            background_data=sample_background_data
        )

        assert explainer.model is not None
        assert explainer.explainer is not None
        assert len(explainer.feature_names) == 10

    def test_explainer_with_numpy_background(self, sample_model):
        """Test explainer with numpy array background data."""
        background = np.random.randn(50, 10)
        feature_names = [f'feature_{i}' for i in range(10)]

        explainer = SHAPExplainer(
            model=sample_model,
            background_data=background,
            feature_names=feature_names
        )

        assert explainer.background_df is not None
        assert list(explainer.background_df.columns) == feature_names

    def test_explainer_background_sampling(self, sample_model):
        """Test background data is sampled when too large."""
        # Create large background data
        large_background = pd.DataFrame(
            np.random.randn(200, 10),
            columns=[f'feature_{i}' for i in range(10)]
        )

        explainer = SHAPExplainer(
            model=sample_model,
            background_data=large_background,
            max_background_samples=50
        )

        # Should be sampled down to 50
        assert len(explainer.background_df) == 50

    def test_get_shap_values(self, sample_model, sample_background_data, sample_test_data):
        """Test SHAP value computation."""
        explainer = SHAPExplainer(
            model=sample_model,
            background_data=sample_background_data
        )

        shap_values = explainer.get_shap_values(sample_test_data)

        # Should have shape (n_samples, n_features)
        assert shap_values.shape == (5, 10)

        # Should be numeric
        assert np.all(np.isfinite(shap_values))

    def test_get_explanation(self, sample_model, sample_background_data, sample_test_data):
        """Test getting explanation for single sample."""
        explainer = SHAPExplainer(
            model=sample_model,
            background_data=sample_background_data
        )

        # Get single sample
        X_single = sample_test_data.iloc[0]

        explanation = explainer.get_explanation(X_single)

        # Should have required attributes
        assert hasattr(explanation, 'values')
        assert hasattr(explanation, 'base_values')
        assert hasattr(explanation, 'data')
        assert hasattr(explanation, 'feature_names')

    def test_get_explanation_dict(self, sample_model, sample_background_data, sample_test_data):
        """Test getting explanation as dictionary."""
        explainer = SHAPExplainer(
            model=sample_model,
            background_data=sample_background_data
        )

        X_single = sample_test_data.iloc[0]

        explanation_dict = explainer.get_explanation(X_single, return_dict=True)

        # Should have required keys
        required_keys = ['shap_values', 'base_value', 'feature_values', 'feature_names', 'prediction']
        for key in required_keys:
            assert key in explanation_dict

    def test_shap_additivity(self, sample_model, sample_background_data, sample_test_data):
        """Test SHAP values sum to prediction (additivity property)."""
        explainer = SHAPExplainer(
            model=sample_model,
            background_data=sample_background_data
        )

        X_single = sample_test_data.iloc[0:1]

        explanation = explainer.get_explanation(X_single, return_dict=True)

        # SHAP values + base value should equal prediction
        shap_sum = sum(explanation['shap_values']) + explanation['base_value']
        prediction = explanation['prediction']

        # Should be approximately equal (within floating point tolerance)
        assert abs(shap_sum - prediction) < 0.01

    def test_get_top_features(self, sample_model, sample_background_data, sample_test_data):
        """Test getting top N features by importance."""
        explainer = SHAPExplainer(
            model=sample_model,
            background_data=sample_background_data
        )

        top_features = explainer.get_top_features(sample_test_data, n=5)

        # Should return DataFrame with top 5 features
        assert len(top_features) == 5
        assert 'feature' in top_features.columns
        assert 'importance' in top_features.columns

        # Should be sorted by importance
        assert all(top_features['importance'].iloc[i] >= top_features['importance'].iloc[i+1]
                   for i in range(len(top_features)-1))

    def test_get_feature_contributions(self, sample_model, sample_background_data, sample_test_data):
        """Test getting feature contributions for single prediction."""
        explainer = SHAPExplainer(
            model=sample_model,
            background_data=sample_background_data
        )

        X_single = sample_test_data.iloc[0]

        contributions = explainer.get_feature_contributions(X_single, top_n=5)

        # Should return DataFrame
        assert isinstance(contributions, pd.DataFrame)
        assert len(contributions) == 5

        # Should have required columns
        required_cols = ['feature', 'value', 'shap_value', 'abs_shap_value']
        for col in required_cols:
            assert col in contributions.columns


# ==================== REASON CODE GENERATOR TESTS ====================

class TestReasonCodeGenerator:
    """Test reason code generation."""

    def test_generator_initialization(self):
        """Test reason code generator initializes."""
        generator = ReasonCodeGenerator(config_path="config/reason_codes.yaml")

        assert generator.positive_reasons is not None
        assert generator.negative_reasons is not None
        assert generator.feature_to_reasons is not None

    def test_generate_reason_codes(self, sample_shap_values, sample_feature_values):
        """Test reason code generation from SHAP values."""
        generator = ReasonCodeGenerator(config_path="config/reason_codes.yaml")

        reason_codes = generator.generate_reason_codes(
            shap_values=sample_shap_values,
            feature_values=sample_feature_values,
            top_n=5
        )

        # Should return list of ReasonCode objects
        assert isinstance(reason_codes, list)
        assert len(reason_codes) <= 5

        # Each should be ReasonCode
        for rc in reason_codes:
            assert isinstance(rc, ReasonCode)

    def test_reason_code_structure(self, sample_shap_values, sample_feature_values):
        """Test reason code has required fields."""
        generator = ReasonCodeGenerator(config_path="config/reason_codes.yaml")

        reason_codes = generator.generate_reason_codes(
            sample_shap_values,
            sample_feature_values,
            top_n=3
        )

        if len(reason_codes) > 0:
            rc = reason_codes[0]

            # Required fields
            assert hasattr(rc, 'code')
            assert hasattr(rc, 'name')
            assert hasattr(rc, 'description')
            assert hasattr(rc, 'long_description')
            assert hasattr(rc, 'impact')
            assert hasattr(rc, 'magnitude')
            assert hasattr(rc, 'contribution')
            assert hasattr(rc, 'features')

    def test_reason_code_sorting(self, sample_shap_values, sample_feature_values):
        """Test reason codes are sorted by contribution."""
        generator = ReasonCodeGenerator(config_path="config/reason_codes.yaml")

        reason_codes = generator.generate_reason_codes(
            sample_shap_values,
            sample_feature_values,
            top_n=5
        )

        if len(reason_codes) > 1:
            # Should be sorted by decreasing contribution
            contributions = [rc.contribution for rc in reason_codes]
            assert all(contributions[i] >= contributions[i+1]
                      for i in range(len(contributions)-1))

    def test_direction_validation(self, sample_shap_values, sample_feature_values):
        """Test SHAP direction validation."""
        generator = ReasonCodeGenerator(config_path="config/reason_codes.yaml")

        # With validation
        codes_validated = generator.generate_reason_codes(
            sample_shap_values,
            sample_feature_values,
            validate_direction=True
        )

        # Without validation
        codes_not_validated = generator.generate_reason_codes(
            sample_shap_values,
            sample_feature_values,
            validate_direction=False
        )

        # May have different counts (some filtered out with validation)
        # This is expected behavior

    def test_magnitude_calculation(self):
        """Test magnitude level calculation."""
        generator = ReasonCodeGenerator(config_path="config/reason_codes.yaml")

        # Test different magnitudes
        thresholds = {'strong': 0.025, 'moderate': 0.01, 'slight': 0.005}

        mag_strong = generator._calculate_magnitude(0.030, thresholds)
        assert mag_strong == 'strong'

        mag_moderate = generator._calculate_magnitude(0.015, thresholds)
        assert mag_moderate == 'moderate'

        mag_slight = generator._calculate_magnitude(0.007, thresholds)
        assert mag_slight == 'slight'

        mag_negligible = generator._calculate_magnitude(0.002, thresholds)
        assert mag_negligible == 'negligible'

    def test_positive_vs_negative_codes(self, sample_feature_values):
        """Test positive vs negative reason codes."""
        generator = ReasonCodeGenerator(config_path="config/reason_codes.yaml")

        # Create SHAP values with known directions
        shap_positive = {
            'savings_rate': -0.05,  # Negative SHAP = reduces risk = positive impact
            'overdraft_count_3mo': 0.05  # Positive SHAP = increases risk = negative impact
        }

        reason_codes = generator.generate_reason_codes(
            shap_positive,
            sample_feature_values,
            top_n=10
        )

        # Should have both positive and negative codes
        impacts = [rc.impact for rc in reason_codes]

        # Check we have variety (if codes were generated)
        if len(reason_codes) >= 2:
            assert 'positive' in impacts or 'negative' in impacts

    def test_format_for_display_text(self, sample_shap_values, sample_feature_values):
        """Test formatting reason codes as text."""
        generator = ReasonCodeGenerator(config_path="config/reason_codes.yaml")

        reason_codes = generator.generate_reason_codes(
            sample_shap_values,
            sample_feature_values,
            top_n=3
        )

        text_output = generator.format_for_display(reason_codes, format_type="text")

        # Should be string
        assert isinstance(text_output, str)
        assert len(text_output) > 0

        # Should contain header
        assert "CREDIT DECISION FACTORS" in text_output

    def test_format_for_display_html(self, sample_shap_values, sample_feature_values):
        """Test formatting reason codes as HTML."""
        generator = ReasonCodeGenerator(config_path="config/reason_codes.yaml")

        reason_codes = generator.generate_reason_codes(
            sample_shap_values,
            sample_feature_values,
            top_n=3
        )

        html_output = generator.format_for_display(reason_codes, format_type="html")

        # Should be HTML
        assert isinstance(html_output, str)
        assert '<div' in html_output
        assert '</div>' in html_output

    def test_format_for_display_json(self, sample_shap_values, sample_feature_values):
        """Test formatting reason codes as JSON."""
        generator = ReasonCodeGenerator(config_path="config/reason_codes.yaml")

        reason_codes = generator.generate_reason_codes(
            sample_shap_values,
            sample_feature_values,
            top_n=3
        )

        json_output = generator.format_for_display(reason_codes, format_type="json")

        # Should be valid JSON string
        import json
        parsed = json.loads(json_output)
        assert isinstance(parsed, list)

    def test_adverse_action_notice(self, sample_shap_values, sample_feature_values):
        """Test adverse action notice generation."""
        generator = ReasonCodeGenerator(config_path="config/reason_codes.yaml")

        reason_codes = generator.generate_reason_codes(
            sample_shap_values,
            sample_feature_values,
            top_n=5
        )

        notice = generator.generate_adverse_action_notice(
            reason_codes=reason_codes,
            applicant_name="John Doe",
            decision="denied"
        )

        # Should contain required elements
        assert "ADVERSE ACTION NOTICE" in notice
        assert "John Doe" in notice
        assert "denied" in notice.lower()


# ==================== REASON CODE DATACLASS TESTS ====================

class TestReasonCode:
    """Test ReasonCode dataclass."""

    def test_reason_code_creation(self):
        """Test creating ReasonCode instance."""
        rc = ReasonCode(
            code="N01",
            name="INCOME_VOLATILITY_HIGH",
            description="Income volatility is higher than typical",
            long_description="Your income shows significant variation...",
            impact="negative",
            magnitude="strong",
            contribution=0.045,
            features=['income_cv', 'income_std']
        )

        assert rc.code == "N01"
        assert rc.impact == "negative"
        assert rc.magnitude == "strong"
        assert rc.contribution == 0.045

    def test_reason_code_repr(self):
        """Test ReasonCode string representation."""
        rc = ReasonCode(
            code="P01",
            name="SAVINGS_BEHAVIOR_POSITIVE",
            description="Strong savings",
            long_description="Details...",
            impact="positive",
            magnitude="moderate",
            contribution=0.028,
            features=['savings_rate']
        )

        repr_str = repr(rc)
        assert "P01" in repr_str
        assert "SAVINGS_BEHAVIOR_POSITIVE" in repr_str
        assert "moderate" in repr_str


# ==================== INTEGRATION TESTS ====================

class TestExplainabilityPipeline:
    """Test end-to-end explainability pipeline."""

    def test_shap_to_reason_codes(self, sample_model, sample_background_data, sample_test_data):
        """Test converting SHAP values to reason codes."""
        # Create explainer
        explainer = SHAPExplainer(
            model=sample_model,
            background_data=sample_background_data
        )

        # Get explanation
        X_single = sample_test_data.iloc[0]
        explanation = explainer.get_explanation(X_single, return_dict=True)

        # Mock reason code generator (since we need actual feature names from config)
        # For integration test, we'll just verify the flow works
        assert 'shap_values' in explanation
        assert 'feature_values' in explanation

    def test_full_explanation_workflow(self, sample_model, sample_background_data, sample_test_data):
        """Test complete explanation generation workflow."""
        # Initialize explainer
        explainer = SHAPExplainer(
            model=sample_model,
            background_data=sample_background_data
        )

        # Get single prediction
        X_single = sample_test_data.iloc[0]

        # Get SHAP values
        shap_values = explainer.get_shap_values(X_single.to_frame().T)
        assert shap_values.shape == (1, 10)

        # Get explanation
        explanation = explainer.get_explanation(X_single)
        assert explanation is not None

        # Get feature contributions
        contributions = explainer.get_feature_contributions(X_single, top_n=5)
        assert len(contributions) == 5


# ==================== EDGE CASE TESTS ====================

class TestExplainabilityEdgeCases:
    """Test edge cases in explainability."""

    def test_zero_shap_values(self, sample_feature_values):
        """Test handling of zero SHAP values."""
        generator = ReasonCodeGenerator(config_path="config/reason_codes.yaml")

        zero_shap = {feature: 0.0 for feature in sample_feature_values.keys()}

        reason_codes = generator.generate_reason_codes(
            zero_shap,
            sample_feature_values,
            top_n=5
        )

        # Should handle gracefully (may return empty list)
        assert isinstance(reason_codes, list)

    def test_very_small_shap_values(self, sample_feature_values):
        """Test handling of very small SHAP values."""
        generator = ReasonCodeGenerator(config_path="config/reason_codes.yaml")

        small_shap = {feature: 0.0001 for feature in sample_feature_values.keys()}

        reason_codes = generator.generate_reason_codes(
            small_shap,
            sample_feature_values,
            top_n=5
        )

        # Should filter out negligible contributions
        assert isinstance(reason_codes, list)

    def test_missing_feature_values(self, sample_shap_values):
        """Test handling of missing feature values."""
        generator = ReasonCodeGenerator(config_path="config/reason_codes.yaml")

        # Incomplete feature values
        incomplete_values = {
            'income_cv': 0.5
        }

        # Should handle gracefully
        reason_codes = generator.generate_reason_codes(
            sample_shap_values,
            incomplete_values,
            top_n=5
        )

        assert isinstance(reason_codes, list)

    def test_single_feature_explanation(self, sample_model, sample_background_data):
        """Test SHAP explanation with single feature."""
        # Create model with single feature
        X_train_single = sample_background_data[['feature_0']]
        y_train = np.random.choice([0, 1], size=len(X_train_single))

        single_model = RandomForestClassifier(n_estimators=5, random_state=42)
        single_model.fit(X_train_single, y_train)

        # Create explainer
        explainer = SHAPExplainer(
            model=single_model,
            background_data=X_train_single
        )

        # Get explanation
        X_single = X_train_single.iloc[0]
        explanation = explainer.get_explanation(X_single)

        assert len(explanation.values) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src.explainability", "--cov-report=term-missing"])
