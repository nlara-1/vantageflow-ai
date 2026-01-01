"""
VantageFlow AI Test Suite

Comprehensive test coverage for:
- Data generation (profiles, transactions, labels)
- Feature engineering (extraction, validation)
- Model training (baseline, XGBoost, predictions)
- Explainability (SHAP, reason codes)

Run all tests:
    pytest tests/ -v

Run with coverage:
    pytest tests/ --cov=src --cov-report=html

Run specific test file:
    pytest tests/test_features.py -v

Run specific test class:
    pytest tests/test_features.py::TestFeatureEngineer -v

Run specific test:
    pytest tests/test_features.py::TestFeatureEngineer::test_feature_extraction -v
"""

__version__ = "1.0.0"
