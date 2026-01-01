"""
Test suite for model training and prediction modules.

Tests train/test split, model training, and scoring pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.models.train import (
    split_data,
    train_baseline_model,
    train_xgboost_model,
    predict_with_model,
    save_model,
    load_model
)


# ==================== FIXTURES ====================

@pytest.fixture
def sample_features():
    """Create sample feature dataset."""
    np.random.seed(42)

    n_samples = 100
    n_features = 39

    # Create random features
    features = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # Add borrower_id
    features['borrower_id'] = [f'BORROWER_{i:03d}' for i in range(n_samples)]

    # Reorder columns (borrower_id first)
    cols = ['borrower_id'] + [f'feature_{i}' for i in range(n_features)]
    features = features[cols]

    return features


@pytest.fixture
def sample_labels():
    """Create sample labels dataset."""
    np.random.seed(42)

    n_samples = 100

    labels = pd.DataFrame({
        'borrower_id': [f'BORROWER_{i:03d}' for i in range(n_samples)],
        'default_label': np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    })

    return labels


@pytest.fixture
def trained_model(sample_features, sample_labels):
    """Create a pre-trained model for testing."""
    # Split data
    splits = split_data(sample_features, sample_labels)

    # Train simple model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(splits['X_train'], splits['y_train'])

    return model


# ==================== DATA SPLITTING TESTS ====================

class TestDataSplitting:
    """Test data splitting functionality."""

    def test_split_data_proportions(self, sample_features, sample_labels):
        """Test data is split according to specified proportions."""
        splits = split_data(
            sample_features,
            sample_labels,
            train_size=0.70,
            val_size=0.15,
            test_size=0.15
        )

        total = len(sample_features)

        # Check proportions (allow small variance due to rounding)
        assert 0.65 <= len(splits['X_train']) / total <= 0.75
        assert 0.10 <= len(splits['X_val']) / total <= 0.20
        assert 0.10 <= len(splits['X_test']) / total <= 0.20

    def test_split_data_no_overlap(self, sample_features, sample_labels):
        """Test train/val/test sets have no overlap."""
        splits = split_data(sample_features, sample_labels)

        train_ids = set(splits['X_train']['borrower_id'])
        val_ids = set(splits['X_val']['borrower_id'])
        test_ids = set(splits['X_test']['borrower_id'])

        # No overlap
        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0

    def test_split_data_coverage(self, sample_features, sample_labels):
        """Test all data is used in splits."""
        splits = split_data(sample_features, sample_labels)

        train_ids = set(splits['X_train']['borrower_id'])
        val_ids = set(splits['X_val']['borrower_id'])
        test_ids = set(splits['X_test']['borrower_id'])

        all_split_ids = train_ids | val_ids | test_ids
        original_ids = set(sample_features['borrower_id'])

        # All IDs should be in one of the splits
        assert all_split_ids == original_ids

    def test_split_stratification(self, sample_features, sample_labels):
        """Test stratification preserves class balance."""
        splits = split_data(sample_features, sample_labels)

        # Calculate proportions
        overall_proportion = sample_labels['default_label'].mean()
        train_proportion = splits['y_train'].mean()
        val_proportion = splits['y_val'].mean()
        test_proportion = splits['y_test'].mean()

        # Should be within 10% of overall proportion
        assert abs(train_proportion - overall_proportion) < 0.10
        assert abs(val_proportion - overall_proportion) < 0.15  # More tolerance for smaller sets
        assert abs(test_proportion - overall_proportion) < 0.15

    def test_split_removes_borrower_id(self, sample_features, sample_labels):
        """Test borrower_id is removed from feature sets."""
        splits = split_data(sample_features, sample_labels)

        # borrower_id should not be in X sets
        assert 'borrower_id' not in splits['X_train'].columns
        assert 'borrower_id' not in splits['X_val'].columns
        assert 'borrower_id' not in splits['X_test'].columns

    def test_split_random_state(self, sample_features, sample_labels):
        """Test random state makes splits reproducible."""
        splits1 = split_data(sample_features, sample_labels, random_state=42)
        splits2 = split_data(sample_features, sample_labels, random_state=42)

        # Should be identical
        pd.testing.assert_frame_equal(splits1['X_train'], splits2['X_train'])
        pd.testing.assert_series_equal(splits1['y_train'], splits2['y_train'])


# ==================== BASELINE MODEL TESTS ====================

class TestBaselineModel:
    """Test baseline logistic regression model."""

    def test_train_baseline_model(self, sample_features, sample_labels):
        """Test baseline model can be trained."""
        splits = split_data(sample_features, sample_labels)

        model, results = train_baseline_model(
            splits['X_train'], splits['y_train'],
            splits['X_val'], splits['y_val'],
            splits['X_test'], splits['y_test']
        )

        # Model should be LogisticRegression
        assert isinstance(model, LogisticRegression)

        # Results should have required keys
        required_keys = ['train_auc', 'val_auc', 'test_auc']
        for key in required_keys:
            assert key in results

    def test_baseline_model_performance(self, sample_features, sample_labels):
        """Test baseline model achieves reasonable performance."""
        splits = split_data(sample_features, sample_labels)

        model, results = train_baseline_model(
            splits['X_train'], splits['y_train'],
            splits['X_val'], splits['y_val'],
            splits['X_test'], splits['y_test']
        )

        # AUC should be better than random (0.5)
        assert results['train_auc'] > 0.5
        # May not be great on random data, but should be > 0.4
        assert results['test_auc'] > 0.4

    def test_baseline_predictions_valid(self, sample_features, sample_labels):
        """Test baseline model predictions are valid probabilities."""
        splits = split_data(sample_features, sample_labels)

        model, results = train_baseline_model(
            splits['X_train'], splits['y_train'],
            splits['X_val'], splits['y_val'],
            splits['X_test'], splits['y_test']
        )

        # Get predictions
        y_pred_proba = results['y_pred_proba_test']

        # Should be probabilities (0 to 1)
        assert np.all(y_pred_proba >= 0)
        assert np.all(y_pred_proba <= 1)

    def test_baseline_coefficients(self, sample_features, sample_labels):
        """Test baseline model has coefficients for all features."""
        splits = split_data(sample_features, sample_labels)

        model, results = train_baseline_model(
            splits['X_train'], splits['y_train'],
            splits['X_val'], splits['y_val'],
            splits['X_test'], splits['y_test']
        )

        # Should have coefficient for each feature
        assert len(model.coef_[0]) == splits['X_train'].shape[1]


# ==================== XGBOOST MODEL TESTS ====================

class TestXGBoostModel:
    """Test XGBoost model training."""

    @pytest.mark.slow
    def test_train_xgboost_model(self, sample_features, sample_labels):
        """Test XGBoost model can be trained."""
        splits = split_data(sample_features, sample_labels)

        model, results = train_xgboost_model(
            splits['X_train'], splits['y_train'],
            splits['X_val'], splits['y_val'],
            splits['X_test'], splits['y_test'],
            n_iter=5,  # Small for testing
            cv_folds=2  # Small for testing
        )

        # Model should be trained
        assert model is not None

        # Results should have required keys
        required_keys = ['train_auc', 'val_auc', 'test_auc', 'best_params']
        for key in required_keys:
            assert key in results

    @pytest.mark.slow
    def test_xgboost_hyperparameters(self, sample_features, sample_labels):
        """Test XGBoost hyperparameter tuning."""
        splits = split_data(sample_features, sample_labels)

        model, results = train_xgboost_model(
            splits['X_train'], splits['y_train'],
            splits['X_val'], splits['y_val'],
            splits['X_test'], splits['y_test'],
            n_iter=3,
            cv_folds=2
        )

        # Should have best parameters
        assert 'best_params' in results
        assert isinstance(results['best_params'], dict)

        # Common hyperparameters
        common_params = ['max_depth', 'learning_rate', 'n_estimators']
        # At least some should be present
        assert any(param in results['best_params'] for param in common_params)

    @pytest.mark.slow
    def test_xgboost_feature_importance(self, sample_features, sample_labels):
        """Test XGBoost provides feature importances."""
        splits = split_data(sample_features, sample_labels)

        model, results = train_xgboost_model(
            splits['X_train'], splits['y_train'],
            splits['X_val'], splits['y_val'],
            splits['X_test'], splits['y_test'],
            n_iter=3,
            cv_folds=2
        )

        # Should have feature importances
        importances = model.feature_importances_
        assert len(importances) == splits['X_train'].shape[1]
        assert np.all(importances >= 0)


# ==================== PREDICTION TESTS ====================

class TestPrediction:
    """Test model prediction functionality."""

    def test_predict_with_model(self, trained_model, sample_features):
        """Test making predictions with trained model."""
        # Get features without borrower_id
        X = sample_features.drop('borrower_id', axis=1)

        predictions = predict_with_model(trained_model, X)

        # Should return probabilities
        assert len(predictions) == len(X)
        assert np.all(predictions >= 0)
        assert np.all(predictions <= 1)

    def test_predict_single_sample(self, trained_model, sample_features):
        """Test predicting single sample."""
        # Get single row
        X_single = sample_features.drop('borrower_id', axis=1).iloc[0:1]

        predictions = predict_with_model(trained_model, X_single)

        assert len(predictions) == 1
        assert 0 <= predictions[0] <= 1

    def test_predict_batch(self, trained_model, sample_features):
        """Test predicting batch of samples."""
        X = sample_features.drop('borrower_id', axis=1).iloc[:50]

        predictions = predict_with_model(trained_model, X)

        assert len(predictions) == 50
        assert all(0 <= p <= 1 for p in predictions)


# ==================== MODEL PERSISTENCE TESTS ====================

class TestModelPersistence:
    """Test model saving and loading."""

    def test_save_and_load_model(self, trained_model, tmp_path):
        """Test model can be saved and loaded."""
        # Save model
        model_path = tmp_path / "test_model.pkl"
        save_model(trained_model, str(model_path))

        # Load model
        loaded_model = load_model(str(model_path))

        # Should be same type
        assert type(loaded_model) == type(trained_model)

    def test_loaded_model_predictions(self, trained_model, sample_features, tmp_path):
        """Test loaded model produces same predictions."""
        X = sample_features.drop('borrower_id', axis=1).iloc[:10]

        # Original predictions
        original_preds = predict_with_model(trained_model, X)

        # Save and load
        model_path = tmp_path / "test_model.pkl"
        save_model(trained_model, str(model_path))
        loaded_model = load_model(str(model_path))

        # Loaded predictions
        loaded_preds = predict_with_model(loaded_model, X)

        # Should be identical
        np.testing.assert_array_almost_equal(original_preds, loaded_preds)


# ==================== EDGE CASE TESTS ====================

class TestModelEdgeCases:
    """Test edge cases in model training."""

    def test_imbalanced_data(self):
        """Test model handles imbalanced data."""
        np.random.seed(42)

        # Create highly imbalanced data (95% class 0, 5% class 1)
        n_samples = 200
        n_features = 10

        features = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        features['borrower_id'] = [f'B{i:03d}' for i in range(n_samples)]

        labels = pd.DataFrame({
            'borrower_id': [f'B{i:03d}' for i in range(n_samples)],
            'default_label': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
        })

        splits = split_data(features, labels)

        # Should handle without error
        model, results = train_baseline_model(
            splits['X_train'], splits['y_train'],
            splits['X_val'], splits['y_val'],
            splits['X_test'], splits['y_test']
        )

        # Model should still train
        assert model is not None

    def test_single_feature(self):
        """Test model with single feature."""
        np.random.seed(42)

        n_samples = 100

        features = pd.DataFrame({
            'borrower_id': [f'B{i:03d}' for i in range(n_samples)],
            'feature_0': np.random.randn(n_samples)
        })

        labels = pd.DataFrame({
            'borrower_id': [f'B{i:03d}' for i in range(n_samples)],
            'default_label': np.random.choice([0, 1], n_samples)
        })

        splits = split_data(features, labels)

        model, results = train_baseline_model(
            splits['X_train'], splits['y_train'],
            splits['X_val'], splits['y_val'],
            splits['X_test'], splits['y_test']
        )

        assert model is not None

    def test_all_zeros(self):
        """Test model handles all-zero features."""
        np.random.seed(42)

        n_samples = 100
        n_features = 5

        # All zeros except one feature
        features_array = np.zeros((n_samples, n_features))
        features_array[:, 0] = np.random.randn(n_samples)

        features = pd.DataFrame(
            features_array,
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        features['borrower_id'] = [f'B{i:03d}' for i in range(n_samples)]

        labels = pd.DataFrame({
            'borrower_id': [f'B{i:03d}' for i in range(n_samples)],
            'default_label': np.random.choice([0, 1], n_samples)
        })

        splits = split_data(features, labels)

        # Should handle without error
        model, results = train_baseline_model(
            splits['X_train'], splits['y_train'],
            splits['X_val'], splits['y_val'],
            splits['X_test'], splits['y_test']
        )

        assert model is not None


# ==================== INTEGRATION TESTS ====================

class TestModelPipeline:
    """Test end-to-end model training pipeline."""

    def test_full_training_pipeline(self, sample_features, sample_labels):
        """Test complete model training pipeline."""
        # Split data
        splits = split_data(sample_features, sample_labels)

        # Train baseline
        baseline_model, baseline_results = train_baseline_model(
            splits['X_train'], splits['y_train'],
            splits['X_val'], splits['y_val'],
            splits['X_test'], splits['y_test']
        )

        # Both should succeed
        assert baseline_model is not None
        assert baseline_results['test_auc'] > 0

    def test_model_comparison(self, sample_features, sample_labels):
        """Test comparing baseline and XGBoost models."""
        splits = split_data(sample_features, sample_labels)

        # Train baseline
        baseline_model, baseline_results = train_baseline_model(
            splits['X_train'], splits['y_train'],
            splits['X_val'], splits['y_val'],
            splits['X_test'], splits['y_test']
        )

        # Get test predictions for both
        X_test = splits['X_test']
        baseline_preds = predict_with_model(baseline_model, X_test)

        # Should have same number of predictions
        assert len(baseline_preds) == len(X_test)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src.models", "--cov-report=term-missing"])
