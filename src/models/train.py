"""
Model training pipeline for VantageFlow AI credit scoring.

Trains baseline (Logistic Regression) and advanced (XGBoost) models
with proper data splitting, hyperparameter tuning, and evaluation.
"""

import argparse
import logging
import os
import json
from datetime import datetime
from typing import Dict, Tuple, Optional, Any, List

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import xgboost as xgb


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def split_data(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train, validation, and test sets.

    Ensures split is done by borrower_id (not individual transactions)
    and stratified by default label to maintain class distribution.

    Args:
        features_df: Features DataFrame with borrower_id
        labels_df: Labels DataFrame with borrower_id and default_label
        train_size: Proportion for training set (default 0.70)
        val_size: Proportion for validation set (default 0.15)
        test_size: Proportion for test set (default 0.15)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)

    Raises:
        ValueError: If sizes don't sum to 1.0 or DataFrames have mismatched borrowers
    """
    # Validate sizes
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError(
            f"Split sizes must sum to 1.0, got {train_size + val_size + test_size}"
        )

    # Merge features and labels on borrower_id
    merged = features_df.merge(
        labels_df[['borrower_id', 'default_label']],
        on='borrower_id',
        how='inner'
    )

    if len(merged) != len(features_df):
        logger.warning(
            f"Feature-label mismatch: {len(features_df)} features, "
            f"{len(labels_df)} labels, {len(merged)} matched"
        )

    # Separate features and labels
    X = merged.drop(columns=['borrower_id', 'default_label'])
    y = merged['default_label']
    borrower_ids = merged['borrower_id']

    logger.info(f"Total samples: {len(X)}")
    logger.info(f"Class distribution: {y.value_counts().to_dict()}")
    logger.info(f"Default rate: {y.mean():.2%}")

    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp, ids_train, ids_temp = train_test_split(
        X, y, borrower_ids,
        test_size=(val_size + test_size),
        random_state=random_state,
        stratify=y
    )

    # Second split: val vs test
    val_proportion = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test, ids_val, ids_test = train_test_split(
        X_temp, y_temp, ids_temp,
        test_size=(1 - val_proportion),
        random_state=random_state,
        stratify=y_temp
    )

    logger.info(f"Train set: {len(X_train)} samples ({y_train.mean():.2%} default)")
    logger.info(f"Val set: {len(X_val)} samples ({y_val.mean():.2%} default)")
    logger.info(f"Test set: {len(X_test)} samples ({y_test.mean():.2%} default)")

    return X_train, X_val, X_test, y_train, y_val, y_test


class BaselineModel:
    """
    Baseline Logistic Regression model with preprocessing.

    Uses StandardScaler for feature normalization, L2 regularization,
    and balanced class weights to handle imbalanced data.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42
    ):
        """
        Initialize baseline model.

        Args:
            C: Inverse of regularization strength (smaller = stronger L2)
            max_iter: Maximum iterations for solver
            random_state: Random seed
        """
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=C,
            penalty='l2',
            class_weight='balanced',  # Handles class imbalance
            max_iter=max_iter,
            random_state=random_state,
            solver='lbfgs'
        )

        self.feature_names = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaselineModel':
        """
        Fit the model.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Self
        """
        logger.info("Training Baseline Model (Logistic Regression)...")

        # Store feature names
        self.feature_names = X.columns.tolist()

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model.fit(X_scaled, y)
        self.is_fitted = True

        logger.info(f"✓ Baseline model trained on {len(X)} samples")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Features

        Returns:
            Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features

        Returns:
            Predicted probabilities (N x 2 array)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance (coefficients).

        Returns:
            DataFrame with feature names and coefficients
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        coefficients = self.model.coef_[0]
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)

        return importance_df


class XGBoostModel:
    """
    XGBoost model with hyperparameter tuning.

    Performs RandomizedSearchCV with 5-fold cross-validation,
    early stopping, and feature importance extraction.
    """

    def __init__(
        self,
        n_iter: int = 50,
        cv_folds: int = 5,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize XGBoost model.

        Args:
            n_iter: Number of iterations for RandomizedSearchCV
            cv_folds: Number of cross-validation folds
            random_state: Random seed
            n_jobs: Number of parallel jobs (-1 = all CPUs)
        """
        self.n_iter = n_iter
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model = None
        self.best_params = None
        self.feature_names = None
        self.is_fitted = False

        # Hyperparameter search space
        self.param_grid = {
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'n_estimators': [100, 200, 300, 500],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2, 0.3],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'reg_alpha': [0, 0.1, 1],  # L1 regularization
            'reg_lambda': [1, 2, 5],  # L2 regularization
        }

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> 'XGBoostModel':
        """
        Fit the model with hyperparameter tuning.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (for early stopping)
            y_val: Validation labels (for early stopping)

        Returns:
            Self
        """
        logger.info("Training XGBoost Model with hyperparameter tuning...")

        # Store feature names
        self.feature_names = X_train.columns.tolist()

        # Calculate scale_pos_weight for class imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        # Base model
        base_model = xgb.XGBClassifier(
            objective='binary:logistic',
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            eval_metric='auc'
        )

        # Stratified K-Fold cross-validation
        cv_splitter = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state
        )

        # RandomizedSearchCV
        logger.info(f"Running RandomizedSearchCV with {self.n_iter} iterations...")

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=self.param_grid,
            n_iter=self.n_iter,
            cv=cv_splitter,
            scoring='roc_auc',
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=1
        )

        # Prepare eval set for early stopping
        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['early_stopping_rounds'] = 50
            fit_params['verbose'] = False

        # Fit
        search.fit(X_train, y_train, **fit_params)

        # Store best model and params
        self.model = search.best_estimator_
        self.best_params = search.best_params_
        self.is_fitted = True

        logger.info(f"✓ Best cross-validation AUC: {search.best_score_:.4f}")
        logger.info(f"✓ Best parameters: {self.best_params}")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Features

        Returns:
            Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features

        Returns:
            Predicted probabilities (N x 2 array)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        return self.model.predict_proba(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from XGBoost.

        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        importance_scores = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)

        return importance_df


class ModelTrainer:
    """
    Orchestrator for training and comparing multiple models.

    Trains baseline and XGBoost models, compares performance,
    saves the best model, and generates training metadata.
    """

    def __init__(
        self,
        output_dir: str = "models/experiments",
        random_state: int = 42
    ):
        """
        Initialize model trainer.

        Args:
            output_dir: Directory to save models and metadata
            random_state: Random seed
        """
        self.output_dir = output_dir
        self.random_state = random_state

        self.baseline_model = None
        self.xgboost_model = None
        self.results = {}

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def train_all(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Train all models and compare performance.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with training results
        """
        logger.info("=" * 70)
        logger.info("TRAINING ALL MODELS")
        logger.info("=" * 70)

        # Train baseline model
        self.baseline_model = BaselineModel(random_state=self.random_state)
        self.baseline_model.fit(X_train, y_train)

        # Evaluate baseline
        baseline_results = self._evaluate_model(
            self.baseline_model,
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            "Baseline (Logistic Regression)"
        )

        # Train XGBoost model
        self.xgboost_model = XGBoostModel(random_state=self.random_state)
        self.xgboost_model.fit(X_train, y_train, X_val, y_val)

        # Evaluate XGBoost
        xgboost_results = self._evaluate_model(
            self.xgboost_model,
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            "XGBoost"
        )

        # Store results
        self.results = {
            "baseline": baseline_results,
            "xgboost": xgboost_results,
            "training_date": datetime.now().isoformat(),
            "random_state": self.random_state
        }

        # Compare models
        self._print_comparison()

        return self.results

    def _evaluate_model(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Evaluate model on train, val, and test sets.

        Args:
            model: Trained model
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
            model_name: Name for logging

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"\nEvaluating {model_name}...")

        results = {}

        for split_name, X, y in [
            ("train", X_train, y_train),
            ("val", X_val, y_val),
            ("test", X_test, y_test)
        ]:
            # Predictions
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X)[:, 1]

            # Metrics
            metrics = {
                "auc": roc_auc_score(y, y_pred_proba),
                "accuracy": accuracy_score(y, y_pred),
                "precision": precision_score(y, y_pred, zero_division=0),
                "recall": recall_score(y, y_pred, zero_division=0),
                "f1": f1_score(y, y_pred, zero_division=0),
                "confusion_matrix": confusion_matrix(y, y_pred).tolist()
            }

            results[split_name] = metrics

            logger.info(f"  {split_name.upper()}: AUC={metrics['auc']:.4f}, "
                       f"Acc={metrics['accuracy']:.4f}, "
                       f"Prec={metrics['precision']:.4f}, "
                       f"Rec={metrics['recall']:.4f}, "
                       f"F1={metrics['f1']:.4f}")

        return results

    def _print_comparison(self) -> None:
        """Print comparison of model performance."""
        logger.info("\n" + "=" * 70)
        logger.info("MODEL COMPARISON (Test Set)")
        logger.info("=" * 70)

        baseline_test = self.results["baseline"]["test"]
        xgboost_test = self.results["xgboost"]["test"]

        logger.info(f"\n{'Metric':<15} {'Baseline':<15} {'XGBoost':<15} {'Winner':<15}")
        logger.info("-" * 70)

        metrics = ["auc", "accuracy", "precision", "recall", "f1"]
        for metric in metrics:
            baseline_val = baseline_test[metric]
            xgboost_val = xgboost_test[metric]
            winner = "XGBoost" if xgboost_val > baseline_val else "Baseline"

            logger.info(f"{metric.upper():<15} {baseline_val:<15.4f} "
                       f"{xgboost_val:<15.4f} {winner:<15}")

    def save_best_model(self, metric: str = "auc") -> str:
        """
        Save the best performing model.

        Args:
            metric: Metric to use for comparison (default: auc)

        Returns:
            Path to saved model
        """
        # Determine best model based on test set performance
        baseline_score = self.results["baseline"]["test"][metric]
        xgboost_score = self.results["xgboost"]["test"][metric]

        if xgboost_score > baseline_score:
            best_model = self.xgboost_model
            model_name = "xgboost"
            logger.info(f"\n✓ XGBoost is best model ({metric}={xgboost_score:.4f})")
        else:
            best_model = self.baseline_model
            model_name = "baseline"
            logger.info(f"\n✓ Baseline is best model ({metric}={baseline_score:.4f})")

        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.output_dir, f"{model_name}_model_{timestamp}.pkl")

        joblib.dump(best_model, model_path)
        logger.info(f"✓ Model saved: {model_path}")

        # Save metadata
        metadata = {
            "model_type": model_name,
            "training_date": self.results["training_date"],
            "best_metric": metric,
            "best_score": max(baseline_score, xgboost_score),
            "test_metrics": self.results[model_name]["test"],
            "random_state": self.random_state
        }

        if model_name == "xgboost":
            metadata["best_params"] = self.xgboost_model.best_params

        metadata_path = os.path.join(self.output_dir, f"{model_name}_metadata_{timestamp}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✓ Metadata saved: {metadata_path}")

        # Save feature importance
        importance_df = best_model.get_feature_importance()
        importance_path = os.path.join(
            self.output_dir,
            f"{model_name}_feature_importance_{timestamp}.csv"
        )
        importance_df.to_csv(importance_path, index=False)
        logger.info(f"✓ Feature importance saved: {importance_path}")

        return model_path


def main():
    """Main function for CLI interface."""
    parser = argparse.ArgumentParser(
        description="Train credit scoring models for VantageFlow AI"
    )
    parser.add_argument(
        "--features",
        type=str,
        default="data/output/features.csv",
        help="Path to features CSV file"
    )
    parser.add_argument(
        "--labels-db",
        type=str,
        default="data/credit_scoring.db",
        help="Path to database with labels"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/experiments",
        help="Directory to save models"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    try:
        # Load features
        logger.info(f"Loading features from {args.features}")
        features_df = pd.read_csv(args.features)
        logger.info(f"✓ Loaded {len(features_df)} borrower features")

        # Load labels
        logger.info(f"Loading labels from {args.labels_db}")
        from sqlalchemy import create_engine
        engine = create_engine(f"sqlite:///{args.labels_db}")
        labels_df = pd.read_sql_query("SELECT borrower_id, default_label FROM labels", engine)
        logger.info(f"✓ Loaded {len(labels_df)} labels")

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            features_df,
            labels_df,
            random_state=args.random_state
        )

        # Train models
        trainer = ModelTrainer(
            output_dir=args.output_dir,
            random_state=args.random_state
        )

        results = trainer.train_all(
            X_train, y_train,
            X_val, y_val,
            X_test, y_test
        )

        # Save best model
        model_path = trainer.save_best_model(metric="auc")

        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Best model saved to: {model_path}")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
