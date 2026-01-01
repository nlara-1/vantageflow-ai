"""
SHAP-based explainability engine for VantageFlow AI credit scoring.

Provides model-agnostic explanations using SHAP (SHapley Additive exPlanations)
with caching, batch processing, and visualization capabilities.
"""

import os
import warnings
from typing import Optional, Union, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


class SHAPExplainer:
    """
    SHAP-based model explainer with caching and visualization.

    Uses TreeExplainer for fast computation with tree-based models (XGBoost).
    Provides individual and batch explanations with waterfall and force plots.
    """

    def __init__(
        self,
        model: Any,
        background_data: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
        max_background_samples: int = 100,
        model_type: str = "tree"
    ):
        """
        Initialize SHAP explainer.

        Args:
            model: Trained model (must have predict method)
            background_data: Background dataset for SHAP computation (sample of training data)
            feature_names: List of feature names (required if background_data is numpy array)
            max_background_samples: Maximum samples to use for background (default 100)
            model_type: Type of explainer ('tree', 'kernel', or 'auto')

        Raises:
            ValueError: If model is invalid or background data is insufficient
        """
        self.model = model
        self.model_type = model_type

        # Convert background data to DataFrame if needed
        if isinstance(background_data, np.ndarray):
            if feature_names is None:
                raise ValueError("feature_names required when background_data is numpy array")
            self.background_df = pd.DataFrame(background_data, columns=feature_names)
        elif isinstance(background_data, pd.DataFrame):
            self.background_df = background_data.copy()
        else:
            raise ValueError("background_data must be pandas DataFrame or numpy array")

        # Store feature names
        self.feature_names = list(self.background_df.columns)

        # Sample background data if too large (for performance)
        if len(self.background_df) > max_background_samples:
            self.background_df = self.background_df.sample(
                n=max_background_samples,
                random_state=42
            )
            print(f"Sampled {max_background_samples} rows from background data for efficiency")

        # Validate model
        if not hasattr(model, 'predict'):
            raise ValueError("Model must have a 'predict' method")

        # Initialize SHAP explainer (cached)
        self.explainer = None
        self._initialize_explainer()

        # Cache for SHAP values
        self._shap_cache = {}

        print(f"✓ SHAP Explainer initialized with {len(self.background_df)} background samples")
        print(f"✓ Features: {len(self.feature_names)}")

    def _initialize_explainer(self) -> None:
        """
        Initialize the appropriate SHAP explainer based on model type.

        Uses TreeExplainer for tree-based models (fast),
        falls back to KernelExplainer for other models (slower).
        """
        try:
            # Try TreeExplainer first (fastest for tree models)
            if self.model_type == "tree" or self.model_type == "auto":
                try:
                    self.explainer = shap.TreeExplainer(
                        self.model,
                        data=self.background_df,
                        feature_perturbation="tree_path_dependent"
                    )
                    print("✓ Using TreeExplainer (optimized for tree-based models)")
                    return
                except Exception as e:
                    if self.model_type == "tree":
                        raise ValueError(f"TreeExplainer failed: {str(e)}")
                    # Fall through to KernelExplainer if auto mode

            # Fall back to KernelExplainer
            if self.model_type == "kernel" or self.model_type == "auto":
                # Wrapper function for prediction
                def predict_fn(X):
                    if isinstance(X, pd.DataFrame):
                        return self.model.predict_proba(X)[:, 1]
                    else:
                        X_df = pd.DataFrame(X, columns=self.feature_names)
                        return self.model.predict_proba(X_df)[:, 1]

                self.explainer = shap.KernelExplainer(
                    predict_fn,
                    self.background_df
                )
                print("✓ Using KernelExplainer (model-agnostic, may be slower)")
                return

        except Exception as e:
            raise RuntimeError(f"Failed to initialize SHAP explainer: {str(e)}")

    def get_shap_values(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        check_additivity: bool = False
    ) -> np.ndarray:
        """
        Compute SHAP values for batch of samples.

        Args:
            X: Input features (DataFrame or numpy array)
            check_additivity: Whether to check SHAP additivity property

        Returns:
            SHAP values array (n_samples x n_features)
        """
        # Convert to DataFrame if needed
        X_df = self._to_dataframe(X)

        # Validate features match
        if list(X_df.columns) != self.feature_names:
            raise ValueError(
                f"Feature mismatch. Expected {self.feature_names}, "
                f"got {list(X_df.columns)}"
            )

        try:
            # Compute SHAP values
            shap_values = self.explainer.shap_values(X_df, check_additivity=check_additivity)

            # Handle different return formats
            if isinstance(shap_values, list):
                # Binary classification may return list
                shap_values = shap_values[1]  # Positive class

            return shap_values

        except Exception as e:
            raise RuntimeError(f"SHAP value computation failed: {str(e)}")

    def get_explanation(
        self,
        X_single: Union[pd.Series, pd.DataFrame, np.ndarray],
        return_dict: bool = False
    ) -> Union[shap.Explanation, Dict[str, Any]]:
        """
        Get explanation for a single prediction.

        Args:
            X_single: Single sample (Series, 1-row DataFrame, or 1D array)
            return_dict: If True, return as dictionary instead of SHAP Explanation

        Returns:
            SHAP Explanation object or dictionary with explanation details
        """
        # Convert to DataFrame
        if isinstance(X_single, pd.Series):
            X_df = X_single.to_frame().T
        elif isinstance(X_single, pd.DataFrame):
            if len(X_single) != 1:
                raise ValueError("X_single must contain exactly 1 sample")
            X_df = X_single
        elif isinstance(X_single, np.ndarray):
            if X_single.ndim == 1:
                X_df = pd.DataFrame([X_single], columns=self.feature_names)
            else:
                if len(X_single) != 1:
                    raise ValueError("X_single must contain exactly 1 sample")
                X_df = pd.DataFrame(X_single, columns=self.feature_names)
        else:
            raise ValueError("X_single must be Series, DataFrame, or numpy array")

        # Get SHAP values
        shap_values = self.get_shap_values(X_df)

        # Get base value (expected value)
        base_value = self.explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1]  # Positive class

        # Create explanation object
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=base_value,
            data=X_df.iloc[0].values,
            feature_names=self.feature_names
        )

        if return_dict:
            # Return as dictionary
            return {
                "shap_values": shap_values[0],
                "base_value": base_value,
                "feature_values": X_df.iloc[0].to_dict(),
                "feature_names": self.feature_names,
                "prediction": self._get_prediction(X_df)
            }

        return explanation

    def generate_waterfall_plot(
        self,
        X_single: Union[pd.Series, pd.DataFrame, np.ndarray],
        max_display: int = 10,
        output_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Generate waterfall plot for single prediction.

        Shows how each feature contributes to pushing the prediction
        from the base value to the final prediction.

        Args:
            X_single: Single sample
            max_display: Maximum number of features to display
            output_path: Path to save plot (optional)
            show: Whether to display plot

        Returns:
            Matplotlib figure
        """
        # Get explanation
        explanation = self.get_explanation(X_single)

        # Create figure
        fig = plt.figure(figsize=(10, 6))

        # Generate waterfall plot
        shap.waterfall_plot(explanation, max_display=max_display, show=False)

        # Get current axis and improve styling
        ax = plt.gca()
        ax.set_title("SHAP Waterfall Plot - Feature Contributions",
                     fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()

        # Save if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Waterfall plot saved: {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def generate_force_plot(
        self,
        X_single: Union[pd.Series, pd.DataFrame, np.ndarray],
        output_path: Optional[str] = None,
        matplotlib: bool = True
    ) -> Any:
        """
        Generate force plot for single prediction.

        Shows features pushing prediction higher (red) or lower (blue).

        Args:
            X_single: Single sample
            output_path: Path to save plot (optional)
            matplotlib: If True, use matplotlib backend; if False, use JavaScript

        Returns:
            SHAP force plot object or matplotlib figure
        """
        # Get explanation
        explanation = self.get_explanation(X_single)

        # Get base value
        base_value = self.explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1]

        if matplotlib:
            # Use matplotlib for static plot
            fig = plt.figure(figsize=(14, 3))
            shap.force_plot(
                base_value,
                explanation.values,
                explanation.data,
                feature_names=self.feature_names,
                matplotlib=True,
                show=False
            )

            plt.tight_layout()

            # Save if path provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                fig.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"✓ Force plot saved: {output_path}")

            return fig
        else:
            # Use JavaScript for interactive plot
            force_plot = shap.force_plot(
                base_value,
                explanation.values,
                explanation.data,
                feature_names=self.feature_names
            )

            # Save as HTML if path provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                shap.save_html(output_path, force_plot)
                print(f"✓ Interactive force plot saved: {output_path}")

            return force_plot

    def generate_summary_plot(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        plot_type: str = "dot",
        max_display: int = 20,
        output_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Generate summary plot showing feature importance across samples.

        Args:
            X: Input features (multiple samples)
            plot_type: Type of plot ('dot', 'bar', 'violin')
            max_display: Maximum features to display
            output_path: Path to save plot
            show: Whether to display plot

        Returns:
            Matplotlib figure
        """
        # Convert to DataFrame
        X_df = self._to_dataframe(X)

        # Compute SHAP values
        shap_values = self.get_shap_values(X_df)

        # Create figure
        fig = plt.figure(figsize=(10, 8))

        # Generate summary plot
        shap.summary_plot(
            shap_values,
            X_df,
            plot_type=plot_type,
            max_display=max_display,
            show=False
        )

        # Improve styling
        ax = plt.gca()
        ax.set_title(f"SHAP Summary Plot - Top {max_display} Features",
                     fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()

        # Save if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Summary plot saved: {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def get_top_features(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        n: int = 10,
        absolute: bool = True
    ) -> pd.DataFrame:
        """
        Get top N most important features based on mean absolute SHAP values.

        Args:
            X: Input features
            n: Number of top features to return
            absolute: If True, use absolute SHAP values; if False, use raw values

        Returns:
            DataFrame with feature importance ranking
        """
        # Convert to DataFrame
        X_df = self._to_dataframe(X)

        # Compute SHAP values
        shap_values = self.get_shap_values(X_df)

        # Calculate mean importance
        if absolute:
            importance = np.abs(shap_values).mean(axis=0)
        else:
            importance = shap_values.mean(axis=0)

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return importance_df.head(n).reset_index(drop=True)

    def get_feature_contributions(
        self,
        X_single: Union[pd.Series, pd.DataFrame, np.ndarray],
        top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get feature contributions for a single prediction.

        Args:
            X_single: Single sample
            top_n: If provided, return only top N features by absolute contribution

        Returns:
            DataFrame with features, values, and SHAP contributions
        """
        # Get explanation
        explanation = self.get_explanation(X_single, return_dict=True)

        # Create DataFrame
        contributions_df = pd.DataFrame({
            'feature': self.feature_names,
            'value': [explanation['feature_values'][f] for f in self.feature_names],
            'shap_value': explanation['shap_values'],
            'abs_shap_value': np.abs(explanation['shap_values'])
        }).sort_values('abs_shap_value', ascending=False)

        if top_n:
            contributions_df = contributions_df.head(top_n)

        return contributions_df.reset_index(drop=True)

    def _to_dataframe(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Convert input to DataFrame.

        Args:
            X: Input features

        Returns:
            DataFrame with proper column names
        """
        if isinstance(X, pd.DataFrame):
            return X
        elif isinstance(X, np.ndarray):
            return pd.DataFrame(X, columns=self.feature_names)
        else:
            raise ValueError("X must be pandas DataFrame or numpy array")

    def _get_prediction(self, X: pd.DataFrame) -> float:
        """
        Get model prediction probability.

        Args:
            X: Input features

        Returns:
            Prediction probability for positive class
        """
        try:
            if hasattr(self.model, 'predict_proba'):
                pred = self.model.predict_proba(X)[0, 1]
            else:
                pred = self.model.predict(X)[0]
            return float(pred)
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")


def create_explainer(
    model: Any,
    X_train: Union[pd.DataFrame, np.ndarray],
    feature_names: Optional[List[str]] = None,
    max_background_samples: int = 100
) -> SHAPExplainer:
    """
    Convenience function to create SHAP explainer.

    Args:
        model: Trained model
        X_train: Training data (or subset) for background
        feature_names: Feature names (required if X_train is numpy array)
        max_background_samples: Maximum background samples

    Returns:
        Initialized SHAPExplainer
    """
    return SHAPExplainer(
        model=model,
        background_data=X_train,
        feature_names=feature_names,
        max_background_samples=max_background_samples
    )


if __name__ == "__main__":
    """
    Example usage: Create SHAP explainer and generate explanations.
    """
    print("SHAP Explainability Engine")
    print("\nExample usage:")
    print("  from src.explainability.shap_engine import SHAPExplainer")
    print("  explainer = SHAPExplainer(model, X_train_sample)")
    print("  explanation = explainer.get_explanation(X_single)")
    print("  explainer.generate_waterfall_plot(X_single, output_path='waterfall.png')")
