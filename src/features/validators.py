"""
Feature validation and compliance checking for VantageFlow AI.

Ensures that features used in credit scoring models comply with fair lending
regulations (ECOA, Fair Housing Act) and do not include prohibited attributes
or high-risk proxies for protected characteristics.
"""

import os
from typing import List, Dict, Set, Optional, Tuple
from datetime import datetime
import warnings

import yaml
import pandas as pd
import numpy as np
from scipy import stats


class FeatureValidationError(Exception):
    """Raised when prohibited features are detected in the feature set."""
    pass


class HighRiskProxyWarning(UserWarning):
    """Warning for features that may be proxies for protected attributes."""
    pass


class FeatureValidator:
    """
    Validate features for compliance with fair lending regulations.

    Loads configuration from feature_config.yaml and validates that:
    1. No prohibited features (race, gender, age, etc.) are used
    2. High-risk proxy features are documented and justified
    3. Features don't have high correlation with protected attributes
    4. Feature documentation is complete
    """

    def __init__(self, config_path: str = "config/feature_config.yaml"):
        """
        Initialize feature validator.

        Args:
            config_path: Path to feature configuration YAML file

        Raises:
            FileNotFoundError: If config file not found
            ValueError: If config file is invalid
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Feature configuration file not found: {config_path}\n"
                "Please ensure config/feature_config.yaml exists."
            )

        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {str(e)}")

        # Extract feature lists from config
        self.prohibited_features = set(self.config.get("prohibited_features", []))
        self.high_risk_proxies = set(self.config.get("high_risk_proxies", []))
        self.allowed_features = set(self.config.get("allowed_features", []))

        # Extract thresholds
        thresholds = self.config.get("correlation_thresholds", {})
        self.protected_correlation_threshold = thresholds.get("protected_attribute_correlation", 0.30)
        self.warning_threshold = thresholds.get("warning_threshold", 0.20)
        self.multicollinearity_threshold = thresholds.get("feature_multicollinearity", 0.90)

        # Audit settings
        self.audit_settings = self.config.get("audit_settings", {})
        self.log_validations = self.audit_settings.get("log_validations", True)

        # Initialize audit log
        self.audit_log = []

    def validate_feature_set(
        self,
        feature_names: List[str],
        raise_on_prohibited: bool = True,
        raise_on_proxy: bool = False
    ) -> Dict[str, any]:
        """
        Validate that feature set contains no prohibited features.

        Args:
            feature_names: List of feature names to validate
            raise_on_prohibited: Raise exception if prohibited features found
            raise_on_proxy: Raise exception if high-risk proxies found

        Returns:
            Dictionary with validation results

        Raises:
            FeatureValidationError: If prohibited features detected and raise_on_prohibited=True
        """
        # Convert to set for faster lookup
        feature_set = set(feature_names)

        # Check for prohibited features
        prohibited_found = feature_set.intersection(self.prohibited_features)

        # Check for high-risk proxies
        proxies_found = feature_set.intersection(self.high_risk_proxies)

        # Check for unknown features (not in allowed list)
        unknown_features = feature_set - self.allowed_features - self.high_risk_proxies

        # Build validation result
        validation_result = {
            "valid": len(prohibited_found) == 0,
            "prohibited_features_found": list(prohibited_found),
            "high_risk_proxies_found": list(proxies_found),
            "unknown_features": list(unknown_features),
            "total_features": len(feature_names),
            "validation_timestamp": datetime.now().isoformat()
        }

        # Log validation if enabled
        if self.log_validations:
            self._log_validation(validation_result)

        # Raise error if prohibited features found
        if prohibited_found and raise_on_prohibited:
            error_msg = (
                f"PROHIBITED FEATURES DETECTED: {', '.join(prohibited_found)}\n\n"
                f"The following features are strictly prohibited under fair lending "
                f"regulations (ECOA, Fair Housing Act):\n"
            )
            for feature in prohibited_found:
                error_msg += f"  - {feature}\n"
            error_msg += (
                f"\nThese features cannot be used in credit scoring models as they "
                f"represent protected characteristics or direct identifiers.\n"
                f"Please remove these features and use only behavioral features "
                f"derived from transaction history."
            )
            raise FeatureValidationError(error_msg)

        # Warn about high-risk proxies
        if proxies_found:
            warning_msg = (
                f"HIGH-RISK PROXY FEATURES DETECTED: {', '.join(proxies_found)}\n"
                f"These features may be correlated with protected characteristics.\n"
                f"Additional fairness testing and documentation required."
            )
            warnings.warn(warning_msg, HighRiskProxyWarning)

            if raise_on_proxy:
                raise FeatureValidationError(
                    f"High-risk proxy features not allowed: {', '.join(proxies_found)}"
                )

        # Warn about unknown features
        if unknown_features:
            warnings.warn(
                f"Unknown features detected (not in allowed list): {', '.join(unknown_features)}\n"
                f"Please add these features to config/feature_config.yaml if they are valid.",
                UserWarning
            )

        return validation_result

    def audit_feature_correlations(
        self,
        features_df: pd.DataFrame,
        protected_attributes_df: pd.DataFrame,
        protected_columns: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Audit correlations between features and protected attributes.

        Args:
            features_df: DataFrame with model features
            protected_attributes_df: DataFrame with protected attributes (for testing only)
            protected_columns: List of protected attribute column names
                             (defaults to ["age", "gender", "ethnicity"])

        Returns:
            Dictionary with correlation audit results
        """
        if protected_columns is None:
            protected_columns = ["age", "gender", "ethnicity"]

        # Ensure DataFrames are aligned
        if len(features_df) != len(protected_attributes_df):
            raise ValueError("Features and protected attributes must have same length")

        correlation_results = {
            "timestamp": datetime.now().isoformat(),
            "num_features": len(features_df.columns),
            "num_protected_attributes": len(protected_columns),
            "correlations": {},
            "high_correlation_features": [],
            "warning_correlation_features": [],
            "passed": True
        }

        # Calculate correlations for each feature with each protected attribute
        for feature in features_df.columns:
            if feature == "borrower_id":
                continue

            feature_correlations = {}

            for protected_attr in protected_columns:
                if protected_attr not in protected_attributes_df.columns:
                    continue

                # Calculate correlation
                corr_value = self._calculate_correlation(
                    features_df[feature],
                    protected_attributes_df[protected_attr]
                )

                feature_correlations[protected_attr] = corr_value

                # Check thresholds
                abs_corr = abs(corr_value)

                if abs_corr > self.protected_correlation_threshold:
                    correlation_results["high_correlation_features"].append({
                        "feature": feature,
                        "protected_attribute": protected_attr,
                        "correlation": corr_value,
                        "threshold": self.protected_correlation_threshold,
                        "severity": "HIGH"
                    })
                    correlation_results["passed"] = False

                elif abs_corr > self.warning_threshold:
                    correlation_results["warning_correlation_features"].append({
                        "feature": feature,
                        "protected_attribute": protected_attr,
                        "correlation": corr_value,
                        "threshold": self.warning_threshold,
                        "severity": "WARNING"
                    })

            correlation_results["correlations"][feature] = feature_correlations

        # Log audit if enabled
        if self.log_validations:
            self._log_audit(correlation_results)

        return correlation_results

    def _calculate_correlation(
        self,
        feature_series: pd.Series,
        protected_series: pd.Series
    ) -> float:
        """
        Calculate correlation between feature and protected attribute.

        Handles both numeric and categorical variables.

        Args:
            feature_series: Feature values
            protected_series: Protected attribute values

        Returns:
            Correlation coefficient
        """
        # Remove NaN values
        mask = ~(feature_series.isna() | protected_series.isna())
        feature_clean = feature_series[mask]
        protected_clean = protected_series[mask]

        if len(feature_clean) < 2:
            return 0.0

        # Check if protected attribute is categorical
        if protected_clean.dtype == 'object' or protected_clean.dtype.name == 'category':
            # Use point-biserial correlation or ANOVA F-statistic
            try:
                # Encode categorical as numeric
                categories = protected_clean.unique()
                if len(categories) == 2:
                    # Binary: use point-biserial correlation
                    encoded = (protected_clean == categories[0]).astype(int)
                    corr, _ = stats.pearsonr(feature_clean, encoded)
                    return corr
                else:
                    # Multi-class: use correlation ratio (eta)
                    groups = [feature_clean[protected_clean == cat] for cat in categories]
                    f_stat, _ = stats.f_oneway(*groups)
                    # Convert F-statistic to correlation-like metric
                    return np.sqrt(f_stat / (f_stat + len(feature_clean) - len(categories)))
            except:
                return 0.0
        else:
            # Both numeric: use Pearson correlation
            try:
                corr, _ = stats.pearsonr(feature_clean, protected_clean)
                return corr
            except:
                return 0.0

    def generate_feature_documentation(
        self,
        feature_names: List[str],
        output_path: str = "docs/feature_documentation.md"
    ) -> str:
        """
        Generate documentation for features used in the model.

        Args:
            feature_names: List of feature names
            output_path: Path to save documentation

        Returns:
            Generated documentation as string
        """
        doc = []
        doc.append("# Feature Documentation - VantageFlow AI Credit Scoring\n")
        doc.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        doc.append(f"**Total Features:** {len(feature_names)}\n\n")

        # Validation summary
        validation = self.validate_feature_set(feature_names, raise_on_prohibited=False)

        doc.append("## Validation Summary\n")
        doc.append(f"- **Status:** {'✅ PASSED' if validation['valid'] else '❌ FAILED'}\n")
        doc.append(f"- **Prohibited Features:** {len(validation['prohibited_features_found'])}\n")
        doc.append(f"- **High-Risk Proxies:** {len(validation['high_risk_proxies_found'])}\n")
        doc.append(f"- **Unknown Features:** {len(validation['unknown_features'])}\n\n")

        if validation['prohibited_features_found']:
            doc.append("### ⚠️ PROHIBITED FEATURES DETECTED\n")
            for feature in validation['prohibited_features_found']:
                doc.append(f"- **{feature}** - MUST BE REMOVED\n")
            doc.append("\n")

        if validation['high_risk_proxies_found']:
            doc.append("### ⚠️ High-Risk Proxy Features\n")
            doc.append("*Require additional fairness testing and documentation*\n\n")
            for feature in validation['high_risk_proxies_found']:
                doc.append(f"- **{feature}**\n")
            doc.append("\n")

        # Feature list
        doc.append("## Feature List\n\n")

        # Group features by category
        feature_groups = self._group_features(feature_names)

        for group_name, features in feature_groups.items():
            doc.append(f"### {group_name}\n\n")
            for feature in features:
                status = self._get_feature_status(feature)
                doc.append(f"- **{feature}** - {status}\n")
            doc.append("\n")

        # Compliance statement
        doc.append("## Compliance Statement\n\n")
        doc.append("This feature set has been validated against fair lending regulations:\n")
        doc.append("- Equal Credit Opportunity Act (ECOA)\n")
        doc.append("- Fair Housing Act (FHA)\n")
        doc.append("- Consumer Financial Protection Bureau (CFPB) guidelines\n\n")

        doc.append("All features are derived from behavioral transaction data and do not ")
        doc.append("directly use protected characteristics (race, gender, age, etc.).\n\n")

        # Review information
        doc.append("## Review Information\n\n")
        doc.append(f"- **Config Version:** {self.config.get('version', 'N/A')}\n")
        doc.append(f"- **Last Config Update:** {self.config.get('last_updated', 'N/A')}\n")
        doc.append(f"- **Approved By:** {self.config.get('approved_by', 'N/A')}\n")
        doc.append(f"- **Next Review:** {self.config.get('next_review_date', 'N/A')}\n")

        # Save documentation
        documentation = ''.join(doc)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(documentation)

        print(f"✓ Feature documentation generated: {output_path}")

        return documentation

    def _group_features(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """
        Group features by category based on naming patterns.

        Args:
            feature_names: List of feature names

        Returns:
            Dictionary mapping category to feature list
        """
        groups = {
            "Income Features": [],
            "Spending Features": [],
            "Financial Health Features": [],
            "Temporal Features": [],
            "Category Features": [],
            "Derived Features": [],
            "Other Features": []
        }

        for feature in feature_names:
            if any(x in feature for x in ["income", "salary"]):
                groups["Income Features"].append(feature)
            elif any(x in feature for x in ["spending", "expense", "discretionary"]):
                groups["Spending Features"].append(feature)
            elif any(x in feature for x in ["balance", "overdraft", "cashflow"]):
                groups["Financial Health Features"].append(feature)
            elif any(x in feature for x in ["count", "days", "months", "transaction"]):
                groups["Temporal Features"].append(feature)
            elif any(x in feature for x in ["rent", "utilities", "groceries", "transportation"]):
                groups["Category Features"].append(feature)
            elif any(x in feature for x in ["ratio", "pct", "score"]):
                groups["Derived Features"].append(feature)
            else:
                groups["Other Features"].append(feature)

        # Remove empty groups
        return {k: v for k, v in groups.items() if v}

    def _get_feature_status(self, feature: str) -> str:
        """
        Get status label for a feature.

        Args:
            feature: Feature name

        Returns:
            Status string
        """
        if feature in self.prohibited_features:
            return "❌ PROHIBITED"
        elif feature in self.high_risk_proxies:
            return "⚠️ High-Risk Proxy"
        elif feature in self.allowed_features:
            return "✅ Allowed"
        else:
            return "❓ Unknown"

    def _log_validation(self, validation_result: Dict) -> None:
        """
        Log validation attempt to audit trail.

        Args:
            validation_result: Validation result dictionary
        """
        self.audit_log.append({
            "type": "validation",
            "timestamp": validation_result["validation_timestamp"],
            "result": validation_result
        })

    def _log_audit(self, audit_result: Dict) -> None:
        """
        Log correlation audit to audit trail.

        Args:
            audit_result: Audit result dictionary
        """
        self.audit_log.append({
            "type": "correlation_audit",
            "timestamp": audit_result["timestamp"],
            "result": audit_result
        })

    def get_audit_log(self) -> List[Dict]:
        """
        Get complete audit log.

        Returns:
            List of audit log entries
        """
        return self.audit_log

    def export_audit_log(self, output_path: str = "data/output/feature_audit_log.json") -> None:
        """
        Export audit log to JSON file.

        Args:
            output_path: Path to save audit log
        """
        import json

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.audit_log, f, indent=2)

        print(f"✓ Audit log exported: {output_path}")


# Convenience functions

def validate_features(
    feature_names: List[str],
    config_path: str = "config/feature_config.yaml"
) -> Dict[str, any]:
    """
    Validate feature set for compliance.

    Args:
        feature_names: List of feature names
        config_path: Path to config file

    Returns:
        Validation results dictionary
    """
    validator = FeatureValidator(config_path)
    return validator.validate_feature_set(feature_names)


def generate_documentation(
    feature_names: List[str],
    output_path: str = "docs/feature_documentation.md",
    config_path: str = "config/feature_config.yaml"
) -> str:
    """
    Generate feature documentation.

    Args:
        feature_names: List of feature names
        output_path: Path to save documentation
        config_path: Path to config file

    Returns:
        Documentation string
    """
    validator = FeatureValidator(config_path)
    return validator.generate_feature_documentation(feature_names, output_path)


if __name__ == "__main__":
    """
    Example usage: Validate features and generate documentation.
    """
    print("Feature Validator - VantageFlow AI")
    print("\nExample usage:")
    print("  from src.features.validators import validate_features")
    print("  result = validate_features(['avg_monthly_income', 'savings_rate'])")
    print("  print(result)")
