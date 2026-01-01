"""
Reason code generation for credit decisions.

Translates SHAP values into business-friendly, regulatory-compliant reason codes
for adverse action notices (FCRA Section 615(a) compliance).
"""

import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

import yaml
import numpy as np
import pandas as pd


@dataclass
class ReasonCode:
    """
    Data class representing a single reason code.

    Attributes:
        code: Unique identifier (e.g., 'N01', 'P01')
        name: Reason code name (e.g., 'INCOME_VOLATILITY_HIGH')
        description: Short description
        long_description: Detailed explanation
        impact: 'positive' or 'negative'
        magnitude: 'strong', 'moderate', or 'slight'
        contribution: Absolute SHAP contribution value
        features: List of contributing features
    """
    code: str
    name: str
    description: str
    long_description: str
    impact: str
    magnitude: str
    contribution: float
    features: List[str]

    def __repr__(self) -> str:
        return f"<ReasonCode {self.code}: {self.name} ({self.magnitude}, {self.contribution:.4f})>"


class ReasonCodeGenerator:
    """
    Generate business-friendly reason codes from SHAP explanations.

    Maps SHAP feature contributions to predefined reason codes with
    magnitude assessment and regulatory compliance.
    """

    def __init__(self, config_path: str = "config/reason_codes.yaml"):
        """
        Initialize reason code generator.

        Args:
            config_path: Path to reason codes configuration file

        Raises:
            FileNotFoundError: If config file not found
            ValueError: If config is invalid
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Reason codes config not found: {config_path}")

        # Load configuration
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config: {str(e)}")

        # Extract reason codes
        self.positive_reasons = self.config.get('positive_reasons', {})
        self.negative_reasons = self.config.get('negative_reasons', {})
        self.magnitude_levels = self.config.get('magnitude_levels', {})

        # Create lookup dictionaries
        self._build_lookups()

        print(f"✓ Loaded {len(self.positive_reasons)} positive and "
              f"{len(self.negative_reasons)} negative reason codes")

    def _build_lookups(self) -> None:
        """Build lookup dictionaries for efficient reason code mapping."""
        # Map features to reason codes
        self.feature_to_reasons = {}

        for reason_type, reasons in [
            ('positive', self.positive_reasons),
            ('negative', self.negative_reasons)
        ]:
            for reason_name, reason_data in reasons.items():
                features = reason_data.get('features', [])
                for feature in features:
                    if feature not in self.feature_to_reasons:
                        self.feature_to_reasons[feature] = []
                    self.feature_to_reasons[feature].append({
                        'name': reason_name,
                        'type': reason_type,
                        'data': reason_data
                    })

    def generate_reason_codes(
        self,
        shap_values: Dict[str, float],
        feature_values: Dict[str, float],
        top_n: int = 5,
        validate_direction: bool = True
    ) -> List[ReasonCode]:
        """
        Generate reason codes from SHAP values and feature values.

        Args:
            shap_values: Dictionary mapping feature name to SHAP value
            feature_values: Dictionary mapping feature name to actual value
            top_n: Number of top reason codes to return
            validate_direction: Whether to validate SHAP direction matches expected

        Returns:
            List of ReasonCode objects, sorted by absolute contribution
        """
        reason_codes = []

        # Process each feature with significant SHAP contribution
        for feature, shap_value in shap_values.items():
            # Skip if contribution is negligible
            if abs(shap_value) < 0.001:
                continue

            # Get feature value
            feature_value = feature_values.get(feature)

            # Find applicable reason codes for this feature
            applicable_reasons = self.feature_to_reasons.get(feature, [])

            for reason_info in applicable_reasons:
                reason_name = reason_info['name']
                reason_type = reason_info['type']
                reason_data = reason_info['data']

                # Check if conditions are met
                if not self._check_conditions(reason_data, feature, feature_value):
                    continue

                # Determine if SHAP direction matches expected impact
                expected_direction = reason_data.get('shap_direction', 'positive')
                actual_direction = 'positive' if shap_value > 0 else 'negative'

                # Validate direction if requested
                if validate_direction and expected_direction != actual_direction:
                    # Skip if direction doesn't match (SHAP disagrees with config)
                    continue

                # Calculate magnitude
                magnitude = self._calculate_magnitude(
                    abs(shap_value),
                    reason_data.get('magnitude_thresholds', {})
                )

                # Create reason code
                reason_code = ReasonCode(
                    code=reason_data.get('code', 'N/A'),
                    name=reason_name,
                    description=reason_data.get('description', ''),
                    long_description=reason_data.get('long_description', ''),
                    impact=reason_data.get('impact', reason_type),
                    magnitude=magnitude,
                    contribution=abs(shap_value),
                    features=[feature]
                )

                reason_codes.append(reason_code)

        # Remove duplicates (keep highest contribution for each reason code name)
        unique_reasons = {}
        for rc in reason_codes:
            if rc.name not in unique_reasons or rc.contribution > unique_reasons[rc.name].contribution:
                unique_reasons[rc.name] = rc

        # Sort by contribution (descending)
        sorted_reasons = sorted(
            unique_reasons.values(),
            key=lambda x: x.contribution,
            reverse=True
        )

        # Return top N
        return sorted_reasons[:top_n]

    def _check_conditions(
        self,
        reason_data: Dict,
        feature: str,
        feature_value: Optional[float]
    ) -> bool:
        """
        Check if feature value meets reason code conditions.

        Args:
            reason_data: Reason code configuration
            feature: Feature name
            feature_value: Feature value

        Returns:
            True if conditions are met (or no conditions specified)
        """
        if feature_value is None:
            return False

        # Get conditions
        conditions = reason_data.get('conditions', [])
        if not conditions:
            return True  # No conditions = always applicable

        # Simple condition parsing (would be more robust in production)
        # For now, just check if feature is in the list (basic validation)
        features = reason_data.get('features', [])
        return feature in features

    def _calculate_magnitude(
        self,
        abs_shap_value: float,
        thresholds: Dict[str, float]
    ) -> str:
        """
        Calculate magnitude level based on SHAP contribution.

        Args:
            abs_shap_value: Absolute SHAP value
            thresholds: Magnitude thresholds (strong, moderate, slight)

        Returns:
            Magnitude level: 'strong', 'moderate', or 'slight'
        """
        # Get thresholds (with defaults)
        strong_threshold = thresholds.get('strong', 0.025)
        moderate_threshold = thresholds.get('moderate', 0.01)
        slight_threshold = thresholds.get('slight', 0.005)

        if abs_shap_value >= strong_threshold:
            return 'strong'
        elif abs_shap_value >= moderate_threshold:
            return 'moderate'
        elif abs_shap_value >= slight_threshold:
            return 'slight'
        else:
            return 'negligible'

    def format_for_display(
        self,
        reason_codes: List[ReasonCode],
        include_technical: bool = False,
        format_type: str = "text"
    ) -> str:
        """
        Format reason codes for display in adverse action notice.

        Args:
            reason_codes: List of ReasonCode objects
            include_technical: Whether to include technical details (SHAP values)
            format_type: Output format ('text', 'html', or 'json')

        Returns:
            Formatted string
        """
        if format_type == "text":
            return self._format_text(reason_codes, include_technical)
        elif format_type == "html":
            return self._format_html(reason_codes, include_technical)
        elif format_type == "json":
            import json
            return json.dumps([self._reason_to_dict(rc) for rc in reason_codes], indent=2)
        else:
            raise ValueError(f"Unknown format_type: {format_type}")

    def _format_text(self, reason_codes: List[ReasonCode], include_technical: bool) -> str:
        """Format as plain text."""
        lines = []
        lines.append("=" * 80)
        lines.append("CREDIT DECISION FACTORS")
        lines.append("=" * 80)
        lines.append("")

        # Add regulatory statement
        regulatory_stmt = self.config.get('adverse_action_notice', {}).get('regulatory_statement', '')
        if regulatory_stmt:
            lines.append(regulatory_stmt.strip())
            lines.append("")

        # Add reason codes
        lines.append("PRIMARY FACTORS (in order of impact):")
        lines.append("-" * 80)
        lines.append("")

        for i, rc in enumerate(reason_codes, 1):
            # Format impact indicator
            impact_indicator = "[-]" if rc.impact == "negative" else "[+]"

            # Format magnitude
            magnitude_text = rc.magnitude.upper()

            lines.append(f"{i}. {impact_indicator} {rc.name}")
            lines.append(f"   Code: {rc.code}")
            lines.append(f"   Impact: {magnitude_text}")
            lines.append(f"   {rc.description}")
            lines.append("")
            lines.append(f"   Details: {rc.long_description}")

            if include_technical:
                lines.append(f"   SHAP Contribution: {rc.contribution:.4f}")
                lines.append(f"   Contributing Features: {', '.join(rc.features)}")

            lines.append("")

        # Add footer
        footer = self.config.get('adverse_action_notice', {}).get('footer', '')
        if footer:
            lines.append("-" * 80)
            lines.append(footer.strip())

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)

    def _format_html(self, reason_codes: List[ReasonCode], include_technical: bool) -> str:
        """Format as HTML."""
        html = ['<div class="adverse-action-notice">']
        html.append('<h2>Credit Decision Factors</h2>')

        # Regulatory statement
        regulatory_stmt = self.config.get('adverse_action_notice', {}).get('regulatory_statement', '')
        if regulatory_stmt:
            html.append(f'<p class="regulatory-statement">{regulatory_stmt.strip()}</p>')

        html.append('<h3>Primary Factors (in order of impact):</h3>')
        html.append('<ol class="reason-codes">')

        for rc in reason_codes:
            impact_class = "positive" if rc.impact == "positive" else "negative"
            html.append(f'<li class="reason-code {impact_class}">')
            html.append(f'  <div class="reason-header">')
            html.append(f'    <strong>{rc.name}</strong> <span class="code">[{rc.code}]</span>')
            html.append(f'    <span class="magnitude {rc.magnitude}">{rc.magnitude.upper()}</span>')
            html.append(f'  </div>')
            html.append(f'  <p class="description">{rc.description}</p>')
            html.append(f'  <p class="long-description">{rc.long_description}</p>')

            if include_technical:
                html.append(f'  <p class="technical">SHAP: {rc.contribution:.4f}, Features: {", ".join(rc.features)}</p>')

            html.append('</li>')

        html.append('</ol>')

        # Footer
        footer = self.config.get('adverse_action_notice', {}).get('footer', '')
        if footer:
            html.append(f'<p class="footer">{footer.strip()}</p>')

        html.append('</div>')

        return '\n'.join(html)

    def _reason_to_dict(self, rc: ReasonCode) -> Dict:
        """Convert ReasonCode to dictionary."""
        return {
            'code': rc.code,
            'name': rc.name,
            'description': rc.description,
            'long_description': rc.long_description,
            'impact': rc.impact,
            'magnitude': rc.magnitude,
            'contribution': rc.contribution,
            'features': rc.features
        }

    def generate_adverse_action_notice(
        self,
        reason_codes: List[ReasonCode],
        applicant_name: Optional[str] = None,
        decision: str = "denied",
        format_type: str = "text"
    ) -> str:
        """
        Generate complete adverse action notice.

        Args:
            reason_codes: List of reason codes
            applicant_name: Applicant name (optional)
            decision: Credit decision ('denied', 'approved', 'approved_with_conditions')
            format_type: Output format

        Returns:
            Formatted adverse action notice
        """
        # Add header with decision
        notice_lines = []
        notice_lines.append("=" * 80)
        notice_lines.append("ADVERSE ACTION NOTICE")
        notice_lines.append("=" * 80)
        notice_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d')}")

        if applicant_name:
            notice_lines.append(f"Applicant: {applicant_name}")

        notice_lines.append(f"Decision: {decision.upper()}")
        notice_lines.append("")

        # Add formatted reason codes
        formatted_reasons = self.format_for_display(reason_codes, format_type=format_type)
        notice_lines.append(formatted_reasons)

        return "\n".join(notice_lines)


def generate_reason_codes_from_shap(
    shap_explanation: Any,
    top_n: int = 5,
    config_path: str = "config/reason_codes.yaml"
) -> List[ReasonCode]:
    """
    Convenience function to generate reason codes from SHAP explanation.

    Args:
        shap_explanation: SHAP Explanation object
        top_n: Number of top reasons to return
        config_path: Path to reason codes config

    Returns:
        List of ReasonCode objects
    """
    # Extract SHAP values and feature values
    shap_values = dict(zip(shap_explanation.feature_names, shap_explanation.values))
    feature_values = dict(zip(shap_explanation.feature_names, shap_explanation.data))

    # Generate reason codes
    generator = ReasonCodeGenerator(config_path)
    return generator.generate_reason_codes(shap_values, feature_values, top_n=top_n)


if __name__ == "__main__":
    """
    Example usage: Generate reason codes from SHAP values.
    """
    print("Reason Code Generator")
    print("\nExample usage:")
    print("  from src.explainability.reason_codes import ReasonCodeGenerator")
    print("  generator = ReasonCodeGenerator()")
    print("  reasons = generator.generate_reason_codes(shap_values, feature_values, top_n=5)")
    print("  print(generator.format_for_display(reasons))")
