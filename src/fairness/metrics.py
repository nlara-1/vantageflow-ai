"""
Fairness metrics and auditing for VantageFlow AI credit scoring.

Implements comprehensive fairness assessments including demographic parity,
equalized odds, equal opportunity, and disparate impact analysis across
protected characteristics.
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Set visualization style
sns.set_style("whitegrid")


class FairnessAuditor:
    """
    Comprehensive fairness auditing for credit scoring models.

    Calculates fairness metrics across protected attributes to ensure
    compliance with anti-discrimination laws and ethical AI principles.
    """

    def __init__(
        self,
        protected_attributes: List[str],
        reference_groups: Optional[Dict[str, str]] = None,
        min_samples_warning: int = 30
    ):
        """
        Initialize fairness auditor.

        Args:
            protected_attributes: List of protected attribute column names
                                (e.g., ['gender', 'race', 'age_group'])
            reference_groups: Dict mapping attribute to reference group value
                            (e.g., {'gender': 'Male', 'race': 'White'})
            min_samples_warning: Minimum samples per group before warning
        """
        self.protected_attributes = protected_attributes
        self.reference_groups = reference_groups or {}
        self.min_samples_warning = min_samples_warning

        self.fairness_metrics = {}
        self.group_statistics = {}

    def calculate_fairness_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        sensitive_features: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive fairness metrics across protected groups.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            sensitive_features: DataFrame with protected attributes

        Returns:
            Dictionary with fairness metrics for each protected attribute
        """
        if sensitive_features is None:
            raise ValueError("sensitive_features DataFrame required for fairness analysis")

        # Validate protected attributes exist
        for attr in self.protected_attributes:
            if attr not in sensitive_features.columns:
                raise ValueError(f"Protected attribute '{attr}' not found in sensitive_features")

        results = {}

        # Calculate metrics for each protected attribute
        for attr in self.protected_attributes:
            attr_results = self._calculate_metrics_for_attribute(
                y_true=y_true,
                y_pred=y_pred,
                y_pred_proba=y_pred_proba,
                sensitive_feature=sensitive_features[attr],
                attribute_name=attr
            )
            results[attr] = attr_results

        self.fairness_metrics = results
        return results

    def _calculate_metrics_for_attribute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray],
        sensitive_feature: pd.Series,
        attribute_name: str
    ) -> Dict[str, Any]:
        """
        Calculate fairness metrics for a single protected attribute.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            sensitive_feature: Series with protected attribute values
            attribute_name: Name of the attribute

        Returns:
            Dictionary with metrics for this attribute
        """
        results = {
            'attribute': attribute_name,
            'groups': {},
            'pairwise_metrics': {},
            'overall_metrics': {}
        }

        # Get unique groups
        groups = sensitive_feature.unique()

        # Determine reference group
        reference_group = self.reference_groups.get(attribute_name)
        if reference_group is None:
            # Use first group as reference
            reference_group = groups[0]

        results['reference_group'] = reference_group

        # Calculate statistics for each group
        for group in groups:
            group_mask = sensitive_feature == group
            group_size = group_mask.sum()

            # Warn if small sample size
            if group_size < self.min_samples_warning:
                warnings.warn(
                    f"Small sample size for {attribute_name}={group}: {group_size} samples. "
                    f"Results may be unreliable.",
                    UserWarning
                )

            # Skip if no samples
            if group_size == 0:
                continue

            # Calculate group statistics
            y_true_group = y_true[group_mask]
            y_pred_group = y_pred[group_mask]

            group_stats = {
                'n_samples': int(group_size),
                'selection_rate': float(y_pred_group.mean()),  # Positive prediction rate
                'positive_rate': float(y_true_group.mean()),   # Actual positive rate
            }

            # Calculate confusion matrix metrics
            if len(y_true_group) > 0:
                cm = confusion_matrix(y_true_group, y_pred_group, labels=[0, 1])

                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()

                    # True Positive Rate (Recall, Sensitivity)
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

                    # False Positive Rate
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

                    # Precision (Positive Predictive Value)
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

                    # False Negative Rate
                    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

                    group_stats.update({
                        'tpr': float(tpr),  # True Positive Rate
                        'fpr': float(fpr),  # False Positive Rate
                        'fnr': float(fnr),  # False Negative Rate
                        'precision': float(precision),
                        'tp': int(tp),
                        'fp': int(fp),
                        'tn': int(tn),
                        'fn': int(fn)
                    })
                else:
                    # Handle degenerate cases
                    group_stats.update({
                        'tpr': 0.0,
                        'fpr': 0.0,
                        'fnr': 0.0,
                        'precision': 0.0
                    })

            results['groups'][group] = group_stats

        # Calculate pairwise fairness metrics vs reference group
        if reference_group in results['groups']:
            ref_stats = results['groups'][reference_group]

            for group in groups:
                if group == reference_group:
                    continue

                if group not in results['groups']:
                    continue

                group_stats = results['groups'][group]

                pairwise = self._calculate_pairwise_metrics(ref_stats, group_stats)
                results['pairwise_metrics'][f"{reference_group}_vs_{group}"] = pairwise

        # Calculate overall fairness metrics
        results['overall_metrics'] = self._calculate_overall_metrics(results['groups'])

        return results

    def _calculate_pairwise_metrics(
        self,
        ref_stats: Dict,
        group_stats: Dict
    ) -> Dict[str, float]:
        """
        Calculate pairwise fairness metrics between reference and comparison group.

        Args:
            ref_stats: Statistics for reference group
            group_stats: Statistics for comparison group

        Returns:
            Dictionary with pairwise fairness metrics
        """
        pairwise = {}

        # Disparate Impact Ratio (selection rate ratio)
        if ref_stats['selection_rate'] > 0:
            pairwise['disparate_impact'] = group_stats['selection_rate'] / ref_stats['selection_rate']
        else:
            pairwise['disparate_impact'] = np.nan

        # Demographic Parity Difference (selection rate difference)
        pairwise['demographic_parity_diff'] = abs(
            group_stats['selection_rate'] - ref_stats['selection_rate']
        )

        # Equalized Odds: TPR and FPR differences
        pairwise['tpr_diff'] = abs(group_stats.get('tpr', 0) - ref_stats.get('tpr', 0))
        pairwise['fpr_diff'] = abs(group_stats.get('fpr', 0) - ref_stats.get('fpr', 0))
        pairwise['equalized_odds_diff'] = max(pairwise['tpr_diff'], pairwise['fpr_diff'])

        # Equal Opportunity: TPR difference only
        pairwise['equal_opportunity_diff'] = pairwise['tpr_diff']

        # Predictive Parity: Precision difference
        pairwise['precision_diff'] = abs(
            group_stats.get('precision', 0) - ref_stats.get('precision', 0)
        )

        return pairwise

    def _calculate_overall_metrics(self, group_stats: Dict) -> Dict[str, Any]:
        """
        Calculate overall fairness metrics across all groups.

        Args:
            group_stats: Statistics for all groups

        Returns:
            Dictionary with overall fairness metrics
        """
        overall = {}

        # Get all selection rates
        selection_rates = [stats['selection_rate'] for stats in group_stats.values()]

        if len(selection_rates) > 0:
            overall['min_selection_rate'] = float(np.min(selection_rates))
            overall['max_selection_rate'] = float(np.max(selection_rates))
            overall['selection_rate_range'] = float(np.max(selection_rates) - np.min(selection_rates))

            # Max disparate impact (worst case)
            if overall['max_selection_rate'] > 0:
                overall['min_disparate_impact'] = overall['min_selection_rate'] / overall['max_selection_rate']
            else:
                overall['min_disparate_impact'] = np.nan

        # Get all TPR values
        tprs = [stats.get('tpr', 0) for stats in group_stats.values()]
        if len(tprs) > 0:
            overall['min_tpr'] = float(np.min(tprs))
            overall['max_tpr'] = float(np.max(tprs))
            overall['tpr_range'] = float(np.max(tprs) - np.min(tprs))

        # Get all FPR values
        fprs = [stats.get('fpr', 0) for stats in group_stats.values()]
        if len(fprs) > 0:
            overall['min_fpr'] = float(np.min(fprs))
            overall['max_fpr'] = float(np.max(fprs))
            overall['fpr_range'] = float(np.max(fprs) - np.min(fprs))

        return overall

    def check_80_rule(
        self,
        attribute: Optional[str] = None
    ) -> Dict[str, bool]:
        """
        Check if model passes the 80% rule (disparate impact > 0.80).

        The 80% rule is a common regulatory threshold for disparate impact.

        Args:
            attribute: Specific attribute to check (if None, checks all)

        Returns:
            Dictionary mapping attribute to pass/fail status
        """
        if not self.fairness_metrics:
            raise ValueError("Must call calculate_fairness_metrics() first")

        results = {}

        attributes_to_check = [attribute] if attribute else self.protected_attributes

        for attr in attributes_to_check:
            if attr not in self.fairness_metrics:
                continue

            metrics = self.fairness_metrics[attr]
            overall = metrics.get('overall_metrics', {})

            # Check minimum disparate impact
            min_di = overall.get('min_disparate_impact', np.nan)

            if not np.isnan(min_di):
                # Pass if disparate impact >= 0.80
                results[attr] = min_di >= 0.80
            else:
                results[attr] = None  # Unable to determine

        return results

    def generate_fairness_report(
        self,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive fairness audit report.

        Args:
            output_path: Path to save report (optional)

        Returns:
            Report as string
        """
        if not self.fairness_metrics:
            raise ValueError("Must call calculate_fairness_metrics() first")

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("FAIRNESS AUDIT REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # 80% Rule check
        rule_80_results = self.check_80_rule()
        report_lines.append("80% RULE COMPLIANCE (Disparate Impact >= 0.80)")
        report_lines.append("-" * 80)
        for attr, passed in rule_80_results.items():
            status = "✓ PASS" if passed else "✗ FAIL" if passed is False else "? UNKNOWN"
            report_lines.append(f"  {attr}: {status}")
        report_lines.append("")

        # Detailed metrics for each attribute
        for attr in self.protected_attributes:
            if attr not in self.fairness_metrics:
                continue

            metrics = self.fairness_metrics[attr]
            report_lines.append("=" * 80)
            report_lines.append(f"PROTECTED ATTRIBUTE: {attr.upper()}")
            report_lines.append("=" * 80)
            report_lines.append(f"Reference Group: {metrics.get('reference_group', 'N/A')}")
            report_lines.append("")

            # Group statistics
            report_lines.append("GROUP STATISTICS")
            report_lines.append("-" * 80)
            report_lines.append(f"{'Group':<20} {'N':<8} {'Sel.Rate':<12} {'TPR':<10} {'FPR':<10} {'Precision':<10}")
            report_lines.append("-" * 80)

            for group, stats in metrics['groups'].items():
                report_lines.append(
                    f"{group:<20} "
                    f"{stats['n_samples']:<8} "
                    f"{stats['selection_rate']:<12.4f} "
                    f"{stats.get('tpr', 0):<10.4f} "
                    f"{stats.get('fpr', 0):<10.4f} "
                    f"{stats.get('precision', 0):<10.4f}"
                )

            report_lines.append("")

            # Pairwise metrics
            if metrics['pairwise_metrics']:
                report_lines.append("PAIRWISE FAIRNESS METRICS")
                report_lines.append("-" * 80)

                for comparison, pairwise in metrics['pairwise_metrics'].items():
                    report_lines.append(f"\n{comparison}:")
                    report_lines.append(f"  Disparate Impact:        {pairwise.get('disparate_impact', 0):.4f}")
                    report_lines.append(f"  Demographic Parity Diff: {pairwise.get('demographic_parity_diff', 0):.4f}")
                    report_lines.append(f"  Equalized Odds Diff:     {pairwise.get('equalized_odds_diff', 0):.4f}")
                    report_lines.append(f"  Equal Opportunity Diff:  {pairwise.get('equal_opportunity_diff', 0):.4f}")
                    report_lines.append(f"  Predictive Parity Diff:  {pairwise.get('precision_diff', 0):.4f}")

            report_lines.append("")

            # Overall metrics
            overall = metrics.get('overall_metrics', {})
            if overall:
                report_lines.append("OVERALL FAIRNESS METRICS")
                report_lines.append("-" * 80)
                report_lines.append(f"  Min Disparate Impact:    {overall.get('min_disparate_impact', 0):.4f}")
                report_lines.append(f"  Selection Rate Range:    {overall.get('selection_rate_range', 0):.4f}")
                report_lines.append(f"  TPR Range:               {overall.get('tpr_range', 0):.4f}")
                report_lines.append(f"  FPR Range:               {overall.get('fpr_range', 0):.4f}")
                report_lines.append("")

        # Interpretation guide
        report_lines.append("=" * 80)
        report_lines.append("INTERPRETATION GUIDE")
        report_lines.append("=" * 80)
        report_lines.append("Disparate Impact: Ratio of selection rates (>= 0.80 to pass 80% rule)")
        report_lines.append("Demographic Parity: Difference in selection rates (lower is better)")
        report_lines.append("Equalized Odds: Max of TPR and FPR differences (lower is better)")
        report_lines.append("Equal Opportunity: Difference in TPR (lower is better)")
        report_lines.append("Predictive Parity: Difference in precision (lower is better)")
        report_lines.append("")
        report_lines.append("=" * 80)

        report_text = "\n".join(report_lines)

        # Save if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"✓ Fairness report saved: {output_path}")

        return report_text

    def plot_fairness_dashboard(
        self,
        output_dir: Optional[str] = None,
        show: bool = True
    ) -> Dict[str, plt.Figure]:
        """
        Create fairness visualization dashboard.

        Args:
            output_dir: Directory to save plots (optional)
            show: Whether to display plots

        Returns:
            Dictionary mapping plot name to figure
        """
        if not self.fairness_metrics:
            raise ValueError("Must call calculate_fairness_metrics() first")

        figures = {}

        for attr in self.protected_attributes:
            if attr not in self.fairness_metrics:
                continue

            metrics = self.fairness_metrics[attr]

            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f"Fairness Dashboard - {attr}", fontsize=16, fontweight='bold')

            # Plot 1: Selection Rates by Group
            self._plot_selection_rates(axes[0, 0], metrics, attr)

            # Plot 2: TPR and FPR by Group
            self._plot_tpr_fpr(axes[0, 1], metrics, attr)

            # Plot 3: Disparate Impact
            self._plot_disparate_impact(axes[1, 0], metrics, attr)

            # Plot 4: Fairness Metrics Heatmap
            self._plot_fairness_heatmap(axes[1, 1], metrics, attr)

            plt.tight_layout()

            # Save if directory provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"fairness_dashboard_{attr}.png")
                fig.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"✓ Dashboard saved: {output_path}")

            figures[attr] = fig

            if show:
                plt.show()
            else:
                plt.close()

        return figures

    def _plot_selection_rates(self, ax, metrics, attr):
        """Plot selection rates by group."""
        groups = list(metrics['groups'].keys())
        selection_rates = [metrics['groups'][g]['selection_rate'] for g in groups]

        ax.bar(groups, selection_rates, color='steelblue', alpha=0.7)
        ax.axhline(y=0.80, color='red', linestyle='--', label='80% Threshold', alpha=0.5)
        ax.set_ylabel('Selection Rate')
        ax.set_title('Selection Rates by Group')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Rotate labels if many groups
        if len(groups) > 3:
            ax.set_xticklabels(groups, rotation=45, ha='right')

    def _plot_tpr_fpr(self, ax, metrics, attr):
        """Plot TPR and FPR by group."""
        groups = list(metrics['groups'].keys())
        tprs = [metrics['groups'][g].get('tpr', 0) for g in groups]
        fprs = [metrics['groups'][g].get('fpr', 0) for g in groups]

        x = np.arange(len(groups))
        width = 0.35

        ax.bar(x - width/2, tprs, width, label='TPR', color='green', alpha=0.7)
        ax.bar(x + width/2, fprs, width, label='FPR', color='red', alpha=0.7)

        ax.set_ylabel('Rate')
        ax.set_title('TPR and FPR by Group')
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        if len(groups) > 3:
            ax.set_xticklabels(groups, rotation=45, ha='right')

    def _plot_disparate_impact(self, ax, metrics, attr):
        """Plot disparate impact ratios."""
        if not metrics['pairwise_metrics']:
            ax.text(0.5, 0.5, 'No pairwise comparisons available',
                   ha='center', va='center', transform=ax.transAxes)
            return

        comparisons = list(metrics['pairwise_metrics'].keys())
        di_ratios = [metrics['pairwise_metrics'][c].get('disparate_impact', 0) for c in comparisons]

        # Shorten comparison names
        short_names = [c.split('_vs_')[-1] for c in comparisons]

        colors = ['green' if di >= 0.80 else 'red' for di in di_ratios]

        ax.barh(short_names, di_ratios, color=colors, alpha=0.7)
        ax.axvline(x=0.80, color='black', linestyle='--', label='80% Rule', linewidth=2)
        ax.set_xlabel('Disparate Impact Ratio')
        ax.set_title('Disparate Impact (vs Reference Group)')
        ax.set_xlim([0, 1.5])
        ax.legend()
        ax.grid(axis='x', alpha=0.3)

    def _plot_fairness_heatmap(self, ax, metrics, attr):
        """Plot fairness metrics heatmap."""
        if not metrics['pairwise_metrics']:
            ax.text(0.5, 0.5, 'No pairwise comparisons available',
                   ha='center', va='center', transform=ax.transAxes)
            return

        comparisons = list(metrics['pairwise_metrics'].keys())
        short_names = [c.split('_vs_')[-1] for c in comparisons]

        # Extract metrics
        metric_names = ['demographic_parity_diff', 'equalized_odds_diff',
                       'equal_opportunity_diff', 'precision_diff']
        metric_labels = ['Demographic\nParity', 'Equalized\nOdds',
                        'Equal\nOpportunity', 'Predictive\nParity']

        data = []
        for comp in comparisons:
            row = [metrics['pairwise_metrics'][comp].get(m, 0) for m in metric_names]
            data.append(row)

        data_array = np.array(data)

        # Create heatmap
        im = ax.imshow(data_array, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.2)

        # Set ticks
        ax.set_xticks(np.arange(len(metric_labels)))
        ax.set_yticks(np.arange(len(short_names)))
        ax.set_xticklabels(metric_labels, fontsize=9)
        ax.set_yticklabels(short_names)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Difference', rotation=270, labelpad=15)

        # Add values
        for i in range(len(short_names)):
            for j in range(len(metric_labels)):
                text = ax.text(j, i, f'{data_array[i, j]:.3f}',
                              ha="center", va="center", color="black", fontsize=8)

        ax.set_title('Fairness Metrics Comparison')


if __name__ == "__main__":
    """
    Example usage: Fairness auditing.
    """
    print("Fairness Auditor")
    print("\nExample usage:")
    print("  from src.fairness.metrics import FairnessAuditor")
    print("  auditor = FairnessAuditor(['gender', 'race', 'age_group'])")
    print("  metrics = auditor.calculate_fairness_metrics(y_true, y_pred, y_pred_proba, sensitive_df)")
    print("  report = auditor.generate_fairness_report()")
    print("  auditor.plot_fairness_dashboard(output_dir='fairness_plots')")
