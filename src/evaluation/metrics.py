"""
Comprehensive evaluation metrics for VantageFlow AI credit scoring.

Provides metrics calculation, visualization, and reporting for binary
classification models including AUC-ROC, KS statistic, Gini coefficient,
calibration metrics, and various plots.
"""

import os
from typing import Dict, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    brier_score_loss, log_loss
)
from sklearn.calibration import calibration_curve
from scipy import stats


# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    y_pred_binary: Optional[np.ndarray] = None,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.

    Args:
        y_true: True binary labels (0/1 or False/True)
        y_pred_proba: Predicted probabilities for positive class
        y_pred_binary: Predicted binary labels (optional, will use threshold if not provided)
        threshold: Threshold for converting probabilities to binary (default 0.5)

    Returns:
        Dictionary with all metrics
    """
    # Convert to numpy arrays
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    # Generate binary predictions if not provided
    if y_pred_binary is None:
        y_pred_binary = (y_pred_proba >= threshold).astype(int)
    else:
        y_pred_binary = np.asarray(y_pred_binary)

    metrics = {}

    # AUC-ROC
    try:
        metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        metrics['auc_roc'] = np.nan

    # Gini coefficient (2*AUC - 1)
    metrics['gini'] = 2 * metrics['auc_roc'] - 1 if not np.isnan(metrics['auc_roc']) else np.nan

    # KS Statistic (Kolmogorov-Smirnov)
    metrics['ks_statistic'] = calculate_ks_statistic(y_true, y_pred_proba)

    # Classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
    metrics['precision'] = precision_score(y_true, y_pred_binary, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred_binary, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred_binary, zero_division=0)

    # Confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    metrics['true_positives'] = int(tp)
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)

    # Additional rates
    metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    metrics['true_negative_rate'] = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity

    # Calibration metrics
    metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
    metrics['log_loss'] = log_loss(y_true, y_pred_proba)

    # Expected Calibration Error (ECE)
    metrics['expected_calibration_error'] = calculate_calibration_error(y_true, y_pred_proba)

    # Positive and negative predictive values
    metrics['positive_predictive_value'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Same as precision
    metrics['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0

    return metrics


def calculate_ks_statistic(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Calculate Kolmogorov-Smirnov (KS) statistic.

    KS measures the maximum separation between cumulative distributions
    of predicted probabilities for positive and negative classes.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities

    Returns:
        KS statistic (0-1, higher is better)
    """
    # Separate predictions by class
    y_true = np.asarray(y_true).astype(bool)
    pos_probs = y_pred_proba[y_true]
    neg_probs = y_pred_proba[~y_true]

    if len(pos_probs) == 0 or len(neg_probs) == 0:
        return 0.0

    # Calculate KS statistic
    ks_stat, _ = stats.ks_2samp(pos_probs, neg_probs)

    return ks_stat


def calculate_calibration_error(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Calculate Expected Calibration Error (ECE).

    ECE measures how well predicted probabilities match actual frequencies.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins for calibration

    Returns:
        Expected Calibration Error
    """
    # Get calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins, strategy='uniform'
    )

    # Calculate ECE
    ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))

    return ece


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: Optional[str] = None,
    title: str = "ROC Curve"
) -> plt.Figure:
    """
    Plot ROC (Receiver Operating Characteristic) curve.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        output_path: Path to save plot (optional)
        title: Plot title

    Returns:
        Matplotlib figure
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot ROC curve
    ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')

    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier (AUC = 0.5)')

    # Styling
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()

    # Save if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curve saved: {output_path}")

    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: Optional[str] = None,
    title: str = "Precision-Recall Curve"
) -> plt.Figure:
    """
    Plot Precision-Recall curve.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        output_path: Path to save plot (optional)
        title: Plot title

    Returns:
        Matplotlib figure
    """
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

    # Calculate baseline (random classifier)
    baseline = y_true.sum() / len(y_true)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot PR curve
    ax.plot(recall, precision, linewidth=2, label='Precision-Recall Curve')

    # Plot baseline
    ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1,
               label=f'Random Classifier (AP = {baseline:.4f})')

    # Styling
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()

    # Save if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Precision-Recall curve saved: {output_path}")

    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
    output_path: Optional[str] = None,
    title: str = "Calibration Curve"
) -> plt.Figure:
    """
    Plot calibration curve (reliability diagram).

    Shows how well predicted probabilities match actual frequencies.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins
        output_path: Path to save plot (optional)
        title: Plot title

    Returns:
        Matplotlib figure
    """
    # Calculate calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins, strategy='uniform'
    )

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot calibration curve
    ax.plot(mean_predicted_value, fraction_of_positives, marker='o',
            linewidth=2, markersize=8, label='Model')

    # Plot perfect calibration
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Calibration')

    # Styling
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])

    # Add ECE annotation
    ece = calculate_calibration_error(y_true, y_pred_proba, n_bins)
    ax.text(0.05, 0.95, f'ECE = {ece:.4f}',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Calibration curve saved: {output_path}")

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred_binary: np.ndarray,
    labels: Optional[list] = None,
    output_path: Optional[str] = None,
    title: str = "Confusion Matrix"
) -> plt.Figure:
    """
    Plot confusion matrix as heatmap.

    Args:
        y_true: True binary labels
        y_pred_binary: Predicted binary labels
        labels: Class labels (default: ['Non-Default', 'Default'])
        output_path: Path to save plot (optional)
        title: Plot title

    Returns:
        Matplotlib figure
    """
    if labels is None:
        labels = ['Non-Default (0)', 'Default (1)']

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=labels, yticklabels=labels, ax=ax,
                annot_kws={'size': 14, 'weight': 'bold'})

    # Styling
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add percentages
    total = cm.sum()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            percentage = cm[i, j] / total * 100
            ax.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                   ha='center', va='center', fontsize=10, color='gray')

    plt.tight_layout()

    # Save if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved: {output_path}")

    return fig


def plot_score_distribution(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: Optional[str] = None,
    title: str = "Score Distribution by Class"
) -> plt.Figure:
    """
    Plot distribution of predicted scores by true class.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        output_path: Path to save plot (optional)
        title: Plot title

    Returns:
        Matplotlib figure
    """
    # Separate by class
    y_true = np.asarray(y_true).astype(bool)
    pos_scores = y_pred_proba[y_true]
    neg_scores = y_pred_proba[~y_true]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histograms
    ax.hist(neg_scores, bins=50, alpha=0.6, label='Non-Default (0)', color='blue', density=True)
    ax.hist(pos_scores, bins=50, alpha=0.6, label='Default (1)', color='red', density=True)

    # Styling
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper center', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add KS statistic
    ks_stat = calculate_ks_statistic(y_true, y_pred_proba)
    ax.text(0.05, 0.95, f'KS Statistic = {ks_stat:.4f}',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Score distribution saved: {output_path}")

    return fig


def generate_evaluation_report(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    y_pred_binary: Optional[np.ndarray] = None,
    model_name: str = "Model",
    output_path: Optional[str] = None
) -> str:
    """
    Generate comprehensive evaluation report.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        y_pred_binary: Predicted binary labels (optional)
        model_name: Name of the model
        output_path: Path to save report (optional)

    Returns:
        Report as string
    """
    # Calculate all metrics
    metrics = calculate_all_metrics(y_true, y_pred_proba, y_pred_binary)

    # Build report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(f"MODEL EVALUATION REPORT - {model_name}")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total Samples: {len(y_true):,}")
    report_lines.append(f"Positive Class (Default): {y_true.sum():,} ({y_true.mean():.2%})")
    report_lines.append(f"Negative Class (Non-Default): {(~y_true.astype(bool)).sum():,} ({(~y_true.astype(bool)).mean():.2%})")
    report_lines.append("")

    # Primary Metrics
    report_lines.append("PRIMARY DISCRIMINATION METRICS")
    report_lines.append("-" * 80)
    report_lines.append(f"  AUC-ROC:          {metrics['auc_roc']:.4f}")
    report_lines.append(f"  Gini Coefficient: {metrics['gini']:.4f}")
    report_lines.append(f"  KS Statistic:     {metrics['ks_statistic']:.4f}")
    report_lines.append("")

    # Classification Metrics
    report_lines.append("CLASSIFICATION METRICS (at 0.5 threshold)")
    report_lines.append("-" * 80)
    report_lines.append(f"  Accuracy:         {metrics['accuracy']:.4f}")
    report_lines.append(f"  Precision:        {metrics['precision']:.4f}")
    report_lines.append(f"  Recall:           {metrics['recall']:.4f}")
    report_lines.append(f"  F1 Score:         {metrics['f1_score']:.4f}")
    report_lines.append("")

    # Confusion Matrix
    report_lines.append("CONFUSION MATRIX")
    report_lines.append("-" * 80)
    report_lines.append(f"  True Negatives:   {metrics['true_negatives']:,}")
    report_lines.append(f"  False Positives:  {metrics['false_positives']:,}")
    report_lines.append(f"  False Negatives:  {metrics['false_negatives']:,}")
    report_lines.append(f"  True Positives:   {metrics['true_positives']:,}")
    report_lines.append("")
    report_lines.append(f"  False Positive Rate: {metrics['false_positive_rate']:.4f}")
    report_lines.append(f"  False Negative Rate: {metrics['false_negative_rate']:.4f}")
    report_lines.append(f"  True Negative Rate:  {metrics['true_negative_rate']:.4f} (Specificity)")
    report_lines.append("")

    # Calibration Metrics
    report_lines.append("CALIBRATION METRICS")
    report_lines.append("-" * 80)
    report_lines.append(f"  Brier Score:      {metrics['brier_score']:.4f}")
    report_lines.append(f"  Log Loss:         {metrics['log_loss']:.4f}")
    report_lines.append(f"  Expected Calibration Error (ECE): {metrics['expected_calibration_error']:.4f}")
    report_lines.append("")

    # Predictive Values
    report_lines.append("PREDICTIVE VALUES")
    report_lines.append("-" * 80)
    report_lines.append(f"  Positive Predictive Value (Precision): {metrics['positive_predictive_value']:.4f}")
    report_lines.append(f"  Negative Predictive Value: {metrics['negative_predictive_value']:.4f}")
    report_lines.append("")

    # Interpretation Guide
    report_lines.append("INTERPRETATION GUIDE")
    report_lines.append("-" * 80)
    report_lines.append("  AUC-ROC: 0.5 = random, 0.7-0.8 = acceptable, 0.8-0.9 = excellent, >0.9 = outstanding")
    report_lines.append("  Gini: 2*AUC - 1, ranges from 0 (random) to 1 (perfect)")
    report_lines.append("  KS Statistic: Max separation between classes, >0.4 is good, >0.5 is excellent")
    report_lines.append("  Brier Score: 0 = perfect calibration, lower is better")
    report_lines.append("  ECE: 0 = perfect calibration, lower is better")
    report_lines.append("")

    report_lines.append("=" * 80)

    report_text = "\n".join(report_lines)

    # Save if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"✓ Evaluation report saved: {output_path}")

    return report_text


def generate_all_plots(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    y_pred_binary: np.ndarray,
    output_dir: str = "data/output/evaluation",
    model_name: str = "model"
) -> Dict[str, str]:
    """
    Generate all evaluation plots.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        y_pred_binary: Predicted binary labels
        output_dir: Directory to save plots
        model_name: Model name for file naming

    Returns:
        Dictionary mapping plot type to file path
    """
    os.makedirs(output_dir, exist_ok=True)

    plot_paths = {}

    # ROC Curve
    roc_path = os.path.join(output_dir, f"{model_name}_roc_curve.png")
    plot_roc_curve(y_true, y_pred_proba, output_path=roc_path)
    plot_paths['roc_curve'] = roc_path

    # Precision-Recall Curve
    pr_path = os.path.join(output_dir, f"{model_name}_precision_recall.png")
    plot_precision_recall_curve(y_true, y_pred_proba, output_path=pr_path)
    plot_paths['precision_recall'] = pr_path

    # Calibration Curve
    cal_path = os.path.join(output_dir, f"{model_name}_calibration.png")
    plot_calibration_curve(y_true, y_pred_proba, output_path=cal_path)
    plot_paths['calibration'] = cal_path

    # Confusion Matrix
    cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred_binary, output_path=cm_path)
    plot_paths['confusion_matrix'] = cm_path

    # Score Distribution
    dist_path = os.path.join(output_dir, f"{model_name}_score_distribution.png")
    plot_score_distribution(y_true, y_pred_proba, output_path=dist_path)
    plot_paths['score_distribution'] = dist_path

    print(f"\n✓ Generated {len(plot_paths)} evaluation plots in {output_dir}")

    return plot_paths


if __name__ == "__main__":
    """
    Example usage: Generate evaluation metrics and plots.
    """
    print("Evaluation Metrics Module")
    print("\nExample usage:")
    print("  from src.evaluation.metrics import calculate_all_metrics, generate_evaluation_report")
    print("  metrics = calculate_all_metrics(y_true, y_pred_proba)")
    print("  report = generate_evaluation_report(y_true, y_pred_proba, model_name='XGBoost')")
