"""
Credit risk label generation for VantageFlow AI.

Assigns credit default labels based on financial health metrics derived
from transaction history, including income stability, savings rate,
overdraft frequency, and spending discipline.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from scipy import stats


class CreditRiskLabeler:
    """
    Assign credit risk labels based on transaction behavior.

    Analyzes transaction patterns to compute financial health scores
    and assign default risk labels with realistic distributions.
    """

    def __init__(
        self,
        default_rate: float = 0.20,
        noise_level: float = 0.05,
        random_seed: Optional[int] = None
    ):
        """
        Initialize credit risk labeler.

        Args:
            default_rate: Target default rate (bottom N% labeled as default)
            noise_level: Amount of random noise to add (0-1)
            random_seed: Random seed for reproducibility
        """
        self.default_rate = default_rate
        self.noise_level = noise_level

        if random_seed is not None:
            np.random.seed(random_seed)

        # Component weights for overall score
        self.weights = {
            "income_stability": 0.30,
            "savings_rate": 0.35,
            "overdraft_frequency": 0.20,
            "spending_discipline": 0.15
        }

    def calculate_income_stability(self, transactions: pd.DataFrame) -> float:
        """
        Calculate income stability score using coefficient of variation.

        Lower CV = more stable income = better score.

        Args:
            transactions: DataFrame of transactions

        Returns:
            Income stability score (0-100, higher is better)
        """
        # Filter income transactions
        income_txns = transactions[transactions["amount"] > 0]

        if len(income_txns) < 2:
            return 0.0  # Not enough data

        # Group by month and sum income
        income_txns["month"] = pd.to_datetime(income_txns["transaction_date"]).dt.to_period("M")
        monthly_income = income_txns.groupby("month")["amount"].sum()

        if len(monthly_income) < 2:
            return 0.0

        # Calculate coefficient of variation
        mean_income = monthly_income.mean()
        std_income = monthly_income.std()

        if mean_income == 0:
            return 0.0

        cv = std_income / mean_income

        # Convert to score (0-100)
        # CV of 0 = perfect stability = 100
        # CV of 0.5+ = very unstable = 0
        score = max(0, min(100, 100 * (1 - cv / 0.5)))

        return score

    def calculate_savings_rate(self, transactions: pd.DataFrame) -> float:
        """
        Calculate savings rate: (income - spending) / income.

        >10% savings rate is considered good.

        Args:
            transactions: DataFrame of transactions

        Returns:
            Savings rate score (0-100, higher is better)
        """
        total_income = transactions[transactions["amount"] > 0]["amount"].sum()
        total_expenses = abs(transactions[transactions["amount"] < 0]["amount"].sum())

        if total_income == 0:
            return 0.0

        savings = total_income - total_expenses
        savings_rate = savings / total_income

        # Convert to score (0-100)
        # 20%+ savings rate = 100
        # 10% savings rate = 50
        # 0% savings rate = 25
        # Negative savings = 0
        if savings_rate >= 0.20:
            score = 100
        elif savings_rate >= 0.10:
            score = 50 + (savings_rate - 0.10) * 500  # Linear from 50 to 100
        elif savings_rate >= 0:
            score = 25 + savings_rate * 250  # Linear from 25 to 50
        else:
            score = max(0, 25 + savings_rate * 50)  # Penalty for negative savings

        return score

    def calculate_overdraft_frequency(self, transactions: pd.DataFrame) -> float:
        """
        Calculate overdraft frequency based on spending patterns.

        Estimates overdrafts by looking for patterns of large expenses
        followed by income (proxy for actual overdraft data).

        Args:
            transactions: DataFrame of transactions

        Returns:
            Overdraft frequency score (0-100, higher is better)
        """
        # Sort by date
        df = transactions.sort_values("transaction_date").copy()
        df["cumulative_balance"] = df["amount"].cumsum()

        # Count negative balance periods (proxy for overdrafts)
        negative_balance_count = (df["cumulative_balance"] < 0).sum()

        # Calculate months of data
        date_range = (df["transaction_date"].max() - df["transaction_date"].min()).days
        months = max(1, date_range / 30)

        # Overdrafts per month
        overdrafts_per_month = negative_balance_count / months

        # Convert to score (0-100)
        # <1 overdraft/month = 100
        # 1-2 overdrafts/month = 50-100
        # 2-5 overdrafts/month = 25-50
        # 5+ overdrafts/month = 0-25
        if overdrafts_per_month < 1:
            score = 100
        elif overdrafts_per_month < 2:
            score = 50 + (2 - overdrafts_per_month) * 50
        elif overdrafts_per_month < 5:
            score = 25 + (5 - overdrafts_per_month) * 8.33
        else:
            score = max(0, 25 - (overdrafts_per_month - 5) * 5)

        return score

    def calculate_spending_discipline(self, transactions: pd.DataFrame) -> float:
        """
        Calculate spending discipline: discretionary spending as % of total.

        <30% discretionary spending is considered disciplined.

        Args:
            transactions: DataFrame of transactions

        Returns:
            Spending discipline score (0-100, higher is better)
        """
        # Filter expense transactions
        expenses = transactions[transactions["amount"] < 0].copy()

        if len(expenses) == 0:
            return 50.0  # Neutral score if no expenses

        total_expenses = abs(expenses["amount"].sum())

        # Discretionary categories
        discretionary_categories = [
            "expense_dining",
            "expense_shopping",
            "expense_entertainment",
            "expense_discretionary"
        ]

        discretionary_expenses = abs(
            expenses[expenses["category"].isin(discretionary_categories)]["amount"].sum()
        )

        if total_expenses == 0:
            return 50.0

        discretionary_pct = discretionary_expenses / total_expenses

        # Convert to score (0-100)
        # <20% = 100
        # 20-30% = 70-100
        # 30-40% = 40-70
        # 40-50% = 20-40
        # 50%+ = 0-20
        if discretionary_pct < 0.20:
            score = 100
        elif discretionary_pct < 0.30:
            score = 70 + (0.30 - discretionary_pct) * 300
        elif discretionary_pct < 0.40:
            score = 40 + (0.40 - discretionary_pct) * 300
        elif discretionary_pct < 0.50:
            score = 20 + (0.50 - discretionary_pct) * 200
        else:
            score = max(0, 20 - (discretionary_pct - 0.50) * 40)

        return score

    def calculate_financial_health_score(
        self,
        transactions: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate overall financial health score.

        Args:
            transactions: DataFrame of transactions

        Returns:
            Dictionary with component scores and overall score
        """
        # Calculate component scores
        income_stability = self.calculate_income_stability(transactions)
        savings_rate = self.calculate_savings_rate(transactions)
        overdraft_frequency = self.calculate_overdraft_frequency(transactions)
        spending_discipline = self.calculate_spending_discipline(transactions)

        # Calculate weighted overall score
        overall_score = (
            income_stability * self.weights["income_stability"] +
            savings_rate * self.weights["savings_rate"] +
            overdraft_frequency * self.weights["overdraft_frequency"] +
            spending_discipline * self.weights["spending_discipline"]
        )

        # Add random noise to avoid perfect separation
        noise = np.random.normal(0, self.noise_level * 100)
        overall_score = max(0, min(100, overall_score + noise))

        return {
            "income_stability": income_stability,
            "savings_rate": savings_rate,
            "overdraft_frequency": overdraft_frequency,
            "spending_discipline": spending_discipline,
            "overall_score": overall_score
        }

    def assign_label(
        self,
        borrower_id: str,
        transactions: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Assign credit risk label to a borrower.

        Args:
            borrower_id: Borrower ID
            transactions: DataFrame of transactions for this borrower

        Returns:
            Dictionary with label information
        """
        # Calculate financial health score
        scores = self.calculate_financial_health_score(transactions)
        overall_score = scores["overall_score"]

        # Calculate default probability (inverse of score)
        # Score 100 = 5% default prob
        # Score 0 = 95% default prob
        default_probability = 0.05 + (100 - overall_score) / 100 * 0.90

        # Add small random noise to probability
        default_probability += np.random.normal(0, 0.02)
        default_probability = max(0.01, min(0.99, default_probability))

        return {
            "borrower_id": borrower_id,
            "default_probability": default_probability,
            "overall_score": overall_score,
            **scores
        }

    def assign_labels_batch(
        self,
        transactions_df: pd.DataFrame,
        score_threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Assign labels to multiple borrowers based on their transactions.

        Args:
            transactions_df: DataFrame with all transactions
            score_threshold: Optional threshold for default (if None, uses percentile)

        Returns:
            DataFrame with labels for all borrowers
        """
        labels = []

        # Group transactions by borrower
        for borrower_id, txns in transactions_df.groupby("borrower_id"):
            label_info = self.assign_label(borrower_id, txns)
            labels.append(label_info)

        labels_df = pd.DataFrame(labels)

        # Determine threshold for default label
        if score_threshold is None:
            # Use percentile based on target default rate
            score_threshold = labels_df["overall_score"].quantile(self.default_rate)

        # Assign default labels (bottom N% = default)
        labels_df["default_label"] = labels_df["overall_score"] < score_threshold

        # Ensure we're close to target default rate
        actual_default_rate = labels_df["default_label"].mean()
        print(f"Target default rate: {self.default_rate:.1%}")
        print(f"Actual default rate: {actual_default_rate:.1%}")
        print(f"Score threshold: {score_threshold:.2f}")

        return labels_df

    def export_to_database(
        self,
        labels_df: pd.DataFrame,
        db_path: str = "data/credit_scoring.db"
    ) -> None:
        """
        Export labels to SQLite database.

        Args:
            labels_df: DataFrame with labels
            db_path: Path to SQLite database
        """
        # Select columns for database
        db_columns = ["borrower_id", "default_label", "default_probability"]
        export_df = labels_df[db_columns].copy()

        # Add label_date
        export_df["label_date"] = datetime.now()

        # Export to database
        engine = create_engine(f"sqlite:///{db_path}")
        export_df.to_sql(
            "labels",
            engine,
            if_exists="append",
            index=False,
            method="multi"
        )

        print(f"\n✓ Exported {len(export_df):,} labels to database")


def validate_labels(
    labels_df: pd.DataFrame,
    transactions_df: pd.DataFrame
) -> Dict[str, any]:
    """
    Validate label distribution and correlation with features.

    Args:
        labels_df: DataFrame with labels
        transactions_df: DataFrame with transactions

    Returns:
        Dictionary with validation metrics
    """
    validation_results = {}

    # Label distribution
    default_rate = labels_df["default_label"].mean()
    validation_results["default_rate"] = default_rate

    # Score distribution
    validation_results["score_stats"] = {
        "mean": labels_df["overall_score"].mean(),
        "median": labels_df["overall_score"].median(),
        "std": labels_df["overall_score"].std(),
        "min": labels_df["overall_score"].min(),
        "max": labels_df["overall_score"].max()
    }

    # Probability distribution by label
    default_probs = labels_df[labels_df["default_label"] == True]["default_probability"]
    non_default_probs = labels_df[labels_df["default_label"] == False]["default_probability"]

    validation_results["probability_by_label"] = {
        "default_mean": default_probs.mean() if len(default_probs) > 0 else 0,
        "non_default_mean": non_default_probs.mean() if len(non_default_probs) > 0 else 0
    }

    # Component score statistics
    for component in ["income_stability", "savings_rate", "overdraft_frequency", "spending_discipline"]:
        if component in labels_df.columns:
            validation_results[f"{component}_stats"] = {
                "mean": labels_df[component].mean(),
                "default_mean": labels_df[labels_df["default_label"] == True][component].mean(),
                "non_default_mean": labels_df[labels_df["default_label"] == False][component].mean()
            }

    # Correlation between components and default label
    numeric_cols = ["income_stability", "savings_rate", "overdraft_frequency",
                    "spending_discipline", "overall_score", "default_probability"]
    correlations = {}
    for col in numeric_cols:
        if col in labels_df.columns:
            # Point-biserial correlation between numeric and binary
            corr, p_value = stats.pointbiserialr(
                labels_df["default_label"],
                labels_df[col]
            )
            correlations[col] = {"correlation": corr, "p_value": p_value}

    validation_results["correlations"] = correlations

    # Check separation between classes
    if len(default_probs) > 0 and len(non_default_probs) > 0:
        # KS statistic (how well separated are the distributions)
        ks_stat, ks_pval = stats.ks_2samp(default_probs, non_default_probs)
        validation_results["separation"] = {
            "ks_statistic": ks_stat,
            "ks_p_value": ks_pval
        }

    return validation_results


def print_validation_report(validation_results: Dict) -> None:
    """
    Print formatted validation report.

    Args:
        validation_results: Validation metrics from validate_labels()
    """
    print("\n" + "=" * 70)
    print("CREDIT RISK LABEL VALIDATION REPORT")
    print("=" * 70)

    # Default rate
    print(f"\n📊 LABEL DISTRIBUTION")
    print("-" * 70)
    print(f"  Default rate: {validation_results['default_rate']:.1%}")

    # Score statistics
    print(f"\n📈 OVERALL SCORE STATISTICS")
    print("-" * 70)
    score_stats = validation_results["score_stats"]
    print(f"  Mean:   {score_stats['mean']:.2f}")
    print(f"  Median: {score_stats['median']:.2f}")
    print(f"  Std:    {score_stats['std']:.2f}")
    print(f"  Range:  {score_stats['min']:.2f} - {score_stats['max']:.2f}")

    # Probability by label
    print(f"\n🎯 DEFAULT PROBABILITY BY LABEL")
    print("-" * 70)
    prob_stats = validation_results["probability_by_label"]
    print(f"  Default (mean):     {prob_stats['default_mean']:.1%}")
    print(f"  Non-default (mean): {prob_stats['non_default_mean']:.1%}")

    # Component scores
    print(f"\n🔍 COMPONENT SCORES")
    print("-" * 70)
    for component in ["income_stability", "savings_rate", "overdraft_frequency", "spending_discipline"]:
        key = f"{component}_stats"
        if key in validation_results:
            stats = validation_results[key]
            print(f"\n  {component.replace('_', ' ').title()}:")
            print(f"    Overall mean:     {stats['mean']:.2f}")
            print(f"    Default mean:     {stats['default_mean']:.2f}")
            print(f"    Non-default mean: {stats['non_default_mean']:.2f}")

    # Correlations
    print(f"\n📉 CORRELATIONS WITH DEFAULT LABEL")
    print("-" * 70)
    for feature, corr_data in validation_results["correlations"].items():
        corr = corr_data["correlation"]
        p_val = corr_data["p_value"]
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"  {feature:<25} {corr:>7.3f} {sig}")

    # Class separation
    if "separation" in validation_results:
        print(f"\n🎲 CLASS SEPARATION (KS Statistic)")
        print("-" * 70)
        sep = validation_results["separation"]
        print(f"  KS Statistic: {sep['ks_statistic']:.3f}")
        print(f"  P-value:      {sep['ks_p_value']:.6f}")
        if sep['ks_statistic'] > 0.5:
            print(f"  → Excellent separation between classes")
        elif sep['ks_statistic'] > 0.3:
            print(f"  → Good separation between classes")
        else:
            print(f"  → Moderate separation between classes")

    print("\n" + "=" * 70)


def generate_and_validate_labels(
    transactions_df: pd.DataFrame,
    db_path: str = "data/credit_scoring.db",
    default_rate: float = 0.20,
    print_report: bool = True
) -> pd.DataFrame:
    """
    Generate labels, validate, and export to database (convenience function).

    Args:
        transactions_df: DataFrame with all transactions
        db_path: Path to SQLite database
        default_rate: Target default rate
        print_report: Whether to print validation report

    Returns:
        DataFrame with labels
    """
    print(f"Generating credit risk labels for {transactions_df['borrower_id'].nunique():,} borrowers...")

    # Generate labels
    labeler = CreditRiskLabeler(default_rate=default_rate, noise_level=0.05)
    labels_df = labeler.assign_labels_batch(transactions_df)

    print(f"✓ Generated {len(labels_df):,} labels")

    # Validate
    validation_results = validate_labels(labels_df, transactions_df)

    if print_report:
        print_validation_report(validation_results)

    # Export to database
    labeler.export_to_database(labels_df, db_path)

    return labels_df


if __name__ == "__main__":
    """
    Example usage: Load transactions and generate labels.
    """
    # This would normally load from database
    print("Example: Generate labels from transaction data")
    print("\nTo use:")
    print("  from src.data_generation.labels import generate_and_validate_labels")
    print("  labels_df = generate_and_validate_labels(transactions_df)")
