"""
Feature engineering pipeline for VantageFlow AI credit scoring.

Extracts 30-40 features from transaction data including income patterns,
spending behavior, financial health indicators, and temporal patterns.
"""

from typing import List, Optional, Dict
from datetime import datetime, timedelta
import warnings

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from src.data.queries import TransactionFeatureExtractor


class FeatureEngineer:
    """
    Feature engineering pipeline for credit scoring.

    Extracts comprehensive features from transaction history including
    income stability, spending patterns, financial health, and temporal features.
    """

    def __init__(self, db_path: str = "data/credit_scoring.db"):
        """
        Initialize feature engineer.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}")
        self.query_extractor = TransactionFeatureExtractor(db_path)

        # Feature names for reference
        self.feature_names = self._get_feature_names()

    def _get_feature_names(self) -> List[str]:
        """
        Get list of all feature names.

        Returns:
            List of feature names
        """
        return [
            # Income features (8)
            "avg_monthly_income",
            "income_std",
            "income_cv",
            "income_trend_3mo",
            "income_trend_6mo",
            "income_frequency_days",
            "min_monthly_income",
            "max_monthly_income",

            # Spending features (8)
            "avg_monthly_spending",
            "spending_std",
            "discretionary_pct",
            "expense_income_ratio",
            "savings_rate",
            "spending_trend_3mo",
            "spending_trend_6mo",
            "spending_volatility",

            # Financial health (7)
            "avg_balance",
            "min_balance",
            "max_balance",
            "overdraft_count_3mo",
            "overdraft_count_6mo",
            "overdraft_rate",
            "avg_net_cashflow",

            # Temporal features (6)
            "transaction_count_per_month",
            "days_since_last_transaction",
            "transaction_count_total",
            "months_of_history",
            "income_transaction_count",
            "expense_transaction_count",

            # Category features (5)
            "rent_to_income_ratio",
            "utilities_to_income_ratio",
            "groceries_to_income_ratio",
            "transportation_to_income_ratio",
            "essential_spending_ratio",

            # Additional derived features (5)
            "income_to_spending_stability_ratio",
            "avg_transaction_size",
            "large_transaction_pct",
            "weekend_spending_pct",
            "financial_health_score",
        ]

    def extract_features(self, borrower_id: str) -> pd.Series:
        """
        Extract all features for a single borrower.

        Args:
            borrower_id: Borrower ID

        Returns:
            Series containing all features
        """
        try:
            # Get raw transaction data
            transactions = self._get_transactions(borrower_id)

            if len(transactions) == 0:
                return self._get_null_features(borrower_id)

            # Extract feature groups
            income_features = self._extract_income_features(transactions)
            spending_features = self._extract_spending_features(transactions)
            health_features = self._extract_financial_health_features(transactions)
            temporal_features = self._extract_temporal_features(transactions)
            category_features = self._extract_category_features(transactions)
            derived_features = self._extract_derived_features(transactions, income_features, spending_features)

            # Combine all features
            features = {
                "borrower_id": borrower_id,
                **income_features,
                **spending_features,
                **health_features,
                **temporal_features,
                **category_features,
                **derived_features
            }

            return pd.Series(features)

        except Exception as e:
            warnings.warn(f"Error extracting features for {borrower_id}: {str(e)}")
            return self._get_null_features(borrower_id)

    def _get_transactions(self, borrower_id: str) -> pd.DataFrame:
        """
        Get all transactions for a borrower.

        Args:
            borrower_id: Borrower ID

        Returns:
            DataFrame with transactions
        """
        query = """
        SELECT *
        FROM transactions
        WHERE borrower_id = :borrower_id
        ORDER BY transaction_date
        """
        df = pd.read_sql_query(query, self.engine, params={"borrower_id": borrower_id})
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])
        return df

    def _extract_income_features(self, transactions: pd.DataFrame) -> Dict:
        """
        Extract income-related features.

        Args:
            transactions: Transaction DataFrame

        Returns:
            Dictionary of income features
        """
        # Filter income transactions
        income_txns = transactions[transactions["amount"] > 0].copy()

        if len(income_txns) == 0:
            return {
                "avg_monthly_income": 0.0,
                "income_std": 0.0,
                "income_cv": 0.0,
                "income_trend_3mo": 0.0,
                "income_trend_6mo": 0.0,
                "income_frequency_days": 0.0,
                "min_monthly_income": 0.0,
                "max_monthly_income": 0.0,
            }

        # Monthly income aggregation
        income_txns["month"] = income_txns["transaction_date"].dt.to_period("M")
        monthly_income = income_txns.groupby("month")["amount"].sum()

        # Basic statistics
        avg_monthly_income = monthly_income.mean()
        income_std = monthly_income.std() if len(monthly_income) > 1 else 0.0
        income_cv = income_std / avg_monthly_income if avg_monthly_income > 0 else 0.0

        # Min/max monthly income
        min_monthly_income = monthly_income.min()
        max_monthly_income = monthly_income.max()

        # Income frequency (average days between income transactions)
        income_dates = income_txns["transaction_date"].sort_values()
        if len(income_dates) > 1:
            income_frequency = (income_dates.max() - income_dates.min()).days / (len(income_dates) - 1)
        else:
            income_frequency = 0.0

        # Income trends (3-month and 6-month)
        income_trend_3mo = self._calculate_trend(monthly_income, window=3)
        income_trend_6mo = self._calculate_trend(monthly_income, window=6)

        return {
            "avg_monthly_income": float(avg_monthly_income),
            "income_std": float(income_std),
            "income_cv": float(income_cv),
            "income_trend_3mo": float(income_trend_3mo),
            "income_trend_6mo": float(income_trend_6mo),
            "income_frequency_days": float(income_frequency),
            "min_monthly_income": float(min_monthly_income),
            "max_monthly_income": float(max_monthly_income),
        }

    def _extract_spending_features(self, transactions: pd.DataFrame) -> Dict:
        """
        Extract spending-related features.

        Args:
            transactions: Transaction DataFrame

        Returns:
            Dictionary of spending features
        """
        # Filter expense transactions
        expense_txns = transactions[transactions["amount"] < 0].copy()

        if len(expense_txns) == 0:
            return {
                "avg_monthly_spending": 0.0,
                "spending_std": 0.0,
                "discretionary_pct": 0.0,
                "expense_income_ratio": 0.0,
                "savings_rate": 0.0,
                "spending_trend_3mo": 0.0,
                "spending_trend_6mo": 0.0,
                "spending_volatility": 0.0,
            }

        # Monthly spending aggregation
        expense_txns["month"] = expense_txns["transaction_date"].dt.to_period("M")
        monthly_spending = expense_txns.groupby("month")["amount"].apply(lambda x: abs(x.sum()))

        # Basic statistics
        avg_monthly_spending = monthly_spending.mean()
        spending_std = monthly_spending.std() if len(monthly_spending) > 1 else 0.0

        # Spending volatility (coefficient of variation)
        spending_volatility = spending_std / avg_monthly_spending if avg_monthly_spending > 0 else 0.0

        # Discretionary spending percentage
        discretionary_categories = [
            "expense_dining", "expense_shopping",
            "expense_entertainment", "expense_discretionary"
        ]
        discretionary_spending = abs(
            expense_txns[expense_txns["category"].isin(discretionary_categories)]["amount"].sum()
        )
        total_spending = abs(expense_txns["amount"].sum())
        discretionary_pct = discretionary_spending / total_spending if total_spending > 0 else 0.0

        # Expense to income ratio and savings rate
        total_income = transactions[transactions["amount"] > 0]["amount"].sum()
        expense_income_ratio = total_spending / total_income if total_income > 0 else 0.0
        savings_rate = (total_income - total_spending) / total_income if total_income > 0 else 0.0

        # Spending trends
        spending_trend_3mo = self._calculate_trend(monthly_spending, window=3)
        spending_trend_6mo = self._calculate_trend(monthly_spending, window=6)

        return {
            "avg_monthly_spending": float(avg_monthly_spending),
            "spending_std": float(spending_std),
            "discretionary_pct": float(discretionary_pct),
            "expense_income_ratio": float(expense_income_ratio),
            "savings_rate": float(savings_rate),
            "spending_trend_3mo": float(spending_trend_3mo),
            "spending_trend_6mo": float(spending_trend_6mo),
            "spending_volatility": float(spending_volatility),
        }

    def _extract_financial_health_features(self, transactions: pd.DataFrame) -> Dict:
        """
        Extract financial health features.

        Args:
            transactions: Transaction DataFrame

        Returns:
            Dictionary of financial health features
        """
        # Calculate running balance
        transactions_sorted = transactions.sort_values("transaction_date").copy()
        transactions_sorted["running_balance"] = transactions_sorted["amount"].cumsum()

        # Balance statistics
        avg_balance = transactions_sorted["running_balance"].mean()
        min_balance = transactions_sorted["running_balance"].min()
        max_balance = transactions_sorted["running_balance"].max()

        # Overdraft detection
        overdraft_periods = (transactions_sorted["running_balance"] < 0).sum()
        total_periods = len(transactions_sorted)
        overdraft_rate = overdraft_periods / total_periods if total_periods > 0 else 0.0

        # Overdraft counts by time period
        recent_3mo = transactions_sorted["transaction_date"].max() - timedelta(days=90)
        recent_6mo = transactions_sorted["transaction_date"].max() - timedelta(days=180)

        overdraft_count_3mo = (
            (transactions_sorted["transaction_date"] >= recent_3mo) &
            (transactions_sorted["running_balance"] < 0)
        ).sum()

        overdraft_count_6mo = (
            (transactions_sorted["transaction_date"] >= recent_6mo) &
            (transactions_sorted["running_balance"] < 0)
        ).sum()

        # Average net cash flow
        avg_net_cashflow = transactions_sorted["amount"].mean()

        return {
            "avg_balance": float(avg_balance),
            "min_balance": float(min_balance),
            "max_balance": float(max_balance),
            "overdraft_count_3mo": int(overdraft_count_3mo),
            "overdraft_count_6mo": int(overdraft_count_6mo),
            "overdraft_rate": float(overdraft_rate),
            "avg_net_cashflow": float(avg_net_cashflow),
        }

    def _extract_temporal_features(self, transactions: pd.DataFrame) -> Dict:
        """
        Extract temporal pattern features.

        Args:
            transactions: Transaction DataFrame

        Returns:
            Dictionary of temporal features
        """
        # Date range
        date_range = (transactions["transaction_date"].max() - transactions["transaction_date"].min()).days
        months_of_history = date_range / 30.0 if date_range > 0 else 0.0

        # Transaction counts
        transaction_count_total = len(transactions)
        transaction_count_per_month = transaction_count_total / months_of_history if months_of_history > 0 else 0.0

        # Days since last transaction
        days_since_last = (datetime.now() - transactions["transaction_date"].max()).days

        # Income vs expense transaction counts
        income_transaction_count = (transactions["amount"] > 0).sum()
        expense_transaction_count = (transactions["amount"] < 0).sum()

        return {
            "transaction_count_per_month": float(transaction_count_per_month),
            "days_since_last_transaction": int(days_since_last),
            "transaction_count_total": int(transaction_count_total),
            "months_of_history": float(months_of_history),
            "income_transaction_count": int(income_transaction_count),
            "expense_transaction_count": int(expense_transaction_count),
        }

    def _extract_category_features(self, transactions: pd.DataFrame) -> Dict:
        """
        Extract category-based spending features.

        Args:
            transactions: Transaction DataFrame

        Returns:
            Dictionary of category features
        """
        # Total income for ratios
        total_income = transactions[transactions["amount"] > 0]["amount"].sum()

        if total_income == 0:
            return {
                "rent_to_income_ratio": 0.0,
                "utilities_to_income_ratio": 0.0,
                "groceries_to_income_ratio": 0.0,
                "transportation_to_income_ratio": 0.0,
                "essential_spending_ratio": 0.0,
            }

        # Category spending
        expenses = transactions[transactions["amount"] < 0]

        rent_spending = abs(expenses[expenses["category"] == "expense_rent"]["amount"].sum())
        utilities_spending = abs(expenses[expenses["category"] == "expense_utilities"]["amount"].sum())
        groceries_spending = abs(expenses[expenses["category"] == "expense_groceries"]["amount"].sum())
        transportation_spending = abs(
            expenses[expenses["category"].isin(["expense_transportation", "expense_gas"])]["amount"].sum()
        )

        # Essential spending (rent + utilities + groceries + transportation)
        essential_spending = rent_spending + utilities_spending + groceries_spending + transportation_spending

        return {
            "rent_to_income_ratio": float(rent_spending / total_income),
            "utilities_to_income_ratio": float(utilities_spending / total_income),
            "groceries_to_income_ratio": float(groceries_spending / total_income),
            "transportation_to_income_ratio": float(transportation_spending / total_income),
            "essential_spending_ratio": float(essential_spending / total_income),
        }

    def _extract_derived_features(
        self,
        transactions: pd.DataFrame,
        income_features: Dict,
        spending_features: Dict
    ) -> Dict:
        """
        Extract derived features from combinations of other features.

        Args:
            transactions: Transaction DataFrame
            income_features: Income features dict
            spending_features: Spending features dict

        Returns:
            Dictionary of derived features
        """
        # Income to spending stability ratio
        income_cv = income_features.get("income_cv", 0.0)
        spending_volatility = spending_features.get("spending_volatility", 0.0)

        if spending_volatility > 0:
            income_to_spending_stability_ratio = income_cv / spending_volatility
        else:
            income_to_spending_stability_ratio = 0.0

        # Average transaction size
        avg_transaction_size = abs(transactions["amount"].mean())

        # Large transaction percentage (>2x average)
        threshold = 2 * avg_transaction_size
        large_transaction_pct = (abs(transactions["amount"]) > threshold).mean()

        # Weekend spending percentage
        transactions["day_of_week"] = transactions["transaction_date"].dt.dayofweek
        weekend_mask = (transactions["day_of_week"] >= 5) & (transactions["amount"] < 0)
        weekend_spending = abs(transactions[weekend_mask]["amount"].sum())
        total_spending = abs(transactions[transactions["amount"] < 0]["amount"].sum())
        weekend_spending_pct = weekend_spending / total_spending if total_spending > 0 else 0.0

        # Financial health score (composite)
        savings_rate = spending_features.get("savings_rate", 0.0)
        overdraft_rate = 0.0  # Will be calculated in health features

        financial_health_score = (
            (savings_rate * 0.4) +  # 40% weight on savings
            ((1 - income_cv) * 0.3) +  # 30% weight on income stability
            ((1 - spending_volatility) * 0.2) +  # 20% weight on spending stability
            ((1 - overdraft_rate) * 0.1)  # 10% weight on avoiding overdrafts
        ) * 100  # Scale to 0-100

        return {
            "income_to_spending_stability_ratio": float(income_to_spending_stability_ratio),
            "avg_transaction_size": float(avg_transaction_size),
            "large_transaction_pct": float(large_transaction_pct),
            "weekend_spending_pct": float(weekend_spending_pct),
            "financial_health_score": float(financial_health_score),
        }

    def _calculate_trend(self, series: pd.Series, window: int) -> float:
        """
        Calculate trend (slope) over a rolling window.

        Args:
            series: Time series data
            window: Window size

        Returns:
            Trend slope
        """
        if len(series) < window:
            return 0.0

        # Get last N periods
        recent = series.tail(window).reset_index(drop=True)

        if len(recent) < 2:
            return 0.0

        # Simple linear regression
        x = np.arange(len(recent))
        y = recent.values

        # Calculate slope
        x_mean = x.mean()
        y_mean = y.mean()

        numerator = ((x - x_mean) * (y - y_mean)).sum()
        denominator = ((x - x_mean) ** 2).sum()

        if denominator == 0:
            return 0.0

        slope = numerator / denominator
        return slope

    def _get_null_features(self, borrower_id: str) -> pd.Series:
        """
        Get null/default features for borrowers with no data.

        Args:
            borrower_id: Borrower ID

        Returns:
            Series with null features
        """
        features = {"borrower_id": borrower_id}
        for feature_name in self.feature_names:
            if "count" in feature_name or "days" in feature_name:
                features[feature_name] = 0
            else:
                features[feature_name] = 0.0

        return pd.Series(features)

    def extract_all_features(
        self,
        borrower_ids: Optional[List[str]] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Extract features for multiple borrowers.

        Args:
            borrower_ids: List of borrower IDs (if None, extracts for all)
            verbose: Whether to print progress

        Returns:
            DataFrame with features for all borrowers
        """
        # Get borrower IDs if not provided
        if borrower_ids is None:
            query = "SELECT DISTINCT borrower_id FROM transactions"
            borrower_ids = pd.read_sql_query(query, self.engine)["borrower_id"].tolist()

        if verbose:
            print(f"Extracting features for {len(borrower_ids):,} borrowers...")

        features_list = []

        for i, borrower_id in enumerate(borrower_ids):
            try:
                features = self.extract_features(borrower_id)
                features_list.append(features)

                if verbose and (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1:,} / {len(borrower_ids):,} borrowers")

            except Exception as e:
                if verbose:
                    warnings.warn(f"Failed to extract features for {borrower_id}: {str(e)}")

        if len(features_list) == 0:
            return pd.DataFrame()

        features_df = pd.DataFrame(features_list)

        if verbose:
            print(f"\n✓ Extracted {len(features_df.columns) - 1} features for {len(features_df):,} borrowers")

        return features_df

    def validate_features(self, features_df: pd.DataFrame) -> Dict:
        """
        Validate feature DataFrame and return quality metrics.

        Args:
            features_df: DataFrame with features

        Returns:
            Dictionary with validation metrics
        """
        validation = {
            "num_borrowers": len(features_df),
            "num_features": len(features_df.columns) - 1,  # Exclude borrower_id
            "null_counts": features_df.isnull().sum().to_dict(),
            "null_percentages": (features_df.isnull().sum() / len(features_df) * 100).to_dict(),
            "feature_stats": {}
        }

        # Statistics for numeric features
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != "borrower_id":
                validation["feature_stats"][col] = {
                    "mean": features_df[col].mean(),
                    "std": features_df[col].std(),
                    "min": features_df[col].min(),
                    "max": features_df[col].max(),
                    "null_count": features_df[col].isnull().sum()
                }

        return validation

    def export_to_csv(
        self,
        features_df: pd.DataFrame,
        output_path: str = "data/output/features.csv"
    ) -> None:
        """
        Export features to CSV file.

        Args:
            features_df: DataFrame with features
            output_path: Output file path
        """
        features_df.to_csv(output_path, index=False)
        print(f"✓ Exported features to {output_path}")


# Convenience functions

def extract_features(
    borrower_id: str,
    db_path: str = "data/credit_scoring.db"
) -> pd.Series:
    """
    Extract features for a single borrower.

    Args:
        borrower_id: Borrower ID
        db_path: Path to database

    Returns:
        Series with features
    """
    engineer = FeatureEngineer(db_path)
    return engineer.extract_features(borrower_id)


def extract_all_features(
    borrower_ids: Optional[List[str]] = None,
    db_path: str = "data/credit_scoring.db",
    output_path: Optional[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Extract features for multiple borrowers.

    Args:
        borrower_ids: List of borrower IDs
        db_path: Path to database
        output_path: Optional CSV output path
        verbose: Whether to print progress

    Returns:
        DataFrame with features
    """
    engineer = FeatureEngineer(db_path)
    features_df = engineer.extract_all_features(borrower_ids, verbose)

    if output_path:
        engineer.export_to_csv(features_df, output_path)

    return features_df


if __name__ == "__main__":
    """
    Example usage: Extract features from transaction data.
    """
    print("Feature Engineering Pipeline")
    print("\nExample usage:")
    print("  from src.features.engineer import extract_all_features")
    print("  features_df = extract_all_features(verbose=True)")
    print("  features_df.to_csv('data/output/features.csv', index=False)")
