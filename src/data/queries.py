"""
SQL queries for feature extraction from transaction data.

Provides functions to extract various features from transaction histories
using SQLite-compatible queries with window functions and aggregations.
"""

from typing import Optional, List
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


class TransactionFeatureExtractor:
    """
    Extract features from transaction data using SQL queries.

    All queries are SQLite-compatible and optimized for performance.
    """

    def __init__(self, db_path: str = "data/credit_scoring.db"):
        """
        Initialize feature extractor.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}")

    def _execute_query(self, query: str, params: dict = None) -> pd.DataFrame:
        """
        Execute SQL query and return DataFrame.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            DataFrame with query results
        """
        try:
            with self.engine.connect() as conn:
                result = pd.read_sql_query(text(query), conn, params=params)
            return result
        except Exception as e:
            raise RuntimeError(f"Query execution failed: {str(e)}")

    def get_monthly_aggregations(self, borrower_id: str) -> pd.DataFrame:
        """
        Get monthly aggregations: income, expenses, net cash flow, transaction count.

        Args:
            borrower_id: Borrower ID

        Returns:
            DataFrame with monthly aggregations
        """
        query = """
        SELECT
            strftime('%Y-%m', transaction_date) AS month,
            SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END) AS total_income,
            SUM(CASE WHEN amount < 0 THEN ABS(amount) ELSE 0 END) AS total_expenses,
            SUM(amount) AS net_cash_flow,
            COUNT(*) AS transaction_count,
            COUNT(CASE WHEN amount > 0 THEN 1 END) AS income_count,
            COUNT(CASE WHEN amount < 0 THEN 1 END) AS expense_count
        FROM transactions
        WHERE borrower_id = :borrower_id
        GROUP BY strftime('%Y-%m', transaction_date)
        ORDER BY month
        """
        return self._execute_query(query, {"borrower_id": borrower_id})

    def get_rolling_features(
        self,
        borrower_id: str,
        window_months: int = 3
    ) -> pd.DataFrame:
        """
        Get rolling window features (e.g., 3-month average income, spending volatility).

        Args:
            borrower_id: Borrower ID
            window_months: Rolling window size in months

        Returns:
            DataFrame with rolling features
        """
        query = f"""
        WITH monthly_stats AS (
            SELECT
                strftime('%Y-%m', transaction_date) AS month,
                SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END) AS income,
                SUM(CASE WHEN amount < 0 THEN ABS(amount) ELSE 0 END) AS expenses
            FROM transactions
            WHERE borrower_id = :borrower_id
            GROUP BY strftime('%Y-%m', transaction_date)
        ),
        ordered_months AS (
            SELECT
                month,
                income,
                expenses,
                ROW_NUMBER() OVER (ORDER BY month) AS row_num
            FROM monthly_stats
        )
        SELECT
            m1.month,
            m1.income,
            m1.expenses,
            AVG(m2.income) AS rolling_avg_income_{window_months}m,
            AVG(m2.expenses) AS rolling_avg_expenses_{window_months}m,
            -- Standard deviation for volatility
            (AVG(m2.expenses * m2.expenses) - AVG(m2.expenses) * AVG(m2.expenses)) AS rolling_var_expenses_{window_months}m
        FROM ordered_months m1
        LEFT JOIN ordered_months m2
            ON m2.row_num BETWEEN m1.row_num - {window_months - 1} AND m1.row_num
        GROUP BY m1.month, m1.income, m1.expenses, m1.row_num
        ORDER BY m1.month
        """
        return self._execute_query(query, {"borrower_id": borrower_id})

    def get_categorical_spending(self, borrower_id: str) -> pd.DataFrame:
        """
        Get spending aggregated by category.

        Args:
            borrower_id: Borrower ID

        Returns:
            DataFrame with spending by category
        """
        query = """
        SELECT
            category,
            COUNT(*) AS transaction_count,
            SUM(ABS(amount)) AS total_amount,
            AVG(ABS(amount)) AS avg_amount,
            MIN(ABS(amount)) AS min_amount,
            MAX(ABS(amount)) AS max_amount,
            SUM(ABS(amount)) * 100.0 / (
                SELECT SUM(ABS(amount))
                FROM transactions
                WHERE borrower_id = :borrower_id AND amount < 0
            ) AS pct_of_total_spending
        FROM transactions
        WHERE borrower_id = :borrower_id AND amount < 0
        GROUP BY category
        ORDER BY total_amount DESC
        """
        return self._execute_query(query, {"borrower_id": borrower_id})

    def get_monthly_categorical_spending(self, borrower_id: str) -> pd.DataFrame:
        """
        Get monthly spending by category (pivot-friendly format).

        Args:
            borrower_id: Borrower ID

        Returns:
            DataFrame with monthly spending by category
        """
        query = """
        SELECT
            strftime('%Y-%m', transaction_date) AS month,
            category,
            COUNT(*) AS transaction_count,
            SUM(ABS(amount)) AS total_amount
        FROM transactions
        WHERE borrower_id = :borrower_id AND amount < 0
        GROUP BY strftime('%Y-%m', transaction_date), category
        ORDER BY month, category
        """
        return self._execute_query(query, {"borrower_id": borrower_id})

    def get_income_stability(self, borrower_id: str) -> pd.DataFrame:
        """
        Calculate income stability metrics including coefficient of variation.

        Args:
            borrower_id: Borrower ID

        Returns:
            DataFrame with income stability metrics
        """
        query = """
        WITH monthly_income AS (
            SELECT
                strftime('%Y-%m', transaction_date) AS month,
                SUM(amount) AS income
            FROM transactions
            WHERE borrower_id = :borrower_id AND amount > 0
            GROUP BY strftime('%Y-%m', transaction_date)
        ),
        income_stats AS (
            SELECT
                AVG(income) AS mean_income,
                COUNT(*) AS month_count,
                -- Calculate standard deviation using the formula: sqrt(E[X^2] - E[X]^2)
                SQRT(AVG(income * income) - AVG(income) * AVG(income)) AS std_income
            FROM monthly_income
        )
        SELECT
            mean_income,
            std_income,
            month_count,
            CASE
                WHEN mean_income > 0 THEN std_income / mean_income
                ELSE NULL
            END AS coefficient_of_variation,
            CASE
                WHEN mean_income > 0 AND std_income / mean_income < 0.2 THEN 'Very Stable'
                WHEN mean_income > 0 AND std_income / mean_income < 0.4 THEN 'Stable'
                WHEN mean_income > 0 AND std_income / mean_income < 0.6 THEN 'Moderate'
                ELSE 'Unstable'
            END AS stability_category
        FROM income_stats
        """
        return self._execute_query(query, {"borrower_id": borrower_id})

    def get_overdraft_periods(self, borrower_id: str) -> pd.DataFrame:
        """
        Detect overdraft periods (negative cumulative balance).

        Args:
            borrower_id: Borrower ID

        Returns:
            DataFrame with overdraft statistics
        """
        query = """
        WITH cumulative_balance AS (
            SELECT
                transaction_date,
                amount,
                SUM(amount) OVER (
                    ORDER BY transaction_date, transaction_id
                ) AS running_balance
            FROM transactions
            WHERE borrower_id = :borrower_id
            ORDER BY transaction_date, transaction_id
        ),
        overdraft_flags AS (
            SELECT
                transaction_date,
                running_balance,
                CASE WHEN running_balance < 0 THEN 1 ELSE 0 END AS is_overdraft
            FROM cumulative_balance
        )
        SELECT
            COUNT(*) AS total_periods,
            SUM(is_overdraft) AS overdraft_periods,
            CAST(SUM(is_overdraft) AS FLOAT) / COUNT(*) AS overdraft_rate,
            MIN(running_balance) AS min_balance
        FROM overdraft_flags
        """
        return self._execute_query(query, {"borrower_id": borrower_id})

    def get_temporal_trends(self, borrower_id: str) -> pd.DataFrame:
        """
        Calculate temporal trends (income and spending slopes over time).

        Uses linear regression approximation: slope = covariance(x, y) / variance(x)

        Args:
            borrower_id: Borrower ID

        Returns:
            DataFrame with trend statistics
        """
        query = """
        WITH monthly_data AS (
            SELECT
                strftime('%Y-%m', transaction_date) AS month,
                SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END) AS income,
                SUM(CASE WHEN amount < 0 THEN ABS(amount) ELSE 0 END) AS expenses,
                ROW_NUMBER() OVER (ORDER BY strftime('%Y-%m', transaction_date)) AS month_num
            FROM transactions
            WHERE borrower_id = :borrower_id
            GROUP BY strftime('%Y-%m', transaction_date)
        ),
        stats AS (
            SELECT
                AVG(month_num) AS mean_x,
                AVG(income) AS mean_income,
                AVG(expenses) AS mean_expenses,
                COUNT(*) AS n
            FROM monthly_data
        ),
        calculations AS (
            SELECT
                SUM((md.month_num - s.mean_x) * (md.income - s.mean_income)) AS cov_income,
                SUM((md.month_num - s.mean_x) * (md.expenses - s.mean_expenses)) AS cov_expenses,
                SUM((md.month_num - s.mean_x) * (md.month_num - s.mean_x)) AS var_x,
                s.mean_income,
                s.mean_expenses,
                s.n
            FROM monthly_data md
            CROSS JOIN stats s
        )
        SELECT
            mean_income,
            mean_expenses,
            n AS month_count,
            CASE
                WHEN var_x > 0 THEN cov_income / var_x
                ELSE 0
            END AS income_trend_slope,
            CASE
                WHEN var_x > 0 THEN cov_expenses / var_x
                ELSE 0
            END AS expense_trend_slope,
            CASE
                WHEN cov_income / var_x > 0 THEN 'Increasing'
                WHEN cov_income / var_x < 0 THEN 'Decreasing'
                ELSE 'Stable'
            END AS income_trend_direction,
            CASE
                WHEN cov_expenses / var_x > 0 THEN 'Increasing'
                WHEN cov_expenses / var_x < 0 THEN 'Decreasing'
                ELSE 'Stable'
            END AS expense_trend_direction
        FROM calculations
        """
        return self._execute_query(query, {"borrower_id": borrower_id})

    def get_transaction_patterns(self, borrower_id: str) -> pd.DataFrame:
        """
        Get transaction timing patterns (day of week, time of month).

        Args:
            borrower_id: Borrower ID

        Returns:
            DataFrame with transaction patterns
        """
        query = """
        SELECT
            -- Day of week patterns
            CAST(strftime('%w', transaction_date) AS INTEGER) AS day_of_week,
            COUNT(*) AS transaction_count,
            AVG(CASE WHEN amount > 0 THEN amount END) AS avg_income,
            AVG(CASE WHEN amount < 0 THEN ABS(amount) END) AS avg_expense,
            -- Day of month patterns
            CAST(strftime('%d', transaction_date) AS INTEGER) AS day_of_month,
            SUM(amount) AS net_amount
        FROM transactions
        WHERE borrower_id = :borrower_id
        GROUP BY day_of_week, day_of_month
        ORDER BY day_of_week, day_of_month
        """
        return self._execute_query(query, {"borrower_id": borrower_id})

    def get_all_features(self, borrower_id: str) -> dict:
        """
        Get all features for a borrower in a single call.

        Args:
            borrower_id: Borrower ID

        Returns:
            Dictionary containing all feature DataFrames
        """
        features = {
            "monthly_aggregations": self.get_monthly_aggregations(borrower_id),
            "rolling_features_3m": self.get_rolling_features(borrower_id, window_months=3),
            "rolling_features_6m": self.get_rolling_features(borrower_id, window_months=6),
            "categorical_spending": self.get_categorical_spending(borrower_id),
            "monthly_categorical_spending": self.get_monthly_categorical_spending(borrower_id),
            "income_stability": self.get_income_stability(borrower_id),
            "overdraft_periods": self.get_overdraft_periods(borrower_id),
            "temporal_trends": self.get_temporal_trends(borrower_id),
            "transaction_patterns": self.get_transaction_patterns(borrower_id)
        }
        return features

    def get_feature_summary(self, borrower_id: str) -> pd.DataFrame:
        """
        Get a single-row summary of key features for modeling.

        Args:
            borrower_id: Borrower ID

        Returns:
            Single-row DataFrame with key features
        """
        query = """
        WITH monthly_data AS (
            SELECT
                strftime('%Y-%m', transaction_date) AS month,
                SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END) AS income,
                SUM(CASE WHEN amount < 0 THEN ABS(amount) ELSE 0 END) AS expenses
            FROM transactions
            WHERE borrower_id = :borrower_id
            GROUP BY strftime('%Y-%m', transaction_date)
        ),
        basic_stats AS (
            SELECT
                COUNT(DISTINCT month) AS num_months,
                AVG(income) AS avg_monthly_income,
                AVG(expenses) AS avg_monthly_expenses,
                AVG(income - expenses) AS avg_monthly_net,
                SQRT(AVG(income * income) - AVG(income) * AVG(income)) AS std_income,
                SQRT(AVG(expenses * expenses) - AVG(expenses) * AVG(expenses)) AS std_expenses
            FROM monthly_data
        ),
        category_stats AS (
            SELECT
                SUM(CASE WHEN category IN ('expense_dining', 'expense_shopping', 'expense_entertainment', 'expense_discretionary')
                    THEN ABS(amount) ELSE 0 END) AS discretionary_spending,
                SUM(CASE WHEN amount < 0 THEN ABS(amount) ELSE 0 END) AS total_spending
            FROM transactions
            WHERE borrower_id = :borrower_id
        ),
        cumulative_data AS (
            SELECT
                SUM(amount) OVER (ORDER BY transaction_date, transaction_id) AS running_balance
            FROM transactions
            WHERE borrower_id = :borrower_id
        ),
        overdraft_stats AS (
            SELECT
                SUM(CASE WHEN running_balance < 0 THEN 1 ELSE 0 END) AS overdraft_count,
                COUNT(*) AS total_transactions
            FROM cumulative_data
        )
        SELECT
            :borrower_id AS borrower_id,
            bs.num_months,
            bs.avg_monthly_income,
            bs.avg_monthly_expenses,
            bs.avg_monthly_net,
            bs.std_income,
            bs.std_expenses,
            CASE
                WHEN bs.avg_monthly_income > 0 THEN bs.std_income / bs.avg_monthly_income
                ELSE NULL
            END AS income_coefficient_of_variation,
            CASE
                WHEN bs.avg_monthly_income > 0
                THEN (bs.avg_monthly_income - bs.avg_monthly_expenses) / bs.avg_monthly_income
                ELSE NULL
            END AS savings_rate,
            CASE
                WHEN cs.total_spending > 0
                THEN cs.discretionary_spending / cs.total_spending
                ELSE NULL
            END AS discretionary_pct,
            CAST(os.overdraft_count AS FLOAT) / bs.num_months AS avg_overdrafts_per_month
        FROM basic_stats bs
        CROSS JOIN category_stats cs
        CROSS JOIN overdraft_stats os
        """
        return self._execute_query(query, {"borrower_id": borrower_id})


# Convenience functions for quick access

def get_monthly_aggregations(
    borrower_id: str,
    db_path: str = "data/credit_scoring.db"
) -> pd.DataFrame:
    """
    Get monthly aggregations for a borrower.

    Args:
        borrower_id: Borrower ID
        db_path: Path to SQLite database

    Returns:
        DataFrame with monthly aggregations
    """
    extractor = TransactionFeatureExtractor(db_path)
    return extractor.get_monthly_aggregations(borrower_id)


def get_feature_summary(
    borrower_id: str,
    db_path: str = "data/credit_scoring.db"
) -> pd.DataFrame:
    """
    Get feature summary for a borrower.

    Args:
        borrower_id: Borrower ID
        db_path: Path to SQLite database

    Returns:
        Single-row DataFrame with key features
    """
    extractor = TransactionFeatureExtractor(db_path)
    return extractor.get_feature_summary(borrower_id)


def get_all_features(
    borrower_id: str,
    db_path: str = "data/credit_scoring.db"
) -> dict:
    """
    Get all features for a borrower.

    Args:
        borrower_id: Borrower ID
        db_path: Path to SQLite database

    Returns:
        Dictionary containing all feature DataFrames
    """
    extractor = TransactionFeatureExtractor(db_path)
    return extractor.get_all_features(borrower_id)


def extract_features_batch(
    borrower_ids: List[str],
    db_path: str = "data/credit_scoring.db",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Extract feature summaries for multiple borrowers.

    Args:
        borrower_ids: List of borrower IDs
        db_path: Path to SQLite database
        verbose: Whether to print progress

    Returns:
        DataFrame with features for all borrowers
    """
    extractor = TransactionFeatureExtractor(db_path)
    features_list = []

    for i, borrower_id in enumerate(borrower_ids):
        try:
            features = extractor.get_feature_summary(borrower_id)
            features_list.append(features)

            if verbose and (i + 1) % 100 == 0:
                print(f"  Processed {i + 1:,} / {len(borrower_ids):,} borrowers")

        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to extract features for {borrower_id}: {str(e)}")

    if len(features_list) == 0:
        return pd.DataFrame()

    result = pd.concat(features_list, ignore_index=True)

    if verbose:
        print(f"\n✓ Extracted features for {len(result):,} borrowers")

    return result


if __name__ == "__main__":
    """
    Example usage: Extract features from transaction data.
    """
    print("Transaction Feature Extractor")
    print("\nExample usage:")
    print("  from src.data.queries import TransactionFeatureExtractor")
    print("  extractor = TransactionFeatureExtractor('data/credit_scoring.db')")
    print("  features = extractor.get_all_features('borrower-123')")
    print("  summary = extractor.get_feature_summary('borrower-123')")
