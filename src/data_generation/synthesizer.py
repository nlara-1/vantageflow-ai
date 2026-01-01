"""
Transaction synthesizer for VantageFlow AI credit scoring system.

Generates realistic banking transaction histories for borrowers with
temporal patterns, seasonal variations, and employment-specific behaviors.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import random

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import Session


# Transaction categories
TRANSACTION_CATEGORIES = {
    # Income
    "income_salary": "Income - Salary",
    "income_gig": "Income - Gig/Freelance",
    "income_benefits": "Income - Benefits",
    "income_bonus": "Income - Bonus",
    "income_other": "Income - Other",

    # Fixed expenses
    "expense_rent": "Expense - Rent/Mortgage",
    "expense_utilities": "Expense - Utilities",
    "expense_insurance": "Expense - Insurance",
    "expense_phone": "Expense - Phone/Internet",
    "expense_subscription": "Expense - Subscriptions",

    # Variable expenses
    "expense_groceries": "Expense - Groceries",
    "expense_dining": "Expense - Dining Out",
    "expense_transportation": "Expense - Transportation",
    "expense_gas": "Expense - Gas/Fuel",
    "expense_shopping": "Expense - Shopping",
    "expense_entertainment": "Expense - Entertainment",
    "expense_healthcare": "Expense - Healthcare",
    "expense_discretionary": "Expense - Discretionary",
}


class TransactionSynthesizer:
    """
    Generate realistic transaction histories for borrowers.

    Creates 12 months of banking transactions with realistic patterns
    based on employment type, income level, and seasonal variations.
    """

    def __init__(
        self,
        borrower_profile: Dict,
        start_date: Optional[datetime] = None,
        months: int = 12,
        random_seed: Optional[int] = None
    ):
        """
        Initialize transaction synthesizer for a borrower.

        Args:
            borrower_profile: Dictionary containing borrower information
            start_date: Start date for transaction history (defaults to 12 months ago)
            months: Number of months to generate
            random_seed: Random seed for reproducibility
        """
        self.profile = borrower_profile
        self.months = months

        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

        # Set start date
        if start_date is None:
            self.start_date = datetime.now() - timedelta(days=365)
        else:
            self.start_date = start_date

        self.end_date = self.start_date + timedelta(days=30 * months)

        # Extract key profile attributes
        self.borrower_id = borrower_profile.get("borrower_id")
        self.income = borrower_profile.get("income", 45000)
        self.employment_status = borrower_profile.get("employment_status", "Full-time")

        # Calculate monthly income
        self.monthly_income = self.income / 12

        # Transaction storage
        self.transactions = []

    def generate_transactions(self) -> List[Dict]:
        """
        Generate all transactions for the borrower.

        Returns:
            List of transaction dictionaries
        """
        try:
            # Generate income transactions
            self._generate_income_transactions()

            # Generate fixed expense transactions
            self._generate_rent_transactions()
            self._generate_utilities_transactions()
            self._generate_insurance_transactions()
            self._generate_subscription_transactions()

            # Generate variable expense transactions
            self._generate_grocery_transactions()
            self._generate_transportation_transactions()
            self._generate_discretionary_transactions()

            # Add seasonal patterns
            self._add_seasonal_bonuses()
            self._add_holiday_spending()

            # Sort transactions by date
            self.transactions.sort(key=lambda x: x["transaction_date"])

            return self.transactions

        except Exception as e:
            raise RuntimeError(f"Error generating transactions: {str(e)}")

    def _add_transaction(
        self,
        date: datetime,
        amount: float,
        category: str,
        description: str
    ) -> None:
        """
        Add a transaction to the list.

        Args:
            date: Transaction date
            amount: Transaction amount (positive for income, negative for expenses)
            category: Transaction category
            description: Transaction description
        """
        self.transactions.append({
            "borrower_id": self.borrower_id,
            "transaction_date": date,
            "amount": round(amount, 2),
            "category": category,
            "description": description
        })

    def _generate_income_transactions(self) -> None:
        """Generate income transactions based on employment type."""
        current_date = self.start_date

        while current_date < self.end_date:
            if self.employment_status == "Full-time":
                # Biweekly salary on 1st and 15th
                self._generate_biweekly_salary(current_date)
            elif self.employment_status == "Part-time":
                # Weekly income
                self._generate_weekly_income(current_date)
            elif self.employment_status in ["Self-employed", "Contract"]:
                # Irregular 1099 income
                self._generate_irregular_income(current_date)
            elif self.employment_status == "Unemployed":
                # Monthly benefits (if applicable)
                self._generate_benefits(current_date)

            # Move to next month
            current_date += timedelta(days=30)

    def _generate_biweekly_salary(self, month_start: datetime) -> None:
        """Generate biweekly salary payments for full-time employees."""
        base_salary = self.monthly_income / 2

        # First paycheck (1st of month)
        first_pay_date = month_start.replace(day=1)
        if first_pay_date >= self.start_date and first_pay_date < self.end_date:
            amount = base_salary * np.random.uniform(0.95, 1.05)  # ±5% noise
            self._add_transaction(
                first_pay_date,
                amount,
                "income_salary",
                "Direct Deposit - Salary"
            )

        # Second paycheck (15th of month)
        try:
            second_pay_date = month_start.replace(day=15)
            if second_pay_date >= self.start_date and second_pay_date < self.end_date:
                amount = base_salary * np.random.uniform(0.95, 1.05)  # ±5% noise
                self._add_transaction(
                    second_pay_date,
                    amount,
                    "income_salary",
                    "Direct Deposit - Salary"
                )
        except ValueError:
            # Handle months with < 15 days (shouldn't happen, but safety)
            pass

    def _generate_weekly_income(self, month_start: datetime) -> None:
        """Generate weekly income for part-time employees."""
        weekly_income = self.monthly_income / 4

        current_date = month_start
        month_end = month_start + timedelta(days=30)

        # Generate weekly payments (every Friday)
        days_until_friday = (4 - current_date.weekday()) % 7
        friday = current_date + timedelta(days=days_until_friday)

        while friday < month_end and friday < self.end_date:
            if friday >= self.start_date:
                amount = weekly_income * np.random.uniform(0.90, 1.10)
                self._add_transaction(
                    friday,
                    amount,
                    "income_salary",
                    "Paycheck - Part-time"
                )
            friday += timedelta(days=7)

    def _generate_irregular_income(self, month_start: datetime) -> None:
        """Generate irregular income for self-employed/contract workers."""
        # 1-4 payments per month with variable amounts
        num_payments = np.random.randint(1, 5)

        for _ in range(num_payments):
            # Random day in month
            day_offset = np.random.randint(0, 30)
            payment_date = month_start + timedelta(days=day_offset)

            if payment_date >= self.start_date and payment_date < self.end_date:
                # Variable amount (±40% of average monthly income / num_payments)
                base_amount = self.monthly_income / num_payments
                amount = base_amount * np.random.uniform(0.6, 1.4)

                self._add_transaction(
                    payment_date,
                    amount,
                    "income_gig",
                    "Payment - Freelance/Contract"
                )

    def _generate_benefits(self, month_start: datetime) -> None:
        """Generate monthly benefit payments for unemployed."""
        if self.income > 0:  # Has some income (unemployment benefits)
            first_of_month = month_start.replace(day=1)
            if first_of_month >= self.start_date and first_of_month < self.end_date:
                # Benefits typically ~60% of previous income, capped
                benefit_amount = min(self.monthly_income * 0.6, 2000)
                self._add_transaction(
                    first_of_month,
                    benefit_amount,
                    "income_benefits",
                    "Unemployment Benefits"
                )

    def _generate_rent_transactions(self) -> None:
        """Generate monthly rent/mortgage payments."""
        # Rent is typically 25-35% of monthly income
        monthly_rent = self.monthly_income * np.random.uniform(0.25, 0.35)

        current_date = self.start_date
        while current_date < self.end_date:
            # Rent due on 1st of month
            try:
                rent_date = current_date.replace(day=1)
                if rent_date >= self.start_date and rent_date < self.end_date:
                    # Slight variation (some months paid a day or two late/early)
                    day_variation = np.random.randint(-2, 3)
                    actual_date = rent_date + timedelta(days=day_variation)

                    self._add_transaction(
                        actual_date,
                        -monthly_rent,  # Negative for expense
                        "expense_rent",
                        "Rent Payment"
                    )
            except ValueError:
                pass

            current_date += timedelta(days=30)

    def _generate_utilities_transactions(self) -> None:
        """Generate utility payments with seasonal variation."""
        # Base utility cost (varies by income/lifestyle)
        base_utilities = self.monthly_income * 0.05  # ~5% of income

        current_date = self.start_date
        while current_date < self.end_date:
            # Seasonal multiplier (higher in summer/winter, lower in spring/fall)
            month = current_date.month
            if month in [6, 7, 8]:  # Summer (high AC)
                seasonal_multiplier = 1.4
            elif month in [12, 1, 2]:  # Winter (high heating)
                seasonal_multiplier = 1.3
            else:  # Spring/Fall
                seasonal_multiplier = 0.9

            utility_amount = base_utilities * seasonal_multiplier * np.random.uniform(0.9, 1.1)

            # Due date varies (typically mid-month)
            due_day = np.random.randint(10, 20)
            try:
                utility_date = current_date.replace(day=due_day)
                if utility_date >= self.start_date and utility_date < self.end_date:
                    self._add_transaction(
                        utility_date,
                        -utility_amount,
                        "expense_utilities",
                        "Utility Payment"
                    )
            except ValueError:
                pass

            current_date += timedelta(days=30)

    def _generate_insurance_transactions(self) -> None:
        """Generate monthly insurance payments."""
        # Insurance ~3-5% of monthly income
        monthly_insurance = self.monthly_income * np.random.uniform(0.03, 0.05)

        current_date = self.start_date
        while current_date < self.end_date:
            # Due around 5th of month
            try:
                insurance_date = current_date.replace(day=5)
                if insurance_date >= self.start_date and insurance_date < self.end_date:
                    self._add_transaction(
                        insurance_date,
                        -monthly_insurance,
                        "expense_insurance",
                        "Insurance Premium"
                    )
            except ValueError:
                pass

            current_date += timedelta(days=30)

    def _generate_subscription_transactions(self) -> None:
        """Generate subscription payments (Netflix, Spotify, gym, etc.)."""
        # 2-5 subscriptions
        num_subscriptions = np.random.randint(2, 6)

        for i in range(num_subscriptions):
            # Random subscription cost $5-50
            sub_cost = np.random.uniform(5, 50)

            # Random billing day
            billing_day = np.random.randint(1, 28)

            current_date = self.start_date
            while current_date < self.end_date:
                try:
                    sub_date = current_date.replace(day=billing_day)
                    if sub_date >= self.start_date and sub_date < self.end_date:
                        self._add_transaction(
                            sub_date,
                            -sub_cost,
                            "expense_subscription",
                            f"Subscription Service {i+1}"
                        )
                except ValueError:
                    pass

                current_date += timedelta(days=30)

    def _generate_grocery_transactions(self) -> None:
        """Generate weekly grocery shopping transactions."""
        current_date = self.start_date

        while current_date < self.end_date:
            # Weekly groceries on varying days (mostly weekends)
            for week in range(4):
                # Random day of week (bias toward weekend)
                if np.random.random() < 0.6:  # 60% weekend
                    day_of_week = np.random.choice([5, 6])  # Sat or Sun
                else:
                    day_of_week = np.random.randint(0, 7)

                days_offset = week * 7 + day_of_week
                grocery_date = current_date + timedelta(days=days_offset)

                if grocery_date >= self.start_date and grocery_date < self.end_date:
                    # $50-150 per trip
                    amount = np.random.uniform(50, 150)
                    self._add_transaction(
                        grocery_date,
                        -amount,
                        "expense_groceries",
                        "Grocery Store"
                    )

            current_date += timedelta(days=30)

    def _generate_transportation_transactions(self) -> None:
        """Generate transportation expenses (gas, transit, etc.)."""
        current_date = self.start_date

        while current_date < self.end_date:
            # 2-4 gas/transit purchases per month
            num_transactions = np.random.randint(2, 5)

            for _ in range(num_transactions):
                day_offset = np.random.randint(0, 30)
                trans_date = current_date + timedelta(days=day_offset)

                if trans_date >= self.start_date and trans_date < self.end_date:
                    # $20-80 per transaction
                    amount = np.random.uniform(20, 80)
                    category = "expense_gas" if np.random.random() < 0.7 else "expense_transportation"
                    desc = "Gas Station" if category == "expense_gas" else "Public Transit"

                    self._add_transaction(
                        trans_date,
                        -amount,
                        category,
                        desc
                    )

            current_date += timedelta(days=30)

    def _generate_discretionary_transactions(self) -> None:
        """Generate discretionary spending (dining, shopping, entertainment)."""
        current_date = self.start_date

        # Discretionary budget ~10-20% of income
        monthly_discretionary = self.monthly_income * np.random.uniform(0.10, 0.20)

        while current_date < self.end_date:
            # 5-15 discretionary transactions per month
            num_transactions = np.random.randint(5, 16)

            for _ in range(num_transactions):
                day_offset = np.random.randint(0, 30)
                disc_date = current_date + timedelta(days=day_offset)

                if disc_date >= self.start_date and disc_date < self.end_date:
                    # Allocate from monthly discretionary budget
                    amount = monthly_discretionary / num_transactions * np.random.uniform(0.5, 2.0)

                    # Choose category
                    category_choice = np.random.choice([
                        "expense_dining",
                        "expense_shopping",
                        "expense_entertainment",
                        "expense_discretionary"
                    ], p=[0.4, 0.3, 0.2, 0.1])

                    descriptions = {
                        "expense_dining": "Restaurant",
                        "expense_shopping": "Retail Store",
                        "expense_entertainment": "Entertainment",
                        "expense_discretionary": "Misc Purchase"
                    }

                    self._add_transaction(
                        disc_date,
                        -amount,
                        category_choice,
                        descriptions[category_choice]
                    )

            current_date += timedelta(days=30)

    def _add_seasonal_bonuses(self) -> None:
        """Add Q4 bonuses and year-end payments for eligible employees."""
        if self.employment_status == "Full-time" and self.income > 40000:
            # 40% chance of year-end bonus
            if np.random.random() < 0.4:
                # Find December in date range
                for year_offset in range(2):
                    bonus_year = self.start_date.year + year_offset
                    bonus_date = datetime(bonus_year, 12, 15)

                    if bonus_date >= self.start_date and bonus_date < self.end_date:
                        # Bonus is 5-15% of annual income
                        bonus_amount = self.income * np.random.uniform(0.05, 0.15)
                        self._add_transaction(
                            bonus_date,
                            bonus_amount,
                            "income_bonus",
                            "Year-End Bonus"
                        )

    def _add_holiday_spending(self) -> None:
        """Add increased spending in November/December."""
        current_date = self.start_date

        while current_date < self.end_date:
            if current_date.month in [11, 12]:  # Holiday season
                # 3-8 extra holiday purchases
                num_purchases = np.random.randint(3, 9)

                for _ in range(num_purchases):
                    day_offset = np.random.randint(0, 30)
                    holiday_date = current_date + timedelta(days=day_offset)

                    if holiday_date >= self.start_date and holiday_date < self.end_date:
                        # Holiday purchases $30-200
                        amount = np.random.uniform(30, 200)
                        self._add_transaction(
                            holiday_date,
                            -amount,
                            "expense_shopping",
                            "Holiday Shopping"
                        )

            current_date += timedelta(days=30)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert transactions to DataFrame.

        Returns:
            DataFrame containing all transactions
        """
        if not self.transactions:
            raise ValueError("No transactions generated. Call generate_transactions() first.")

        return pd.DataFrame(self.transactions)

    def export_to_database(self, db_path: str = "data/credit_scoring.db") -> None:
        """
        Export transactions to SQLite database.

        Args:
            db_path: Path to SQLite database
        """
        if not self.transactions:
            raise ValueError("No transactions generated. Call generate_transactions() first.")

        df = self.to_dataframe()
        engine = create_engine(f"sqlite:///{db_path}")

        df.to_sql(
            "transactions",
            engine,
            if_exists="append",
            index=False,
            method="multi"
        )


def generate_transactions_for_borrower(
    borrower_profile: Dict,
    db_path: str = "data/credit_scoring.db",
    months: int = 12
) -> pd.DataFrame:
    """
    Generate and export transactions for a single borrower.

    Args:
        borrower_profile: Borrower profile dictionary
        db_path: Path to SQLite database
        months: Number of months of history to generate

    Returns:
        DataFrame containing generated transactions
    """
    synthesizer = TransactionSynthesizer(borrower_profile, months=months)
    synthesizer.generate_transactions()
    synthesizer.export_to_database(db_path)
    return synthesizer.to_dataframe()


def generate_transactions_batch(
    borrower_profiles: List[Dict],
    db_path: str = "data/credit_scoring.db",
    months: int = 12,
    verbose: bool = True
) -> int:
    """
    Generate transactions for multiple borrowers.

    Args:
        borrower_profiles: List of borrower profile dictionaries
        db_path: Path to SQLite database
        months: Number of months of history to generate
        verbose: Whether to print progress

    Returns:
        Total number of transactions generated
    """
    total_transactions = 0

    for i, profile in enumerate(borrower_profiles):
        try:
            df = generate_transactions_for_borrower(profile, db_path, months)
            total_transactions += len(df)

            if verbose and (i + 1) % 100 == 0:
                print(f"  Processed {i + 1:,} / {len(borrower_profiles):,} borrowers "
                      f"({total_transactions:,} transactions)")

        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to generate transactions for borrower "
                      f"{profile.get('borrower_id', 'unknown')}: {str(e)}")

    if verbose:
        print(f"\n✓ Generated {total_transactions:,} transactions for "
              f"{len(borrower_profiles):,} borrowers")

    return total_transactions


if __name__ == "__main__":
    """
    Example usage: Generate transactions for sample borrowers.
    """
    # Sample borrower profile
    sample_profile = {
        "borrower_id": "test-123",
        "income": 50000,
        "employment_status": "Full-time"
    }

    print("Generating sample transactions...")
    synthesizer = TransactionSynthesizer(sample_profile, months=12)
    synthesizer.generate_transactions()

    df = synthesizer.to_dataframe()
    print(f"\n✓ Generated {len(df):,} transactions")
    print(f"\nSample transactions:")
    print(df.head(20).to_string(index=False))

    # Summary statistics
    print(f"\n{'='*70}")
    print("TRANSACTION SUMMARY")
    print(f"{'='*70}")
    print(f"Total transactions: {len(df):,}")
    print(f"Date range: {df['transaction_date'].min()} to {df['transaction_date'].max()}")
    print(f"\nIncome: ${df[df['amount'] > 0]['amount'].sum():,.2f}")
    print(f"Expenses: ${abs(df[df['amount'] < 0]['amount'].sum()):,.2f}")
    print(f"Net: ${df['amount'].sum():,.2f}")
    print(f"\nTransactions per month: {len(df) / 12:.1f}")
    print(f"\nCategory breakdown:")
    print(df['category'].value_counts().to_string())
