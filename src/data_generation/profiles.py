"""
Borrower profile generation for VantageFlow AI credit scoring system.

This module generates realistic synthetic borrower profiles with diverse
demographic characteristics for fairness testing and model training.
"""

import uuid
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import random

import numpy as np
import pandas as pd
from faker import Faker


# Initialize Faker for realistic synthetic data
fake = Faker()
Faker.seed(42)  # For reproducibility
np.random.seed(42)


# Distribution parameters and weights
EMPLOYMENT_TYPES = {
    "Full-time": 0.55,      # 55% full-time employment
    "Part-time": 0.15,      # 15% part-time
    "Self-employed": 0.12,  # 12% self-employed (1099)
    "Contract": 0.08,       # 8% contract work
    "Unemployed": 0.10      # 10% unemployed
}

EDUCATION_LEVELS = {
    "High School": 0.28,
    "Some College": 0.18,
    "Associate's": 0.10,
    "Bachelor's": 0.30,
    "Master's": 0.11,
    "Doctorate": 0.03
}

REGIONS = {
    "Urban": 0.55,
    "Suburban": 0.30,
    "Rural": 0.15
}

GENDERS = {
    "Male": 0.48,
    "Female": 0.48,
    "Non-binary": 0.02,
    "Prefer not to say": 0.02
}

ETHNICITIES = {
    "White": 0.60,
    "Hispanic/Latino": 0.18,
    "Black/African American": 0.13,
    "Asian": 0.06,
    "Native American": 0.01,
    "Other": 0.01,
    "Prefer not to say": 0.01
}


def generate_income(
    employment_type: str,
    education_level: str,
    age: int,
    mean_income: float = 45000,
    std_income: float = 25000
) -> float:
    """
    Generate realistic annual income based on demographics.

    Uses lognormal distribution with adjustments for employment type,
    education level, and age to create realistic income patterns.

    Args:
        employment_type: Type of employment
        education_level: Highest education level
        age: Borrower's age
        mean_income: Mean annual income
        std_income: Standard deviation of income

    Returns:
        Annual income in dollars
    """
    # Convert mean and std to lognormal parameters
    variance = std_income ** 2
    mu = np.log(mean_income ** 2 / np.sqrt(variance + mean_income ** 2))
    sigma = np.sqrt(np.log(variance / mean_income ** 2 + 1))

    # Base income from lognormal distribution
    base_income = np.random.lognormal(mu, sigma)

    # Adjust for employment type
    employment_multipliers = {
        "Full-time": 1.0,
        "Part-time": 0.4,
        "Self-employed": 0.9,
        "Contract": 0.85,
        "Unemployed": 0.1  # Minimal income (unemployment benefits, etc.)
    }
    base_income *= employment_multipliers.get(employment_type, 1.0)

    # Adjust for education level
    education_multipliers = {
        "High School": 0.7,
        "Some College": 0.8,
        "Associate's": 0.85,
        "Bachelor's": 1.1,
        "Master's": 1.3,
        "Doctorate": 1.5
    }
    base_income *= education_multipliers.get(education_level, 1.0)

    # Adjust for age (career progression)
    if age < 25:
        age_multiplier = 0.6
    elif age < 35:
        age_multiplier = 0.9
    elif age < 50:
        age_multiplier = 1.1
    elif age < 65:
        age_multiplier = 1.0
    else:
        age_multiplier = 0.7  # Retirement or reduced hours

    base_income *= age_multiplier

    # Ensure minimum income
    return max(base_income, 0)


def weighted_choice(choices: Dict[str, float]) -> str:
    """
    Select an item from a dictionary of choices with weights.

    Args:
        choices: Dictionary mapping choice to probability weight

    Returns:
        Selected choice
    """
    items = list(choices.keys())
    weights = list(choices.values())
    return random.choices(items, weights=weights, k=1)[0]


def generate_borrower_profile() -> Dict:
    """
    Generate a single realistic borrower profile.

    Returns:
        Dictionary containing borrower profile data
    """
    # Generate basic demographic information
    age = np.random.randint(18, 80)
    gender = weighted_choice(GENDERS)
    ethnicity = weighted_choice(ETHNICITIES)
    education_level = weighted_choice(EDUCATION_LEVELS)
    employment_type = weighted_choice(EMPLOYMENT_TYPES)
    region = weighted_choice(REGIONS)

    # Generate income based on demographics
    income = generate_income(employment_type, education_level, age)

    # Generate location data
    if region == "Urban":
        zip_code = fake.postcode()
    elif region == "Suburban":
        zip_code = fake.postcode()
    else:  # Rural
        zip_code = fake.postcode()

    # Create borrower profile
    profile = {
        "borrower_id": str(uuid.uuid4()),
        "created_at": fake.date_time_between(start_date="-2y", end_date="now"),
        "age": age,
        "income": round(income, 2),
        "employment_status": employment_type,
        "education_level": education_level,
        "gender": gender,
        "ethnicity": ethnicity,
        "zip_code": zip_code,
        "region": region
    }

    return profile


def generate_borrower_profiles(n_profiles: int = 10000) -> pd.DataFrame:
    """
    Generate multiple borrower profiles.

    Args:
        n_profiles: Number of profiles to generate

    Returns:
        DataFrame containing all borrower profiles
    """
    print(f"Generating {n_profiles:,} borrower profiles...")

    profiles = [generate_borrower_profile() for _ in range(n_profiles)]
    df = pd.DataFrame(profiles)

    print(f"✓ Generated {len(df):,} borrower profiles")

    return df


def validate_distributions(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Validate that generated data matches expected distributions.

    Args:
        df: DataFrame containing borrower profiles

    Returns:
        Dictionary containing validation statistics
    """
    validation_results = {}

    # Income statistics
    validation_results["income"] = {
        "mean": df["income"].mean(),
        "median": df["income"].median(),
        "std": df["income"].std(),
        "min": df["income"].min(),
        "max": df["income"].max()
    }

    # Employment type distribution
    employment_dist = df["employment_status"].value_counts(normalize=True).to_dict()
    validation_results["employment"] = employment_dist

    # Education level distribution
    education_dist = df["education_level"].value_counts(normalize=True).to_dict()
    validation_results["education"] = education_dist

    # Region distribution
    region_dist = df["region"].value_counts(normalize=True).to_dict()
    validation_results["region"] = region_dist

    # Gender distribution
    gender_dist = df["gender"].value_counts(normalize=True).to_dict()
    validation_results["gender"] = gender_dist

    # Ethnicity distribution
    ethnicity_dist = df["ethnicity"].value_counts(normalize=True).to_dict()
    validation_results["ethnicity"] = ethnicity_dist

    # Age statistics
    validation_results["age"] = {
        "mean": df["age"].mean(),
        "median": df["age"].median(),
        "min": df["age"].min(),
        "max": df["age"].max()
    }

    return validation_results


def print_validation_report(validation_results: Dict[str, Dict]) -> None:
    """
    Print a formatted validation report.

    Args:
        validation_results: Validation statistics from validate_distributions()
    """
    print("\n" + "=" * 70)
    print("BORROWER PROFILE VALIDATION REPORT")
    print("=" * 70)

    # Income statistics
    print("\n📊 INCOME DISTRIBUTION")
    print("-" * 70)
    income_stats = validation_results["income"]
    print(f"  Mean:   ${income_stats['mean']:>12,.2f}")
    print(f"  Median: ${income_stats['median']:>12,.2f}")
    print(f"  Std:    ${income_stats['std']:>12,.2f}")
    print(f"  Range:  ${income_stats['min']:>12,.2f} - ${income_stats['max']:>12,.2f}")

    # Age statistics
    print("\n👤 AGE DISTRIBUTION")
    print("-" * 70)
    age_stats = validation_results["age"]
    print(f"  Mean:   {age_stats['mean']:>6.1f} years")
    print(f"  Median: {age_stats['median']:>6.1f} years")
    print(f"  Range:  {age_stats['min']:>6.0f} - {age_stats['max']:>6.0f} years")

    # Employment distribution
    print("\n💼 EMPLOYMENT TYPE DISTRIBUTION")
    print("-" * 70)
    for emp_type, pct in sorted(validation_results["employment"].items(),
                                 key=lambda x: x[1], reverse=True):
        print(f"  {emp_type:<20} {pct:>6.1%}")

    # Education distribution
    print("\n🎓 EDUCATION LEVEL DISTRIBUTION")
    print("-" * 70)
    for edu_level, pct in sorted(validation_results["education"].items(),
                                  key=lambda x: x[1], reverse=True):
        print(f"  {edu_level:<20} {pct:>6.1%}")

    # Region distribution
    print("\n🌍 REGION DISTRIBUTION")
    print("-" * 70)
    for region, pct in sorted(validation_results["region"].items(),
                               key=lambda x: x[1], reverse=True):
        print(f"  {region:<20} {pct:>6.1%}")

    # Gender distribution
    print("\n⚧ GENDER DISTRIBUTION")
    print("-" * 70)
    for gender, pct in sorted(validation_results["gender"].items(),
                               key=lambda x: x[1], reverse=True):
        print(f"  {gender:<20} {pct:>6.1%}")

    # Ethnicity distribution
    print("\n🌐 ETHNICITY DISTRIBUTION")
    print("-" * 70)
    for ethnicity, pct in sorted(validation_results["ethnicity"].items(),
                                  key=lambda x: x[1], reverse=True):
        print(f"  {ethnicity:<30} {pct:>6.1%}")

    print("\n" + "=" * 70)


def export_to_database(
    df: pd.DataFrame,
    db_path: str = "data/credit_scoring.db",
    table_name: str = "borrowers"
) -> None:
    """
    Export borrower profiles to SQLite database.

    Args:
        df: DataFrame containing borrower profiles
        db_path: Path to SQLite database
        table_name: Name of the table to write to
    """
    from sqlalchemy import create_engine

    print(f"\nExporting {len(df):,} profiles to database: {db_path}")

    # Create database engine
    engine = create_engine(f"sqlite:///{db_path}")

    # Prepare data for export (remove region as it's not in the schema)
    df_export = df.drop(columns=["region"], errors="ignore")

    # Export to database
    df_export.to_sql(
        table_name,
        engine,
        if_exists="append",
        index=False,
        method="multi"
    )

    print(f"✓ Successfully exported {len(df):,} borrower profiles to {table_name} table")


def generate_and_export(
    n_profiles: int = 10000,
    db_path: str = "data/credit_scoring.db",
    validate: bool = True,
    print_report: bool = True
) -> pd.DataFrame:
    """
    Generate borrower profiles and export to database (convenience function).

    Args:
        n_profiles: Number of profiles to generate
        db_path: Path to SQLite database
        validate: Whether to validate distributions
        print_report: Whether to print validation report

    Returns:
        DataFrame containing generated profiles
    """
    # Generate profiles
    df = generate_borrower_profiles(n_profiles)

    # Validate distributions
    if validate:
        validation_results = validate_distributions(df)
        if print_report:
            print_validation_report(validation_results)

    # Export to database
    export_to_database(df, db_path)

    return df


if __name__ == "__main__":
    """
    Example usage: Generate 10,000 borrower profiles and export to database.
    """
    # Generate profiles
    profiles_df = generate_and_export(
        n_profiles=10000,
        db_path="data/credit_scoring.db",
        validate=True,
        print_report=True
    )

    # Display sample profiles
    print("\n" + "=" * 70)
    print("SAMPLE BORROWER PROFILES")
    print("=" * 70)
    print(profiles_df.head(10).to_string(index=False))

    # Save to CSV for inspection
    profiles_df.to_csv("data/output/borrower_profiles.csv", index=False)
    print(f"\n✓ Saved profiles to data/output/borrower_profiles.csv")
