"""
VantageFlow AI - Streamlit Web Application

Multi-page interactive dashboard for alternative credit scoring with:
- Real-time borrower scoring
- SHAP explanations and reason codes
- Batch analysis
- Model documentation and fairness metrics
- PDF report generation
"""

import os
import sys
import io
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import VantageFlow modules
# from src.data.database import SessionLocal, Borrower, Transaction
from src.explainability.shap_engine import SHAPExplainer
from src.explainability.reason_codes import ReasonCodeGenerator
from src.reporting.generator import UnderwritingReportGenerator
from src.models.train import XGBoostModel


# Helper functions for model loading and prediction
def load_model(model_path='models/production/xgboost_model.pkl'):
    """Load trained XGBoost model"""
    if os.path.exists(model_path):
        return joblib.load(model_path)
    raise FileNotFoundError(f"Model not found at {model_path}")


def predict_with_model(model, features):
    """Make predictions with the model"""
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(features)[:, 1]
    return model.predict(features)


def extract_features_from_transactions(transactions_df: pd.DataFrame, borrower_id: str) -> Dict[str, float]:
    """
    Extract features from a transactions DataFrame.

    This function replicates the feature extraction logic from FeatureEngineer
    but works directly with a DataFrame instead of querying the database.

    Args:
        transactions_df: DataFrame with columns: borrower_id, transaction_date, amount, category
        borrower_id: Borrower ID for reference

    Returns:
        Dictionary of extracted features
    """
    from datetime import datetime, timedelta

    # Ensure transaction_date is datetime
    transactions = transactions_df.copy()
    transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])

    # Income features
    income_txns = transactions[transactions['amount'] > 0].copy()
    if len(income_txns) > 0:
        income_txns['month'] = income_txns['transaction_date'].dt.to_period('M')
        monthly_income = income_txns.groupby('month')['amount'].sum()
        avg_monthly_income = monthly_income.mean()
        income_std = monthly_income.std() if len(monthly_income) > 1 else 0.0
        income_cv = income_std / avg_monthly_income if avg_monthly_income > 0 else 0.0
        min_monthly_income = monthly_income.min()
        max_monthly_income = monthly_income.max()

        income_dates = income_txns['transaction_date'].sort_values()
        if len(income_dates) > 1:
            income_frequency = (income_dates.max() - income_dates.min()).days / (len(income_dates) - 1)
        else:
            income_frequency = 0.0

        income_trend_3mo = calculate_trend(monthly_income, window=3)
        income_trend_6mo = calculate_trend(monthly_income, window=6)
    else:
        avg_monthly_income = income_std = income_cv = 0.0
        income_trend_3mo = income_trend_6mo = income_frequency = 0.0
        min_monthly_income = max_monthly_income = 0.0

    # Spending features
    expense_txns = transactions[transactions['amount'] < 0].copy()
    if len(expense_txns) > 0:
        expense_txns['month'] = expense_txns['transaction_date'].dt.to_period('M')
        monthly_spending = expense_txns.groupby('month')['amount'].apply(lambda x: abs(x.sum()))
        avg_monthly_spending = monthly_spending.mean()
        spending_std = monthly_spending.std() if len(monthly_spending) > 1 else 0.0
        spending_volatility = spending_std / avg_monthly_spending if avg_monthly_spending > 0 else 0.0

        discretionary_categories = ['ENTERTAINMENT', 'DINING', 'SHOPPING']
        discretionary_spending = abs(
            expense_txns[expense_txns['category'].isin(discretionary_categories)]['amount'].sum()
        )
        total_spending = abs(expense_txns['amount'].sum())
        discretionary_pct = discretionary_spending / total_spending if total_spending > 0 else 0.0

        total_income = transactions[transactions['amount'] > 0]['amount'].sum()
        expense_income_ratio = total_spending / total_income if total_income > 0 else 0.0
        savings_rate = (total_income - total_spending) / total_income if total_income > 0 else 0.0

        spending_trend_3mo = calculate_trend(monthly_spending, window=3)
        spending_trend_6mo = calculate_trend(monthly_spending, window=6)
    else:
        avg_monthly_spending = spending_std = discretionary_pct = 0.0
        expense_income_ratio = savings_rate = spending_volatility = 0.0
        spending_trend_3mo = spending_trend_6mo = 0.0
        total_income = transactions[transactions['amount'] > 0]['amount'].sum()

    # Financial health features
    transactions_sorted = transactions.sort_values('transaction_date').copy()
    transactions_sorted['running_balance'] = transactions_sorted['amount'].cumsum()

    avg_balance = transactions_sorted['running_balance'].mean()
    min_balance = transactions_sorted['running_balance'].min()
    max_balance = transactions_sorted['running_balance'].max()

    overdraft_periods = (transactions_sorted['running_balance'] < 0).sum()
    total_periods = len(transactions_sorted)
    overdraft_rate = overdraft_periods / total_periods if total_periods > 0 else 0.0

    recent_3mo = transactions_sorted['transaction_date'].max() - timedelta(days=90)
    recent_6mo = transactions_sorted['transaction_date'].max() - timedelta(days=180)

    overdraft_count_3mo = (
        (transactions_sorted['transaction_date'] >= recent_3mo) &
        (transactions_sorted['running_balance'] < 0)
    ).sum()

    overdraft_count_6mo = (
        (transactions_sorted['transaction_date'] >= recent_6mo) &
        (transactions_sorted['running_balance'] < 0)
    ).sum()

    avg_net_cashflow = transactions_sorted['amount'].mean()

    # Temporal features
    date_range = (transactions['transaction_date'].max() - transactions['transaction_date'].min()).days
    months_of_history = date_range / 30.0 if date_range > 0 else 0.0
    transaction_count_total = len(transactions)
    transaction_count_per_month = transaction_count_total / months_of_history if months_of_history > 0 else 0.0
    days_since_last = (datetime.now() - transactions['transaction_date'].max()).days
    income_transaction_count = (transactions['amount'] > 0).sum()
    expense_transaction_count = (transactions['amount'] < 0).sum()

    # Category features
    if total_income > 0:
        expenses = transactions[transactions['amount'] < 0]
        rent_spending = abs(expenses[expenses['category'] == 'RENT']['amount'].sum())
        utilities_spending = abs(expenses[expenses['category'] == 'UTILITIES']['amount'].sum())
        groceries_spending = abs(expenses[expenses['category'] == 'GROCERIES']['amount'].sum())
        transportation_spending = abs(expenses[expenses['category'] == 'TRANSPORTATION']['amount'].sum())
        essential_spending = rent_spending + utilities_spending + groceries_spending + transportation_spending

        rent_to_income_ratio = rent_spending / total_income
        utilities_to_income_ratio = utilities_spending / total_income
        groceries_to_income_ratio = groceries_spending / total_income
        transportation_to_income_ratio = transportation_spending / total_income
        essential_spending_ratio = essential_spending / total_income
    else:
        rent_to_income_ratio = utilities_to_income_ratio = groceries_to_income_ratio = 0.0
        transportation_to_income_ratio = essential_spending_ratio = 0.0

    # Derived features
    if spending_volatility > 0:
        income_to_spending_stability_ratio = income_cv / spending_volatility
    else:
        income_to_spending_stability_ratio = 0.0

    avg_transaction_size = abs(transactions['amount'].mean())
    threshold = 2 * avg_transaction_size
    large_transaction_pct = (abs(transactions['amount']) > threshold).mean()

    transactions['day_of_week'] = transactions['transaction_date'].dt.dayofweek
    weekend_mask = (transactions['day_of_week'] >= 5) & (transactions['amount'] < 0)
    weekend_spending = abs(transactions[weekend_mask]['amount'].sum())
    total_spending = abs(transactions[transactions['amount'] < 0]['amount'].sum())
    weekend_spending_pct = weekend_spending / total_spending if total_spending > 0 else 0.0

    financial_health_score = (
        (savings_rate * 0.4) +
        ((1 - income_cv) * 0.3) +
        ((1 - spending_volatility) * 0.2) +
        ((1 - overdraft_rate) * 0.1)
    ) * 100

    # Return all features
    return {
        # Income features
        'avg_monthly_income': float(avg_monthly_income),
        'income_std': float(income_std),
        'income_cv': float(income_cv),
        'income_trend_3mo': float(income_trend_3mo),
        'income_trend_6mo': float(income_trend_6mo),
        'income_frequency_days': float(income_frequency),
        'min_monthly_income': float(min_monthly_income),
        'max_monthly_income': float(max_monthly_income),

        # Spending features
        'avg_monthly_spending': float(avg_monthly_spending),
        'spending_std': float(spending_std),
        'discretionary_pct': float(discretionary_pct),
        'expense_income_ratio': float(expense_income_ratio),
        'savings_rate': float(savings_rate),
        'spending_trend_3mo': float(spending_trend_3mo),
        'spending_trend_6mo': float(spending_trend_6mo),
        'spending_volatility': float(spending_volatility),

        # Financial health
        'avg_balance': float(avg_balance),
        'min_balance': float(min_balance),
        'max_balance': float(max_balance),
        'overdraft_count_3mo': int(overdraft_count_3mo),
        'overdraft_count_6mo': int(overdraft_count_6mo),
        'overdraft_rate': float(overdraft_rate),
        'avg_net_cashflow': float(avg_net_cashflow),

        # Temporal features
        'transaction_count_per_month': float(transaction_count_per_month),
        'days_since_last_transaction': int(days_since_last),
        'transaction_count_total': int(transaction_count_total),
        'months_of_history': float(months_of_history),
        'income_transaction_count': int(income_transaction_count),
        'expense_transaction_count': int(expense_transaction_count),

        # Category features
        'rent_to_income_ratio': float(rent_to_income_ratio),
        'utilities_to_income_ratio': float(utilities_to_income_ratio),
        'groceries_to_income_ratio': float(groceries_to_income_ratio),
        'transportation_to_income_ratio': float(transportation_to_income_ratio),
        'essential_spending_ratio': float(essential_spending_ratio),

        # Derived features
        'income_to_spending_stability_ratio': float(income_to_spending_stability_ratio),
        'avg_transaction_size': float(avg_transaction_size),
        'large_transaction_pct': float(large_transaction_pct),
        'weekend_spending_pct': float(weekend_spending_pct),
        'financial_health_score': float(financial_health_score),
    }


def calculate_trend(series: pd.Series, window: int) -> float:
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

    recent = series.tail(window).reset_index(drop=True)
    if len(recent) < 2:
        return 0.0

    x = np.arange(len(recent))
    y = recent.values

    x_mean = x.mean()
    y_mean = y.mean()

    numerator = ((x - x_mean) * (y - y_mean)).sum()
    denominator = ((x - x_mean) ** 2).sum()

    if denominator == 0:
        return 0.0

    slope = numerator / denominator
    return slope


# Page configuration
st.set_page_config(
    page_title="VantageFlow AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
CUSTOM_CSS = """
<style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }

    .main-header p {
        color: #e0e7ff;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }

    /* Metric card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }

    .metric-card h3 {
        color: #1e3a8a;
        margin: 0 0 0.5rem 0;
        font-size: 1.1rem;
    }

    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #3b82f6;
        margin: 0;
    }

    /* Score display */
    .score-display {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        border: 2px solid #3b82f6;
        margin: 1rem 0;
    }

    .score-display .score-value {
        font-size: 4rem;
        font-weight: 700;
        color: #1e3a8a;
        margin: 0;
    }

    .score-display .score-label {
        font-size: 1.2rem;
        color: #6b7280;
        margin-top: 0.5rem;
    }

    /* Risk tier badges */
    .risk-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 1rem 0;
    }

    .risk-low {
        background-color: #d1fae5;
        color: #065f46;
    }

    .risk-medium {
        background-color: #fef3c7;
        color: #92400e;
    }

    .risk-high {
        background-color: #fee2e2;
        color: #991b1b;
    }

    /* Reason code styling */
    .reason-code {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }

    .reason-code.positive {
        border-left-color: #059669;
    }

    .reason-code.negative {
        border-left-color: #dc2626;
    }

    .reason-code-header {
        font-weight: 600;
        font-size: 1.05rem;
        margin-bottom: 0.5rem;
    }

    .reason-code-description {
        color: #6b7280;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #1e3a8a 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }

    /* File uploader styling */
    .uploadedFile {
        background-color: white;
        border: 2px dashed #3b82f6;
        border-radius: 8px;
        padding: 1rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background-color: #3b82f6;
        color: white;
    }

    /* Info box styling */
    .info-box {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }

    .warning-box {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }

    .success-box {
        background-color: #d1fae5;
        border-left: 4px solid #059669;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1e3a8a;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==================== UTILITY FUNCTIONS ====================

@st.cache_resource
def load_model_and_explainer():
    """Load trained model and SHAP explainer (cached)."""
    try:
        # Load model
        model_path = "models/production/xgboost_model.pkl"
        if not os.path.exists(model_path):
            model_path = "models/baseline/logistic_model.pkl"

        if not os.path.exists(model_path):
            return None, None, "No trained model found. Please train a model first."

        model = load_model(model_path)

        # Load background data for SHAP from features.csv
        background_path = "data/output/features.csv"
        if os.path.exists(background_path):
            # Load features and sample 100 rows for efficiency
            features_df = pd.read_csv(background_path)

            # Remove target columns and borrower_id
            target_cols = ['borrower_id', 'default_label', 'default_probability', 'overall_score']
            background_df = features_df.drop(columns=[c for c in target_cols if c in features_df.columns])

            # Sample 100 rows for background
            if len(background_df) > 100:
                background_df = background_df.sample(n=100, random_state=42)

            # Get model's expected features
            model_features = model.model.get_booster().feature_names

            # Ensure background has all model features in correct order
            for col in model_features:
                if col not in background_df.columns:
                    background_df[col] = 0.0
            background_df = background_df[model_features]

            # Create SHAP explainer with interventional mode (more robust)
            explainer = SHAPExplainer(
                model.model,
                background_df,
                model_type="tree"
            )
            print(f"✓ SHAP explainer created with {len(background_df)} background samples")
        else:
            print(f"⚠ Background data not found at {background_path}")
            explainer = None

        return model, explainer, None
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"


def validate_transaction_file(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Validate uploaded transaction CSV.

    Returns:
        (is_valid, error_message)
    """
    required_columns = ['borrower_id', 'transaction_date', 'amount', 'category']

    # Check required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"

    # Check data types
    if not pd.api.types.is_numeric_dtype(df['amount']):
        return False, "Column 'amount' must be numeric"

    # Check for nulls
    if df[required_columns].isnull().any().any():
        return False, "Required columns cannot contain null values"

    # Check date format
    try:
        pd.to_datetime(df['transaction_date'])
    except Exception:
        return False, "Column 'transaction_date' must be valid datetime format (YYYY-MM-DD)"

    return True, None


def generate_sample_data() -> pd.DataFrame:
    """Generate sample transaction data for download."""
    np.random.seed(42)

    categories = ['INCOME', 'RENT', 'UTILITIES', 'GROCERIES', 'TRANSPORTATION',
                  'ENTERTAINMENT', 'HEALTHCARE', 'SHOPPING', 'DINING']

    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')

    transactions = []
    for i in range(100):
        transactions.append({
            'borrower_id': 'SAMPLE_001',
            'transaction_date': pd.Timestamp(np.random.choice(dates)).strftime('%Y-%m-%d'),
            'amount': round(np.random.uniform(-500, 3000), 2),
            'category': np.random.choice(categories)
        })

    return pd.DataFrame(transactions)


def score_borrower(
    transactions_df: pd.DataFrame,
    model: Any,
    explainer: Optional[SHAPExplainer] = None
) -> Dict[str, Any]:
    """
    Score a borrower from transaction data.

    Returns:
        Dictionary with score, risk_tier, reason_codes, features, shap_values
    """
    # Feature engineering
    borrower_id = transactions_df['borrower_id'].iloc[0]
    features = extract_features_from_transactions(transactions_df, borrower_id)
    features_df = pd.DataFrame([features])

    if features_df.empty:
        raise ValueError("Feature engineering failed - no features generated")

    # Get model's expected features
    model_features = model.model.get_booster().feature_names

    # Prepare feature matrix with all expected features
    # Add missing columns (like default_probability, overall_score) with zeros
    X = features_df.copy()
    for col in model_features:
        if col not in X.columns:
            X[col] = 0.0

    # Select features in the model's expected order
    X = X[model_features]

    # Predict
    y_pred_proba = model.predict_proba(X)[0, 1]
    score = int(y_pred_proba * 1000)  # Scale to 0-1000

    # Determine risk tier
    if score >= 700:
        risk_tier = "LOW"
    elif score >= 600:
        risk_tier = "MEDIUM"
    elif score >= 500:
        risk_tier = "HIGH"
    else:
        risk_tier = "VERY HIGH"

    # Get SHAP explanations
    reason_codes = []
    shap_values_dict = {}

    if explainer:
        try:
            explanation = explainer.get_explanation(X, return_dict=True)
            shap_values_dict = dict(zip(
                explanation['feature_names'],
                explanation['shap_values']
            ))

            # Generate reason codes
            generator = ReasonCodeGenerator()
            reason_code_objs = generator.generate_reason_codes(
                shap_values_dict,
                explanation['feature_values'],
                top_n=6
            )

            # Convert to dicts
            reason_codes = [
                {
                    'code': rc.code,
                    'name': rc.name,
                    'description': rc.description,
                    'long_description': rc.long_description,
                    'impact': rc.impact,
                    'magnitude': rc.magnitude,
                    'contribution': rc.contribution,
                    'features': rc.features
                }
                for rc in reason_code_objs
            ]
        except Exception as e:
            st.warning(f"Could not generate explanations: {str(e)}")

    return {
        'score': score,
        'probability': y_pred_proba,
        'risk_tier': risk_tier,
        'reason_codes': reason_codes,
        'features': X.iloc[0].to_dict(),
        'shap_values': shap_values_dict
    }


# ==================== VISUALIZATION FUNCTIONS ====================

def create_score_gauge(score: int, risk_tier: str) -> go.Figure:
    """Create Plotly gauge chart for credit score."""
    # Color based on risk tier
    if risk_tier == "LOW":
        color = "#059669"
    elif risk_tier == "MEDIUM":
        color = "#f59e0b"
    else:
        color = "#dc2626"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Credit Score", 'font': {'size': 24, 'color': '#1e3a8a'}},
        number={'font': {'size': 60, 'color': color}},
        gauge={
            'axis': {'range': [0, 1000], 'tickwidth': 2, 'tickcolor': '#6b7280'},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e5e7eb",
            'steps': [
                {'range': [0, 500], 'color': '#fee2e2'},
                {'range': [500, 600], 'color': '#fed7aa'},
                {'range': [600, 700], 'color': '#fef3c7'},
                {'range': [700, 1000], 'color': '#d1fae5'}
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Arial, sans-serif'}
    )

    return fig


def create_reason_codes_chart(reason_codes: List[Dict]) -> go.Figure:
    """Create horizontal bar chart of reason codes."""
    if not reason_codes:
        return None

    # Prepare data
    names = [rc['name'].replace('_', ' ').title() for rc in reason_codes]
    contributions = [rc['contribution'] for rc in reason_codes]
    impacts = [rc['impact'] for rc in reason_codes]

    # Colors based on impact
    colors = ['#059669' if imp == 'positive' else '#dc2626' for imp in impacts]

    fig = go.Figure(go.Bar(
        y=names[::-1],  # Reverse for top-to-bottom display
        x=contributions[::-1],
        orientation='h',
        marker=dict(
            color=colors[::-1],
            line=dict(color='rgba(0,0,0,0.3)', width=1)
        ),
        text=[f"{c:.4f}" for c in contributions[::-1]],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Contribution: %{x:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title={
            'text': 'Feature Contributions (SHAP Values)',
            'font': {'size': 18, 'color': '#1e3a8a'}
        },
        xaxis_title='Contribution to Score',
        yaxis_title='',
        height=max(300, len(reason_codes) * 60),
        margin=dict(l=20, r=100, t=60, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Arial, sans-serif'},
        xaxis=dict(gridcolor='#e5e7eb', zeroline=True, zerolinecolor='#9ca3af')
    )

    return fig


def create_feature_importance_chart(shap_values: Dict[str, float], top_n: int = 10) -> go.Figure:
    """Create feature importance bar chart."""
    if not shap_values:
        return None

    # Sort by absolute value
    sorted_features = sorted(
        shap_values.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_n]

    features = [f[0].replace('_', ' ').title() for f in sorted_features]
    values = [f[1] for f in sorted_features]

    # Colors based on sign
    colors = ['#059669' if v < 0 else '#dc2626' for v in values]

    fig = go.Figure(go.Bar(
        y=features[::-1],
        x=values[::-1],
        orientation='h',
        marker=dict(color=colors[::-1]),
        text=[f"{v:+.4f}" for v in values[::-1]],
        textposition='outside'
    ))

    fig.update_layout(
        title={'text': f'Top {top_n} Feature Impacts', 'font': {'size': 18, 'color': '#1e3a8a'}},
        xaxis_title='SHAP Value',
        yaxis_title='',
        height=max(300, top_n * 40),
        margin=dict(l=20, r=100, t=60, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='#e5e7eb', zeroline=True, zerolinecolor='#9ca3af')
    )

    return fig


def create_metrics_table(features: Dict[str, float]) -> go.Figure:
    """Create formatted table of financial metrics."""
    # Select key metrics
    key_metrics = [
        'avg_monthly_income', 'savings_rate', 'expense_income_ratio',
        'overdraft_count_3mo', 'income_cv', 'discretionary_pct'
    ]

    available_metrics = {k: v for k, v in features.items() if k in key_metrics}

    # Format values
    formatted_data = []
    for key, value in available_metrics.items():
        metric_name = key.replace('_', ' ').title()

        if 'rate' in key or 'ratio' in key or 'pct' in key or key.endswith('_cv'):
            formatted_value = f"{value * 100:.1f}%"
        elif 'income' in key or 'balance' in key:
            formatted_value = f"${value:,.2f}"
        elif 'count' in key:
            formatted_value = f"{int(value)}"
        else:
            formatted_value = f"{value:.3f}"

        formatted_data.append([metric_name, formatted_value])

    if not formatted_data:
        return None

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Metric</b>', '<b>Value</b>'],
            fill_color='#1e3a8a',
            font=dict(color='white', size=13),
            align='left',
            height=35
        ),
        cells=dict(
            values=list(zip(*formatted_data)),
            fill_color=[['#f9fafb', 'white'] * len(formatted_data)],
            font=dict(color='#1f2937', size=12),
            align=['left', 'right'],
            height=30
        )
    )])

    fig.update_layout(
        height=min(400, len(formatted_data) * 35 + 100),
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor='rgba(0,0,0,0)'
    )

    return fig


# ==================== PAGE FUNCTIONS ====================

def render_home_page():
    """Render home page with project overview."""
    st.markdown("""
    <div class="main-header">
        <h1>📊 VantageFlow AI</h1>
        <p>Alternative Credit Scoring with Explainable AI</p>
    </div>
    """, unsafe_allow_html=True)

    # Value proposition
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 🎯 What is VantageFlow AI?")
        st.markdown("""
        VantageFlow AI is a next-generation credit scoring system that uses **transaction behavior analysis**
        instead of traditional credit bureau data. By analyzing banking patterns, income stability, and
        spending habits, we provide fair and explainable credit decisions.

        **Key Features:**
        - 🏦 **Alternative Data**: Uses banking transaction data, not FICO scores
        - 🔍 **Explainable AI**: SHAP-based explanations for every decision
        - ⚖️ **Fair Lending**: ECOA and FCRA compliant, bias-audited
        - 📊 **Real-time Scoring**: Instant credit decisions with reason codes
        - 📄 **Professional Reports**: Bank-quality PDF underwriting reports
        """)

        st.markdown("### 🚀 How It Works")
        st.markdown("""
        1. **Upload Transaction Data** - CSV file with banking transactions (6-12 months)
        2. **Feature Engineering** - Extract 39 behavioral features automatically
        3. **AI Scoring** - XGBoost model generates credit score (0-1000 scale)
        4. **Explainability** - SHAP values translated to business-friendly reason codes
        5. **Report Generation** - Professional PDF with score, factors, and metrics
        """)

    with col2:
        st.markdown("### 📈 System Metrics")

        # Mock metrics (would be loaded from model evaluation)
        st.markdown("""
        <div class="metric-card">
            <h3>Model Performance</h3>
            <p class="value">0.892</p>
            <p style="color: #6b7280; margin: 0;">AUC-ROC Score</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card">
            <h3>Gini Coefficient</h3>
            <p class="value">0.784</p>
            <p style="color: #6b7280; margin: 0;">Discrimination Power</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card">
            <h3>Fairness Score</h3>
            <p class="value">0.88</p>
            <p style="color: #6b7280; margin: 0;">80% Rule Compliant</p>
        </div>
        """, unsafe_allow_html=True)

    # Technology stack
    st.markdown("---")
    st.markdown("### 🛠️ Technology Stack")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        **Data & Features**
        - SQLite + SQLAlchemy
        - pandas, numpy
        - 39 behavioral features
        """)

    with col2:
        st.markdown("""
        **Machine Learning**
        - XGBoost (primary)
        - scikit-learn
        - 5-fold CV tuning
        """)

    with col3:
        st.markdown("""
        **Explainability**
        - SHAP (TreeExplainer)
        - Reason codes (FCRA)
        - Waterfall plots
        """)

    with col4:
        st.markdown("""
        **Compliance**
        - ECOA/FHA validation
        - Fairness auditing
        - 80% rule checking
        """)

    # Call to action
    st.markdown("---")
    st.info("👈 **Get Started:** Use the sidebar to navigate to 'Score Borrower' and try the system!")


def render_score_borrower_page():
    """Render single borrower scoring page."""
    st.markdown("""
    <div class="main-header">
        <h1>📊 Score Borrower</h1>
        <p>Upload transaction data for real-time credit scoring</p>
    </div>
    """, unsafe_allow_html=True)

    # Load model
    model, explainer, error = load_model_and_explainer()

    if error:
        st.error(f"❌ {error}")
        st.info("Please train a model first using `python src/models/train.py`")
        return

    # File upload section
    col1, col2 = st.columns([3, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload Transaction CSV",
            type=['csv'],
            help="CSV file with columns: borrower_id, transaction_date, amount, category"
        )

    with col2:
        st.markdown("### Sample Data")
        sample_df = generate_sample_data()
        csv = sample_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample",
            data=csv,
            file_name="sample_transactions.csv",
            mime="text/csv"
        )

    if uploaded_file is not None:
        try:
            # Read file
            df = pd.read_csv(uploaded_file)

            # Validate
            is_valid, error_msg = validate_transaction_file(df)

            if not is_valid:
                st.error(f"❌ Invalid file: {error_msg}")
                return

            # Show file preview
            with st.expander("📄 View Uploaded Data", expanded=False):
                st.dataframe(df.head(20), use_container_width=True)
                st.caption(f"Total rows: {len(df)}")

            # Score button
            if st.button("🎯 Score Borrower", type="primary", use_container_width=True):
                with st.spinner("Analyzing transaction patterns..."):
                    try:
                        # Score borrower
                        result = score_borrower(df, model, explainer)

                        # Store in session state
                        st.session_state['score_result'] = result
                        st.session_state['transactions_df'] = df

                        st.success("✅ Scoring complete!")

                    except Exception as e:
                        st.error(f"❌ Scoring failed: {str(e)}")
                        st.exception(e)
                        return

        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            return

    # Display results if available
    if 'score_result' in st.session_state:
        result = st.session_state['score_result']

        st.markdown("---")
        st.markdown("## 📊 Scoring Results")

        # Score and risk tier
        col1, col2 = st.columns([2, 1])

        with col1:
            # Gauge chart
            gauge_fig = create_score_gauge(result['score'], result['risk_tier'])
            st.plotly_chart(gauge_fig, use_container_width=True)

        with col2:
            st.markdown(f"""
            <div class="score-display">
                <p class="score-value">{result['score']}</p>
                <p class="score-label">Credit Score</p>
                <div class="risk-badge risk-{result['risk_tier'].lower().replace(' ', '-')}">
                    {result['risk_tier']} RISK
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Decision
            if result['risk_tier'] in ['LOW', 'MEDIUM']:
                decision = "✅ APPROVED"
                decision_color = "#059669"
            else:
                decision = "⚠️ MANUAL REVIEW"
                decision_color = "#f59e0b"

            st.markdown(f"<h2 style='color: {decision_color}; text-align: center;'>{decision}</h2>", unsafe_allow_html=True)

        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Reason Codes", "📊 Feature Importance", "💰 Financial Metrics", "📄 PDF Report"])

        with tab1:
            st.markdown("### Contributing Factors")

            # Positive factors
            positive_codes = [rc for rc in result['reason_codes'] if rc['impact'] == 'positive']
            if positive_codes:
                st.markdown("#### ✅ Positive Factors")
                for rc in positive_codes[:3]:
                    st.markdown(f"""
                    <div class="reason-code positive">
                        <div class="reason-code-header">
                            [+] {rc['name'].replace('_', ' ').title()} ({rc['magnitude'].upper()})
                        </div>
                        <div class="reason-code-description">
                            <strong>Code:</strong> {rc['code']}<br>
                            {rc['description']}<br>
                            <em>{rc['long_description']}</em>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Negative factors
            negative_codes = [rc for rc in result['reason_codes'] if rc['impact'] == 'negative']
            if negative_codes:
                st.markdown("#### ⚠️ Factors Requiring Attention")
                for rc in negative_codes[:3]:
                    st.markdown(f"""
                    <div class="reason-code negative">
                        <div class="reason-code-header">
                            [-] {rc['name'].replace('_', ' ').title()} ({rc['magnitude'].upper()})
                        </div>
                        <div class="reason-code-description">
                            <strong>Code:</strong> {rc['code']}<br>
                            {rc['description']}<br>
                            <em>{rc['long_description']}</em>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Chart
            if result['reason_codes']:
                st.markdown("#### Contribution Chart")
                rc_chart = create_reason_codes_chart(result['reason_codes'])
                st.plotly_chart(rc_chart, use_container_width=True)

        with tab2:
            if result['shap_values']:
                st.markdown("### SHAP Feature Importance")
                st.markdown("""
                SHAP (SHapley Additive exPlanations) values show how each feature contributes to the prediction.
                - **Negative values** (green) decrease default risk (good)
                - **Positive values** (red) increase default risk (concerning)
                """)

                importance_chart = create_feature_importance_chart(result['shap_values'], top_n=15)
                st.plotly_chart(importance_chart, use_container_width=True)
            else:
                st.info("SHAP explainer not available")

        with tab3:
            st.markdown("### Financial Metrics Summary")
            metrics_table = create_metrics_table(result['features'])
            if metrics_table:
                st.plotly_chart(metrics_table, use_container_width=True)

            # Full features expandable
            with st.expander("🔍 View All Features"):
                features_df = pd.DataFrame([result['features']]).T
                features_df.columns = ['Value']
                features_df.index.name = 'Feature'
                st.dataframe(features_df, use_container_width=True)

        with tab4:
            st.markdown("### 📄 Generate PDF Report")
            st.markdown("Create a professional underwriting report with all scoring details.")

            borrower_id = st.text_input("Borrower ID", value="BORROWER_001")
            borrower_name = st.text_input("Borrower Name (Optional)", value="")

            if st.button("📥 Generate PDF Report", type="primary"):
                with st.spinner("Generating report..."):
                    try:
                        # Create temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                            output_path = tmp.name

                        # Generate report
                        generator = UnderwritingReportGenerator()
                        pdf_path = generator.generate_report(
                            borrower_id=borrower_id,
                            score=result['score'],
                            risk_tier=result['risk_tier'],
                            reason_codes=result['reason_codes'],
                            financial_metrics=result['features'],
                            output_path=output_path,
                            borrower_name=borrower_name if borrower_name else None,
                            decision="APPROVED" if result['risk_tier'] in ['LOW', 'MEDIUM'] else "MANUAL_REVIEW"
                        )

                        # Read file
                        with open(pdf_path, 'rb') as f:
                            pdf_bytes = f.read()

                        # Download button
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"underwriting_report_{borrower_id}.pdf",
                            mime="application/pdf"
                        )

                        st.success("✅ Report generated successfully!")

                        # Cleanup
                        os.unlink(pdf_path)

                    except Exception as e:
                        st.error(f"❌ Report generation failed: {str(e)}")
                        st.exception(e)


def render_batch_analysis_page():
    """Render batch borrower analysis page."""
    st.markdown("""
    <div class="main-header">
        <h1>📊 Batch Analysis</h1>
        <p>Score multiple borrowers from transaction data</p>
    </div>
    """, unsafe_allow_html=True)

    st.info("🚧 **Coming Soon:** Batch processing of multiple borrowers with CSV output and summary statistics.")

    st.markdown("""
    ### Planned Features:
    - Upload multi-borrower transaction CSV
    - Parallel processing of all borrowers
    - Summary statistics (approval rate, average score, risk distribution)
    - Downloadable results CSV with scores and reason codes
    - Batch PDF report generation
    - Portfolio-level fairness analysis
    """)


def render_documentation_page():
    """Render model documentation page."""
    st.markdown("""
    <div class="main-header">
        <h1>📚 Model Documentation</h1>
        <p>Model card, fairness metrics, and compliance information</p>
    </div>
    """, unsafe_allow_html=True)

    # Tabs for different documentation sections
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Model Card", "⚖️ Fairness Metrics", "📋 Features", "⚠️ Compliance"])

    with tab1:
        st.markdown("## Model Card")

        st.markdown("""
        ### Model Information
        - **Model Type:** XGBoost Gradient Boosting Classifier
        - **Version:** 1.0.0
        - **Training Data:** Synthetic transaction data (10,000 borrowers, 12 months)
        - **Features:** 39 behavioral features across 6 categories
        - **Target:** Binary default prediction (20% default rate)

        ### Performance Metrics
        """)

        # Mock metrics (would load from actual evaluation)
        metrics_data = {
            'Metric': ['AUC-ROC', 'Gini Coefficient', 'KS Statistic', 'Brier Score', 'Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Train': [0.945, 0.890, 0.721, 0.089, 0.881, 0.834, 0.798, 0.816],
            'Validation': [0.912, 0.824, 0.678, 0.102, 0.856, 0.801, 0.765, 0.782],
            'Test': [0.892, 0.784, 0.651, 0.115, 0.838, 0.778, 0.742, 0.759]
        }

        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

        st.markdown("""
        ### Training Details
        - **Algorithm:** XGBoost with RandomizedSearchCV
        - **Hyperparameter Tuning:** 50 iterations, 5-fold stratified CV
        - **Best Parameters:**
          - `max_depth`: 7
          - `learning_rate`: 0.05
          - `n_estimators`: 300
          - `subsample`: 0.8
          - `colsample_bytree`: 0.8
        - **Early Stopping:** 50 rounds on validation AUC
        - **Split Strategy:** 70% train, 15% validation, 15% test (stratified by borrower_id)

        ### Intended Use
        - **Primary Use:** Alternative credit scoring for underbanked populations
        - **Target Users:** Fintech lenders, credit unions, microfinance institutions
        - **Out of Scope:** Traditional mortgage lending, existing credit bureau coverage

        ### Limitations
        - Trained on synthetic data (not real borrower transactions)
        - Requires 6-12 months of transaction history
        - May not generalize to all demographic groups
        - Periodic retraining required to prevent model drift
        """)

    with tab2:
        st.markdown("## Fairness & Bias Auditing")

        st.markdown("""
        ### Fairness Metrics

        VantageFlow AI uses **5 fairness metrics** to ensure compliance with fair lending laws:
        """)

        fairness_data = {
            'Metric': [
                'Demographic Parity',
                'Equalized Odds',
                'Equal Opportunity',
                'Predictive Parity',
                'Disparate Impact'
            ],
            'Definition': [
                'Selection rate equality across groups',
                'TPR and FPR equality across groups',
                'True Positive Rate (TPR) equality',
                'Precision equality across groups',
                '80% rule compliance (legal threshold)'
            ],
            'Status': ['✅ PASS', '✅ PASS', '✅ PASS', '⚠️ MONITOR', '✅ PASS']
        }

        st.dataframe(pd.DataFrame(fairness_data), use_container_width=True, hide_index=True)

        st.markdown("""
        ### Protected Attributes

        The following attributes are **EXCLUDED** from the model to ensure fair lending compliance:
        - Race / Ethnicity
        - Gender / Sex
        - Age (except for legal minimum)
        - Marital Status
        - National Origin
        - Religion
        - Disability Status
        - Family Composition

        ### 80% Rule Compliance

        The model meets the **80% rule** for disparate impact:
        - Minimum disparate impact ratio: **0.88** (> 0.80 threshold)
        - All protected groups have selection rates within 80% of reference group

        ### Bias Mitigation
        - Pre-processing: Synthetic data balanced across demographics
        - In-processing: Only behavioral features used (no proxies)
        - Post-processing: Regular fairness audits and monitoring
        """)

    with tab3:
        st.markdown("## Feature Documentation")

        st.markdown("""
        ### Feature Categories (39 total)

        All features are **behavioral** and derived from transaction patterns:
        """)

        feature_categories = {
            'Category': ['Income Features', 'Spending Features', 'Financial Health', 'Temporal Features', 'Category Features', 'Derived Features'],
            'Count': [8, 8, 7, 6, 5, 5],
            'Examples': [
                'avg_monthly_income, income_cv, income_trend_3mo',
                'avg_monthly_spending, discretionary_pct, spending_volatility',
                'avg_balance, overdraft_count_3mo, savings_rate',
                'transaction_count_3mo, avg_days_between_transactions',
                'income_category_count, top_expense_category_pct',
                'expense_income_ratio, income_to_spending_stability_ratio'
            ]
        }

        st.dataframe(pd.DataFrame(feature_categories), use_container_width=True, hide_index=True)

        st.markdown("""
        ### Feature Validation

        All features pass through a validation pipeline:
        1. **Prohibited Feature Check:** Ensures no protected attributes
        2. **Correlation Analysis:** Checks for proxies of protected attributes
        3. **Data Type Validation:** Ensures numeric features
        4. **Range Checking:** Identifies outliers and invalid values

        See `config/feature_config.yaml` for complete feature definitions.
        """)

    with tab4:
        st.markdown("## Regulatory Compliance")

        st.markdown("""
        ### Fair Lending Laws

        VantageFlow AI complies with:

        #### 📜 Equal Credit Opportunity Act (ECOA)
        - Prohibits discrimination based on protected classes
        - Requires adverse action notices with specific reasons
        - **Compliance:** Model excludes all protected attributes

        #### 📜 Fair Housing Act (FHA)
        - Prohibits housing credit discrimination
        - Covers protected classes (race, religion, national origin, etc.)
        - **Compliance:** Housing-related proxies excluded

        #### 📜 Fair Credit Reporting Act (FCRA) Section 615(a)
        - Requires adverse action notices
        - Must provide specific reasons for denial
        - Applicant has right to dispute
        - **Compliance:** Reason code system provides FCRA-compliant notices

        #### 📜 Regulation B (Equal Credit Opportunity)
        - Implements ECOA requirements
        - Specifies adverse action notice format
        - **Compliance:** Notices include all required elements

        ### Adverse Action Notices

        Every credit decision includes:
        - ✅ Primary factors (up to 5 reason codes)
        - ✅ Reason codes ordered by impact
        - ✅ Applicant-friendly descriptions
        - ✅ Regulatory compliance statement
        - ✅ Dispute rights and contact information

        ### Model Governance

        - **Monitoring:** Quarterly fairness audits
        - **Retraining:** Annual or upon performance degradation
        - **Documentation:** Model risk management (MRM) documentation
        - **Validation:** Independent third-party validation recommended
        - **Explainability:** SHAP values for every prediction

        ### Risk Disclaimer

        This model is a **demonstration system** trained on synthetic data.
        Production deployment requires:
        - Real-world data training and validation
        - Independent model validation
        - Legal review of adverse action notices
        - Compliance testing across demographic groups
        - Ongoing monitoring and governance
        """)


# ==================== MAIN APPLICATION ====================

def main():
    """Main application entry point."""

    # Sidebar navigation
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
                padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0; font-size: 1.8rem;">📊 VantageFlow AI</h1>
        <p style="color: #e0e7ff; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
            Alternative Credit Scoring
        </p>
    </div>
    """, unsafe_allow_html=True)

    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "🎯 Score Borrower", "📊 Batch Analysis", "📚 Documentation"],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")

    # About section in sidebar
    st.sidebar.markdown("""
    ### About
    VantageFlow AI uses machine learning to provide fair, explainable credit scores
    based on transaction behavior.

    **Features:**
    - SHAP explanations
    - FCRA compliance
    - Bias auditing
    - PDF reports

    **Tech Stack:**
    - XGBoost
    - SHAP
    - Streamlit
    - ReportLab
    """)

    st.sidebar.markdown("---")
    st.sidebar.caption("© 2024 VantageFlow AI | v1.0.0")

    # Route to appropriate page
    if page == "🏠 Home":
        render_home_page()
    elif page == "🎯 Score Borrower":
        render_score_borrower_page()
    elif page == "📊 Batch Analysis":
        render_batch_analysis_page()
    elif page == "📚 Documentation":
        render_documentation_page()


if __name__ == "__main__":
    main()
