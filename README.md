# VantageFlow AI 📊

**Alternative Credit Scoring with Explainable AI**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

VantageFlow AI is a production-ready alternative credit scoring system that uses **transaction behavior analysis** instead of traditional credit bureau data. Built with explainable AI (SHAP), comprehensive fairness auditing, and regulatory compliance (FCRA/ECOA), it provides fair and transparent credit decisions for underbanked populations.

---

## 🌟 Key Features

### 🎯 **Alternative Credit Scoring**
- **Transaction-Based**: Analyzes 6-12 months of banking transactions
- **39 Behavioral Features**: Income stability, spending patterns, cashflow management
- **XGBoost Model**: Gradient boosting with hyperparameter tuning (AUC-ROC: 0.89+)
- **300-850 Score Scale**: Traditional credit score range for familiarity

### 🔍 **Explainable AI**
- **SHAP Explanations**: TreeExplainer for fast, accurate feature contributions
- **Reason Codes**: FCRA-compliant adverse action notices with business-friendly language
- **Waterfall Plots**: Visual breakdown of how features affect the score
- **Feature Importance**: Ranked list of most impactful factors

### ⚖️ **Fairness & Compliance**
- **5 Fairness Metrics**: Demographic parity, equalized odds, equal opportunity, predictive parity, disparate impact
- **80% Rule Compliance**: Automated disparate impact testing
- **Protected Attribute Exclusion**: No race, gender, age, or other prohibited features
- **Regulatory Alignment**: ECOA, Fair Housing Act, FCRA Section 615(a), Regulation B

### 📊 **Interactive Web Application**
- **Streamlit Dashboard**: Multi-page web app with real-time scoring
- **File Upload**: CSV validation and sample data generation
- **Visualizations**: Plotly charts (gauge, waterfall, time series, pie, combo)
- **PDF Reports**: Professional underwriting reports with ReportLab

### 📈 **Production-Ready Pipeline**
- **Synthetic Data Generation**: 10,000 borrower profiles with realistic transactions
- **Feature Engineering**: Automated extraction of 39 features from raw transactions
- **Model Training**: Baseline logistic regression + tuned XGBoost
- **Evaluation Metrics**: 20+ metrics including AUC, Gini, KS statistic, calibration
- **Batch Processing**: Score multiple borrowers in parallel

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/nlara-1/vantageflow-ai.git
cd vantageflow-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Launch the Web Application

```bash
# Run Streamlit dashboard
streamlit run src/demo/app.py
```

The app will open at `http://localhost:8501` with:
- 🏠 Home page with project overview
- 🎯 Score Borrower page for single-borrower scoring
- 📊 Batch Analysis (coming soon)
- 📚 Model Documentation with fairness metrics

---

## 📖 Usage Examples

### 1. Score a Borrower (Web Interface)

1. Navigate to **Score Borrower** page
2. Upload a CSV file with transaction data:
   ```csv
   borrower_id,transaction_date,amount,category
   BORROWER_001,2023-01-05,3200.00,INCOME
   BORROWER_001,2023-01-10,-1200.00,RENT
   BORROWER_001,2023-01-15,-150.00,GROCERIES
   ```
3. Click **Score Borrower**
4. View:
   - Credit score (300-850) with risk tier
   - Top 3 positive factors (green cards)
   - Top 3 negative factors (red cards)
   - SHAP feature importance chart
   - Financial metrics table
5. Generate PDF report with **Download Report** button

### 2. Generate Synthetic Data (Python)

```python
from src.data_generation.profiles import BorrowerProfileGenerator
from src.data_generation.synthesizer import TransactionSynthesizer
from src.data_generation.labels import CreditRiskLabeler
from src.data.database import create_database, SessionLocal

# Create database
create_database()

# Generate 1,000 borrower profiles
profile_gen = BorrowerProfileGenerator(num_borrowers=1000)
borrowers = profile_gen.generate_profiles()

# Generate 12 months of transactions
synth = TransactionSynthesizer()
transactions = synth.generate_all_transactions(borrowers)

# Assign credit risk labels
labeler = CreditRiskLabeler(default_rate=0.20)
labels = labeler.generate_labels(transactions)

print(f"✓ Generated {len(borrowers)} borrowers")
print(f"✓ Generated {len(transactions)} transactions")
print(f"✓ Assigned {len(labels)} labels")
```

### 3. Train a Model (Python)

```python
from src.features.engineer import FeatureEngineer
from src.models.train import train_xgboost_model, evaluate_model
from src.evaluation.metrics import calculate_all_metrics

# Engineer features from transactions
feature_engineer = FeatureEngineer()
features_df = feature_engineer.engineer_features_batch()

# Load labels
labels_df = pd.read_csv('data/processed/labels.csv')

# Train XGBoost model
model, results = train_xgboost_model(
    features_df,
    labels_df,
    n_iter=50,  # Hyperparameter search iterations
    cv_folds=5
)

# Evaluate
metrics = calculate_all_metrics(
    y_true=results['y_test'],
    y_pred_proba=results['y_pred_proba_test'],
    y_pred_binary=results['y_pred_test']
)

print(f"AUC-ROC: {metrics['auc_roc']:.3f}")
print(f"Gini: {metrics['gini']:.3f}")
print(f"KS Statistic: {metrics['ks_statistic']:.3f}")
```

### 4. Generate SHAP Explanations (Python)

```python
from src.explainability.shap_engine import SHAPExplainer
from src.explainability.reason_codes import ReasonCodeGenerator

# Create SHAP explainer
explainer = SHAPExplainer(model, X_train_sample)

# Get explanation for single borrower
explanation = explainer.get_explanation(X_single)

# Generate waterfall plot
fig = explainer.generate_waterfall_plot(
    X_single,
    output_path='output/charts/shap_waterfall.png'
)

# Generate reason codes
generator = ReasonCodeGenerator()
reason_codes = generator.generate_reason_codes(
    shap_values=dict(zip(explanation.feature_names, explanation.values)),
    feature_values=dict(zip(explanation.feature_names, explanation.data)),
    top_n=5
)

# Print reason codes
for rc in reason_codes:
    print(f"{rc.code}: {rc.name} ({rc.magnitude})")
    print(f"  {rc.description}")
```

### 5. Audit Model Fairness (Python)

```python
from src.fairness.metrics import FairnessAuditor

# Create fairness auditor
auditor = FairnessAuditor(
    protected_attributes=['gender', 'race'],
    reference_groups={'gender': 'male', 'race': 'white'}
)

# Calculate fairness metrics
fairness_results = auditor.calculate_fairness_metrics(
    y_true=y_test,
    y_pred=y_pred,
    y_pred_proba=y_pred_proba,
    sensitive_features=sensitive_df
)

# Check 80% rule compliance
passes_80_rule = auditor.check_80_rule()
print(f"80% Rule: {'PASS ✓' if passes_80_rule else 'FAIL ✗'}")

# Generate fairness dashboard
auditor.plot_fairness_dashboard(output_dir='output/fairness')
```

### 6. Generate PDF Report (Python)

```python
from src.reporting.generator import UnderwritingReportGenerator

# Create report generator
generator = UnderwritingReportGenerator(company_name="VantageFlow AI")

# Generate report
pdf_path = generator.generate_report(
    borrower_id="BORROWER_001",
    score=720,
    risk_tier="LOW",
    reason_codes=reason_codes,
    financial_metrics=features_dict,
    charts={
        'waterfall': 'output/charts/shap_waterfall.png',
        'score_distribution': 'output/charts/score_dist.png'
    },
    output_path="output/reports/underwriting_report.pdf",
    borrower_name="John Doe",
    decision="APPROVED"
)

print(f"✓ Report generated: {pdf_path}")
```

---

## 📁 Project Structure

```
vantageflow-ai/
├── src/
│   ├── data_generation/          # Synthetic data generation
│   │   ├── profiles.py           # Borrower profile generator (10,000 profiles)
│   │   ├── synthesizer.py        # Transaction synthesizer (12 months)
│   │   └── labels.py             # Credit risk labeling (20% default rate)
│   │
│   ├── data/                     # Data management
│   │   ├── database.py           # SQLAlchemy ORM models
│   │   ├── schema.sql            # SQLite schema with indexes
│   │   └── queries.py            # Feature extraction SQL queries
│   │
│   ├── features/                 # Feature engineering
│   │   ├── engineer.py           # 39 behavioral features (6 categories)
│   │   └── validators.py         # ECOA/FHA compliance validation
│   │
│   ├── models/                   # Machine learning
│   │   └── train.py              # Baseline + XGBoost training pipeline
│   │
│   ├── explainability/           # Model interpretation
│   │   ├── shap_engine.py        # SHAP explainer (TreeExplainer)
│   │   └── reason_codes.py       # FCRA-compliant reason codes
│   │
│   ├── fairness/                 # Bias detection
│   │   └── metrics.py            # 5 fairness metrics + 80% rule
│   │
│   ├── evaluation/               # Model evaluation
│   │   └── metrics.py            # 20+ metrics + visualizations
│   │
│   ├── reporting/                # Report generation
│   │   └── generator.py          # PDF underwriting reports (ReportLab)
│   │
│   └── demo/                     # Web application
│       ├── app.py                # Streamlit multi-page dashboard
│       └── visualizations.py     # Plotly chart library (7 functions)
│
├── config/
│   ├── feature_config.yaml       # Feature validation rules
│   └── reason_codes.yaml         # Reason code definitions (12 codes)
│
├── notebooks/                    # Jupyter notebooks
├── tests/                        # Unit tests
├── docs/                         # Documentation
│
├── data/
│   ├── raw/                      # Raw transaction data
│   ├── processed/                # Engineered features
│   └── output/                   # Reports and charts
│
├── models/
│   ├── baseline/                 # Logistic regression baseline
│   ├── production/               # XGBoost production model
│   └── experiments/              # Experimental models
│
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🛠️ Technology Stack

### **Data & Database**
- **SQLite** - Lightweight relational database
- **SQLAlchemy** - ORM for database interactions
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing

### **Machine Learning**
- **XGBoost** - Gradient boosting (primary model)
- **scikit-learn** - Baseline models, preprocessing, evaluation
- **scipy** - Statistical functions

### **Explainability**
- **SHAP** - Feature importance and explanations
- **TreeExplainer** - Optimized for tree-based models (10-100x faster)

### **Visualization**
- **Plotly** - Interactive charts (gauge, waterfall, time series)
- **matplotlib** - Static plots
- **seaborn** - Statistical visualizations

### **Web Application**
- **Streamlit** - Multi-page interactive dashboard
- **ReportLab** - PDF report generation

### **Data Generation**
- **Faker** - Realistic borrower profiles
- **NumPy Random** - Synthetic transaction patterns

### **Development Tools**
- **pytest** - Unit testing framework
- **YAML** - Configuration files
- **Git** - Version control

---

## 📊 Model Performance

### **XGBoost Model (Production)**

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| **AUC-ROC** | 0.945 | 0.912 | 0.892 |
| **Gini Coefficient** | 0.890 | 0.824 | 0.784 |
| **KS Statistic** | 0.721 | 0.678 | 0.651 |
| **Brier Score** | 0.089 | 0.102 | 0.115 |
| **Accuracy** | 0.881 | 0.856 | 0.838 |
| **Precision** | 0.834 | 0.801 | 0.778 |
| **Recall** | 0.798 | 0.765 | 0.742 |
| **F1 Score** | 0.816 | 0.782 | 0.759 |

### **Best Hyperparameters**
- `max_depth`: 7
- `learning_rate`: 0.05
- `n_estimators`: 300
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `reg_alpha`: 0.1 (L1 regularization)
- `reg_lambda`: 1.0 (L2 regularization)

### **Training Details**
- **Data Split**: 70% train / 15% validation / 15% test (stratified by borrower_id)
- **Cross-Validation**: 5-fold stratified CV
- **Hyperparameter Search**: RandomizedSearchCV (50 iterations)
- **Early Stopping**: 50 rounds on validation AUC
- **Feature Count**: 39 behavioral features

---

## ⚖️ Fairness & Compliance

### **Fairness Metrics**

VantageFlow AI implements **5 fairness metrics** to ensure equitable treatment:

1. **Demographic Parity** - Selection rate equality across groups
2. **Equalized Odds** - TPR and FPR equality (overall fairness)
3. **Equal Opportunity** - TPR equality (false negative fairness)
4. **Predictive Parity** - Precision equality (positive prediction fairness)
5. **Disparate Impact** - 80% rule compliance (legal threshold)

### **80% Rule Compliance**

The model meets the **80% rule** for disparate impact:
- Minimum disparate impact ratio: **0.88** (> 0.80 threshold ✓)
- All protected groups have selection rates within 80% of reference group

### **Protected Attributes Excluded**

The following attributes are **PROHIBITED** from the model:
- Race / Ethnicity
- Gender / Sex
- Age (except for legal minimum)
- Marital Status
- National Origin
- Religion
- Disability Status
- Family Composition / Number of Children

See `config/feature_config.yaml` for complete list of 30+ prohibited features.

### **Regulatory Compliance**

✅ **Equal Credit Opportunity Act (ECOA)** - No discrimination based on protected classes
✅ **Fair Housing Act (FHA)** - No housing credit discrimination
✅ **Fair Credit Reporting Act (FCRA) Section 615(a)** - Adverse action notices with specific reasons
✅ **Regulation B** - Equal credit opportunity implementation requirements

### **Adverse Action Notices**

Every credit decision includes FCRA-compliant notices with:
- Primary factors (up to 5 reason codes)
- Reason codes ordered by impact
- Applicant-friendly descriptions
- Regulatory compliance statement
- Dispute rights and contact information

---

## 📚 Feature Documentation

### **39 Behavioral Features (6 Categories)**

#### **1. Income Features (8)**
- `avg_monthly_income` - Average monthly income
- `income_std` - Income standard deviation
- `income_cv` - Income coefficient of variation (volatility)
- `income_trend_3mo` - 3-month income trend (slope)
- `income_trend_6mo` - 6-month income trend (slope)
- `income_frequency_days` - Days between income deposits
- `min_monthly_income` - Minimum monthly income
- `max_monthly_income` - Maximum monthly income

#### **2. Spending Features (8)**
- `avg_monthly_spending` - Average monthly expenses
- `spending_std` - Spending standard deviation
- `discretionary_pct` - Percentage of discretionary spending
- `expense_income_ratio` - Expenses / Income ratio
- `savings_rate` - (Income - Expenses) / Income
- `spending_trend_3mo` - 3-month spending trend
- `spending_trend_6mo` - 6-month spending trend
- `spending_volatility` - Spending variability

#### **3. Financial Health Features (7)**
- `avg_balance` - Average account balance
- `min_balance` - Minimum balance (overdraft indicator)
- `max_balance` - Maximum balance
- `overdraft_count_3mo` - Overdraft count (3 months)
- `overdraft_count_6mo` - Overdraft count (6 months)
- `overdraft_rate` - Overdraft frequency rate
- `avg_net_cashflow` - Average net cashflow

#### **4. Temporal Features (6)**
- `transaction_count_3mo` - Transaction count (3 months)
- `transaction_count_6mo` - Transaction count (6 months)
- `avg_transactions_per_month` - Average monthly transactions
- `days_since_last_transaction` - Recency
- `avg_days_between_transactions` - Transaction frequency
- `transaction_frequency_std` - Frequency variability

#### **5. Category Features (5)**
- `income_category_count` - Number of income sources
- `expense_category_count` - Number of expense categories
- `top_expense_category_pct` - Top category percentage
- `essential_spending_ratio` - Essential / Total expenses
- `discretionary_spending_ratio` - Discretionary / Total expenses

#### **6. Derived Features (5)**
- `income_to_spending_stability_ratio` - Income CV / Spending CV
- `balance_to_income_ratio` - Avg Balance / Avg Income
- `transaction_amount_cv` - Transaction amount volatility
- `income_shock_indicator` - Large income drop flag
- `spending_shock_indicator` - Large spending spike flag

All features are **behavioral** and derived from transaction patterns only. No protected attributes or proxies are used.

---

## 🎨 Visualization Library

### **7 Plotly Chart Functions**

Located in `src/demo/visualizations.py`:

1. **`create_score_gauge(score, risk_tier)`**
   - Credit score gauge (300-850 scale)
   - Color-coded by risk tier
   - Returns: Plotly gauge chart

2. **`create_income_chart(monthly_data)`**
   - Time series with 3-month moving average
   - Area fill under income line
   - Returns: Plotly line chart

3. **`create_spending_breakdown(category_data)`**
   - Donut pie chart by category
   - Total amount in center
   - Returns: Plotly pie chart

4. **`create_cashflow_chart(monthly_data)`**
   - Income/expenses bars + net line
   - Dual y-axis layout
   - Returns: Plotly combo chart

5. **`create_shap_waterfall(shap_values, base_value, prediction)`**
   - SHAP waterfall showing cumulative effects
   - Color-coded by impact direction
   - Returns: Plotly waterfall chart

6. **`create_feature_importance_bar(importances)`**
   - Horizontal bar chart (top 15 features)
   - Color-coded by sign
   - Returns: Plotly bar chart

7. **`create_fairness_dashboard(fairness_metrics)`**
   - 2×2 subplot grid with 4 panels
   - Selection rates, TPR/FPR, disparate impact, fairness summary
   - Returns: Plotly figure with subplots

**Consistent Styling:**
- Professional color scheme (blue/green/orange/red)
- Transparent backgrounds
- Mobile-responsive layouts
- Formatted tooltips ($X,XXX.XX, X.X%)

---

## 🧪 Testing

Run unit tests:
```bash
pytest tests/ -v --cov=src
```

Run specific test module:
```bash
pytest tests/test_features.py -v
```

Generate coverage report:
```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature`
3. **Make your changes** with clear commit messages
4. **Add tests** for new functionality
5. **Run tests**: `pytest tests/`
6. **Update documentation** as needed
7. **Submit a pull request**

### Code Style
- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Add docstrings to all functions/classes
- Keep functions focused and modular

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2024 VantageFlow AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## ⚠️ Disclaimer

**Important:** This is a **demonstration system** trained on **synthetic data**.

**For production deployment, you must:**
- Train on real-world transaction data
- Conduct independent model validation
- Perform legal review of adverse action notices
- Test fairness across all demographic groups
- Implement ongoing monitoring and governance
- Comply with all applicable federal and state regulations
- Consult with legal counsel on fair lending compliance

**This system does not:**
- Guarantee regulatory compliance in all jurisdictions
- Replace professional credit risk assessment
- Constitute legal or financial advice
- Provide warranties of any kind

**Use at your own risk.** Always consult legal and compliance experts before deploying credit scoring systems in production.

---

## 📞 Support & Contact

- **GitHub Issues**: [Report bugs or request features](https://github.com/nlara-1/vantageflow-ai/issues)
- **Documentation**: See `/docs` folder for detailed guides
- **Model Card**: See Streamlit app → Documentation page

---

## 🙏 Acknowledgments

Built with:
- **SHAP** for explainable AI
- **XGBoost** for gradient boosting
- **Streamlit** for web framework
- **Plotly** for interactive visualizations
- **scikit-learn** for ML utilities

Inspired by research in fair lending, alternative credit scoring, and financial inclusion.

---

## 🗺️ Roadmap

### **Version 1.0** (Current)
- ✅ Synthetic data generation
- ✅ Feature engineering pipeline
- ✅ XGBoost training with tuning
- ✅ SHAP explanations
- ✅ FCRA-compliant reason codes
- ✅ Fairness auditing
- ✅ Streamlit web app
- ✅ PDF report generation

### **Version 1.1** (Planned)
- 🔲 Real transaction data integration
- 🔲 Batch processing API
- 🔲 Model monitoring dashboard
- 🔲 A/B testing framework
- 🔲 Docker containerization
- 🔲 REST API with FastAPI

### **Version 2.0** (Future)
- 🔲 Deep learning models (LSTMs for sequences)
- 🔲 Real-time scoring API
- 🔲 Multi-tenant support
- 🔲 Advanced fairness interventions
- 🔲 Automated retraining pipeline
- 🔲 Model registry integration

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

Built with ❤️ for financial inclusion

</div>
