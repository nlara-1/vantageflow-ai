# VantageFlow AI - User Guide

**Complete Guide to Using the VantageFlow AI Credit Scoring System**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Web Application Guide](#web-application-guide)
4. [Scoring a Borrower](#scoring-a-borrower)
5. [Understanding Your Results](#understanding-your-results)
6. [Generating Reports](#generating-reports)
7. [Python API Usage](#python-api-usage)
8. [Troubleshooting](#troubleshooting)
9. [FAQs](#faqs)

---

## Introduction

VantageFlow AI is an alternative credit scoring system that analyzes **banking transaction patterns** instead of traditional credit bureau data. It's designed for:

- **Lenders** evaluating creditworthiness without traditional credit history
- **Underbanked populations** who lack FICO scores
- **Financial institutions** seeking fair, transparent credit decisions
- **Researchers** studying alternative credit scoring methodologies

### What Makes VantageFlow AI Different?

✅ **Transaction-Based** - Analyzes spending behavior, not credit history
✅ **Explainable** - Every score includes clear reason codes
✅ **Fair** - No race, gender, or age data used
✅ **Compliant** - Meets FCRA/ECOA regulatory requirements
✅ **Interactive** - Web dashboard for easy exploration

---

## Getting Started

### Prerequisites

- **Python 3.8+**
- **pip** package manager
- **Web browser** (Chrome, Firefox, or Safari)
- **6-12 months** of transaction data (CSV format)

### Installation

```bash
# Clone the repository
git clone https://github.com/nlara-1/vantageflow-ai.git
cd vantageflow-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Launch the Web Application

```bash
streamlit run src/demo/app.py
```

The dashboard will open at **http://localhost:8501**

---

## Web Application Guide

### Navigation

The VantageFlow AI dashboard has **4 main pages**:

#### 🏠 **Home**
- Project overview
- Key features
- System metrics (AUC, Gini, Fairness Score)
- Technology stack

#### 🎯 **Score Borrower**
- Upload transaction data (CSV)
- Get instant credit score (300-850)
- View reason codes and explanations
- Download PDF reports

#### 📊 **Batch Analysis** (Coming Soon)
- Score multiple borrowers at once
- Portfolio-level analytics
- Bulk PDF report generation

#### 📚 **Model Documentation**
- Model card with performance metrics
- Fairness audit results
- Feature descriptions
- Regulatory compliance information

---

## Scoring a Borrower

### Step 1: Prepare Your Data

Create a CSV file with the following columns:

| Column | Description | Format | Example |
|--------|-------------|--------|---------|
| `borrower_id` | Unique borrower identifier | String | BORROWER_001 |
| `transaction_date` | Date of transaction | YYYY-MM-DD | 2023-06-15 |
| `amount` | Transaction amount | Float | 3200.00 (income) or -150.00 (expense) |
| `category` | Transaction category | String | INCOME, RENT, GROCERIES, etc. |

**Example CSV:**
```csv
borrower_id,transaction_date,amount,category
BORROWER_001,2023-01-05,3200.00,INCOME
BORROWER_001,2023-01-10,-1200.00,RENT
BORROWER_001,2023-01-15,-150.00,GROCERIES
BORROWER_001,2023-01-20,-80.00,TRANSPORTATION
BORROWER_001,2023-02-05,3200.00,INCOME
BORROWER_001,2023-02-10,-1200.00,RENT
```

### Step 2: Upload Your File

1. Navigate to the **🎯 Score Borrower** page
2. Click **"Browse files"** or drag and drop your CSV
3. Wait for validation (instant)

**Validation Checks:**
- ✅ Required columns present
- ✅ Date format valid (YYYY-MM-DD)
- ✅ Amount is numeric
- ✅ No missing values in required columns

If validation fails, you'll see a clear error message.

### Step 3: Download Sample Data (Optional)

Don't have test data? Click **📥 Download Sample** to get 100 sample transactions with realistic patterns.

### Step 4: Score the Borrower

1. Click **🎯 Score Borrower** button
2. Wait 2-5 seconds for processing
3. View your results!

---

## Understanding Your Results

### Credit Score

**Range:** 300-850 (same as traditional FICO scores)

| Score Range | Risk Tier | Meaning |
|-------------|-----------|---------|
| **750-850** | Excellent | Very low default risk |
| **700-749** | Good | Low default risk |
| **650-699** | Fair | Moderate default risk |
| **600-649** | Poor | Higher default risk |
| **300-599** | Very Poor | High default risk |

**Color Coding:**
- 🟢 **Green** = Excellent/Good (approve)
- 🟡 **Yellow** = Fair (conditional approval)
- 🔴 **Red** = Poor/Very Poor (deny or review)

### Decision Guidance

Based on the risk tier:
- ✅ **LOW/MEDIUM** → Typically approved
- ⚠️ **HIGH/VERY HIGH** → Manual review recommended

### Reason Codes

Reason codes explain **why** a borrower received their score.

#### Positive Factors (Green Cards)

These **help** the borrower's score:

**Example:**
```
[+] INCOME_STABILITY_STRONG (MODERATE)
Code: P01
Your income shows consistent patterns with low variability,
indicating stable employment or reliable income sources.
```

**Common Positive Codes:**
- **P01** - Income Stability Strong
- **P02** - Savings Behavior Positive
- **P03** - Expense Management Good
- **P04** - Income Growth Positive
- **P05** - Financial Buffer Adequate
- **P06** - Overdraft Free

#### Negative Factors (Red Cards)

These **hurt** the borrower's score:

**Example:**
```
[-] INCOME_VOLATILITY_HIGH (STRONG)
Code: N01
Your income shows significant month-to-month variation,
indicating irregular income patterns that may affect repayment ability.
```

**Common Negative Codes:**
- **N01** - Income Volatility High
- **N02** - Overdraft Frequent
- **N03** - Negative Cashflow Persistent
- **N04** - Income Insufficient
- **N05** - Expense Control Weak
- **N06** - Financial Buffer Minimal

### Magnitude Levels

Each reason code has a **magnitude** indicating its impact:

- **STRONG** - Significant impact (SHAP value ≥ 0.025)
- **MODERATE** - Moderate impact (SHAP value ≥ 0.01)
- **SLIGHT** - Minor impact (SHAP value ≥ 0.005)

### Feature Importance Chart

The horizontal bar chart shows the top 15 features affecting the score:

- **Red bars** - Increase default risk (negative for borrower)
- **Green bars** - Decrease default risk (positive for borrower)
- **Longer bars** - Stronger influence on score

### Financial Metrics Table

Key metrics extracted from transaction data:

| Metric | Description |
|--------|-------------|
| **Avg Monthly Income** | Average income per month |
| **Savings Rate** | (Income - Expenses) / Income |
| **Expense Income Ratio** | Total expenses / Total income |
| **Overdraft Count (3mo)** | Number of overdrafts in last 3 months |
| **Income CV** | Coefficient of variation (volatility) |
| **Discretionary Pct** | % of spending on non-essentials |

---

## Generating Reports

### PDF Underwriting Report

Professional report suitable for loan files and compliance.

**To Generate:**

1. Score a borrower first
2. Navigate to **📄 PDF Report** tab
3. Enter **Borrower ID** (e.g., BORROWER_001)
4. Optionally enter **Borrower Name**
5. Click **📥 Generate PDF Report**
6. Wait 2-3 seconds
7. Click **📥 Download PDF Report**

**Report Contents:**

1. **Header**
   - Company name
   - Report date
   - Borrower ID/name
   - Decision (APPROVED/DENIED/MANUAL_REVIEW)

2. **Score Display**
   - Large credit score (300-850)
   - Risk tier badge
   - Color-coded for quick assessment

3. **Positive Factors**
   - Top 3 positive reason codes
   - Code ID, magnitude, description, details

4. **Negative Factors**
   - Top 3 negative reason codes
   - Code ID, magnitude, description, details

5. **Financial Metrics**
   - Table of key financial indicators
   - Formatted (currency, percentages)

6. **Charts** (if provided)
   - SHAP waterfall plot
   - Score distribution
   - Income trends

7. **Disclaimer**
   - FCRA/ECOA compliance statement
   - Dispute rights
   - Contact information

**Report Format:**
- **File Type:** PDF
- **Layout:** Letter size (8.5" × 11")
- **Pages:** 3-5 pages typical
- **Quality:** 300 DPI, professional fonts
- **Compliance:** FCRA Section 615(a) compliant

---

## Python API Usage

For programmatic access, use the Python API directly.

### Score a Single Borrower

```python
import pandas as pd
from src.features.engineer import FeatureEngineer
from src.models.train import load_model, predict_with_model
from src.explainability.shap_engine import SHAPExplainer
from src.explainability.reason_codes import ReasonCodeGenerator

# Load transaction data
transactions = pd.read_csv('borrower_transactions.csv')

# Extract features
engineer = FeatureEngineer()
features = engineer.engineer_features_from_dataframe(transactions)

# Load trained model
model = load_model('models/production/xgboost_model.pkl')

# Get prediction
X = features.drop('borrower_id', axis=1)
probability = predict_with_model(model, X)[0]
score = int(probability * 1000)  # Scale to 0-1000

# Determine risk tier
if score >= 700:
    risk_tier = "LOW"
elif score >= 600:
    risk_tier = "MEDIUM"
else:
    risk_tier = "HIGH"

print(f"Credit Score: {score}")
print(f"Risk Tier: {risk_tier}")
```

### Generate SHAP Explanations

```python
# Create SHAP explainer
explainer = SHAPExplainer(model, X_train_sample)

# Get explanation for single borrower
explanation = explainer.get_explanation(X.iloc[0])

# Generate reason codes
generator = ReasonCodeGenerator()
reason_codes = generator.generate_reason_codes(
    shap_values=dict(zip(explanation.feature_names, explanation.values)),
    feature_values=dict(zip(explanation.feature_names, explanation.data)),
    top_n=5
)

# Print reason codes
for rc in reason_codes:
    symbol = "[+]" if rc.impact == "positive" else "[-]"
    print(f"{symbol} {rc.name} ({rc.magnitude})")
    print(f"  {rc.description}")
```

### Generate PDF Report

```python
from src.reporting.generator import UnderwritingReportGenerator

# Create generator
generator = UnderwritingReportGenerator()

# Generate report
pdf_path = generator.generate_report(
    borrower_id="BORROWER_001",
    score=score,
    risk_tier=risk_tier,
    reason_codes=[rc.__dict__ for rc in reason_codes],
    financial_metrics=features.iloc[0].to_dict(),
    output_path="reports/underwriting_report.pdf",
    borrower_name="John Doe",
    decision="APPROVED"
)

print(f"Report saved: {pdf_path}")
```

---

## Troubleshooting

### Common Issues

#### 1. **"Model not found" error**

**Problem:** No trained model available.

**Solution:**
```bash
# Train a model first
python -m src.data_generation.profiles
python -m src.data_generation.synthesizer
python -m src.data_generation.labels
python -m src.features.engineer
python -m src.models.train
```

#### 2. **"Invalid CSV format" error**

**Problem:** CSV missing required columns or has wrong format.

**Solution:**
- Check that all 4 columns are present: `borrower_id`, `transaction_date`, `amount`, `category`
- Ensure dates are in YYYY-MM-DD format
- Ensure amounts are numeric
- Remove header rows beyond the first row

#### 3. **"Insufficient transaction data" error**

**Problem:** Not enough transactions to extract features.

**Solution:**
- Need at least **6 months** of transaction data
- Need at least **30 transactions** total
- Should have mix of income and expenses

#### 4. **Streamlit won't start**

**Problem:** Port 8501 already in use.

**Solution:**
```bash
# Use different port
streamlit run src/demo/app.py --server.port 8502
```

#### 5. **"SHAP explainer not available" warning**

**Problem:** Background data not found for SHAP.

**Solution:**
- This is optional and doesn't affect scoring
- To enable: create `data/processed/X_train_sample.csv` with feature data

### Getting Help

1. **Check the docs** - Read this guide and README.md
2. **Search issues** - https://github.com/nlara-1/vantageflow-ai/issues
3. **Open new issue** - Provide error message and steps to reproduce
4. **Review logs** - Check terminal output for detailed errors

---

## FAQs

### General Questions

**Q: Is this a real credit scoring system?**
A: It's a demonstration system using synthetic data. For production use, train on real transaction data and get legal review.

**Q: How accurate is it?**
A: On synthetic data: AUC-ROC = 0.892, Gini = 0.784. Real-world accuracy depends on your data quality.

**Q: Does it use traditional credit scores?**
A: No! It only uses transaction data (no FICO, credit bureau, or credit history).

**Q: What data do I need?**
A: 6-12 months of banking transactions with: date, amount, category, borrower ID.

### Technical Questions

**Q: What machine learning model does it use?**
A: XGBoost gradient boosting with hyperparameter tuning via RandomizedSearchCV.

**Q: How are explanations generated?**
A: SHAP (SHapley Additive exPlanations) with TreeExplainer, then mapped to business-friendly reason codes.

**Q: Is it fair?**
A: Yes - no protected attributes used, automated fairness testing, 80% rule compliance.

**Q: Can I customize it?**
A: Yes - modify `config/reason_codes.yaml` and `config/feature_config.yaml`.

### Compliance Questions

**Q: Is it FCRA compliant?**
A: Yes - adverse action notices follow FCRA Section 615(a) requirements.

**Q: Does it discriminate?**
A: No - protected attributes (race, gender, age) are explicitly excluded.

**Q: Can I use it for lending decisions?**
A: With legal review, yes. Always consult compliance experts before production deployment.

**Q: What about Regulation B (ECOA)?**
A: System is designed to comply - no prohibited bases, adverse action notices provided.

### Data Questions

**Q: What transaction categories are supported?**
A: Common categories: INCOME, RENT, UTILITIES, GROCERIES, TRANSPORTATION, ENTERTAINMENT, DINING, SHOPPING, HEALTHCARE, etc.

**Q: Do I need all categories?**
A: No - system adapts to available categories. But more diversity is better.

**Q: Can I use bank statement PDFs?**
A: Not directly - you'll need to extract transactions to CSV first.

**Q: How do I handle multiple income sources?**
A: Use category tags: INCOME_SALARY, INCOME_FREELANCE, INCOME_BUSINESS, etc.

---

## Next Steps

**For New Users:**
1. ✅ Download sample data
2. ✅ Score a test borrower
3. ✅ Generate a PDF report
4. ✅ Explore the Model Documentation page

**For Advanced Users:**
1. ✅ Prepare your own transaction data
2. ✅ Train a custom model with your data
3. ✅ Customize reason codes for your use case
4. ✅ Integrate into your lending workflow

**For Developers:**
1. ✅ Read [DEVELOPMENT.md](DEVELOPMENT.md)
2. ✅ Run the test suite
3. ✅ Explore the Python API
4. ✅ Contribute improvements

---

## Support

**Documentation:**
- README.md - Project overview
- DEVELOPMENT.md - Developer guide
- This guide - User instructions

**Community:**
- GitHub Issues - Bug reports & features
- GitHub Discussions - Questions & ideas

**Professional Services:**
- Contact for custom implementations
- Training and consulting available

---

**Happy Scoring! 🎯**

*VantageFlow AI - Built with ❤️ for financial inclusion*
