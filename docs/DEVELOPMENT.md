# VantageFlow AI - Development Guide

**Complete Guide for Contributors and Developers**

---

## Table of Contents

1. [Development Setup](#development-setup)
2. [Project Architecture](#project-architecture)
3. [Code Organization](#code-organization)
4. [Development Workflow](#development-workflow)
5. [Testing](#testing)
6. [Code Style](#code-style)
7. [Adding New Features](#adding-new-features)
8. [Performance Optimization](#performance-optimization)
9. [Deployment](#deployment)
10. [Contributing](#contributing)

---

## Development Setup

### Prerequisites

- **Python 3.8-3.11** (3.11 recommended for best performance)
- **pip 21.0+**
- **Git 2.30+**
- **8GB RAM minimum** (16GB recommended for model training)
- **5GB disk space**

### Initial Setup

```bash
# Clone repository
git clone https://github.com/nlara-1/vantageflow-ai.git
cd vantageflow-ai

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt  # If available

# Or install individually
pip install pytest pytest-cov black flake8 mypy ipython ipdb
```

### Environment Configuration

Create `.env` file in project root:

```bash
# Development settings
ENV=development
DEBUG=True

# Database
DATABASE_PATH=data/vantageflow_dev.db

# Model paths
MODEL_PATH=models/development/xgboost_model.pkl
BASELINE_MODEL_PATH=models/development/baseline_model.pkl

# Logging
LOG_LEVEL=DEBUG
LOG_PATH=logs/vantageflow_dev.log

# Streamlit
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Testing
TEST_DATA_PATH=data/test/
PYTEST_ADDOPTS=-v --tb=short

# Random seeds for reproducibility
RANDOM_SEED=42
NUMPY_SEED=42
TF_SEED=42
```

### Verify Installation

```bash
# Run quick tests
pytest tests/test_features.py::TestFeatureEngineer::test_engineer_initialization -v

# Check imports
python -c "import src.features.engineer; print('✓ Imports work')"

# Start app (should open without errors)
streamlit run src/demo/app.py --server.headless true
```

---

## Project Architecture

### High-Level Architecture

```
┌─────────────────┐
│  Web Interface  │  Streamlit Dashboard (src/demo/)
│   (Streamlit)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Reporting     │  PDF Generation (src/reporting/)
│   (ReportLab)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Explainability  │  SHAP + Reason Codes (src/explainability/)
│  (SHAP, YAML)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Evaluation    │  Metrics + Fairness (src/evaluation/, src/fairness/)
│  (scikit-learn) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Prediction    │  Model Training + Inference (src/models/)
│    (XGBoost)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Features     │  Feature Engineering (src/features/)
│    (pandas)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│      Data       │  Data Access + Generation (src/data/, src/data_generation/)
│  (SQLAlchemy)   │
└─────────────────┘
```

### Design Patterns

**1. Pipeline Pattern**
- Each module outputs data for the next stage
- Loose coupling via dataframes and pickle files
- Easy to test each stage independently

**2. Configuration Pattern**
- YAML files for reason codes and feature validation
- Separate config from code
- Easy to modify without code changes

**3. Factory Pattern**
- Model creation functions (train_baseline_model, train_xgboost_model)
- Explainer creation (create_explainer)
- Report generation (UnderwritingReportGenerator)

**4. Strategy Pattern**
- Multiple fairness metrics with common interface
- Pluggable model types (baseline, XGBoost)
- Swappable visualization functions

---

## Code Organization

### Directory Structure

```
vantageflow-ai/
├── src/                          # Source code
│   ├── data_generation/          # Synthetic data generation
│   │   ├── profiles.py           # Borrower profile generation
│   │   ├── synthesizer.py        # Transaction synthesis
│   │   └── labels.py             # Credit risk labeling
│   │
│   ├── data/                     # Data access layer
│   │   ├── database.py           # SQLAlchemy ORM models
│   │   ├── schema.sql            # Database schema
│   │   └── queries.py            # Feature extraction queries
│   │
│   ├── features/                 # Feature engineering
│   │   ├── engineer.py           # 39 feature extraction
│   │   └── validators.py         # ECOA/FHA validation
│   │
│   ├── models/                   # Machine learning
│   │   └── train.py              # Model training + prediction
│   │
│   ├── explainability/           # Model interpretation
│   │   ├── shap_engine.py        # SHAP computation
│   │   └── reason_codes.py       # Reason code generation
│   │
│   ├── fairness/                 # Bias detection
│   │   └── metrics.py            # Fairness metrics
│   │
│   ├── evaluation/               # Model evaluation
│   │   └── metrics.py            # Performance metrics
│   │
│   ├── reporting/                # Report generation
│   │   └── generator.py          # PDF reports
│   │
│   └── demo/                     # Web application
│       ├── app.py                # Streamlit dashboard
│       └── visualizations.py     # Plotly charts
│
├── config/                       # Configuration files
│   ├── feature_config.yaml       # Feature validation rules
│   └── reason_codes.yaml         # Reason code definitions
│
├── tests/                        # Test suite
│   ├── conftest.py               # Pytest fixtures
│   ├── test_data_generation.py  # Data generation tests
│   ├── test_features.py          # Feature engineering tests
│   ├── test_models.py            # Model training tests
│   └── test_explainability.py    # SHAP + reason codes tests
│
├── data/                         # Data storage
│   ├── raw/                      # Raw transaction data
│   ├── processed/                # Engineered features
│   └── output/                   # Reports and charts
│
├── models/                       # Model artifacts
│   ├── baseline/                 # Baseline models
│   ├── production/               # Production models
│   └── experiments/              # Experimental models
│
├── docs/                         # Documentation
│   ├── USER_GUIDE.md             # User guide
│   └── DEVELOPMENT.md            # This file
│
├── notebooks/                    # Jupyter notebooks
├── scripts/                      # Utility scripts
├── logs/                         # Application logs
│
├── requirements.txt              # Python dependencies
├── pytest.ini                    # Pytest configuration
├── .gitignore                    # Git ignore rules
└── README.md                     # Project overview
```

### Module Dependencies

```
demo/ → reporting/ → explainability/ → evaluation/ → models/ → features/ → data/
                                           ↓
                                      fairness/
```

**Dependency Rules:**
1. Lower layers don't depend on higher layers
2. Config depends on nothing (pure data)
3. Tests can depend on any layer
4. Demo is the top layer (depends on everything)

---

## Development Workflow

### 1. Create Feature Branch

```bash
# Update main
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/your-feature-name

# Or bug fix branch
git checkout -b fix/bug-description
```

### 2. Make Changes

```bash
# Edit files
vim src/features/engineer.py

# Run tests frequently
pytest tests/test_features.py -v

# Check code style
black src/features/engineer.py --check
flake8 src/features/engineer.py
```

### 3. Write Tests

```python
# tests/test_features.py

def test_your_new_feature(sample_transactions):
    """Test your new feature extraction."""
    engineer = FeatureEngineer()
    features = engineer.engineer_features_from_dataframe(sample_transactions)

    # Your assertions
    assert 'new_feature' in features.columns
    assert features['new_feature'].dtype == np.float64
    assert features['new_feature'].notnull().all()
```

### 4. Update Documentation

```python
# Add docstrings
def your_new_function(param1: str, param2: int) -> pd.DataFrame:
    """
    Brief description of what the function does.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        DataFrame with columns: [col1, col2, ...]

    Raises:
        ValueError: When param2 is negative

    Example:
        >>> result = your_new_function("test", 10)
        >>> len(result)
        10
    """
    # Implementation
    pass
```

### 5. Run Full Test Suite

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html

# Specific test
pytest tests/test_features.py::TestFeatureEngineer::test_feature_extraction -v
```

### 6. Commit Changes

```bash
# Stage changes
git add src/features/engineer.py tests/test_features.py

# Commit with descriptive message
git commit -m "Add transaction frequency feature

- Calculates days between transactions
- Handles edge cases (single transaction)
- Includes comprehensive tests
- Updates documentation"
```

### 7. Push and Create PR

```bash
# Push to GitHub
git push origin feature/your-feature-name

# Create pull request on GitHub
# Request review from maintainers
```

---

## Testing

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_data_generation.py  # 31 tests
├── test_features.py          # 30 tests
├── test_models.py            # 23 tests
└── test_explainability.py    # 31 tests

Total: 115 tests
```

### Running Tests

```bash
# All tests (fast, <1 minute)
pytest tests/ -v

# Exclude slow tests
pytest tests/ -m "not slow"

# Specific file
pytest tests/test_features.py -v

# Specific class
pytest tests/test_features.py::TestFeatureEngineer -v

# Specific test
pytest tests/test_features.py::TestFeatureEngineer::test_feature_extraction -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing

# Parallel execution (if pytest-xdist installed)
pytest tests/ -n auto

# Stop on first failure
pytest tests/ -x

# Show print statements
pytest tests/ -s

# Verbose output with locals
pytest tests/ -vv --showlocals
```

### Writing Good Tests

**1. Use Fixtures**

```python
@pytest.fixture
def sample_data():
    """Reusable test data."""
    return pd.DataFrame({
        'feature_1': [1, 2, 3],
        'feature_2': [4, 5, 6]
    })

def test_function(sample_data):
    """Test uses fixture."""
    result = process_data(sample_data)
    assert len(result) == 3
```

**2. Test Edge Cases**

```python
def test_empty_input():
    """Test empty DataFrame."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError):
        process_data(empty_df)

def test_single_row():
    """Test single row input."""
    single_row = pd.DataFrame([{'col': 1}])
    result = process_data(single_row)
    assert len(result) == 1
```

**3. Use Parametrize**

```python
@pytest.mark.parametrize("input,expected", [
    (0.5, "MEDIUM"),
    (0.8, "LOW"),
    (0.2, "HIGH"),
])
def test_risk_tier(input, expected):
    """Test risk tier calculation."""
    assert calculate_risk_tier(input) == expected
```

**4. Test Reproducibility**

```python
def test_reproducibility():
    """Test function produces same output."""
    np.random.seed(42)
    result1 = generate_random_data()

    np.random.seed(42)
    result2 = generate_random_data()

    pd.testing.assert_frame_equal(result1, result2)
```

---

## Code Style

### PEP 8 Compliance

```bash
# Check style
flake8 src/ --max-line-length=100

# Auto-format
black src/ --line-length=100

# Check type hints
mypy src/ --ignore-missing-imports
```

### Naming Conventions

```python
# Modules: lowercase_with_underscores
import feature_engineer

# Classes: CapitalizedWords
class FeatureEngineer:
    pass

# Functions: lowercase_with_underscores
def calculate_features():
    pass

# Constants: UPPERCASE_WITH_UNDERSCORES
MAX_FEATURES = 100

# Private: _leading_underscore
def _internal_helper():
    pass
```

### Type Hints

```python
from typing import List, Dict, Optional, Union, Tuple

def process_features(
    data: pd.DataFrame,
    feature_names: List[str],
    threshold: Optional[float] = None
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Always use type hints."""
    pass
```

### Docstrings

**Google Style:**

```python
def complex_function(param1: str, param2: int = 10) -> pd.DataFrame:
    """
    One-line summary.

    Longer description explaining what the function does,
    when to use it, and any important caveats.

    Args:
        param1: Description of param1. Can be multi-line
            if needed with proper indentation.
        param2: Description of param2. Default: 10.

    Returns:
        DataFrame with the following columns:
        - col1 (float): Description of col1
        - col2 (str): Description of col2

    Raises:
        ValueError: If param2 is negative.
        KeyError: If required columns missing.

    Example:
        >>> result = complex_function("test", 20)
        >>> len(result)
        20

    Note:
        This function is expensive for large inputs.
    """
    if param2 < 0:
        raise ValueError("param2 must be non-negative")

    # Implementation
    return pd.DataFrame()
```

---

## Adding New Features

### 1. Adding a New Behavioral Feature

**Step 1: Update Feature Engineer**

```python
# src/features/engineer.py

def _calculate_new_feature(self, transactions_df: pd.DataFrame) -> float:
    """Calculate your new feature."""
    # Implementation
    return value

def _get_feature_names(self) -> List[str]:
    """Add to feature list."""
    return [
        # ... existing features
        'new_feature_name'
    ]
```

**Step 2: Add to Allowed Features**

```yaml
# config/feature_config.yaml
allowed_features:
  - avg_monthly_income
  - new_feature_name  # Add here
```

**Step 3: Write Tests**

```python
# tests/test_features.py

def test_new_feature_calculation(sample_transactions):
    """Test new feature is calculated correctly."""
    engineer = FeatureEngineer()
    features = engineer.engineer_features_from_dataframe(sample_transactions)

    assert 'new_feature_name' in features.columns
    assert features['new_feature_name'].dtype == np.float64
    # More specific assertions
```

### 2. Adding a New Reason Code

**Step 1: Define in Config**

```yaml
# config/reason_codes.yaml

positive_reasons:
  NEW_POSITIVE_REASON:
    code: "P08"
    description: "Short description for UI"
    long_description: "Detailed explanation for borrowers..."
    features: [feature_1, feature_2]
    conditions: ["feature_1 > 0.5"]
    impact: "positive"
    shap_direction: "negative"  # Negative SHAP = reduces risk
    magnitude_thresholds:
      strong: 0.03
      moderate: 0.01
      slight: 0.005
    regulatory_note: "Compliance justification"
```

**Step 2: Test**

```python
# tests/test_explainability.py

def test_new_reason_code():
    """Test new reason code generation."""
    generator = ReasonCodeGenerator()

    shap_values = {'feature_1': -0.04}  # Negative SHAP
    feature_values = {'feature_1': 0.6}  # Meets condition

    codes = generator.generate_reason_codes(shap_values, feature_values)

    # Should include new code
    code_names = [rc.name for rc in codes]
    assert 'NEW_POSITIVE_REASON' in code_names
```

### 3. Adding a New Visualization

**Step 1: Create Chart Function**

```python
# src/demo/visualizations.py

def create_your_new_chart(
    data: pd.DataFrame,
    title: str = "Your Chart Title"
) -> go.Figure:
    """
    Create your new chart type.

    Args:
        data: Input data with required columns
        title: Chart title

    Returns:
        Plotly Figure object

    Example:
        >>> fig = create_your_new_chart(data)
        >>> fig.show()
    """
    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(...))

    # Update layout
    fig.update_layout(
        title={'text': f'<b>{title}</b>', 'font': {'size': 18}},
        **LAYOUT_CONFIG
    )

    return fig
```

**Step 2: Add to Streamlit App**

```python
# src/demo/app.py

from src.demo.visualizations import create_your_new_chart

# In your page function
chart = create_your_new_chart(data)
st.plotly_chart(chart, use_container_width=True)
```

---

## Performance Optimization

### Profiling

```python
# Profile code
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here
result = slow_function()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

### Common Optimizations

**1. Vectorize Operations**

```python
# Slow (loop)
for i, row in df.iterrows():
    df.loc[i, 'result'] = row['a'] + row['b']

# Fast (vectorized)
df['result'] = df['a'] + df['b']
```

**2. Use Caching**

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_computation(param):
    # Cached for repeated calls
    return result
```

**3. Batch Processing**

```python
# Process in chunks
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    process_chunk(chunk)
```

**4. Optimize SHAP**

```python
# Use TreeExplainer (fast)
explainer = shap.TreeExplainer(model, data)

# Sample background data
background = X_train.sample(100, random_state=42)
```

---

## Deployment

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ src/
COPY config/ config/
COPY models/ models/

# Expose Streamlit port
EXPOSE 8501

# Run application
CMD ["streamlit", "run", "src/demo/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:

```bash
docker build -t vantageflow-ai .
docker run -p 8501:8501 vantageflow-ai
```

### Production Checklist

- [ ] All tests passing
- [ ] Code coverage >80%
- [ ] No flake8 warnings
- [ ] Type hints complete
- [ ] Documentation updated
- [ ] Performance tested
- [ ] Security review
- [ ] Error logging configured
- [ ] Monitoring setup
- [ ] Backup strategy

---

## Contributing

### Pull Request Process

1. **Fork** the repository
2. **Create** feature branch
3. **Make** changes with tests
4. **Run** full test suite
5. **Update** documentation
6. **Submit** pull request
7. **Address** review feedback

### Code Review Checklist

- [ ] Code follows PEP 8
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
- [ ] Performance acceptable
- [ ] Security considerations addressed

---

**Happy Coding! 🚀**

*VantageFlow AI - Built with ❤️ for financial inclusion*
