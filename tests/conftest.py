"""
Pytest configuration and shared fixtures.

Provides common fixtures and pytest hooks for the entire test suite.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# ==================== PYTEST CONFIGURATION ====================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# ==================== SHARED FIXTURES ====================

@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture(scope="session")
def test_models_dir(tmp_path_factory):
    """Create temporary directory for test models."""
    return tmp_path_factory.mktemp("test_models")


@pytest.fixture(scope="session")
def test_output_dir(tmp_path_factory):
    """Create temporary directory for test outputs."""
    return tmp_path_factory.mktemp("test_output")


@pytest.fixture
def random_seed():
    """Provide consistent random seed for tests."""
    return 42


# ==================== PYTEST HOOKS ====================

def pytest_collection_modifyitems(config, items):
    """Modify test items during collection."""
    # Auto-add markers based on test names
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid.lower() or "pipeline" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)

        # Mark slow tests
        if "test_train_xgboost" in item.nodeid or "test_full_pipeline" in item.nodeid:
            item.add_marker(pytest.mark.slow)


def pytest_report_header(config):
    """Add custom header to pytest output."""
    return [
        "VantageFlow AI Test Suite v1.0.0",
        "=" * 60,
        "Testing modules: data_generation, features, models, explainability",
        "Target coverage: >80%"
    ]
