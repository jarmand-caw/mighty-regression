import os
import pandas as pd
import pytest
from mighty_regression.log_init import initialize_logger

initialize_logger()

@pytest.fixture(scope="session")
def directory():
    return os.path.dirname(__file__)


@pytest.fixture(scope="session")
def data(directory):
    return pd.read_csv(os.path.join(directory, "test_data/test.csv"))