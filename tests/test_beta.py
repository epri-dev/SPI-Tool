import pandas as pd
import pytest
import os

from spi_tool import BetaConfig, load_beta_input, run_beta_analysis

CURRENT_DIRECTORY = os.path.dirname(__file__)


def test_get_data_valid_file():
    filename = os.path.join(CURRENT_DIRECTORY, "data/valid_carbon_prices.csv")
    loaded_data = load_beta_input(filename)
    assert not loaded_data.data.empty
    assert "date" in loaded_data.data.index.name
    assert loaded_data.unit == "2022 $/MTCO2e"


def test_get_data_with_duplicates():
    filename = os.path.join(CURRENT_DIRECTORY, "data/duplicated_dates.csv")
    loaded_data = load_beta_input(filename)
    assert loaded_data.warnings
    assert "Duplicated dates found" in loaded_data.warnings[0]


def test_get_data_invalid_file():
    filename = "tests/data/invalid_file.csv"
    with pytest.raises(Exception):
        load_beta_input(filename)


def test_beta_library_api_generates_samples():
    filename = os.path.join(CURRENT_DIRECTORY, "data/valid_carbon_prices.csv")
    loaded_data = load_beta_input(filename)
    analysis = run_beta_analysis(
        loaded_data.data,
        BetaConfig(
            alpha=2,
            beta=5,
            random_seed=42,
            n_samples=4,
            scenario_bound_1="value1",
            scenario_bound_2="value2",
        ),
    )

    assert list(analysis.prediction_df.index) == [
        "sample_1",
        "sample_2",
        "sample_3",
        "sample_4",
    ]
    assert list(analysis.output_df.columns) == [
        "sample_1",
        "sample_2",
        "sample_3",
        "sample_4",
    ]
