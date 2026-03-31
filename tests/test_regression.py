import os
import pandas as pd

from spi_tool import RegressionConfig, load_regression_input, run_regression_analysis

CURRENT_DIRECTORY = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_DIRECTORY)
DEFAULT_LOAD_FILE = os.path.join(
    PROJECT_ROOT,
    "src",
    "spi_tool",
    "resources",
    "data",
    "miso-daily-demand.csv",
)


def test_load_regression_input_with_duplicates():
    filename = os.path.join(CURRENT_DIRECTORY, "data/historical_demand.csv")
    loaded_data = load_regression_input(filename, label="Load")
    assert len(loaded_data.data.index) == 366


def test_regression_library_api_using_lag_1():
    loaded_data = load_regression_input(DEFAULT_LOAD_FILE, label="Load")
    analysis = run_regression_analysis(
        loaded_data.data,
        RegressionConfig(label="Load", regression_y_term="lag_1"),
    )

    assert analysis.output_df.shape == (3107, 10)
    assert analysis.output_df.index[0] == pd.Timestamp("2023-12-31")
    assert analysis.output_df.index[-1] == pd.Timestamp("2032-07-02")
    pd.testing.assert_frame_equal(
        analysis.output_df[["sample_1", "sample_10"]].head(3),
        pd.DataFrame(
            {
                "sample_1": [
                    1651806.0,
                    1813397.676700982,
                    1865300.4611644207,
                ],
                "sample_10": [
                    1651806.0,
                    1657912.8064094293,
                    1588483.4483864766,
                ],
            },
            index=pd.DatetimeIndex(
                ["2023-12-31", "2024-01-01", "2024-01-02"],
                name="date",
            ),
        ),
    )


def test_lognormal_regression_library_api_using_lag_1():
    loaded_data = load_regression_input(DEFAULT_LOAD_FILE, label="Load")
    analysis = run_regression_analysis(
        loaded_data.data,
        RegressionConfig(
            label="Load",
            regression_y_term="lag_1",
            regression_kind="lognormal",
        ),
    )

    assert analysis.output_df.shape == (3107, 10)
    assert analysis.output_df.index[0] == pd.Timestamp("2023-12-31")
    assert analysis.output_df.index[-1] == pd.Timestamp("2032-07-02")
    pd.testing.assert_frame_equal(
        analysis.output_df[["sample_1", "sample_10"]].head(3),
        pd.DataFrame(
            {
                "sample_1": [
                    1651806.0,
                    1809062.4064356107,
                    1861863.7960334213,
                ],
                "sample_10": [
                    1651806.0,
                    1657237.8093551004,
                    1593375.3515773404,
                ],
            },
            index=pd.DatetimeIndex(
                ["2023-12-31", "2024-01-01", "2024-01-02"],
                name="date",
            ),
        ),
    )
