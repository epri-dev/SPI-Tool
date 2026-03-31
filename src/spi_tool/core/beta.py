from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy

from .common import LoadedData, extract_single_unit


@dataclass(frozen=True)
class BetaConfig:
    alpha: float = 2.0
    beta: float = 5.0
    random_seed: int = 42
    n_samples: int = 10
    scenario_bound_1: str | None = None
    scenario_bound_2: str | None = None

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError("alpha must be greater than 0")
        if self.beta <= 0:
            raise ValueError("beta must be greater than 0")
        if self.n_samples < 1:
            raise ValueError("n_samples must be at least 1")


@dataclass(frozen=True)
class BetaRunResult:
    prediction_df: pd.DataFrame
    output_df: pd.DataFrame
    scenario_bound_1: str
    scenario_bound_2: str


def load_beta_input(source: Any) -> LoadedData:
    try:
        df = pd.read_csv(source, header=[0, 1], index_col=0, parse_dates=True)
    except Exception as exc:
        raise ValueError(f"Error reading file: {exc}") from exc

    unit = extract_single_unit(df.columns)
    warnings: list[str] = []

    df.index.name = "date"
    if df.index.duplicated().any():
        duplicated_dates = df.index[df.index.duplicated()].unique()
        warnings.append(
            "Duplicated dates found.\n"
            f"Averaging values for dates: {list(duplicated_dates)}"
        )

    df.columns = [str(column[0]).strip() for column in df.columns]
    if len(df.columns) < 2:
        raise ValueError("Data must include at least two scenario columns.")

    normalized_df = df.ffill().bfill().resample("1YS").mean().interpolate()
    return LoadedData(data=normalized_df, unit=unit, warnings=tuple(warnings))


def resolve_beta_bounds(
    input_df: pd.DataFrame,
    scenario_bound_1: str | None = None,
    scenario_bound_2: str | None = None,
) -> tuple[str, str]:
    if input_df is None or input_df.empty:
        raise ValueError("input_df must contain at least one scenario column.")

    lower_bound = scenario_bound_1 or input_df.columns[0]
    upper_bound = scenario_bound_2 or input_df.columns[-1]

    if lower_bound not in input_df.columns:
        raise KeyError(f"Unknown lower bound scenario: {lower_bound}")
    if upper_bound not in input_df.columns:
        raise KeyError(f"Unknown upper bound scenario: {upper_bound}")

    return lower_bound, upper_bound


def generate_beta_scaling_factors(config: BetaConfig) -> pd.DataFrame:
    rng = np.random.RandomState(config.random_seed)
    samples = [f"sample_{index + 1}" for index in range(config.n_samples)]
    random_values = rng.rand(config.n_samples)

    prediction_df = pd.DataFrame(
        data=scipy.stats.beta.ppf(random_values, config.alpha, config.beta),
        index=samples,
        columns=["scaling_factor"],
    )
    prediction_df.index.name = "sample"
    return prediction_df


def generate_beta_samples(
    input_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    *,
    scenario_bound_1: str | None = None,
    scenario_bound_2: str | None = None,
) -> tuple[pd.DataFrame, str, str]:
    lower_bound, upper_bound = resolve_beta_bounds(
        input_df,
        scenario_bound_1=scenario_bound_1,
        scenario_bound_2=scenario_bound_2,
    )

    low_df = input_df[lower_bound]
    high_df = input_df[upper_bound]

    output_samples = {}
    for sample in prediction_df.index:
        output_samples[sample] = (high_df - low_df) * prediction_df.loc[
            sample, "scaling_factor"
        ] + low_df

    return pd.DataFrame(output_samples), lower_bound, upper_bound


def run_beta_analysis(
    input_df: pd.DataFrame,
    config: BetaConfig,
) -> BetaRunResult:
    prediction_df = generate_beta_scaling_factors(config)
    output_df, lower_bound, upper_bound = generate_beta_samples(
        input_df,
        prediction_df,
        scenario_bound_1=config.scenario_bound_1,
        scenario_bound_2=config.scenario_bound_2,
    )
    return BetaRunResult(
        prediction_df=prediction_df,
        output_df=output_df,
        scenario_bound_1=lower_bound,
        scenario_bound_2=upper_bound,
    )
