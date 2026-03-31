from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal
import warnings

import numpy as np
import pandas as pd
import scipy

from .common import LoadedData, extract_single_unit


RegressionKind = Literal["normal", "lognormal"]


@dataclass(frozen=True)
class RegressionConfig:
    label: str = "Load"
    regression_kind: RegressionKind = "normal"
    regression_y_term: str = "lag_1"
    use_day_type: bool = False
    use_month: bool = False
    shift: bool = True
    end_date: pd.Timestamp | date | datetime | None = None
    random_seed: int = 42
    n_samples: int = 10
    annual_growth_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.regression_kind not in {"normal", "lognormal"}:
            raise ValueError("regression_kind must be either 'normal' or 'lognormal'")
        if self.regression_y_term != "lag_1":
            raise ValueError("Only lag_1 regression is currently supported")
        if self.n_samples < 1:
            raise ValueError("n_samples must be at least 1")
        if self.annual_growth_rate < 0:
            raise ValueError("annual_growth_rate must be non-negative")

    @property
    def normalized_label(self) -> str:
        return self.label.lower()


@dataclass(frozen=True)
class RegressionFitResult:
    prediction_df: pd.DataFrame
    grouped_pairs: dict[str, tuple[np.ndarray, np.ndarray]]
    predictions_by_index: dict[str, dict[str, float]]

    @property
    def indices(self) -> list[str]:
        return list(self.grouped_pairs)


@dataclass(frozen=True)
class RegressionRunResult:
    processed_df: pd.DataFrame
    fit_result: RegressionFitResult
    output_df: pd.DataFrame
    resolved_end_date: pd.Timestamp

    @property
    def prediction_df(self) -> pd.DataFrame:
        return self.fit_result.prediction_df


def linear_regression(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if len(x) <= 2 or len(y) <= 2:
        raise ValueError("At least three observations are required for regression.")

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)

    y_predictions = intercept + slope * x

    steyx = np.sqrt(np.sum((y - y_predictions) ** 2) / (len(y) - 2))
    mean_reversion = -float(slope)
    long_run_mean = float(intercept / -slope)
    volatility = abs(float((steyx / long_run_mean) * 100))

    return {
        "slope": slope,
        "intercept": intercept,
        "p_value": p_value,
        "r_value": r_value,
        "steyx": steyx,
        "mean_reversion": mean_reversion,
        "long_run_mean": long_run_mean,
        "volatility": volatility,
    }


def load_regression_input(source: Any, *, label: str = "Load") -> LoadedData:
    try:
        input_df = pd.read_csv(source, header=[0, 1], parse_dates=True)
    except Exception as exc:
        raise ValueError(f"Error reading file: {exc}") from exc

    unit = extract_single_unit(input_df.columns, skip=1)
    input_df.columns = [str(column[0]).strip() for column in input_df.columns]

    required_columns = ["date", label.lower()]
    if not all(column in input_df.columns for column in required_columns):
        raise ValueError(
            "Data is missing required columns:\n\n\n"
            f"{','.join(required_columns)}\n\n\n"
            "Instead found the following:\n\n\n"
            f"{','.join(input_df.columns)}"
        )

    warnings_list: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            parsed_dates = pd.to_datetime(input_df["date"])
    except Exception as exc:
        raise ValueError(f"Error parsing date column: {exc}") from exc

    warnings_list.extend(
        f"Warning when parsing date column: {warning.message}"
        for warning in caught_warnings
    )

    duplicated_dates = parsed_dates[parsed_dates.duplicated()].unique()
    if len(duplicated_dates) > 0:
        warnings_list.insert(
            0,
            "Duplicated dates found.\n"
            f"Averaging values for dates: {list(duplicated_dates)}",
        )

    try:
        normalized_df = (
            input_df.assign(date=parsed_dates)
            .sort_values("date")
            .set_index("date")
            .dropna()
            .resample("1D")
            .mean()
            .interpolate()
        )
    except Exception as exc:
        raise ValueError(f"Error parsing date column: {exc}") from exc

    return LoadedData(data=normalized_df, unit=unit, warnings=tuple(warnings_list))


def build_parameter_index(
    df: pd.DataFrame,
    *,
    use_day_type: bool,
    use_month: bool,
) -> pd.Series:
    if use_day_type and use_month:
        return df["day_type"] + "-" + df["month"]
    if use_day_type:
        return df["day_type"]
    if use_month:
        return df["month"]
    return pd.Series("all", index=df.index)


def prepare_regression_dataframe(
    input_df: pd.DataFrame,
    *,
    label: str = "Load",
    use_day_type: bool = False,
    use_month: bool = False,
    shift: bool = True,
) -> pd.DataFrame:
    normalized_label = label.lower()
    if normalized_label not in input_df.columns:
        raise KeyError(f"Expected column '{normalized_label}' in input_df")

    df = (
        input_df.resample("1D")
        .mean()
        .interpolate()
        .reset_index()
        .assign(date=lambda value: pd.to_datetime(value["date"]))
        .set_index("date")
    )

    df = (
        df.assign(
            month=lambda value: value.index.strftime("%b"),
            day_type=lambda value: np.where(
                value.index.dayofweek < 5, "Weekday", "Weekend"
            ),
            normal_values=lambda value: value[normalized_label],
            lognormal_values=lambda value: np.log(value["normal_values"].values),
            normal_lag_1_values=lambda value: value["normal_values"].shift(1),
            lognormal_lag_1_values=lambda value: value["lognormal_values"].shift(1),
        )
        .assign(
            month=lambda value: value["month"].shift(1 if shift else 0),
            day_type=lambda value: value["day_type"].shift(1 if shift else 0),
        )
        .dropna()
    )

    return df.assign(
        parameter_index=build_parameter_index(
            df,
            use_day_type=use_day_type,
            use_month=use_month,
        )
    )


def ordered_parameter_indices(
    *,
    use_day_type: bool,
    use_month: bool,
) -> list[str]:
    if use_day_type and use_month:
        return [
            f"{day_type}-{calendar.month_abbr[month]}"
            for month in range(1, 13)
            for day_type in ["Weekday", "Weekend"]
        ]
    if use_day_type:
        return ["Weekday", "Weekend"]
    if use_month:
        return [calendar.month_abbr[month] for month in range(1, 13)]
    return ["all"]


def group_regression_pairs(
    processed_df: pd.DataFrame,
    *,
    regression_kind: RegressionKind,
    regression_y_term: str,
    use_day_type: bool,
    use_month: bool,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    grouped_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for index in ordered_parameter_indices(
        use_day_type=use_day_type,
        use_month=use_month,
    ):
        if index == "all":
            group_df = processed_df
        else:
            group_df = processed_df.query("parameter_index == @index")

        grouped_pairs[index] = (
            group_df[f"{regression_kind}_values"].values,
            group_df[f"{regression_kind}_{regression_y_term}_values"].values,
        )
    return grouped_pairs


def fit_regression_parameters(
    processed_df: pd.DataFrame,
    *,
    regression_kind: RegressionKind,
    regression_y_term: str = "lag_1",
    use_day_type: bool = False,
    use_month: bool = False,
) -> RegressionFitResult:
    grouped_pairs = group_regression_pairs(
        processed_df,
        regression_kind=regression_kind,
        regression_y_term=regression_y_term,
        use_day_type=use_day_type,
        use_month=use_month,
    )

    results = []
    for index, (x_values, y_values) in grouped_pairs.items():
        try:
            results.append(linear_regression(x_values, y_values))
        except ValueError as exc:
            raise ValueError(
                f"Unable to fit regression for group '{index}': {exc}"
            ) from exc

    prediction_df = pd.DataFrame(index=list(grouped_pairs), data=results)
    predictions_by_index = {
        index: dict(prediction_df.loc[index]) for index in prediction_df.index
    }

    return RegressionFitResult(
        prediction_df=prediction_df,
        grouped_pairs=grouped_pairs,
        predictions_by_index=predictions_by_index,
    )


def resolve_prediction_index(
    dt: pd.Timestamp,
    *,
    use_day_type: bool,
    use_month: bool,
) -> str:
    if use_day_type and use_month:
        day_type = "Weekday" if dt.day_of_week < 5 else "Weekend"
        month = calendar.month_abbr[dt.month]
        return f"{day_type}-{month}"
    if use_day_type:
        return "Weekday" if dt.day_of_week < 5 else "Weekend"
    if use_month:
        return calendar.month_abbr[dt.month]
    return "all"


def resolve_end_date(
    input_df: pd.DataFrame,
    end_date: pd.Timestamp | date | datetime | None,
) -> pd.Timestamp:
    if end_date is not None:
        return pd.Timestamp(end_date)
    return pd.Timestamp(input_df.index[-1]) + pd.Timedelta(days=len(input_df.index))


def generate_regression_sample(
    predictions: list[dict[str, float]],
    *,
    processed_df: pd.DataFrame,
    regression_kind: RegressionKind,
    rng: np.random.RandomState,
) -> np.ndarray:
    output_values = np.full(
        len(predictions), processed_df["normal_values"].values[-1]
    ).astype("float64")

    randomness = scipy.stats.norm.ppf(rng.rand(len(output_values)))

    slopes = np.array([prediction["slope"] for prediction in predictions])
    steyxes = np.array([prediction["steyx"] for prediction in predictions])
    intercepts = np.array([prediction["intercept"] for prediction in predictions])

    for index in range(1, len(output_values)):
        if regression_kind == "normal":
            output_values[index] = (
                intercepts[index]
                + slopes[index] * output_values[index - 1]
                + steyxes[index] * randomness[index]
            )
        else:
            output_values[index] = np.exp(
                intercepts[index]
                + slopes[index] * np.log(output_values[index - 1])
                + steyxes[index] * randomness[index]
            )

    return output_values


def generate_regression_samples(
    input_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    fit_result: RegressionFitResult,
    *,
    config: RegressionConfig,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    resolved_end_date = resolve_end_date(input_df, config.end_date)
    output_dates = pd.date_range(
        start=processed_df.index[-1], end=resolved_end_date, freq="D"
    )

    output_df = pd.DataFrame(data={"date": output_dates}).set_index("date")
    predictions = [
        fit_result.predictions_by_index[
            resolve_prediction_index(
                output_date,
                use_day_type=config.use_day_type,
                use_month=config.use_month,
            )
        ]
        for output_date in output_dates
    ]

    long_run_mean = input_df[config.normalized_label].mean()
    growth_multiplier = (
        long_run_mean
        * np.arange(len(output_df))
        * (config.annual_growth_rate / 365 / 100)
    )

    rng = np.random.RandomState(config.random_seed)
    for sample_index in range(config.n_samples):
        output_values = generate_regression_sample(
            predictions,
            processed_df=processed_df,
            regression_kind=config.regression_kind,
            rng=rng,
        )
        output_df[f"sample_{sample_index + 1}"] = output_values + growth_multiplier

    return output_df, resolved_end_date


def run_regression_analysis(
    input_df: pd.DataFrame,
    config: RegressionConfig,
) -> RegressionRunResult:
    processed_df = prepare_regression_dataframe(
        input_df,
        label=config.label,
        use_day_type=config.use_day_type,
        use_month=config.use_month,
        shift=config.shift,
    )
    fit_result = fit_regression_parameters(
        processed_df,
        regression_kind=config.regression_kind,
        regression_y_term=config.regression_y_term,
        use_day_type=config.use_day_type,
        use_month=config.use_month,
    )
    output_df, resolved_end_date = generate_regression_samples(
        input_df,
        processed_df,
        fit_result,
        config=config,
    )
    return RegressionRunResult(
        processed_df=processed_df,
        fit_result=fit_result,
        output_df=output_df,
        resolved_end_date=resolved_end_date,
    )
