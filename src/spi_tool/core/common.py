from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class LoadedData:
    data: pd.DataFrame
    unit: str
    warnings: tuple[str, ...] = ()


def extract_single_unit(columns, *, skip: int = 0) -> str:
    units = []
    for column in columns[skip:]:
        if not isinstance(column, tuple) or len(column) < 2:
            continue
        unit = str(column[1]).strip()
        if unit:
            units.append(unit)

    unique_units = list(dict.fromkeys(units))
    if not unique_units:
        return ""
    if len(unique_units) > 1:
        raise ValueError(
            "Expected a single unit in the CSV header, "
            f"but found multiple units: {', '.join(unique_units)}"
        )
    return unique_units[0]


def warnings_to_message(messages: Iterable[str]) -> str:
    return "\n".join(message for message in messages if message)
