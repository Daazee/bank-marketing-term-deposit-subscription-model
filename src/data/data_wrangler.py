"""Wrangling operations reproduced from the data preparation notebook."""

import pandas as pd

TARGET_COLUMN = "y"
EXCLUDED_COLUMNS = ("duration", "campaign", "month", "day_of_week")


def standardize_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with lower-case names and periods replaced by underscores."""
    result = data.copy()
    result.columns = result.columns.str.replace(".", "_", regex=False).str.lower()
    return result


def wrangle_data(
    data: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
    excluded_columns: tuple[str, ...] = EXCLUDED_COLUMNS,
) -> pd.DataFrame:
    """Standardize column names and remove the columns excluded in notebook 02.

    The source notebook calls all four exclusions leakage. ``duration`` is only
    known after a call; campaign count and calendar fields may be known earlier.
    They are retained as exclusions here to preserve the experiment.
    """
    result = standardize_columns(data)
    if target_column not in result.columns:
        raise ValueError(f"Target column {target_column!r} is not present")
    return result.drop(columns=[c for c in excluded_columns if c in result.columns])
