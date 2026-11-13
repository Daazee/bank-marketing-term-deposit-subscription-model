
from pathlib import Path
import pandas as pd


def load_data(filepath: str | Path, *, sep: str = ",", **read_csv_kwargs: object) -> pd.DataFrame:
    """Load a CSV file into a DataFrame with a useful missing-file error."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    try:
        return pd.read_csv(path, sep=sep, **read_csv_kwargs)
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise ValueError(f"Could not read dataset at {path}: {exc}") from exc
