
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split

@dataclass
class DataPartitions:
    """Raw feature frames and target series for each data partition."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


class DataSplitter:

    def __init__(
        self,
        target_column: str = "y",
        random_state: int = 42,
        test_size: float = 0.2,
        validation_size: float = 0.2,
    ):
        self.target_column = target_column
        self.random_state = random_state
        self.test_size = test_size
        self.validation_size = validation_size

    def split(self, data: pd.DataFrame) -> DataPartitions:

        if self.target_column not in data.columns:
            raise ValueError(
                f"Target column {self.target_column!r} is missing"
            )

        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        # Calculate validation proportion relative to remaining data
        validation_ratio = self.validation_size / (1 - self.test_size)

        X_train, X_validation, y_train, y_validation = train_test_split(
            X_train_val,
            y_train_val,
            test_size=validation_ratio,
            random_state=self.random_state,
            stratify=y_train_val,
        )

        return DataPartitions(
            X_train=X_train,
            X_validation=X_validation,
            X_test=X_test,
            y_train=y_train,
            y_validation=y_validation,
            y_test=y_test,
        )
