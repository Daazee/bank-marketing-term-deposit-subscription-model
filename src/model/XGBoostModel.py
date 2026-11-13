from pathlib import Path
from typing import Any

import joblib
import numpy as np
import xgboost as xgb


class XGBoostModel:
    """Small wrapper around the ``xgboost.train`` API used in notebook 04."""

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        num_boost_round: int = 10,
        feature_names: list[str] | None = None,
    ):
        self.params = dict(params or {})
        self.num_boost_round = num_boost_round
        self.feature_names = feature_names
        self.model: xgb.Booster | None = None

    def _matrix(self, X: Any, y: Any | None = None) -> xgb.DMatrix:
        names = list(X.columns) if hasattr(X, "columns") else self.feature_names
        return xgb.DMatrix(X, label=y, feature_names=names)

    def fit(self, X_train: Any, y_train: Any) -> "XGBoostModel":
        self.model = xgb.train(
            self.params,
            self._matrix(X_train, y_train),
            num_boost_round=self.num_boost_round,
        )
        return self

    def predict(self, X: Any, threshold) -> Any:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def predict_proba(self, X: Any) -> Any:
        if self.model is None:
            raise RuntimeError("Model has not been fitted or loaded")
        positive_probability = self.model.predict(self._matrix(X))
        return np.column_stack((1.0 - positive_probability, positive_probability))

    def save_model(self, filepath: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save a model before fitting")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load_model(cls, filepath: str | Path) -> "XGBoostModel":
        model = joblib.load(filepath)
        if not isinstance(model, cls):
            raise TypeError(f"Saved object at {filepath} is not an XGBoostModel")
        return model
