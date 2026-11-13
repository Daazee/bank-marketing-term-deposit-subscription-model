"""Train-fitted feature encoding used by notebook 03."""
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

class FeatureEngineer:
    """Reproduce the notebook's stratified split and DictVectorizer encoding."""

    def __init__(self):
        self.vectorizer = DictVectorizer(sparse=False)

    def fit_transform(self, X_train: pd.DataFrame):
        return self.vectorizer.fit_transform(
            X_train.to_dict(orient="records")
        )

    def transform(self, X: pd.DataFrame):
        if not hasattr(self.vectorizer, "vocabulary_"):
            raise RuntimeError(
                "FeatureEngineer must be fitted first"
            )

        return self.vectorizer.transform(
            X.to_dict(orient="records")
        )

    def get_feature_names(self) -> list[str]:
        return self.vectorizer.get_feature_names_out().tolist()
