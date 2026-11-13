"""FastAPI endpoints for bank marketing subscription predictions."""

import tomllib
from pathlib import Path
from typing import Any, Literal
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
with (PROJECT_ROOT / "pyproject.toml").open("rb") as config_file:
    config = tomllib.load(config_file)

model_settings = config["model"]["settings"]
model_name: str = model_settings["model-name"]
model_directory = PROJECT_ROOT / model_settings["model-path"]
model_path = max(
    model_directory.glob(f"*{model_name}*.bin"),
    key=lambda path: path.stat().st_mtime,
    default=None,
)
if model_path is None:
    raise FileNotFoundError(
        f"No model matching {model_name!r} found in {model_directory}"
    )

pipeline = joblib.load(model_path)
vectorizer_path = model_directory / "dv.bin"
vectorizer = joblib.load(vectorizer_path)

app = FastAPI(title="Bank Marketing Campaign Prediction")


class BankMarketingInput(BaseModel):
    """Validated bank client and campaign attributes accepted by the API."""

    model_config = ConfigDict(extra="forbid")

    age: int = Field(ge=18, le=120)
    job: Literal[
        "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
        "retired", "self-employed", "services", "student", "technician",
        "unemployed", "unknown",
    ]
    marital: Literal["divorced", "married", "single", "unknown"]
    education: Literal[
        "basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
        "professional.course", "university.degree", "unknown",
    ]
    default: Literal["no", "yes", "unknown"]
    housing: Literal["no", "yes", "unknown"]
    loan: Literal["no", "yes", "unknown"]
    contact: Literal["cellular", "telephone"]
    month: Literal[
        "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep",
        "oct", "nov", "dec",
    ]
    day_of_week: Literal["mon", "tue", "wed", "thu", "fri"]
    campaign: int = Field(ge=1)
    pdays: int = Field(ge=0)
    previous: int = Field(ge=0)
    poutcome: Literal["failure", "nonexistent", "success"]
    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: float


@app.get("/health")
def health() -> dict[str, Any]:
    """Report service and configured model readiness."""
    return {
        "status": "ok",
        "model_name": model_name,
        "model_loaded": pipeline is not None and vectorizer is not None,
    }


@app.post("/predict")
def predict(bankMarketingInput: BankMarketingInput) -> dict[str, Any]:
    """Return the model's estimated probability of term-deposit subscription."""
    record = bankMarketingInput.model_dump()
    # Match the training data preparation: these fields were excluded before
    # fitting the DictVectorizer and model pipeline.
    for column in ("month", "day_of_week", "campaign"):
        record.pop(column)

    frame = pd.DataFrame([record])
    encoded = vectorizer.transform(frame.to_dict(orient="records"))
    probability = float(pipeline.predict_proba(encoded)[0, 1])
    return {
        "conversion_probability": round(probability, 6),
        "conversion": probability >= 0.5,
    }


#sample request returning true
# {
#   "age": 58,
#   "job": "admin.",
#   "marital": "divorced",
#   "education": "basic.4y",
#   "default": "no",
#   "housing": "yes",
#   "loan": "no",
#   "contact": "cellular",
#   "month": "jan",
#   "day_of_week": "mon",
#   "campaign": 1,
#   "pdays": 0,
#   "previous": 0,
#   "poutcome": "nonexistent",
#   "emp_var_rate": 0,
#   "cons_price_idx": 92.893,
#   "cons_conf_idx": -46.2,
#   "euribor3m": 1.344,
#   "nr_employed": 0
# }


#sample request returning false
{
  "age": 18,
  "job": "admin.",
  "marital": "divorced",
  "education": "basic.4y",
  "default": "no",
  "housing": "no",
  "loan": "no",
  "contact": "cellular",
  "month": "jan",
  "day_of_week": "mon",
  "campaign": 1,
  "pdays": 0,
  "previous": 0,
  "poutcome": "failure",
  "emp_var_rate": 0,
  "cons_price_idx": 0,
  "cons_conf_idx": 0,
  "euribor3m": 0,
  "nr_employed": 0
}