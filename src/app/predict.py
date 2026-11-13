import os
import tomllib
from pathlib import Path
from typing import Any
from fastapi import FastAPI
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
import joblib
from utils.file_utils import get_latest_model

toml_path = Path("pyproject.toml")

with toml_path.open("rb") as f:
    data = tomllib.load(f)

# Access values like a standard dictionary
project_name = data["project"]["name"]
model_name = data["model.settings"]["model-name"]



# MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).with_name("pipeline.bin")))
# with MODEL_PATH.open("rb") as model_file:
#     pipeline_path = get_latest_model("logistic_model_pipeline_with_standardscaler")
#     pipeline = joblib.load(pipeline_path)

app = FastAPI(title="Bank Marketing Campaign Prediction")

class BankMarketingInput(BaseModel):
    """Validated input features for bank marketing predictions."""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid"
    )

    # Bank client data
    age: int = Field(ge=18, le=120)
    job: Literal[
        "admin.", "blue-collar", "entrepreneur", "housemaid",
        "management", "retired", "self-employed", "services",
        "student", "technician", "unemployed", "unknown"
    ]
    marital: Literal["divorced", "married", "single", "unknown"]
    education: Literal[
        "basic.4y", "basic.6y", "basic.9y", "high.school",
        "illiterate", "professional.course",
        "university.degree", "unknown"
    ]
    default: Literal["no", "yes", "unknown"]
    housing: Literal["no", "yes", "unknown"]
    loan: Literal["no", "yes", "unknown"]

    # Current campaign contact information
    contact: Literal["cellular", "telephone"]
    month: Literal[
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec"
    ]
    day_of_week: Literal["mon", "tue", "wed", "thu", "fri"]

    # Other campaign attributes
    campaign: int = Field(ge=1)
    pdays: int = Field(ge=0)
    previous: int = Field(ge=0)
    poutcome: Literal["failure", "nonexistent", "success"]

    # Social and economic context
    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: float


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_name": model_name
        }


# @app.post("/predict")
# def predict(bankMarketingInput: BankMarketingInput) -> dict[str, Any]:
#     record = normalize_record(lead.model_dump())
#     probability = float(pipeline.predict_proba([record])[0, 1])
#     return {
#         "conversion_probability": round(probability, 6),
#         "conversion": probability >= 0.5,
#     }