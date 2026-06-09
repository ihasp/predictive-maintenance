import sys
from pathlib import Path
from fastapi.applications import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel, Field
import uvicorn

from model import train_model
from model.train_model import TARGET_COLUMNS
from server.spare_parts import FailureMode, optimize_inventory

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


app: FastAPI = FastAPI()

MODEL_PATH = ROOT_DIR / "model" / "maintenance_model.pkl"
MODEL = joblib.load(MODEL_PATH)


class Telemetry(BaseModel):
    machine_id: str
    product_type: str = Field(default="M", pattern="^[LMHlmh]$")
    air_temp: float
    process_temp: float
    rotational_speed: int
    torque: float
    tool_wear: int


@app.post("/telemetry")
async def receive_telemetry(data: Telemetry):
    input_df = build_model_input(data)

    prediction_probs = MODEL.predict_proba(input_df)
    predicted_modes = TARGET_COLUMNS[: len(prediction_probs)]

    risks: dict[FailureMode, float] = {
        FailureMode(failure_mode): float(prediction_probs[index][0][1])
        for index, failure_mode in enumerate(predicted_modes)
    }

    for failure_mode in TARGET_COLUMNS:
        risks.setdefault(FailureMode(failure_mode), 0.0)

    optimization = optimize_inventory(risks)

    return {
        "status": "received",
        "machine_id": data.machine_id,
        "failure_risks": {
            mode.value if hasattr(mode, "value") else mode: round(risk, 3)
            for mode, risk in risks.items()
        },
        "action": optimization["overall_action"],
        "inventory_optimization": optimization,
    }


def build_model_input(data: Telemetry) -> pd.DataFrame:
    row = {
        "Air temperature [K]": data.air_temp,
        "Process temperature [K]": data.process_temp,
        "Rotational speed [rpm]": data.rotational_speed,
        "Torque [Nm]": data.torque,
        "Tool wear [min]": data.tool_wear,
    }

    if hasattr(MODEL, "named_steps"):
        row["Type"] = data.product_type.upper()

    return pd.DataFrame([row])


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
