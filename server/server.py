import sys
from pathlib import Path

import joblib
import pandas as pd
import uvicorn
from fastapi.applications import FastAPI
from pydantic import BaseModel, Field
from spare_parts import optimize_inventory

TARGET_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

app: FastAPI = FastAPI()

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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

    risks: dict[str, float] = {
        failure_mode: float(prediction_probs[index][0][1])
        for index, failure_mode in enumerate(predicted_modes)
    }

    for failure_mode in TARGET_COLUMNS:
        risks.setdefault(failure_mode, 0.0)

    optimization = optimize_inventory(risks)
    action = optimization["overall_action"]

    if action != "NO_ACTION":
        print(f"\n⚠️  WARNING: Machine {data.machine_id} requires {action}!")
        for mode, risk in risks.items():
            if risk > 0.05:
                print(f"    -> Elevated risk of {mode}: {risk:.1%}")
        print("-" * 50)

    return {
        "status": "received",
        "machine_id": data.machine_id,
        "failure_risks": {mode: round(risk, 3) for mode, risk in risks.items()},
        "action": action,
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
