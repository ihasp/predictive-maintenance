import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi.applications import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
import uvicorn

from model import train_model

app: FastAPI = FastAPI()

MODEL_PATH = ROOT_DIR / "model" / "maintenance_model.pkl"
MODEL = joblib.load(MODEL_PATH)


class Telemetry(BaseModel):
    machine_id: str
    air_temp: float
    process_temp: float
    rotational_speed: int
    torque: float
    tool_wear: int


@app.post("/telemetry")
async def receive_telemetry(data: Telemetry):
    # Przygotowanie danych do formatu, który rozumie model
    input_df = pd.DataFrame(
        [
            {
                "Air temperature [K]": data.air_temp,
                "Process temperature [K]": data.process_temp,
                "Rotational speed [rpm]": data.rotational_speed,
                "Torque [Nm]": data.torque,
                "Tool wear [min]": data.tool_wear,
            }
        ]
    )

    # Predykcja (multi-output)
    prediction_probs = MODEL.predict_proba(input_df)

    # Przykładowa logika dla TWF (Tool Wear Failure)
    # prediction_probs[0] to zazwyczaj TWF w Twoim modelu
    twf_risk = prediction_probs[0][0][1]

    response = {"status": "received", "twf_risk": round(twf_risk, 2), "action": "NONE"}

    if twf_risk > 0.5:
        # Logika biznesowa: zamówienie części i wymiana
        response["action"] = "REPLACE_TOOL"
        response["message"] = "Risk high! Ordering new tool and scheduling maintenance."
        # Tutaj mógłbyś dodać wpis do bazy danych lub wysłać maila do UR

    return response


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
