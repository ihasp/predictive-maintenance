import random
import time
from typing import Any

import requests
from requests.models import Response

URL = "http://127.0.0.1:8000/telemetry"


class MachineSimulator:
    def __init__(self, machine_id, product_type="M") -> None:
        self.machine_id = machine_id
        self.product_type = product_type
        self.tool_wear = 0
        self.air_temp: float = 298.0

    def generate_data(self) -> dict[str, Any]:
        # Symulacja fizyki: Moment obrotowy i prędkość są ze sobą powiązane
        torque: float = random.uniform(30, 50)
        # Wyższe obroty = wyższa temperatura procesu
        rotational_speed: float = 1500 + random.uniform(-100, 100)
        self.air_temp += random.uniform(-0.1, 0.1)
        process_temp: float = self.air_temp + 10 + (torque / 10)

        # Narzędzie zużywa się z każdym cyklem
        self.tool_wear += random.uniform(0.1, 0.5)

        # Losowa awaria (np. nagły skok zużycia)
        if random.random() > 0.98:
            self.tool_wear += 20

        return {
            "machine_id": self.machine_id,
            "product_type": self.product_type,
            "air_temp": round(self.air_temp, 1),
            "process_temp": round(process_temp, 1),
            "rotational_speed": int(rotational_speed),
            "torque": round(torque, 1),
            "tool_wear": int(self.tool_wear),
        }

    def reset_tool(self) -> None:
        self.tool_wear = 0


if __name__ == "__main__":
    sim: MachineSimulator = MachineSimulator(machine_id="CNC-BIELSKO-01")
    while True:
        data: dict[str, Any] = sim.generate_data()
        try:
            response: Response = requests.post(URL, json=data)
            print(f"Dane wysłane: {data} | Odpowiedź API: {response.json()}")

            if response.json().get("action") == "RESERVE_FROM_STOCK":
                sim.reset_tool()
                print("--- CZESC ZAREZERWOWANA, WYMIANA ZAPLANOWANA ---")
        except Exception as e:
            print(f"Błąd połączenia: {e}")

        time.sleep(1)  # Przesyłaj dane co sekundę
