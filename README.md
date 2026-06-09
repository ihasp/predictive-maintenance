## Project Goal

The classic approach to maintenance often leads to two problems:

- reacting too late to failures and costly downtime,
- excessive spare parts inventory, which freezes capital.

The goal of the project is to demonstrate a prototype that recommends inventory actions based on failure risk: observation, part reservation, standard stock replenishment, or urgent ordering.

## How it works

1. `machine_simulation/machine.py` generates sample machine data.
2. `server/server.py` receives telemetry via the `/telemetry` endpoint.
3. The model in `model/maintenance_model.pkl` calculates the risk of failure modes.
4. `server/spare_parts.py` maps failure modes to spare parts and inventory decisions.
5. The API returns the risk, recommended action, order cost, and estimated downtime cost.

## Running the project

```powershell
.\.venv\Scripts\python.exe server\server.py
```

In a second terminal:

```powershell
.\.venv\Scripts\python.exe machine_simulation\machine.py
```
