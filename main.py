import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from helpers.dataset import move_dataset
from helpers.dataset import use_original_path


def optimize_spare_parts(data_path):
    print("Wczytywanie i przygotowywanie danych...")
    # Wczytanie danych z pliku pobranego z Kaggle
    if data_path is None:
        ValueError("Dataset not defined, check if it moved properly")

    df = pd.read_csv(data_path)

    # Cechy wejściowe (sensory)
    features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    # Typy awarii (Target) - mapujemy je na konkretne części zamienne
    failure_types = ["TWF", "HDF", "PWF", "OSF"]

    spare_parts_mapping = {
        "TWF": "Narzędzia robocze (wymiana eksploatacyjna)",
        "HDF": "Komponenty układu chłodzenia",
        "PWF": "Części układu zasilania i napędu",
        "OSF": "Elementy konstrukcyjne i łożyska",
    }

    X = df[features]
    y = df[failure_types]

    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Trenowanie modelu Predictive Maintenance (Random Forest)...")
    # Trenowanie modelu MultiOutput (przewiduje każdy typ awarii niezależnie)
    model = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)

    # Ewaluacja modelu
    print("\nRaport klasyfikacji (jakość przewidywania na danych testowych):")
    y_pred = model.predict(X_test)
    for i, col in enumerate(failure_types):
        print(f"\n--- Typ awarii: {col} ({spare_parts_mapping[col]}) ---")
        print(classification_report(y_test.iloc[:, i], y_pred[:, i], zero_division=0))

    # Symulacja: Działanie systemu dla nowych odczytów z maszyn na hali
    print("\n=== SYMULACJA ZAMÓWIEŃ CZĘŚCI ZAMIENNYCH ===")

    # Przykładowe aktualne odczyty z 3 maszyn na hali produkcyjnej
    current_sensors = pd.DataFrame(
        {
            "Air temperature [K]": [298.1, 303.5, 298.9],
            "Process temperature [K]": [308.6, 313.0, 309.1],
            "Rotational speed [rpm]": [
                1551,
                1340,
                2800,
            ],  # Maszyna 3: nienaturalnie wysokie obroty
            "Torque [Nm]": [
                42.8,
                65.2,
                15.0,
            ],  # Maszyna 2: bardzo wysoki moment obrotowy
            "Tool wear [min]": [45, 215, 10],  # Maszyna 2: mocno zużyte narzędzie
        }
    )

    # Przewidywanie PRAWDOPODOBIEŃSTWA poszczególnych awarii
    probabilities = model.predict_proba(current_sensors)

    # Próg decyzyjny (np. jeśli ryzyko awarii > 40%, zamawiamy część)
    THRESHOLD = 0.40

    for machine_id in range(len(current_sensors)):
        print(f"\n[ Maszyna ID: {machine_id + 1} ]")
        orders_placed = False

        for i, failure_mode in enumerate(failure_types):
            # predict_proba zwraca listę tablic dla każdego targetu
            prob_of_failure = probabilities[i][machine_id][1]

            if prob_of_failure > THRESHOLD:
                part_needed = spare_parts_mapping[failure_mode]
                print(
                    f" -> ALARM! Ryzyko awarii {failure_mode}: {prob_of_failure * 100:.1f}%."
                )
                print(
                    f" -> AKCJA: Automatyczne przesunięcie z magazynu/zamówienie: {part_needed}"
                )
                orders_placed = True

        if not orders_placed:
            print(" -> Status: OK. Brak konieczności rezerwacji części.")


def main() -> None:
    move_dataset()
    optimize_spare_parts(data_path="helpers/dataset/ai4i2020.csv")


if __name__ == "__main__":
    main()
