from pathlib import Path
import shutil

import kagglehub


def download_dataset() -> str:
    dataset_path = kagglehub.dataset_download(
        "stephanmatzka/predictive-maintenance-dataset-ai4i-2020"
    )
    return dataset_path


def use_original_path() -> str:
    dataset_org_path = kagglehub.dataset_download(
        "stephanmatzka/predictive-maintenance-dataset-ai4i-2020"
    )

    file_path = dataset_org_path + "\\ai4i2020.csv"
    return file_path


def move_dataset() -> None:
    dataset_path = download_dataset()
    source: str = dataset_path + "\\ai4i2020.csv"

    print(f"Original dataset path: {source}")

    base_dir: Path = Path(__file__).resolve().parent
    destination: Path = base_dir / "dataset" / "ai4i2020.csv"

    destination.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy(source, destination)
    print(f"Dataset moved to: {destination}")
