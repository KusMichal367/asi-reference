"""Pakiet modelu (Sprint 3): wspólny zapis i odczyt modelu z metadata.json."""

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

PACKAGE_VERSION = 1
METADATA_FILE = "metadata.json"
SKLEARN_FILE = "model.joblib"
AUTOGLUON_DIR = "model"
LIBRARIES = ("python", "numpy", "pandas", "scikit-learn", "joblib", "autogluon.tabular")
REQUIRED_KEYS = (
    "package_version",
    "model_name",
    "model_format",
    "features",
    "target",
    "positive_class",
    "metrics",
    "git_sha",
    "data_fingerprint",
    "library_versions",
)


class ModelPackageError(ValueError):
    """Pakiet modelu jest niekompletny albo uszkodzony."""


# -=-=-= metadane -=-=-=


def git_sha(root: Path | None = None) -> str:

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def file_fingerprint(path: Path | str) -> str:

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def library_versions() -> dict[str, str]:
    import platform

    versions = {"python": platform.python_version()}
    for name in LIBRARIES[1:]:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            continue
    return versions


def build_metadata(
    *,
    model_name: str,
    model_format: str,
    features: list[str] | tuple[str, ...],
    target: str,
    positive_class: str,
    metrics: dict[str, Any],
    data_fingerprint: str,
    git_commit: str | None = None,
) -> dict[str, Any]:
    return {
        "package_version": PACKAGE_VERSION,
        "model_name": model_name,
        "model_format": model_format,
        "features": [str(name) for name in features],
        "target": target,
        "positive_class": str(positive_class),
        "metrics": metrics,
        "git_sha": git_commit or git_sha(),
        "data_fingerprint": data_fingerprint,
        "library_versions": library_versions(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def validate_metadata(metadata: dict[str, Any]) -> list[str]:
    problems = [f"brak klucza: {key}" for key in REQUIRED_KEYS if key not in metadata]
    if metadata.get("model_format") not in ("sklearn", "autogluon"):
        problems.append(f"nieznany model_format: {metadata.get('model_format')}")
    if not metadata.get("features"):
        problems.append("pusta lista features")
    return problems


# -=-=-= zapis i odczyt -=-=-=


def save_package(model: Any, package_dir: Path | str, metadata: dict[str, Any]) -> Path:

    problems = validate_metadata(metadata)
    if problems:
        raise ModelPackageError("; ".join(problems))

    package_dir = Path(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)

    if metadata["model_format"] == "sklearn":
        joblib.dump(model, package_dir / SKLEARN_FILE)
    else:
        model.clone(path=str(package_dir / AUTOGLUON_DIR), return_clone=False, dirs_exist_ok=True)

    (package_dir / METADATA_FILE).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return package_dir


def load_package(package_dir: Path | str) -> tuple[Any, dict[str, Any]]:

    package_dir = Path(package_dir)
    metadata_path = package_dir / METADATA_FILE
    if not metadata_path.exists():
        raise ModelPackageError(f"brak {METADATA_FILE} w {package_dir}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    problems = validate_metadata(metadata)
    if problems:
        raise ModelPackageError("; ".join(problems))

    if metadata["model_format"] == "sklearn":
        model = joblib.load(package_dir / SKLEARN_FILE)
    else:
        from autogluon.tabular import TabularPredictor

        model = TabularPredictor.load(str(package_dir / AUTOGLUON_DIR))
    return model, metadata


# -=-=-= predykcja -=-=-=


def predict(model: Any, metadata: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:

    missing = [name for name in metadata["features"] if name not in df.columns]
    if missing:
        raise ModelPackageError(f"brak cech wejściowych: {missing}")

    X = df[metadata["features"]]
    proba = model.predict_proba(X)
    if isinstance(proba, pd.DataFrame):
        positive = proba[1] if 1 in proba.columns else proba.iloc[:, -1]
    else:
        positive = proba[:, 1]

    probability = pd.Series(positive, index=X.index, dtype=float)
    return pd.DataFrame(
        {"prediction": (probability >= 0.5).astype(int), "probability": probability},
        index=X.index,
    )


# -=-=-= W&B Artifact -=-=-=


def log_package_artifact(
    package_dir: Path | str,
    *,
    project: str,
    entity: str | None = None,
    name: str = "churn-model",
    aliases: tuple[str, ...] = ("production",),
) -> str:

    import wandb

    package_dir = Path(package_dir)
    _, metadata = load_package(package_dir)

    with wandb.init(project=project, entity=entity, job_type="package-model") as run:
        artifact = wandb.Artifact(name=name, type="model", metadata=metadata)
        artifact.add_dir(str(package_dir))
        run.log_artifact(artifact, aliases=list(aliases))
        return f"{run.entity}/{run.project}/{name}"


def download_package_artifact(reference: str, target_dir: Path | str | None = None) -> Path:

    import wandb

    artifact = wandb.Api().artifact(reference, type="model")
    return Path(artifact.download(root=str(target_dir) if target_dir else None))
