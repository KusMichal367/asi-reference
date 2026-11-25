"""Benchmark pakietu modelu (Sprint 3): rozmiar, pamięć, czas wczytania i predykcji.

Każdy kandydat jest mierzony jako gotowy pakiet (``save_package``), wczytany w świeżym
procesie Pythona przez ``load_package`` i odpytywany przez ``predict`` — dokładnie tą samą
ścieżką, której użyją smoke test i API w Sprincie 4. Dzięki temu porównujemy to, co
faktycznie trafi do wdrożenia, a nie katalog treningowy.

Uruchomienie pomiaru dla jednego pakietu (wynik JSON na stdout):
    python -m asi_project.benchmark --package-dir data/06_models/candidates/x --data test.csv
"""

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from asi_project.metrics import compute_metrics
from asi_project.model_package import load_package, predict

DEFAULT_REPEATS = 100
BATCH_REPEATS = 3


def package_size_mb(package_dir: Path | str) -> float:
    """Rozmiar całego pakietu (model + metadata.json) w MB."""
    files = [path for path in Path(package_dir).rglob("*") if path.is_file()]
    return round(sum(path.stat().st_size for path in files) / (1024 * 1024), 4)


def _rss_mb() -> float | None:
    """Pamięć rezydentna bieżącego procesu (RSS). Obejmuje pamięć natywną (LightGBM, CatBoost),
    której nie widzi ``tracemalloc``."""
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process().memory_info().rss / (1024 * 1024)


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def measure_package(
    package_dir: Path | str,
    data: pd.DataFrame,
    target: str,
    repeats: int = DEFAULT_REPEATS,
) -> dict[str, Any]:
    """Pomiar w bieżącym procesie. Wywoływać w świeżym procesie (``measure_in_fresh_process``),
    inaczej pamięć i czas wczytania są zaniżone przez już zaimportowane biblioteki."""
    y_true = data[target].astype(int)
    X = data.drop(columns=[target])

    gc.collect()
    rss_before = _rss_mb()
    t0 = time.perf_counter()
    model, metadata = load_package(package_dir)
    load_time_s = time.perf_counter() - t0

    # Rozgrzewka: AutoGluon dociąga część modeli leniwie przy pierwszej predykcji.
    predict(model, metadata, X.iloc[:1])
    rss_after = _rss_mb()

    single_ms = []
    for i in range(repeats):
        row = X.iloc[[i % len(X)]]
        t0 = time.perf_counter()
        predict(model, metadata, row)
        single_ms.append((time.perf_counter() - t0) * 1000)

    batch_s = []
    for _ in range(BATCH_REPEATS):
        t0 = time.perf_counter()
        result = predict(model, metadata, X)
        batch_s.append(time.perf_counter() - t0)

    metrics = compute_metrics(y_true, result["prediction"])
    return {
        "model_name": metadata["model_name"],
        "model_format": metadata["model_format"],
        "artifact_size_mb": package_size_mb(package_dir),
        "load_time_s": round(load_time_s, 4),
        "load_memory_mb": (round(rss_after - rss_before, 2) if rss_before is not None else None),
        "latency_single_p50_ms": round(statistics.median(single_ms), 3),
        "latency_single_p95_ms": round(_percentile(single_ms, 0.95), 3),
        "batch_ms_per_row": round(min(batch_s) / len(X) * 1000, 4),
        "n_rows": len(X),
        "f1_from_package": round(metrics["f1"], 6),
    }


def measure_in_fresh_process(
    package_dir: Path | str,
    data_csv: Path | str,
    target: str,
    repeats: int = DEFAULT_REPEATS,
    timeout_s: int = 600,
) -> dict[str, Any]:
    """Uruchamia ``measure_package`` w nowym procesie i zwraca wynik (JSON)."""
    command = [
        sys.executable,
        "-m",
        "asi_project.benchmark",
        "--package-dir",
        str(package_dir),
        "--data",
        str(data_csv),
        "--target",
        target,
        "--repeats",
        str(repeats),
    ]
    # Nowy proces musi widzieć pakiet asi_project także bez `pip install -e .` (np. w pytest).
    src_dir = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [src_dir, env.get("PYTHONPATH")]))
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=timeout_s, check=False, env=env
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Pomiar {package_dir} nie powiódł się:\n{completed.stderr[-2000:]}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--data", required=True, help="CSV z cechami i kolumną targetu")
    parser.add_argument("--target", required=True)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    args = parser.parse_args(argv)

    data = pd.read_csv(args.data)
    result = measure_package(args.package_dir, data, args.target, args.repeats)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
