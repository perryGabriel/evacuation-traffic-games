from __future__ import annotations

import json
from pathlib import Path

from evacuation_vug.types import ExperimentConfig


ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    text = p.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(f"Config {p} is not JSON and PyYAML is unavailable") from exc
        return yaml.safe_load(text)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    raw = load_yaml(path)
    return ExperimentConfig(
        name=raw["name"],
        horizon=int(raw["horizon"]),
        seed=int(raw.get("seed", 0)),
        network_config=raw["network_config"],
        generation={k: float(v) for k, v in raw["generation"].items()},
        surrogate=raw["surrogate"],
        policies=list(raw["policies"]),
        output_dir=raw.get("output_dir", f"results/{raw['name']}"),
    )
