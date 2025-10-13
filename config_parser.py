from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from omegaconf import OmegaConf, DictConfig

@dataclass
class Hyperparams:
    sigma: float = 0.001
    total_species: int = 10
    epochs: int = 100

@dataclass
class Sampling:
    temperature: float = 0.6
    top_p: float = 0.95
    max_new_tokens: int = 1024
    engine_mem_fraction: float = 0.75

@dataclass
class Saves:
    save_path: str = "./saves/model"

@dataclass
class Logging:
    # set to None to disable logging
    logs_save_path: Optional[str] = "logs/model_outputs.txt"

@dataclass
class Config:
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    dtype: str = "bfloat16"
    data_path: str = "data/demo_dataset.json"
    hyperparams: Hyperparams = field(default_factory=Hyperparams)
    sampling: Sampling = field(default_factory=Sampling)
    saves: Saves = field(default_factory=Saves)
    logging: Logging = field(default_factory=Logging)


# ---------- Parser ----------
def load_config(yaml_path: str | Path) -> Config:
    """
    Load and validate the YAML config into the typed dataclasses.
    - Enforces field types and Literal constraints.
    - Converts the string "None" (case-insensitive) in logs_save_path to None.
    """
    # Load YAML (as DictConfig)
    cfg_dc: DictConfig = OmegaConf.load(str(yaml_path))

    # Build structured schema for validation
    schema = OmegaConf.structured(Config)
    # Merge YAML into schema (validates & fills defaults)
    merged: DictConfig = OmegaConf.merge(schema, cfg_dc)

    # Normalize "None" -> None for logs_save_path
    if "logging" in merged and "logs_save_path" in merged.logging:
        lsp = merged.logging.logs_save_path
        if isinstance(lsp, str) and lsp.strip().lower() == "none":
            merged.logging.logs_save_path = None

    # Optional: additional sanity checks
    _validate_ranges(merged)

    # Convert to dataclass tree
    return OmegaConf.to_object(merged)


def _validate_ranges(cfg: DictConfig) -> None:
    if not (0.0 < cfg.sampling.temperature <= 2.0):
        raise ValueError(f"sampling.temperature out of range: {cfg.sampling.temperature}")
    if not (0.0 < cfg.sampling.top_p <= 1.0):
        raise ValueError(f"sampling.top_p out of range: {cfg.sampling.top_p}")
    if cfg.sampling.max_new_tokens <= 0:
        raise ValueError("sampling.max_new_tokens must be positive")
    if not (0.0 < cfg.sampling.engine_mem_fraction <= 1.0):
        raise ValueError("sampling.engine_mem_fraction must be in (0, 1]")
    if cfg.hyperparams.sigma <= 0:
        raise ValueError("hyperparams.sigma must be positive")
    if cfg.hyperparams.total_species <= 0:
        raise ValueError("hyperparams.total_species must be positive")
    if cfg.hyperparams.epochs <= 0:
        raise ValueError("hyperparams.epochs must be positive")
