"""Pydantic config schema for experiment configurations."""

import argparse
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    path: str
    target: str
    test_size: float = Field(default=0.2, gt=0, lt=1)
    drop_columns: list[str] = []
    data_version: str = "v1"


class ModelConfig(BaseModel):
    name: str
    params: dict[str, Any] = {}


class ExperimentConfig(BaseModel):
    experiment_name: str
    seeds: list[int] = Field(default=[42, 123, 456, 789, 1024], min_length=1)
    data: DataConfig
    model: ModelConfig
    cv_splits: int = Field(default=5, ge=2)
    scoring: str = "roc_auc"

    @classmethod
    def from_yaml(cls, path: str) -> Self:
        raw = yaml.safe_load(Path(path).read_text())
        return cls(**raw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    print(cfg.model_dump_json(indent=2))
