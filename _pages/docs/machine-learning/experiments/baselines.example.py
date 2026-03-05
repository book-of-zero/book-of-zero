"""Baseline evaluation with DummyClassifier and LogisticRegression.

Requires config.py (from config.example.py) on the import path.
"""

import argparse

import mlflow
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import ExperimentConfig


def _make_baselines(seed: int) -> dict[str, BaseEstimator]:
    """Construct baseline estimators with per-seed random state."""
    return {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logistic": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=seed)),
        ]),
    }


def main(config_path: str) -> None:
    cfg = ExperimentConfig.from_yaml(config_path)

    df = pd.read_parquet(cfg.data.path)
    X = df.drop(columns=[cfg.data.target] + cfg.data.drop_columns)
    y = df[cfg.data.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.data.test_size, stratify=y, random_state=cfg.seeds[0]
    )

    mlflow.set_experiment(cfg.experiment_name)

    for seed in cfg.seeds:
        baselines = _make_baselines(seed)
        for name, model in baselines.items():
            with mlflow.start_run(run_name=f"baseline_{name}_seed{seed}"):
                mlflow.log_params({"baseline": name, "seed": seed, "scoring": cfg.scoring})

                cv = StratifiedKFold(n_splits=cfg.cv_splits, shuffle=True, random_state=seed)
                scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=cfg.scoring)

                mlflow.log_metric(f"mean_{cfg.scoring}", scores.mean())
                mlflow.log_metric(f"std_{cfg.scoring}", scores.std())

                for i, score in enumerate(scores):
                    mlflow.log_metric(f"fold_{cfg.scoring}", score, step=i)

                print(f"[seed {seed}] {name}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
