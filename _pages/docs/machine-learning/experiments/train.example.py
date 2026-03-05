"""Multi-seed training with cross-validation and MLflow tracking.

Requires config.py (from config.example.py) on the import path.
"""

import argparse
import subprocess
import tempfile
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from xgboost import XGBClassifier

from config import ExperimentConfig


def require_clean_worktree() -> str:
    """Fail fast if the working tree has uncommitted changes."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, check=True,
    )
    if result.stdout.strip():
        raise RuntimeError(
            "Uncommitted changes detected — commit or stash before running experiments.\n"
            f"{result.stdout}"
        )
    commit_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return commit_hash


def main(config_path: str) -> None:
    commit_hash = require_clean_worktree()
    cfg = ExperimentConfig.from_yaml(config_path)

    df = pd.read_parquet(cfg.data.path)
    X = df.drop(columns=[cfg.data.target] + cfg.data.drop_columns)
    y = df[cfg.data.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.data.test_size, stratify=y, random_state=cfg.seeds[0]
    )

    mlflow.set_experiment(cfg.experiment_name)

    for seed in cfg.seeds:
        with mlflow.start_run(run_name=f"{cfg.model.name}_seed{seed}"):
            mlflow.log_params(cfg.model.params)
            mlflow.log_params({
                "seed": seed,
                "model_name": cfg.model.name,
                "scoring": cfg.scoring,
            })
            mlflow.set_tag("git_commit", commit_hash)
            mlflow.set_tag("data_version", cfg.data.data_version)

            start_time = time.monotonic()

            model = XGBClassifier(**cfg.model.params, random_state=seed)
            cv = StratifiedKFold(n_splits=cfg.cv_splits, shuffle=True, random_state=seed)

            results = cross_validate(
                model, X_train, y_train, cv=cv, scoring=cfg.scoring
            )

            oof_preds = cross_val_predict(
                model, X_train, y_train, cv=cv, method="predict_proba"
            )[:, 1]

            mlflow.log_metric("training_duration_seconds", time.monotonic() - start_time)
            mlflow.log_metric(f"mean_{cfg.scoring}", results["test_score"].mean())
            mlflow.log_metric(f"std_{cfg.scoring}", results["test_score"].std())

            for i, score in enumerate(results["test_score"]):
                mlflow.log_metric(f"fold_{cfg.scoring}", score, step=i)

            with tempfile.TemporaryDirectory() as tmp_dir:
                oof_path = Path(tmp_dir) / f"oof_seed{seed}.parquet"
                pd.DataFrame({"oof_pred": oof_preds}).to_parquet(oof_path)
                mlflow.log_artifact(str(oof_path))

    with mlflow.start_run(run_name="config_snapshot"):
        mlflow.log_artifact(config_path)

    print(f"Logged {len(cfg.seeds)} seed runs to experiment '{cfg.experiment_name}'")
    fold_scores = results["test_score"]
    print(f"Last seed CV: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
