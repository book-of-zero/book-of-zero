"""Final held-out test evaluation for a frozen experiment config.

Requires config.py (from config.example.py) on the import path.
"""

import argparse
import tempfile
from collections import defaultdict
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, get_scorer, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import ExperimentConfig

THRESHOLD_MIN = 0.0
THRESHOLD_MAX = 1.0


def evaluate_seed(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cfg: ExperimentConfig,
    seed: int,
    threshold: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Fit on the full training split, then evaluate once on the held-out test set.

    Returns metrics dict and confusion matrix as a labeled DataFrame.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBClassifier(**cfg.model.params, random_state=seed)),
    ])
    pipe.fit(X_train, y_train)

    scorer = get_scorer(cfg.scoring)
    primary_score = scorer(pipe, X_test, y_test)

    y_score = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        f"test_{cfg.scoring}": primary_score,
        "test_precision": precision_score(y_test, y_pred, zero_division=0),
        "test_recall": recall_score(y_test, y_pred, zero_division=0),
        "test_f1": f1_score(y_test, y_pred, zero_division=0),
    }

    cm_df = pd.DataFrame(
        confusion_matrix(y_test, y_pred),
        index=["actual_negative", "actual_positive"],
        columns=["predicted_negative", "predicted_positive"],
    )

    return metrics, cm_df


def main(config_path: str, threshold: float) -> None:
    if not THRESHOLD_MIN <= threshold <= THRESHOLD_MAX:
        raise ValueError(
            f"Threshold must be between {THRESHOLD_MIN} and {THRESHOLD_MAX}, got {threshold}"
        )

    cfg = ExperimentConfig.from_yaml(config_path)

    df = pd.read_parquet(cfg.data.path)
    X = df.drop(columns=[cfg.data.target] + cfg.data.drop_columns)
    y = df[cfg.data.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.data.test_size, stratify=y, random_state=cfg.seeds[0]
    )

    mlflow.set_experiment(cfg.experiment_name)
    summary: dict[str, list[float]] = defaultdict(list)

    for seed in cfg.seeds:
        with mlflow.start_run(run_name=f"test_{cfg.model.name}_seed{seed}"):
            mlflow.log_params(cfg.model.params)
            mlflow.log_params({
                "seed": seed,
                "model_name": cfg.model.name,
                "scoring": cfg.scoring,
                "threshold": threshold,
            })
            mlflow.set_tag("data_version", cfg.data.data_version)

            metrics, cm_df = evaluate_seed(X_train, y_train, X_test, y_test, cfg, seed, threshold)
            for name, value in metrics.items():
                mlflow.log_metric(name, value)
                summary[name].append(value)

            with tempfile.TemporaryDirectory() as tmp_dir:
                cm_path = Path(tmp_dir) / f"confusion_matrix_seed{seed}.csv"
                cm_df.to_csv(cm_path)
                mlflow.log_artifact(str(cm_path))

            print(
                f"[seed {seed}] "
                f"{cfg.scoring}={metrics[f'test_{cfg.scoring}']:.4f}, "
                f"precision={metrics['test_precision']:.4f}, "
                f"recall={metrics['test_recall']:.4f}, "
                f"f1={metrics['test_f1']:.4f}"
            )

    with mlflow.start_run(run_name="test_summary"):
        mlflow.log_params({"scoring": cfg.scoring, "threshold": threshold})
        for name, values in summary.items():
            mlflow.log_metric(f"mean_{name}", float(np.mean(values)))
            mlflow.log_metric(f"std_{name}", float(np.std(values)))

    print("\nHeld-out test summary")
    for name, values in summary.items():
        print(f"{name}: {np.mean(values):.4f} +/- {np.std(values):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for precision/recall/F1 on the held-out test set",
    )
    args = parser.parse_args()
    main(args.config, args.threshold)
