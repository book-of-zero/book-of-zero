---
layout: post
title: "Experiments: run rigorous ML experiments"
nav_order: 6
---

Same model, same data, three runs — 0.89, 0.85, 0.87. Without experiment discipline, none of these are defensible.

This guide covers the practices that make ML experiments reproducible, comparable, and auditable: seed control, data splitting, configuration management, tracking, and ablation. The goal is not to track everything — it is to track enough that any result can be explained, challenged, and reproduced. The examples use binary classification with scikit-learn (`roc_auc`, `stratify`, `predict_proba`). The principles apply to regression and multi-class — substitute the metric and splitting strategy.

---

## On this page

- [Key concepts](#key-concepts)
- [Reproducibility contract](#reproducibility-contract)
  - [Source control](#source-control)
  - [Data immutability](#data-immutability)
  - [Seeds](#seeds)
  - [Environment pinning](#environment-pinning)
  - [Determinism boundaries](#determinism-boundaries)
- [Data splitting](#data-splitting)
  - [Three sets, three purposes](#three-sets-three-purposes)
  - [No leakage](#no-leakage)
  - [Group-aware splits](#group-aware-splits)
  - [Early stopping](#early-stopping)
- [Configuration management](#configuration-management)
  - [Config structure](#config-structure)
  - [Config model](#config-model)
- [Experiment tracking](#experiment-tracking)
  - [Autologging](#autologging)
  - [Explicit logging](#explicit-logging)
  - [Organizing runs](#organizing-runs)
- [Workflow: running an experiment](#workflow-running-an-experiment)
  - [Single run](#single-run)
  - [Baselines](#baselines)
  - [Error analysis](#error-analysis)
  - [Hyperparameter search](#hyperparameter-search)
  - [Ablation study](#ablation-study)
- [Common pitfalls](#common-pitfalls)

---

## Key concepts

- **Experiment**: a question you are trying to answer ("does model A beat model B?", "does feature X help?"). One independent variable, one fixed evaluation protocol.
- **Run**: a single execution with a fixed configuration. An experiment produces many runs.
- **Ablation**: removing one component at a time to measure its individual contribution. If performance drops, the component matters.
- **Seed**: the starting state for random number generators. Fixing seeds makes stochastic processes repeatable.
- **Baseline**: a reference model that sets the performance floor and bar. A trivial baseline (e.g., predict the majority class) proves the features carry signal; a simple baseline (e.g., logistic regression) proves the added complexity of a fancier model is justified.
- **Config-driven experiment**: all parameters live in configuration files — the config is the single source of truth for what a run did.

---

## Reproducibility contract

A result that cannot be reproduced cannot be defended. Reproducibility has five layers: source control, data immutability, seed control, environment pinning, and awareness of determinism boundaries.

### Source control

MLflow automatically logs the git commit hash, but this is a false guarantee if you run experiments with uncommitted changes. The "dirty working tree" vulnerability means the logged hash points to code that is different from what actually ran. 

To enforce strict source control, either fail the run if `git status --porcelain` is not empty, or explicitly log the `git diff` as an MLflow artifact.

### Data immutability

Pinning code and environment is useless if the underlying data changes. Treat training data as immutable. Overwriting a file like `features_v1.parquet` permanently breaks reproducibility for every prior experiment that relied on it.

Use append-only storage or a data versioning tool like DVC to enforce data immutability. If a dataset must change, write it as `features_v2.parquet` and update the experiment configuration.

### Seeds

Without fixed seeds, the same model trained twice produces different scores — random forests sample features randomly, data shuffling changes the split order, and stochastic estimators start from different states. Seeds anchor the random number generators to a known starting point.

Control randomness by passing `random_state` explicitly to every estimator and splitter. Scikit-learn's randomness flows through numpy, and each `random_state` parameter creates an independent random number generator scoped to that object — no shared global state, no accidental coupling between components:

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

cv = StratifiedKFold(n_splits=cfg.cv_splits, shuffle=True, random_state=seed)
model = RandomForestClassifier(n_estimators=500, random_state=seed)
```

Global seeding with `np.random.seed()` is a legacy pattern that modifies shared state and causes unintended side effects across unrelated components — avoid it. `PYTHONHASHSEED` is sometimes mentioned in reproducibility guides, but it controls Python set iteration order — irrelevant for scikit-learn, whose randomness flows through numpy.

A single seed tells you the pipeline runs end-to-end. It does not tell you whether the result is robust — a different seed might give a substantially different score. To make a defensible claim, you need the distribution.

**Running multiple seeds is mandatory for publishable results.** 3–5 seeds is often insufficient: conclusions can flip with different seeds, and statistical power is too low to detect real differences.

For classical ML with cross-validation, each seed × fold combination produces an observation, so K-fold CV × S seeds gives K × S paired data points. With 5-fold CV and 10 seeds you get 50 paired observations — enough for robust statistical testing.

<p align="center">
  <img src="{{ "/assets/images/plots/seed_variance.svg" | relative_url }}" alt="Box plot showing ROC AUC score distributions across 10 seeds for three models">
</p>

| Setting | Minimum seeds | Paired observations (5-fold CV) |
|---------|--------------|-------------------------------|
| Exploratory | 5 | 25 |
| Publishable comparison | 10 | 50 |
| High-stakes or noisy | 20+ | 100+ |

If the standard deviation across seeds is large relative to the difference between models, the comparison is inconclusive — add more seeds or reconsider the experimental design. Use power analysis (see [Evaluation: statistical significance]({{ site.baseurl }}/docs/machine-learning/evaluation/evaluation/#statistical-significance)) to determine exactly how many you need.

### Environment pinning

Pin everything that can change between machines or over time:

- **Python version**: specify in `pyproject.toml` and the Dockerfile (`ARG PYTHON_VERSION=3.13`).
- **Dependencies**: use `uv.lock`. The Dockerfile enforces this with `uv sync --locked` — if the lockfile is missing or stale, the build fails.
- **scikit-learn version**: a minor version upgrade can change default hyperparameters, convergence criteria, or internal random number consumption. A model trained on 1.4 may not reproduce identically on 1.5 even with the same seed. The lockfile handles this, but document the version in your experiment logs.

The Dockerfile from the [Docker guide]({{ site.baseurl }}/docs/containerization/docker/docker/) already enforces Python version and dependency pinning via `uv sync --locked`.

### Determinism boundaries

Full determinism is not always achievable — and sometimes not worth the cost. Know where it breaks:

- **Nested parallelism**: `RandomForestClassifier` with `random_state` set is deterministic regardless of `n_jobs`. But nesting parallel calls — for example `GridSearchCV(n_jobs=-1)` wrapping `RandomForestClassifier(n_jobs=-1)` — can produce non-deterministic results because the joblib worker scheduling is not guaranteed to be consistent. Fix by parallelizing at one level only (typically the outer loop).
- **Floating-point non-associativity**: `(a + b) + c ≠ a + (b + c)` in floating point. Different hardware or parallelism strategies can change the order of operations. This is typically negligible for classical ML but can compound in iterative algorithms (gradient boosting over many rounds, iterative SVD).

**Practical stance**: enforce seeds (per-estimator `random_state` on every estimator and splitter), pin the environment (see [Environment pinning](#environment-pinning)), and accept that bit-exact reproduction across different hardware is sometimes infeasible. Document the platform when it matters.

---

## Data splitting

The split protocol determines what evidence you can claim. Get it wrong and metrics look good in the notebook but collapse in production — the model was learning from information it should never have seen.

### Three sets, three purposes

- **Training set**: used to fit the model. Every parameter update happens here.
- **Validation set**: used to select hyperparameters and compare model variants. In cross-validation, each fold's held-out portion serves as the validation set — no separate validation split is needed.
- **Test set**: held out from all decisions. Evaluated once, at the end, on the final selected configuration only. If you use the test set to pick between models or tune hyperparameters, it becomes a second validation set and your reported metrics are optimistically biased.

Split before anything else. The test set must be created before any preprocessing, feature engineering, or model training. The snippets below use `cfg`, a Pydantic config object — defined in [Configuration management](#configuration-management):

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=cfg.data.test_size, stratify=y, random_state=cfg.seeds[0]
)
```

`train_test_split` shuffles the data before splitting (`shuffle=True` by default). This is correct for i.i.d. tabular data — without shuffling, if the dataset is sorted by date, by class, or by source, the split is biased and the test set is not representative. `stratify=y` goes further: it shuffles while preserving the class distribution in both sets, so neither set ends up with a skewed label ratio. This is critical for imbalanced datasets where a plain random split could leave one class underrepresented in the test set. When `stratify` is set, shuffle is always on.

`random_state` controls the shuffle order. The test split uses a fixed seed (`cfg.seeds[0]`) so the test set is identical across all seed runs — comparisons are fair because every model variant is evaluated against the same held-out data.

All cross-validation, hyperparameter search, and model comparison happens on `(X_train, y_train)` only. The test set `(X_test, y_test)` is not touched until final evaluation.

For time-series or sequential data, shuffling must be disabled — random splits create future leakage where the model trains on future data and predicts the past. Use time-based splits instead: train on data before a cutoff date, validate and test on data after. scikit-learn provides `TimeSeriesSplit` for cross-validation on temporal data.

### No leakage

Data leakage means information from the validation or test set bleeds into training — the model gets answers it would not have in production. This is the most common source of overly optimistic results. Common sources:

- **Preprocessing fit on the full dataset**: fitting a scaler, encoder, or imputer on all data before splitting leaks test-set statistics into training.
- **Feature selection on the full dataset**: selecting features based on correlation with the target using all data gives the model advance knowledge of validation and test patterns.
- **Target encoding without proper folds**: encoding categorical features using target means from the full dataset leaks the target into every fold.

The fix: use a scikit-learn `Pipeline` to bundle preprocessing and model into a single estimator. When passed to `cross_validate` or `GridSearchCV`, the pipeline fits preprocessing on each training fold and transforms the validation fold — no leakage:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", XGBClassifier(**cfg.model.params, random_state=seed)),
])

# cross_validate fits the pipeline on each training fold
# and evaluates on the held-out fold — preprocessing is inside the loop
results = cross_validate(pipe, X_train, y_train, cv=cv, scoring=cfg.scoring)
```

If your preprocessing is too complex for a pipeline (for example, heavy feature engineering with external data), use the training indices from each fold explicitly and transform the validation indices separately. The rule is the same: fit on train, transform on validation, never the reverse.

### Group-aware splits

If the dataset contains multiple samples per entity — multiple images per patient, multiple transactions per user, multiple readings per sensor — a random split scatters the same entity across train and test. The model memorizes entity-specific patterns rather than learning generalizable ones: with patient data, it learns patient A's baseline lab values rather than the disease signal. Reported metrics can be inflated by 5–30%.

`train_test_split` does not support group constraints. Use `GroupShuffleSplit` for the initial split and `GroupKFold` for cross-validation:

```python
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

# Initial split respecting groups
gss = GroupShuffleSplit(n_splits=1, test_size=cfg.data.test_size, random_state=cfg.seeds[0])
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Cross-validation respecting groups
cv = GroupKFold(n_splits=cfg.cv_splits)
results = cross_validate(pipe, X_train, y_train, cv=cv, groups=groups[train_idx], scoring=cfg.scoring)
```

If your data has no natural grouping (each row is an independent observation), standard `train_test_split` and `StratifiedKFold` are correct. When groups and class imbalance both matter, use `StratifiedGroupKFold`.

### Early stopping

Iterative algorithms like gradient boosting (XGBoost, LightGBM) or deep learning require a separate validation set inside the training loop to monitor performance and halt training before overfitting begins. This is distinct from the validation folds used in cross-validation for model selection — early stopping uses a dedicated split within each training fold.

When using cross-validation, passing this internal validation set through a scikit-learn `Pipeline` can be notoriously tricky, as pipelines do not natively map sample-level `fit_params` to the estimator for the held-out fold.

To implement early stopping during cross-validation properly, you must either manually iterate over the fold indices and pass `eval_set=[(X_val, y_val)]` directly to the estimator's `fit` method, or use a library-specific integration (like `xgboost.sklearn`):

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

cv = StratifiedKFold(n_splits=cfg.cv_splits, shuffle=True, random_state=seed)

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # Fit preprocessing on training fold only
    scaler = StandardScaler()
    X_fold_train_scaled = scaler.fit_transform(X_fold_train)
    X_fold_val_scaled = scaler.transform(X_fold_val)

    # Pass eval_set directly to the estimator
    model = XGBClassifier(**cfg.model.params, random_state=seed, early_stopping_rounds=20)
    model.fit(
        X_fold_train_scaled, y_fold_train,
        eval_set=[(X_fold_val_scaled, y_fold_val)],
        verbose=False
    )

    # Evaluate on the held-out fold
    score = roc_auc_score(y_fold_val, model.predict_proba(X_fold_val_scaled)[:, 1])
    print(f"Fold {fold}: {score:.4f} (stopped at iteration {model.best_iteration})")
```

Never use the final test set for early stopping — this turns the test set into a validation set and leaks information directly into the model's stopping criteria.

---

## Configuration management

Hard-coded hyperparameters buried in a training script are impossible to track and easy to forget. Externalizing parameters into YAML files means each config is a complete, diffable record of what a run did. Pydantic validates on load, provides IDE autocompletion, and serializes to a dict for MLflow. Both Pydantic and PyYAML are already transitive dependencies of MLflow — nothing extra to install.

### Config structure

One YAML file per experiment configuration. Each file is a complete, self-contained record of what a run did — readable, diffable, and loggable as an MLflow artifact:

```yaml
# configs/baseline.yaml
experiment_name: model_comparison

seeds: [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999]

data:
  path: data/processed/features_v1.parquet
  target: is_fraud
  test_size: 0.2

model:
  name: xgboost
  params:
    n_estimators: 500
    max_depth: 6
    learning_rate: 0.1

cv_splits: 5
scoring: roc_auc
```

For ablation variants, copy the baseline and change one thing. Duplication is acceptable — readability and reproducibility matter more than DRY for experiment configs.

### Config model

Define the config as a Pydantic model. Invalid configs fail immediately on load with a clear error — not silently at runtime halfway through a training run:

```python
from pydantic import BaseModel, Field
import yaml
from pathlib import Path


class DataConfig(BaseModel):
    path: str
    target: str
    test_size: float = Field(default=0.2, gt=0, lt=1)
    drop_columns: list[str] = []


class ModelConfig(BaseModel):
    name: str
    params: dict = {}


class ExperimentConfig(BaseModel):
    experiment_name: str
    seeds: list[int] = Field(default=[42, 123, 456, 789, 1024], min_length=1)
    data: DataConfig
    model: ModelConfig
    cv_splits: int = Field(default=5, ge=2)
    scoring: str = "roc_auc"

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        raw = yaml.safe_load(Path(path).read_text())
        return cls(**raw)
```

Load and use:

```python
config_path = "configs/baseline.yaml"
cfg = ExperimentConfig.from_yaml(config_path)

# IDE autocomplete — cfg.model.name, cfg.data.path, cfg.scoring
# Log model params to MLflow (flat dict, MLflow-compatible)
mlflow.log_params(cfg.model.params)
# Log the config file itself as an artifact (reproduces the run)
mlflow.log_artifact(config_path)
```

Run experiments by pointing to a config file:

```bash
python train.py --config configs/baseline.yaml
python train.py --config configs/ablation_no_text.yaml
```

---

## Experiment tracking

MLflow records parameters, metrics, and artifacts for every run. The tracking server provides a UI for comparing runs and a programmatic API for querying results.

### Autologging

For scikit-learn, `mlflow.sklearn.autolog()` captures parameters, metrics, and model artifacts automatically — including child runs for `GridSearchCV` and `RandomizedSearchCV`, where each parameter combination is tracked as a nested run:

```python
import mlflow
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

mlflow.sklearn.autolog(max_tuning_runs=20)

mlflow.set_experiment(cfg.experiment_name)

with mlflow.start_run(run_name=f"{cfg.model.name}_gridsearch"):
    estimator = XGBClassifier(**cfg.model.params)
    param_grid = {"max_depth": [3, 6, 9], "learning_rate": [0.01, 0.1, 0.3]}

    search = GridSearchCV(estimator, param_grid, cv=cfg.cv_splits, scoring=cfg.scoring)
    search.fit(X_train, y_train)
    # MLflow logs all param combinations as child runs automatically
```

`max_tuning_runs` controls how many child runs are logged for hyperparameter searches — it defaults to 5, which is too low for most experiments. Set it to match the number of parameter combinations you want to inspect.

Autolog captures: estimator parameters (via `get_params(deep=True)`), training metrics, best score, best parameters, and the fitted model artifact. Use autologging with `GridSearchCV` when the search space is small and enumerable. For larger or continuous search spaces, use Optuna (see [Hyperparameter search](#hyperparameter-search)).

### Explicit logging

Add explicit logging for config snapshots, custom metrics, tags, and artifacts that autolog does not cover. `mlflow.log_params()` only accepts flat dicts — nested dicts get stringified into unreadable blobs. Log each level explicitly:

```python
import time
import mlflow
import pandas as pd
from sklearn.model_selection import cross_validate, cross_val_predict, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

config_path = "configs/baseline.yaml"
cfg = ExperimentConfig.from_yaml(config_path)

df = pd.read_parquet(cfg.data.path)
X, y = df.drop(columns=[cfg.data.target]), df[cfg.data.target]

# Hold out the test set before any training or tuning
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=cfg.data.test_size, stratify=y, random_state=cfg.seeds[0]
)

mlflow.set_experiment(cfg.experiment_name)

for seed in cfg.seeds:
    with mlflow.start_run(run_name=f"{cfg.model.name}_seed{seed}"):
        # Flat dicts only — log model params and metadata separately
        mlflow.log_params(cfg.model.params)
        mlflow.log_params({"seed": seed, "model_name": cfg.model.name, "scoring": cfg.scoring})

        # Tags for filtering in the UI (git hash is auto-logged)
        mlflow.set_tag("data_version", "v1")

        start_time = time.time()

        model = XGBClassifier(**cfg.model.params, random_state=seed)
        cv = StratifiedKFold(n_splits=cfg.cv_splits, shuffle=True, random_state=seed)
        
        results = cross_validate(model, X_train, y_train, cv=cv, scoring=cfg.scoring)
        
        # Out-of-fold predictions enable ensembling and deep error analysis without retraining
        oof_preds = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        
        mlflow.log_metric("training_duration_seconds", time.time() - start_time)

        pd.DataFrame({"oof_pred": oof_preds}).to_parquet(f"oof_seed{seed}.parquet")
        mlflow.log_artifact(f"oof_seed{seed}.parquet")

        mlflow.log_metric(f"mean_{cfg.scoring}", results["test_score"].mean())
        mlflow.log_metric(f"std_{cfg.scoring}", results["test_score"].std())

        # Per-fold scores — needed for paired statistical tests
        for i, score in enumerate(results["test_score"]):
            mlflow.log_metric(f"fold_{cfg.scoring}", score, step=i)

# Log the config file once as an artifact (outside the seed loop)
with mlflow.start_run(run_name="config_snapshot"):
    mlflow.log_artifact(config_path)
```

| Category | What | How |
|----------|------|-----|
| Parameters | Seed, model hyperparams, scoring | `mlflow.log_params(...)` or autolog |
| Metrics | CV mean, std, per-fold scores, test metrics, duration | `mlflow.log_metric(...)` or autolog |
| Artifacts | Model, config file, OOF predictions | `mlflow.log_artifact(...)` or autolog |
| Tags | Data version, custom metadata | `mlflow.set_tag(...)` (git hash is auto-logged) |

### Organizing runs

Structure runs so that comparison is natural:

- **Experiment** = comparable runs on the same data. One MLflow experiment per task where runs share the same input dataset and evaluation protocol (e.g., `feature_selection_ablation`, `model_comparison_v2`). The organizing principle is input data consistency — if runs use different datasets, they belong in different experiments.
- **Run** = one execution. For seed sweeps, each run corresponds to one seed × one configuration. For hyperparameter searches, use a parent run with nested child runs (one per trial) — the MLflow UI renders the hierarchy so you can compare trials within a search and compare searches within an experiment.
- **Params** = inputs that affect model behavior. Log seed, model hyperparameters, and scoring metric with `log_param()` — these are immutable per run and define what the run did.
- **Tags** = metadata for filtering. Use tags for data version and any cross-cutting attribute you want to filter or group by later. Git hash is auto-logged by MLflow as `mlflow.source.git.commit` — no need to set it manually. Note that `mlflow.set_tag("data_version", "v1")` is a label, not a guarantee — it does not track what the data actually contained. For true data reproducibility, use a data versioning tool like DVC alongside your experiment tracker so that any result can be traced back to the exact dataset that produced it.
- **Run name** = human-readable label. Use a descriptive pattern like `xgboost_seed42` so the MLflow UI is scannable without clicking into each run.

Avoid dumping unrelated runs into the same MLflow experiment — it makes the UI unusable and comparisons meaningless. When in doubt, create a new experiment. They are cheap.

---

## Workflow: running an experiment

### Single run

The minimal loop: load config, train, evaluate, log.

```bash
python train.py --config configs/baseline.yaml
```

A single run validates that the pipeline works end-to-end. It is not evidence of anything — one seed, one configuration, one data split.

### Baselines

Before comparing complex models, establish what "good" means. Two baselines set the floor and the bar:

- **Trivial baseline**: `DummyClassifier(strategy="most_frequent")` predicts the majority class. Any model must beat this — if it does not, the features carry no signal or there is a data problem.
- **Simple baseline**: `LogisticRegression` (or `Ridge` for regression). A tuned linear model is often competitive. If a complex model beats logistic regression by less than the standard deviation across seeds, the added complexity is not justified.

```python
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

baselines = {
    "dummy": DummyClassifier(strategy="most_frequent"),
    "logistic": LogisticRegression(max_iter=1000, random_state=seed),
}

# Run inside the seed loop (for seed in cfg.seeds:)
for name, model in baselines.items():
    cv = StratifiedKFold(n_splits=cfg.cv_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=cfg.scoring)
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")
```

Run baselines first. If a complex model cannot beat a tuned logistic regression by a meaningful margin, the complexity is not worth the cost.

### Error analysis

Before tuning hyperparameters or adding features, understand what the model gets wrong. A model stuck at 0.85 AUC might be failing on a single error category that no amount of hyperparameter tuning will fix — but that a new feature or data cleaning step would resolve in minutes. Error analysis tells you where the ceiling is and what to try next.

The method: take your baseline model's predictions on the validation set and sample ~100 misclassified examples. For each, write down a short error category — the reason the model got it wrong. Then count:

| Error category | Count | % of errors | Ceiling if fixed |
|----------------|-------|-------------|------------------|
| Missing address data | 34 | 34% | +0.03 AUC |
| Mislabeled ground truth | 22 | 22% | +0.02 AUC |
| Rare merchant category | 18 | 18% | +0.01 AUC |
| Ambiguous cases | 15 | 15% | — |
| Other | 11 | 11% | — |

The "ceiling if fixed" column is the key insight: if 34% of errors come from missing address data, fixing that category (imputation, new feature, data collection) can improve performance by roughly that proportion of the current error. Hyperparameter tuning, by contrast, optimizes within the current representation — it cannot fix a missing feature or a labeling problem.

This analysis directly informs what to try next:

- **Missing or noisy features** → feature engineering or data collection, not more tuning.
- **Mislabeled examples** → data cleaning before any model changes.
- **Ambiguous cases** → irreducible error; do not chase this.
- **Systematic failure on a subgroup** → stratified evaluation, targeted features, or more training data for that subgroup.

Use `sklearn.metrics.confusion_matrix` for a quick overview, then inspect actual examples for the error categories that matter most. Log the error analysis table as an MLflow artifact — it documents the reasoning behind your next experiment, not just the result.

### Hyperparameter search

Once the pipeline works and baselines are established, hyperparameter search finds the configuration that best exploits the data. Search is not training — it is meta-optimization over training runs. The search space lives in code (distributions cannot be expressed in YAML); the config file holds fixed parameters. The full lifecycle:

1. **Search** — Optuna explores the search space, MLflow logs every trial.
2. **Freeze** — write the best params into a new config YAML.
3. **Evaluate** — run the frozen config across all seeds as a single run.

The config records *what worked*. The code defines *how you searched*. Both are logged to MLflow.

#### 1. Search

Use Optuna for Bayesian optimization with MLflow tracking:

```python
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
    }

    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        mlflow.log_params(params)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBClassifier(**params, random_state=cfg.seeds[0])),
        ])
        cv = StratifiedKFold(n_splits=cfg.cv_splits, shuffle=True, random_state=cfg.seeds[0])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=cfg.scoring)
        mean_score = scores.mean()

        mlflow.log_metric(f"val_{cfg.scoring}", mean_score)

    return mean_score

mlflow.set_experiment(cfg.experiment_name)

with mlflow.start_run(run_name="hparam_search"):
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg.seeds[0]),
    )
    study.optimize(objective, n_trials=100)

    mlflow.log_params(study.best_params)
    mlflow.log_metric(f"best_val_{cfg.scoring}", study.best_value)
```

Each trial is a nested MLflow run under a parent search run, so the full history is visible in the MLflow UI.

<p align="center">
  <img src="{{ "/assets/images/plots/optuna_history.svg" | relative_url }}" alt="Hyperparameter search history showing trial scores and running best with plateau annotation">
</p>

#### 2. Freeze

Write the best params to a new config file — this is the reproducible record:

```python
import yaml

best_cfg = cfg.model_copy(deep=True)
best_cfg.model.params = study.best_params

config_path = "configs/best_xgboost.yaml"
Path(config_path).write_text(
    yaml.dump(best_cfg.model_dump(), default_flow_style=False)
)
```

#### 3. Evaluate

Run the frozen config across all seeds to get the validation results you report:

```bash
python train.py --config configs/best_xgboost.yaml
```

Then evaluate the held-out test set — this is the final, unbiased estimate. Use the same pipeline that was used during cross-validation so preprocessing is consistent. Train across all seeds to confirm stability:

```python
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

test_aucs = []

for seed in cfg.seeds:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBClassifier(**study.best_params, random_state=seed)),
    ])
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    test_aucs.append(roc_auc_score(y_test, y_prob))

with mlflow.start_run(run_name="final_test_evaluation"):
    mlflow.log_params(study.best_params)
    mlflow.log_metric("mean_test_auc", np.mean(test_aucs))
    mlflow.log_metric("std_test_auc", np.std(test_aucs))
```

If the mean test AUC is substantially lower than the cross-validation AUC, the model is overfitting to the training distribution. If the standard deviation across seeds is large, the model is unstable. Do not go back and tune — that turns the test set into a validation set.

**Small datasets**: when the dataset is too small for a reliable held-out test set (roughly n < 1,000), the holdout estimate has high variance. Use nested cross-validation instead: an outer loop evaluates generalization while an inner loop selects hyperparameters. This gives an unbiased performance estimate without sacrificing data. The cost is computational — 5-fold outer × 5-fold inner × 100 trials is expensive, so reserve nested CV for settings where every sample counts.

**Search strategy**: Bayesian optimization (Optuna's TPE sampler) outperforms grid search and random search in most settings — it focuses trials on promising regions rather than sampling blindly. Random search is a reasonable baseline when the search space is small. Grid search is wasteful — it spends trials on unimportant dimensions.

**Reproducibility**: `TPESampler(seed=...)` makes the search deterministic, but only with sequential execution. Parallel optimization (`n_jobs > 1`) reseeds internally to avoid duplicate suggestions across workers, breaking reproducibility. The search evaluates each trial on a single seed's CV folds — this adds variance to the selection. The multi-seed evaluation in step 3 guards against lucky picks: if the frozen config degrades across 10 seeds, the search result was not robust.

**Budget**: start with 100 trials. Check the optimization history — if the best score plateaus after 30 trials, more trials will not help. If it is still improving at 100 trials, the search space may be too large or the problem too noisy. Be aware of the total cost: 100 Optuna trials × 5-fold CV = 500 model fits for the search alone, plus 10 seeds × 5-fold CV = 50 fits for the final evaluation. Start with fewer seeds or trials during exploration and scale up for the final run.

### Ablation study

Ablation answers: "does this component actually help?"

1. **Start from the full system** — the configuration with all components enabled. This is your control.
2. **Remove one component at a time** and re-run. Each ablation variant changes exactly one thing so the effect is attributable. This is leave-one-component-out (LOCO) ablation — the standard approach. Its limitation: it cannot detect interaction effects. If two components are individually weak but jointly critical (synergy), or if one component compensates when another is removed (masquerade), LOCO will miss it. Use a factorial design when you suspect interactions.
3. **Run each variant across multiple seeds** (same seeds for all variants).
4. **Compare to the control** using the same metric and statistical test (see [Evaluation: comparing results]({{ site.baseurl }}/docs/machine-learning/evaluation/evaluation/#comparing-results)).

Run all variants with a simple loop:

```bash
for config in configs/ablation_*.yaml; do
    python train.py --config "$config"
done
```

Report ablation results as a table: each row is a variant, columns show the metric mean ± std, and the delta from the control. This is the format reviewers expect:

| Variant | AUC (mean ± std) | Δ vs. control | p-value |
|---------|-------------------|---------------|---------|
| Full system (control) | 0.891 ± 0.012 | — | — |
| − text features | 0.864 ± 0.015 | −0.027 | 0.003 |
| − interaction terms | 0.887 ± 0.013 | −0.004 | 0.42 |
| − feature selection | 0.879 ± 0.014 | −0.012 | 0.04 |

<p align="center">
  <img src="{{ "/assets/images/plots/ablation.svg" | relative_url }}" alt="Ablation study bar chart showing AUC delta vs control with significance markers">
</p>

For statistical tests and how to read comparison tables, see [Evaluation: comparing results]({{ site.baseurl }}/docs/machine-learning/evaluation/evaluation/#comparing-results).

---

## Common pitfalls

- **Single-seed conclusions.** One seed proves nothing. A model can win on seed 42 and lose on seed 123. Always run multiple seeds and report the distribution.
- **Overfitting the validation set.** If you run 200 hyperparameter trials optimizing validation AUC, the best validation AUC is biased upward. The test set must be held out from the entire search process — evaluate it once, at the end, on the selected configuration only.
- **Comparing across different splits.** If model A was evaluated on a random 80/20 split and model B on 5-fold CV, the comparison is meaningless. Fix the evaluation protocol before comparing.
- **Leakage through preprocessing.** Fitting a scaler, encoder, or imputer on the full dataset before splitting leaks information from the test set into training. All preprocessing must be fit inside the cross-validation loop, on the training fold only.
- **Selective reporting.** Running many experiments and reporting only the best result is p-hacking. Report all variants, including the ones that did not work — failed experiments are information.
- **Confounding seed with configuration.** If you change the learning rate and the seed at the same time, you cannot attribute the result to either. Change one variable at a time, or use a full factorial design with multiple seeds per configuration.
- **Not logging enough to reproduce.** If you cannot re-run a result six months later from the logged config, the logging was insufficient. Log the complete config, the git hash, and the environment. Test reproduction before you need it.
- **Not analyzing errors before iterating.** Jumping from a baseline score to hyperparameter tuning skips the most informative step. Error analysis on ~100 misclassified examples often reveals that the dominant error category (missing features, mislabeled data, a broken preprocessing step) cannot be fixed by tuning — and that fixing it improves performance more than any hyperparameter change would (see [Error analysis](#error-analysis)).
- **Not documenting compute resources.** A result that takes 8 GPU-hours to reproduce but does not say so wastes the reviewer's time — or makes the result irreproducible for anyone without equivalent hardware. Log runtime, hardware (CPU/GPU, memory), and total compute cost alongside metrics.

---
