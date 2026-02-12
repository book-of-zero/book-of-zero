---
layout: post
title: "Metrics: measure what matters"
nav_order: 5
---

Optimize for the wrong metric and a model that looks great on paper fails in production. The goal is not to know every metric — it is to pick the right few for each phase of the workflow and read the numbers correctly.

---

## On this page

- [Key concepts](#key-concepts)
- [The confusion matrix](#the-confusion-matrix)
- [Workflow: classification](#workflow-classification)
  - [Baseline](#baseline)
  - [Model selection](#model-selection)
  - [Threshold tuning and evaluation](#threshold-tuning-and-evaluation)
  - [Production monitoring](#production-monitoring)
- [Workflow: regression](#workflow-regression)
  - [Baseline](#baseline-1)
  - [Model selection and evaluation](#model-selection-and-evaluation)
  - [Production monitoring](#production-monitoring-1)
- [Reading the numbers](#reading-the-numbers)
- [Custom metrics](#custom-metrics)

---

## Key concepts

- **Confusion matrix**: A 2×2 table that counts true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN). Every classification metric is a function of these four numbers.
- **Threshold**: The probability cutoff that turns a model's continuous output into a binary decision. Moving the threshold trades precision for recall.
- **Threshold-independent metric**: Evaluates the model across all thresholds (e.g., AUC). Use for model comparison — it separates "is this a good model?" from "where do I set the cutoff?"
- **Threshold-dependent metric**: Evaluates the model at one specific cutoff (e.g., precision at 0.5). Use for final evaluation — it answers "how will this perform at the operating point I deploy?"
- **Precision**: Of everything the model flagged as positive, how many actually were. High precision = few false alarms.
- **Recall**: Of all actual positives, how many the model caught. High recall = few missed cases.
- **Log loss (cross-entropy)**: Measures how well predicted probabilities match actual outcomes. Unlike AUC or MCC, it penalizes poorly calibrated probabilities — a confident wrong prediction is punished heavily. For a direct check of calibration (whether a predicted 70% is actually right 70% of the time), use calibration curves.

---

## The confusion matrix

Read every classification metric from this table:

```
                  Predicted positive    Predicted negative
Actual positive        TP                     FN
Actual negative        FP                     TN
```

- **TP (true positive)**: correctly caught.
- **FN (false negative)**: missed — the model said "no" but it was "yes." Cost depends on domain (missed fraud, missed diagnosis).
- **FP (false positive)**: false alarm — the model said "yes" but it was "no." Cost depends on domain (unnecessary investigation, wasted resources).
- **TN (true negative)**: correctly ignored.

Every metric below is just a different way of combining these four numbers to answer a specific question.

---

## Workflow: classification

### Baseline

Before training any model, compute the baseline: what score does a naive predictor achieve?

- **Majority-class classifier**: always predicts the dominant class. This is your accuracy baseline. If your model barely beats it, the model is not useful.
- **Random classifier**: AUC-ROC = 0.5, MCC = 0. Any trained model should clear these comfortably.

Establish the baseline first. It prevents you from celebrating a "92% accuracy" on a dataset that is 91% one class — this is the **accuracy paradox.** A model that predicts the majority class for every input achieves high accuracy on imbalanced data while being completely useless. This is why accuracy alone is insufficient for imbalanced problems. Always pair it with metrics that reflect performance on the minority class: recall, precision, F1, MCC, or AUC-PR.

### Model selection

You are comparing models or hyperparameter sets. Use threshold-independent metrics so the comparison is fair — you are judging the model's ranking ability, not a particular cutoff.

To compare reliably, evaluate with cross-validation — typically 5-fold. Sklearn uses stratified k-fold by default for classifiers, preserving class proportions in each fold. A single train/test split can be noisy; cross-validation gives you a mean and standard deviation for each metric.

**Balanced data → AUC-ROC.**

AUC-ROC measures ranking quality: if you pick a random positive and a random negative, AUC is the probability that the model scores the positive higher. AUC = 1.0 is perfect separation, 0.5 is random — the model cannot tell the two apart. Technically, it plots recall (true positive rate) vs false positive rate across all thresholds. AUC says nothing about whether predicted probabilities are calibrated.

How to interpret:
- 0.9–1.0: outstanding discriminator.
- 0.8–0.9: strong — usually production-ready.
- 0.7–0.8: acceptable — may be enough depending on the problem.
- 0.5–0.7: poor — revisit features or model choice.

**Imbalanced data → AUC-PR.**

AUC-ROC can look misleadingly good on imbalanced data. The reason is FPR dilution: `FPR = FP / (FP + TN)`, and when negatives dominate, TN is huge — so even many false positives barely move the FPR. The ROC curve hugs the upper-left corner and AUC inflates. AUC-PR focuses on the positive (minority) class only, so it gives a more honest picture when positives are rare.

How to interpret: AUC-PR has no universal "good" threshold — the baseline depends on class prevalence. A random classifier scores AUC-PR ≈ the positive class rate (e.g., 0.01 if 1% positive). Any trained model should far exceed that. Compare AUC-PR between models, not against a fixed number.

Use AUC-PR for fraud detection, disease screening, defect detection — any problem where the positive class is under ~10-20% of the data.

**When in doubt → MCC.**

MCC (Matthews Correlation Coefficient) uses all four quadrants of the confusion matrix. It ranges from −1 (perfect inversion) through 0 (random) to +1 (perfect). Unlike accuracy and F1, MCC can only score high if the model handles both positives and negatives well — F1 ignores TN entirely, so a model that over-predicts the positive class can still get a decent F1. MCC catches that. It works regardless of class balance.

`MCC = (TP × TN − FP × FN) / √((TP + FP)(TP + FN)(TN + FP)(TN + FN))`

How to interpret:
- 0.7–1.0: strong.
- 0.4–0.7: moderate — may be enough depending on the problem.
- 0.2–0.4: weak — the model has some signal but not much.
- 0.0–0.2: negligible — effectively noise.
- Slightly negative (e.g., −0.05): likely sampling noise, not meaningful — treat as random-level performance.
- Strongly negative (e.g., −0.5): systematic inversion — the model's predictions are anti-correlated with reality. Check for label bugs or data leakage. If deliberate, flip the predictions.

**When probabilities matter → log loss.**

AUC and MCC evaluate whether the model ranks or classifies correctly — neither tells you if the predicted probabilities are well-matched to reality. Log loss (cross-entropy) penalizes confident wrong predictions heavily. Note: log loss is a proper scoring rule that reflects both calibration and discrimination together — it cannot isolate one from the other. For a direct assessment of calibration, use calibration curves (reliability diagrams).

`Log loss = −mean(y × log(p) + (1 − y) × log(1 − p))`

How to interpret: log loss = 0 is perfect. The baseline is the log loss of always predicting the class prevalence (e.g., always predicting 0.01 for a 1% positive rate). Lower is better. There are no universal "good" bands — compare against the baseline and between models. A sudden rise in log loss often means the model is becoming overconfident on wrong predictions.

Use log loss when downstream decisions depend on the probability itself, not just the class label — risk scoring, bid optimization, or any system that uses predicted probabilities as inputs. A model can have a high AUC but poorly calibrated probabilities; log loss catches this.

**Multiclass.**

All metrics above extend to multiclass problems. The key decision is how to average across classes — and it changes the story. Imagine three classes with 1000, 100, and 10 samples:

- **`macro`**: compute the metric per class, then take the unweighted mean. The 10-sample class counts as much as the 1000-sample class. Use this when rare classes are important — it surfaces poor performance on them.
- **`weighted`**: same as macro, but weighted by the number of true instances per class. The 1000-sample class dominates. Use this when the score should reflect overall performance proportional to class frequency.
- **`micro`**: pool all TP, FP, FN globally, then compute. For multiclass (not multilabel), micro F1 = micro precision = micro recall = accuracy.

MCC works for multiclass without averaging — pass predictions directly to `sklearn.metrics.matthews_corrcoef`. Note: for multiclass, the minimum MCC is between −1 and 0 (depending on class distribution), not exactly −1.

AUC-ROC extends via one-vs-rest (`ovr`) or one-vs-one (`ovo`) and requires probability scores. For imbalanced multiclass, prefer `ovo` with `average='macro'` — it is insensitive to class imbalance.

### Threshold tuning and evaluation

You picked a model. Now you choose the operating point — the threshold where the model makes actual decisions.

This is where you align the metric with the **business cost of errors**.

**False negatives are expensive → optimize recall.**

`Recall = TP / (TP + FN)`. Lowering the threshold catches more positives but increases false alarms.

Use when: a miss is dangerous or costly (medical screening, security threats). Accept more false positives to catch more true positives.

**False positives are expensive → optimize precision.**

`Precision = TP / (TP + FP)`. Raising the threshold reduces false alarms but misses more positives.

Use when: each positive prediction triggers an expensive action (manual review, production rollback, customer intervention).

**Both matter roughly equally → use F1.**

`F1 = 2 × (precision × recall) / (precision + recall)`. The harmonic mean penalizes extreme imbalances — a model with 99% precision and 1% recall gets a low F1.

F1 assumes false positives and false negatives cost the same. If they do not, use F-beta with a beta > 1 (favor recall) or beta < 1 (favor precision).

**How to pick the threshold in practice:**

1. Plot precision and recall as a function of threshold.
2. Identify the region where both are acceptable for your use case.
3. Pick the threshold that matches your cost trade-off.
4. Report precision, recall, and F1 at that threshold — not just the "best F1."

### Production monitoring

Models degrade. Data distributions shift, upstream features change, and user behavior evolves.

Performance metrics (F1, precision, recall) require ground truth labels. In many production settings, labels are delayed or unavailable — this is one of the hardest problems in production ML. When labels are delayed, rely on feature drift and prediction drift as proxy signals until labels arrive.

- Track the same metric you optimized at threshold tuning, computed on live predictions once labels are available.
- Set alerts for significant drops. A simple starting point is a fixed threshold (e.g., F1 drops more than 5% from a rolling average). More robust approaches use statistical tests (e.g., PSI, KS test) that account for sample size and natural variance.
- Watch for **training-serving skew**: features computed differently at training time and serving time. This is a common and hard-to-detect failure mode.
- Monitor feature and prediction distributions against a **reference baseline** (typically the training data distribution). This does not require labels and can catch degradation before performance metrics show it.

---

## Workflow: regression

### Baseline

- **Mean predictor**: always predicts the mean of the target. Your model should beat this comfortably.
- For time series: the naive baseline is "predict the last observed value."

### Model selection and evaluation

Unlike classification, regression does not have a threshold step. The same metrics serve both model selection and evaluation. As with classification, use cross-validation (typically 5-fold) to get reliable estimates — sklearn uses standard k-fold by default for regressors.

**Default: MAE.**

The average size of the error, in the same unit as the target. Easy to interpret: "the model is off by 2.3 units on average." MAE treats all errors equally — it is robust to outliers and is the right default unless you have a specific reason to penalize large errors more.

`MAE = mean(|y − ŷ|)`

**When big misses are costly: RMSE.**

RMSE punishes large errors disproportionately — the squaring step makes sure a few big misses raise RMSE much more than they raise MAE. Use RMSE when a large error is much worse than a small one: demand forecasting (big underestimate = stockout), energy grid planning, structural engineering.

`RMSE = √(mean((y − ŷ)²))`

**For scale-independent comparison: R².**

R² answers one question: what proportion of the variance in the target does the model explain? R² = 1.0 is perfect, R² = 0 means the model is no better than predicting the mean.

`R² = 1 − (SS_res / SS_tot)`, where `SS_res = Σ(y − ŷ)²` and `SS_tot = Σ(y − ȳ)²`.

Use R² when comparing models across different targets or scales (MAE and RMSE are in the target's units, so they are not directly comparable across different problems). R² is also sklearn's default `.score()` for regressors.

How to interpret:
- R² = 1.0: perfect — the model explains all variance.
- 0.7–1.0: strong — the model captures most of the signal.
- 0.4–0.7: moderate — useful but missing substantial variance. Check for missing features or nonlinearity.
- 0.0–0.4: weak — the model explains little. Consider whether the problem is inherently noisy or the features are insufficient.
- R² < 0: the model is worse than always predicting the mean. Something is fundamentally wrong.

**Interpreting MAE and RMSE together.** RMSE is always ≥ MAE. The gap between them tells you something: if RMSE ≈ MAE, errors are uniform — the model misses by roughly the same amount everywhere. If RMSE >> MAE, a few large errors are pulling RMSE up — investigate the outliers, as they may reveal a data quality issue, a subpopulation the model cannot handle, or a feature gap. Report both: MAE tells you the typical error, RMSE tells you about the worst-case behavior.

### Production monitoring

As with classification, performance metrics require ground truth labels. When labels are delayed, monitor feature drift and prediction drift as proxy signals.

- Track MAE (or RMSE, depending on what you optimized) on live predictions once labels are available.
- Watch for **error drift**: if MAE rises steadily, investigate — possible causes include concept drift (the relationship between features and target is shifting), data quality issues, population shift, or upstream pipeline changes.
- Monitor **error percentiles** (P90, P99) over time. Rising tail percentiles reveal growing outlier errors that MAE alone can mask — a subpopulation may be diverging from training data.
- Compare live R² against the training R². A large drop signals that the model's explanatory power has degraded.
- Check for **feature drift**: if the distribution of key input features shifts significantly from training, the model's predictions may degrade before the target metric shows it.

---

## Reading the numbers

Metrics are only useful if you interpret them in context.

- **Compare train and validation metrics.** If the training score is high and the validation score is low, the model is overfitting — add regularization, reduce features, or get more data. If both are low, it is underfitting. A small gap with high scores on both is a good fit.
- **Always compare to the baseline.** An F1 of 0.85 means nothing in isolation. If the majority-class baseline gives 0.80, the model adds almost no value.
- **Check both classes.** A high overall score can hide poor performance on the minority class. Always look at per-class precision and recall.
- **Compare MAE and RMSE.** If RMSE is much larger than MAE, you have an outlier problem — do not average it away.
- **Negative MCC: check the magnitude.** Slightly negative (near zero) is likely noise. Strongly negative means the model is systematically inverted — check for label errors, data leakage, or a flipped target. If the inversion is confirmed, flip the predictions.
- **Context determines "good enough."** An AUC of 0.75 can be strong for a hard medical imaging task and terrible for a spam filter. There is no universal threshold for a "good" score.
- **Suspiciously perfect scores signal leakage.** If a model achieves an AUC of 0.99 on a non-trivial problem, the most likely explanation is data leakage — not a great model. Check for target information bleeding into features, preprocessing applied before the train/test split, or duplicate records across splits. Similarly, tuning hyperparameters on the test set inflates all metrics — always use a separate validation set or cross-validation.
- **Read the standard deviation from cross-validation.** A mean AUC of 0.83 ± 0.02 is more trustworthy than 0.85 ± 0.08. High variance across folds means the model's performance depends heavily on which data it sees — the model may be unstable or the dataset too small. If the standard deviation is large relative to the difference between models, you cannot confidently pick a winner.
- **Test set class balance must match deployment.** Precision, recall, and F1 depend on the ratio of positives to negatives. If your test set has 10% positives but production has 1%, precision will be worse in production than your test results suggest. AUC-ROC is invariant to class balance shifts, but threshold-dependent metrics are not.
- **Small test sets lie.** A metric computed on 100 samples has wide confidence intervals. If you cannot increase the test set, use bootstrap resampling to quantify uncertainty.

---

## Custom metrics

Metrics like accuracy and F1 assume all errors cost the same. In practice they rarely do. When the business cost of a false negative is fundamentally different from a false positive, you need cost-sensitive evaluation.

**Try the simple options first.** Before building a custom metric, check whether sklearn's built-in mechanisms are enough. `class_weight` (available in `LogisticRegression`, `SVC`, `RandomForestClassifier`, etc.) lets you set per-class weights that modify the training loss — e.g., `class_weight={0: 1, 1: 50}` encodes a 50:1 cost ratio directly. `sample_weight` allows per-sample weighting in both training and evaluation. These handle many cost-sensitive scenarios without writing any custom code.

**When to build a custom metric:**

- A false negative costs 100× more than a false positive (or vice versa) and F-beta does not capture the ratio well enough.
- The business cares about a quantity that no standard metric measures (e.g., revenue impact per prediction, time-to-detection).
- You need to combine multiple signals into a single optimization target (e.g., revenue-weighted accuracy). Note: for constraints like latency, the standard approach is to treat them as satisficing metrics (must be below a threshold) and optimize accuracy separately, rather than combining them into one formula.

**How to build one:**

1. **Write down the cost matrix.** Assign a concrete cost (dollars, hours, risk) to each cell of the confusion matrix. If you cannot quantify it, estimate the ratio — "a missed fraud case costs roughly 50× more than a false alert" is enough.
2. **Define the formula.** `Weighted cost = (FN × cost_fn) + (FP × cost_fp)`. Minimize this. This assumes correct predictions (TP, TN) have zero or equal cost — if a true positive has direct value (e.g., revenue from an approved loan), include those terms too. For regression, replace the uniform loss with a domain-specific penalty (e.g., asymmetric loss that penalizes underestimates more than overestimates).
3. **Make it a scorer.** Wrap it so your training framework can optimize against it (e.g., `sklearn.metrics.make_scorer`). If the metric is not differentiable, use it for evaluation and cross-validation — optimize a differentiable proxy during training.
4. **Validate it against intuition.** Compute the custom metric on a few examples where you know the right answer. If the metric disagrees with human judgment, the cost model is wrong — fix the costs, not the judgment.

**Keep the standard metrics alongside it.** A custom metric optimizes for business value, but standard metrics (precision, recall, MCC) remain your sanity check. If the custom metric improves while MCC drops, something is off.
