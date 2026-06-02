# 03 — Methodology

## Validation: match the deployment distribution, not the training distribution

The first and most important design choice. Because the test species mixture differs
structurally from training (see [02](02_eda_findings.md)), i.i.d. k-fold CV reports
performance on the wrong population. We construct a held-out validation set
**stratified to match the test species proportions** (2,688 train / 672 validation):

```python
from src.data.dataset import load_validation_split
X_train, X_val, y_train, y_val, species_train, species_val = load_validation_split()
```

All model selection is anchored to the **masked mean AUC across all 8 antibiotics**
on this split (`src/utils/metrics.py`), never to a single antibiotic or species, and
never to out-of-fold scores (which proved ~6–12 points optimistic).

## Handling the data's structure

- **Sparsity / feature selection** (`src/features/reducers.py`): drop zero-variance
  features; supervised reduction via **PLS** and per-species PLS; univariate
  (`f_classif`) and LightGBM-importance multi-stage selection. Unsupervised reduction
  (PCA/KernelPCA/NMF) was tried and generally underperformed supervised PLS.
- **Missing labels** (`src/utils/loss.py`): `MaskedBCEWithLogitsLoss` computes loss
  only over observed labels; the metric masks identically.
- **Species imbalance** (`src/models/lightgbm_baseline.py`): per-sample weights that
  upweight test-dominant species and downweight P. aeruginosa.

## Models

Gradient-boosted trees are the workhorse (sparse-native, robust on 3,360 rows):
**LightGBM**, **XGBoost**, **CatBoost**, plus a **PyTorch MLP** for diversity.
Semi-supervised extensions: **self-training** (pseudo-labeling high-confidence test
points) and **transductive dimensionality reduction** (fitting the reducer on
train+test features together). `experiments/controlled_experiment.py` provides a
standardized CV + metrics harness so approaches are compared on equal footing.

## Ensembling

Predictions from diverse pipelines are combined by **rank averaging** (average of
per-antibiotic rank-transformed scores), which is scale-free and robust. The best
submission — `experiments/run_mega_blend.py` — rank-averages three diverse pipelines
(PLS-LGB, a species-aware blend, and a tuned LightGBM), then applies species
reweighting and the intrinsic-resistance rules as a deterministic post-step.

Why not stacking? See [04 — Results & Lessons](04_results_and_lessons.md): the
meta-learner overfit the shifted-distribution fold artifacts and did not generalize.

## Reading path through the code

`src/data/dataset.py` → `src/features/reducers.py` →
`experiments/controlled_experiment.py` → `experiments/run_mega_blend.py` →
`experiments/run_self_training.py`.
