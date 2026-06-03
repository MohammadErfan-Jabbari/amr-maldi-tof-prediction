"""Reproduce the headline result on the distribution-matched holdout.

Run:  uv run python experiments/reproduce_holdout_eval.py

The Kaggle leaderboard closed, so the public/private LB numbers can't be re-queried.
This script instead reproduces the *approach* on the held-out validation split that
mirrors the test species distribution (2,688 train / 672 val), which is the signal we
actually controlled during the competition. It trains three diverse gradient-boosted
models, blends them by rank averaging, applies the intrinsic-resistance rules, and
reports masked mean AUC overall, per antibiotic, and per species.

Outputs: reports/holdout_eval.json + reports/figures/holdout_auc_by_species.png
"""
import os
# Cap threads BEFORE importing numerical libraries. This machine has 96 cores and each
# booster defaults to all of them, causing catastrophic thread oversubscription. 8 threads
# per fit is plenty here and avoids the contention that otherwise stalls the run.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

from pathlib import Path
import json
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.dataset import load_validation_split, ANTIBIOTICS  # noqa: E402

from lightgbm import LGBMClassifier  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402
from catboost import CatBoostClassifier  # noqa: E402

SPECIES = {0: "E. coli", 1: "K. pneumoniae", 2: "P. mirabilis", 3: "P. aeruginosa"}
# Species reweighting toward the test distribution (mega_blend setting).
SPECIES_WEIGHTS = {0: 1.5, 1: 3.0, 2: 1.5, 3: 0.05}
INTRINSIC = {3: ["Ampicillin", "Amoxicillin_Clavulanic_acid", "Ertapenem",
                 "Cefotaxime", "Cefuroxime"], 2: ["Imipenem"]}
AB_IDX = {ab: i for i, ab in enumerate(ANTIBIOTICS)}


def make_models():
    return {
        "lgb": LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.03,
                              subsample=0.8, colsample_bytree=0.5, min_child_samples=25,
                              n_jobs=8, verbose=-1),
        "xgb": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.03,
                             subsample=0.8, colsample_bytree=0.5, tree_method="hist",
                             n_jobs=8, eval_metric="logloss"),
        "cat": CatBoostClassifier(iterations=300, depth=6, learning_rate=0.03,
                                  thread_count=8, verbose=0),
    }


def train_predict(model, X_tr, y_tr, sp_tr, X_val):
    """One model per antibiotic, masking NaN labels, with species sample weights."""
    preds = np.zeros((len(X_val), len(ANTIBIOTICS)))
    for j in range(len(ANTIBIOTICS)):
        m = ~np.isnan(y_tr[:, j])
        if len(np.unique(y_tr[m, j])) < 2:
            continue
        w = np.array([SPECIES_WEIGHTS[int(s)] for s in sp_tr[m]])
        model.fit(X_tr[m], y_tr[m, j].astype(int), sample_weight=w)
        preds[:, j] = model.predict_proba(X_val)[:, 1]
    return preds


def rank_average(pred_list):
    out = np.zeros_like(pred_list[0])
    for j in range(out.shape[1]):
        out[:, j] = np.mean([rankdata(p[:, j]) / len(p) for p in pred_list], axis=0)
    return out


def apply_intrinsic(pred, species_ids):
    pred = pred.copy()
    for sid, abx in INTRINSIC.items():
        mask = species_ids == sid
        for ab in abx:
            pred[mask, AB_IDX[ab]] = 1.0
    return pred


def masked_auc_per_antibiotic(y, p):
    aucs = {}
    for j, ab in enumerate(ANTIBIOTICS):
        m = ~np.isnan(y[:, j])
        if m.sum() > 0 and len(np.unique(y[m, j])) == 2:
            aucs[ab] = roc_auc_score(y[m, j], p[m, j])
    return aucs


def main():
    X_tr, X_val, y_tr, y_val, sp_tr, sp_val = load_validation_split()
    sp_val = np.asarray(sp_val).astype(int)
    print(f"train {X_tr.shape}  val {X_val.shape}  (val matches test species mix)")

    individual = {}
    preds = []
    for name, model in make_models().items():
        p = train_predict(model, X_tr, y_tr, sp_tr, X_val)
        preds.append(p)
        aucs = masked_auc_per_antibiotic(y_val, p)
        individual[name] = float(np.mean(list(aucs.values())))
        print(f"  {name:<4} holdout mean AUC: {individual[name]:.4f}")

    blend = rank_average(preds)
    blend_aucs = masked_auc_per_antibiotic(y_val, blend)
    blend_mean = float(np.mean(list(blend_aucs.values())))

    final = apply_intrinsic(blend, sp_val)
    final_aucs = masked_auc_per_antibiotic(y_val, final)
    final_mean = float(np.mean(list(final_aucs.values())))

    print("\n=== blend (rank average of 3 models) ===")
    print(f"mean AUC                 : {blend_mean:.4f}")
    print(f"mean AUC + intrinsic rules: {final_mean:.4f}")

    print("\n=== per-antibiotic AUC (final) ===")
    for ab, a in final_aucs.items():
        print(f"  {ab:<32}{a:.4f}")

    print("\n=== per-species mean AUC (final) ===")
    per_species = {}
    for sid, name in SPECIES.items():
        idx = sp_val == sid
        if idx.sum() == 0:
            continue
        sub = masked_auc_per_antibiotic(y_val[idx], final[idx])
        if sub:
            per_species[name] = float(np.mean(list(sub.values())))
            print(f"  {name:<16}{per_species[name]:.4f}  (n={int(idx.sum())})")

    results = {
        "individual_models": individual,
        "blend_mean_auc": blend_mean,
        "blend_plus_intrinsic_mean_auc": final_mean,
        "per_antibiotic": final_aucs,
        "per_species": per_species,
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "holdout_eval.json").write_text(json.dumps(results, indent=2))

    # Figure: per-species mean AUC
    (ROOT / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    names = list(per_species.keys())
    vals = [per_species[n] for n in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(names, vals, color="#1a365d")
    ax.axvline(final_mean, color="#e53e3e", ls="--", lw=1.5,
               label=f"overall {final_mean:.3f}")
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("mean AUC (distribution-matched holdout)")
    ax.set_title("Per-species performance — final pipeline")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "figures" / "holdout_auc_by_species.png", dpi=130)
    print("\nsaved reports/holdout_eval.json + reports/figures/holdout_auc_by_species.png")


if __name__ == "__main__":
    main()
