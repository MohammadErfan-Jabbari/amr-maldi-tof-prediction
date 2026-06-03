"""Recompute every headline number used in the README/docs/poster from the raw data.

Run:  uv run python scripts/verify_claims.py

This is the single source of truth for the quantitative claims in this repo. If a
number here disagrees with the docs, the docs are wrong. Intrinsic-resistance rules
mirror `experiments/run_mega_blend.py::apply_intrinsic_rules` exactly.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"

ANTIBIOTICS = [
    "Ampicillin", "Levofloxacin", "Ciprofloxacin", "Imipenem",
    "Amoxicillin_Clavulanic_acid", "Ertapenem", "Cefotaxime", "Cefuroxime",
]
SPECIES = {0: "E. coli", 1: "K. pneumoniae", 2: "P. mirabilis", 3: "P. aeruginosa"}
# Intrinsic resistance rules — identical to run_mega_blend.apply_intrinsic_rules
INTRINSIC = {
    3: ["Ampicillin", "Amoxicillin_Clavulanic_acid", "Ertapenem", "Cefotaxime", "Cefuroxime"],
    2: ["Imipenem"],
}


def main():
    train = pd.read_csv(RAW / "train.csv")
    test = pd.read_csv(RAW / "test.csv")
    maldi = [c for c in train.columns if c.startswith("maldi_feature_")]

    print("=" * 64)
    print("DATASET SHAPE")
    print("=" * 64)
    print(f"train rows           : {len(train)}")
    print(f"test rows            : {len(test)}")
    print(f"MALDI features       : {len(maldi)}")
    print(f"antibiotic targets   : {len(ANTIBIOTICS)}")

    print("\n" + "=" * 64)
    print("SPECIES DISTRIBUTION SHIFT (train -> test)")
    print("=" * 64)
    tr = train["species_id"].value_counts(normalize=True) * 100
    te = test["species_id"].value_counts(normalize=True) * 100
    print(f"{'species':<16}{'train %':>10}{'test %':>10}")
    for sid, name in SPECIES.items():
        print(f"{name:<16}{tr.get(sid, 0):>9.1f}%{te.get(sid, 0):>9.1f}%")

    print("\n" + "=" * 64)
    print("FEATURE SPARSITY & ZERO-VARIANCE")
    print("=" * 64)
    Xtr = train[maldi].to_numpy(np.float32)
    Xte = test[maldi].to_numpy(np.float32)
    Xall = np.vstack([Xtr, Xte])
    print(f"zeros in train matrix    : {(Xtr == 0).mean() * 100:.2f}%")
    print(f"zeros in train+test      : {(Xall == 0).mean() * 100:.2f}%")
    zv = (Xtr.var(axis=0) < 1e-5).sum()
    print(f"zero-variance features   : {zv} of {len(maldi)} = {zv / len(maldi) * 100:.1f}%  (var < 1e-5)")
    print("  (NOTE: the old docs' '18.8% / 1128 constant features' and '765' figures are both"
          " unverified at this threshold — report the measured value above.)")

    print("\n" + "=" * 64)
    print("MISSING LABELS (train, % NaN per antibiotic)")
    print("=" * 64)
    for ab in ANTIBIOTICS:
        pct = train[ab].isna().mean() * 100
        flag = "  <- semi-supervised" if pct > 10 else ""
        print(f"{ab:<32}{pct:>6.1f}%{flag}")
    pa = train[train["species_id"] == 3]
    print(f"\nAmox/Clav missing within P. aeruginosa: "
          f"{pa['Amoxicillin_Clavulanic_acid'].isna().mean() * 100:.1f}%")

    print("\n" + "=" * 64)
    print("INTRINSIC RESISTANCE = FREE PREDICTIONS (test set)")
    print("=" * 64)
    n_test = len(test)
    n_cells = n_test * len(ANTIBIOTICS)
    free_cells = 0
    samples_with_free = np.zeros(n_test, dtype=bool)
    sid = test["species_id"].to_numpy()
    for species_id, abx in INTRINSIC.items():
        mask = sid == species_id
        free_cells += mask.sum() * len(abx)
        samples_with_free |= mask
        print(f"{SPECIES[species_id]:<16} {mask.sum():>4} samples x {len(abx)} drugs "
              f"= {mask.sum() * len(abx):>4} cells")
    print("-" * 40)
    print(f"deterministic cells      : {free_cells} of {n_cells} = {free_cells / n_cells * 100:.1f}% of cells")
    print(f"samples with >=1 free    : {samples_with_free.sum()} of {n_test} = "
          f"{samples_with_free.sum() / n_test * 100:.1f}% of samples")

    print("\n" + "=" * 64)
    print("ANTIBIOTIC CORRELATIONS (pairwise, non-missing)")
    print("=" * 64)
    for a, b in [("Levofloxacin", "Ciprofloxacin"),
                 ("Imipenem", "Ertapenem"),
                 ("Ertapenem", "Cefotaxime")]:
        sub = train[[a, b]].dropna()
        print(f"{a} <-> {b}: r = {sub[a].corr(sub[b]):.3f}  (n={len(sub)})")


if __name__ == "__main__":
    main()
