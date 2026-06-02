# 01 — Problem & Data

## The task

Given a bacterial isolate's **MALDI-TOF mass spectrum**, predict its **resistance
(0/1) to 8 antibiotics**. This is a *multi-label* classification problem, scored by
the **mean ROC-AUC across the 8 antibiotics** (Kaggle public LB = 40% of test,
private = 60%).

The 8 targets (with training resistance rate):

| Antibiotic | Class | Train resistance |
|---|---|---|
| Ampicillin | aminopenicillin | 88.2% |
| Amoxicillin/Clavulanic acid | penicillin + inhibitor | 29.9% |
| Cefotaxime | 3rd-gen cephalosporin | 73.8% |
| Cefuroxime | 2nd-gen cephalosporin | 76.4% |
| Ciprofloxacin | fluoroquinolone | 61.1% |
| Ertapenem | carbapenem | 64.5% |
| Imipenem | carbapenem | 38.9% |
| Levofloxacin | fluoroquinolone | 62.4% |

## The data

- **6,000 features** per sample: binned MALDI-TOF m/z intensities (~3 Da/bin over the
  ~2–20 kDa range dominated by ribosomal proteins). The matrix is **93.3% zeros** —
  natural sparsity, plus ~18.8% of features are constant (zero-variance) and removable.
- **3,360 train / 1,000 test** samples across **4 bacterial species**.
- **Labels are partially missing** (e.g. 42.8% missing for Amoxicillin/Clavulanic
  acid, and ~96% missing for that drug within P. aeruginosa). Missingness is **not
  random** — it tracks species and clinical relevance, so it must be *masked*, never
  imputed or treated as negative.

## Why it's hard (the three structural traps)

1. **Covariate shift between train and test.** The species mixtures are nearly
   inverted (see [02 — EDA Findings](02_eda_findings.md)). Any validation scheme that
   samples i.i.d. from training reports performance on the wrong population.
2. **Extreme sparsity + modest sample size.** 6,000 sparse features over 3,360 rows
   favors tree models with native sparse handling over neural networks.
3. **Missing, non-random labels.** The semi-supervised structure means the loss and
   the metric must both ignore NaN targets per antibiotic.

These three properties drive every methodological choice in
[03 — Methodology](03_methodology.md).
