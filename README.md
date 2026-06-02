# Predicting Antibiotic Resistance from MALDI-TOF Mass Spectra

Multi-label prediction of bacterial resistance to **8 antibiotics** directly from
**MALDI-TOF** mass spectra — a master's ML project built around the
[Kaggle AMR Prediction competition](https://www.kaggle.com/competitions/antimicrobial-resistance-prediction-from-maldi-tof).

**Best result:** public leaderboard **0.838** mean AUC (rank-averaged ensemble);
private-set rank improved on the public rank, i.e. the chosen model *generalized*.

> **The interesting part of this project is not the score — it is the diagnosis.**
> Most of the work was discovering *why* obvious approaches fail on this dataset:
> a severe train→test population shift that breaks standard cross-validation, a
> proxy-metric trap, and an ensemble method (stacking) that looks best on paper and
> generalizes worst. Those lessons are documented in
> [`docs/04_results_and_lessons.md`](docs/04_results_and_lessons.md).

---

## Why this problem matters

Antimicrobial resistance (AMR) is a leading global-health threat. MALDI-TOF spectra
are **already acquired routinely** for microbial identification in clinical labs. If
antibiotic resistance can be read off the *same* spectrum, clinicians could choose
effective therapy days earlier than culture-based susceptibility testing allows. The
research question: **how much resistance signal is actually recoverable from these
spectra, and what is the honest performance ceiling?**

---

## The data in one screen

| Aspect | Value |
|---|---|
| Input | 6,000 binned MALDI-TOF m/z features (~3 Da/bin), **93.3% zeros** |
| Output | binary resistance for 8 antibiotics (multi-label) |
| Samples | 3,360 train · 1,000 test |
| Species | 4 bacterial species, **strong train↔test distribution shift** |
| Labels | partially missing (42.8% for Amoxicillin/Clavulanic acid) — **semi-supervised** |
| Metric | mean ROC-AUC across the 8 antibiotics |

![Species distribution shift](reports/figures/phase4_species_distribution_comparison.png)

*The defining difficulty: the training and test species mixtures are almost inverted.
P. aeruginosa is 43% of train but 3% of test; K. pneumoniae is 28% of train but 51% of test.*

---

## The four findings (the scientific core)

1. **Distribution shift is the whole game.** Train and test have nearly opposite
   species mixtures. Standard k-fold CV scores the *training* mixture; the leaderboard
   scores the *test* mixture, so out-of-fold AUC was systematically optimistic by
   ~6–12 points. The fix was a held-out validation set explicitly **stratified to match
   the test species proportions** — the prerequisite for any trustworthy decision.

2. **A per-subgroup proxy metric does not optimize the global objective.** Targeting
   the hardest, most test-dominant species (K. pneumoniae) raised its AUC but *lowered*
   the leaderboard, because the gain was paid for by E. coli and P. mirabilis. Model
   selection must be anchored to the **masked mean AUC across all 8 antibiotics**.

3. **The simplest ensemble generalized best.** Stacking reached the highest internal
   out-of-fold score but did not transfer — the meta-learner exploited fold-level
   artifacts produced by the shifted distribution. **Rank averaging**, which learns no
   combination, cannot learn those artifacts, and won.

4. **Domain knowledge supplies free, exact predictions.** Two species have *intrinsic*
   (deterministic) resistance: P. aeruginosa to five antibiotics, P. mirabilis to
   Imipenem. Hard-coding these removes variance without adding bias — ~4.3% of all test
   prediction cells (≈22% of test samples get at least one free label).

Full discussion: [`docs/04_results_and_lessons.md`](docs/04_results_and_lessons.md).

---

## Skills demonstrated

| Area | What's in here |
|---|---|
| **Modeling** | LightGBM, XGBoost, CatBoost, PyTorch MLP; PLS / PCA / KernelPCA / NMF |
| **Validation under shift** | distribution-matched holdout, species sample-reweighting |
| **Semi-supervised** | self-training / pseudo-labeling, transductive dimensionality reduction |
| **Ensembling** | rank averaging vs. weighted averaging vs. stacking (with the failure analysis) |
| **Missing labels** | masked BCE loss + masked metrics (custom, `src/utils/`) |
| **From-scratch components** | `MaskedBCEWithLogitsLoss`, `PerSpeciesPLS`, `MultiStageSelector`, `LGBImportanceSelector`, a controlled-experiment CV framework |
| **Domain integration** | intrinsic-resistance rules from microbiology |
| **Tooling** | `uv`, scikit-learn pipelines, custom JSON experiment + submission tracking |

---

## Reproduce

```bash
uv sync                                          # build the environment (Python ≥ 3.11)
# place competition CSVs in raw/  (see data/README.md)
uv run python scripts/smoke_test.py              # sanity check
uv run python experiments/run_mega_blend.py      # reproduce the best submission
```

All paths resolve relative to the repo root, so the scripts run from a fresh clone.

---

## Repository layout

```
src/             cleaned importable package (data · features · models · training · inference · utils)
experiments/     run_mega_blend.py (best) + key approaches; archive/ holds the full exploration
scripts/eda/     8-phase exploratory data analysis
docs/            problem, EDA findings, methodology, results & lessons (+ insights/, research/)
reports/figures/ curated figures
poster/          academic poster (LaTeX → PDF/PNG)
outputs/         curated submission CSVs
data/            how to obtain the data (data not distributed)
```

---

## Results (confirmed leaderboard submissions)

| Approach | Public LB (mean AUC) |
|---|---|
| LightGBM baseline | 0.832 |
| + species reweight + intrinsic rules | 0.833 |
| **Mega-blend (rank-average of 3 diverse pipelines)** | **0.838** |

On the **private** test set the best model scored **0.81307** (public 0.83862), and its
private leaderboard rank *improved* on its public rank — evidence it generalized rather
than overfitting the public split.

Stacking, K. pneumoniae-targeted weighting, self-training, transductive DR, and
per-species models were all explored; none beat the rank-averaged blend. Their
*negative* results are the most instructive part — see the lessons doc.

---

## Limitations & honest ceiling

The ~0.838 plateau appears structural: the model still encodes the
P. aeruginosa-dominated training representation, and loss reweighting alone cannot
manufacture K. pneumoniae signal that the training data underrepresents. Breaking
through would likely require species-conditioned inference, explicit domain
adaptation, or transductive methods that use test-set features during training —
none of which were solved within the competition window.

---

## References

- Competition: [Kaggle — AMR Prediction from MALDI-TOF](https://www.kaggle.com/competitions/antimicrobial-resistance-prediction-from-maldi-tof)
- Weis et al., *Direct antimicrobial resistance prediction from clinical MALDI-TOF mass spectra* (Nature Medicine, 2022)
- BorgwardtLab, [`maldi_amr`](https://github.com/BorgwardtLab/maldi_amr)

---

**Author:** Mohammad Erfan Jabbari · License: [MIT](LICENSE) · *Educational / research project.*
