# 04 — Results & Lessons

This is the heart of the project. The final score (public LB ~0.838) matters less than
the chain of failures that produced it.

## Timeline (what was tried → what happened → what was learned)

1. **Baseline, OOF-as-oracle.** LightGBM on the 6,000 sparse features reached LB ~0.832
   quickly, but every decision was made against out-of-fold AUC — which, under the
   species shift, was systematically ~6–12 points optimistic relative to the leaderboard.
2. **Stacking looked best, generalized worst.** A LightGBM meta-learner over base-model
   out-of-fold predictions produced the highest *internal* mean AUC seen (~0.95 OOF) but
   did **not** transfer; its leaderboard score fell below the simple blend. The
   meta-learner had calibrated on fold-level artifacts created by the shifted distribution.
3. **A simple blend quietly won.** Rank-averaging three independently trained pipelines
   scored LB ~0.838 — and was nearly under-valued because it was being judged against the
   inflated stacking OOF numbers.
4. **The wrong-metric trap.** After building a distribution-matched holdout, effort shifted
   to improving K. pneumoniae AUC (51% of test, the weakest species). Validation K. pn rose,
   but the leaderboard *fell* — the gain was paid for by E. coli and P. mirabilis.
5. **Metric realignment.** Model selection was re-anchored to masked mean AUC across all 8
   antibiotics; the rank-averaged blend was restored as best. No later method
   (species-specific models, transductive PCA, MLP variants) beat it.
6. **Biology as a shortcut.** Intrinsic-resistance rules fixed ~4.3% of test cells exactly
   — small, but free and bias-free.
7. **Structural ceiling.** ~0.838 appears to be where loss-reweighting alone plateaus; the
   model still encodes the P. aeruginosa-dominant training representation.

## Four lessons (stated generally)

**1. Validation distribution must mirror the deployment distribution, not the training
distribution.** When test covariates differ structurally from training, i.i.d. CV reports
performance on a population that will never be scored. A distribution-matched holdout was the
*minimum prerequisite* for trustworthy model selection; its absence invalidated several
iterations of "improvements."

**2. Optimizing a per-subgroup proxy does not optimize the global objective under imbalance.**
Improving the hardest, most test-dominant subgroup is locally rational but globally wrong when
the gain is reallocated capacity away from other subgroups. Optimize the metric you are scored
on, directly.

**3. Learned ensemble combination amplifies distributional leakage; non-learned combination does
not.** Stacking's large gap between internal OOF and held-out/leaderboard performance was not
ordinary variance overfitting — the meta-learner exploited fold artifacts that exist only under
the training distribution. Rank averaging learns no weights, so it has nothing to overfit.

**4. Deterministic domain priors belong as hard constraints on a known subset.** For organisms
with biologically guaranteed resistance, the answer is an invariant, not an inference. Overriding
model output on exactly those cells removes variance without adding bias.

## What worked vs. what didn't

| Worked | Didn't |
|---|---|
| Rank averaging | Stacking / meta-learners (overfit the shift) |
| Distribution-matched validation | Trusting out-of-fold AUC |
| Model diversity (LGB + XGB + CatBoost + MLP) | Neural nets alone |
| Supervised reduction (PLS) | Unsupervised reduction (PCA/KPCA) |
| Masking missing labels | Optimizing one species' AUC |
| Intrinsic-resistance rules | Treating missing labels as random |

## Pooled vs. within-species performance (reproduced)

`experiments/reproduce_holdout_eval.py` re-derives the pipeline on the distribution-matched
holdout (mean AUC 0.808). The aggregate hides the real structure: within each test-dominant
species the blend scores only 0.65–0.70 (E. coli 0.65, K. pneumoniae 0.69, P. mirabilis 0.70),
versus ~0.81 pooled. A large part of the pooled score is the model separating *species* —
which are themselves predictive of resistance — rather than discriminating resistance among
isolates of the *same* species. Per antibiotic, the carbapenem/aminopenicillin signals are
strong (Imipenem 0.96, Ampicillin 0.88) while the fluoroquinolones are weak (Levofloxacin
0.70, Ciprofloxacin 0.71). This decomposition is the honest account of where the ceiling
comes from, and it is the kind of result the headline metric alone would hide.

## A note on confirmed vs. internal numbers

Public-LB values reported here (~0.832 / ~0.833 / ~0.838) correspond to logged Kaggle
submissions (`docs/submissions/submissions_log.md`). The stacking OOF figure is an *internal*
out-of-fold estimate illustrating the overfit gap; it is presented qualitatively, not as a
confirmed leaderboard result.
