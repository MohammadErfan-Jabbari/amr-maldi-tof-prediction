# Experiments

Curated entry points at the top level; the full exploration is preserved in
[`archive/`](archive/).

## Canonical / representative (top level)

| Script | What it is |
|---|---|
| `run_mega_blend.py` | **Best submission** (public LB ~0.838): rank-average of 3 diverse pipelines + species reweighting + intrinsic-resistance rules |
| `run_self_training.py` | Semi-supervised: pseudo-labeling high-confidence test points |
| `transductive_base.py` | Transductive dimensionality reduction (fit reducer on train+test features) |
| `run_species_specific.py` | Per-species models |
| `run_miracle_v2.py` | Large diversity ensemble (did not beat the mega-blend) |
| `controlled_experiment.py` | Standardized CV + metrics harness used to compare approaches fairly |
| `ensemble_utils.py` | Averaging / rank-averaging / (and the stacking that overfit) |
| `evaluate_blends.py` | Rank candidate blends by validation AUC |
| `run_phase1.py` | Representative dimensionality-reduction ablation |

## archive/

Earlier phase-by-phase experiments and one-off variants, kept for provenance. Paths
were made portable, but their results are superseded by `run_mega_blend.py`. They are
not maintained — read them to trace the exploration, not to reproduce the best score.
