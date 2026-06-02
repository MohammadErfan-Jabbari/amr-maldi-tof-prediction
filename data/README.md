# Data

The competition data is **not** distributed in this repository (Kaggle terms +
size). Download it from the competition page and place the CSVs in `raw/`.

## Get the data

```bash
# Requires the Kaggle CLI and accepted competition rules
kaggle competitions download -c antimicrobial-resistance-prediction-from-maldi-tof
unzip antimicrobial-resistance-prediction-from-maldi-tof.zip -d raw/
```

Competition: https://www.kaggle.com/competitions/antimicrobial-resistance-prediction-from-maldi-tof

## Expected layout

```
raw/
├── train.csv               # 3,360 samples × (sample_id, species_id, 6000 MALDI features, 8 antibiotic labels)
├── test.csv                # 1,000 samples (features only)
├── sample_submission.csv
└── species_mapping.csv     # species_id → species name
```

- **Features**: `maldi_feature_0 … maldi_feature_5999` — binned MALDI-TOF m/z intensities (~93.3% zeros).
- **Targets**: 8 antibiotics, binary resistance (0/1), **partially missing** (semi-supervised).
- **Species**: 4 bacterial species, with a strong train↔test distribution shift.

## Processed cache

`data/processed/val_split.npz` is a cached, distribution-matched train/validation
split (2,688 / 672). It is regenerated automatically on first use:

```python
from src.data.dataset import load_validation_split
X_train, X_val, y_train, y_val, species_train, species_val = load_validation_split()
```
