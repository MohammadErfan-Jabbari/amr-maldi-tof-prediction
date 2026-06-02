#!/usr/bin/env python3
"""
Minimal LightGBM baseline for AMR prediction.

Trains 8 separate LightGBM models (one per antibiotic) with:
- Constant feature removal
- Species-stratified 5-fold CV
- Per-fold and overall AUC tracking
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import load_train_data, ANTIBIOTICS
from utils.metrics import mean_auc


# Antibiotics as list for iteration
ANTIBIOTIC_LIST = ANTIBIOTICS
SPECIES_NAMES = {0: "E.coli", 1: "K.pneumoniae", 2: "P.mirabilis", 3: "P.aeruginosa"}


def compute_sample_weights(species: np.ndarray) -> np.ndarray:
    """
    Compute sample weights to counteract species distribution shift.
    Downweight P. aeruginosa (43% train → 3% test).

    Args:
        species: Array of species IDs

    Returns:
        Array of sample weights
    """
    return np.where(species == 3, 0.3, 1.0)  # P. aeruginosa = 0.3x


def remove_constant_features(X_train, X_test, threshold=1e-5):
    """Remove features with variance below threshold."""
    variances = X_train.var(axis=0)
    constant_mask = variances > threshold
    n_removed = (~constant_mask).sum()

    print(f"Removing {n_removed} constant/near-constant features (var < {threshold})")
    print(f"Remaining features: {constant_mask.sum()}")

    return X_train[:, constant_mask], X_test[:, constant_mask], constant_mask


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def train_lightgbm_baseline(data_dir: str = str(_PROJECT_ROOT / "raw")):
    """
    Train LightGBM baseline with species-stratified 5-fold CV.

    Returns:
        models: Dict of trained LightGBM models (one per antibiotic)
        oof_preds: Out-of-fold predictions (n_samples, n_antibiotics)
        feature_mask: Boolean mask of non-constant features
    """
    print("=" * 60)
    print("LightGBM Baseline Training")
    print("=" * 60)

    # Load data
    print("\n[Step 1] Loading data...")
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    test_df = pd.read_csv(f"{data_dir}/test.csv")

    # Extract features, labels, species
    feature_cols = [f"maldi_feature_{i}" for i in range(6000)]
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    y_train = train_df[ANTIBIOTIC_LIST].values
    species_train = train_df["species_id"].values

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Labels: {y_train.shape}")

    # Remove constant features
    print("\n[Step 2] Removing constant features...")
    X_train_filtered, X_test_filtered, feature_mask = remove_constant_features(X_train, X_test)

    # Initialize OOF predictions array
    oof_preds = np.full_like(y_train, np.nan, dtype=float)

    # Train one model per antibiotic
    print("\n[Step 3] Training models (5-fold species-stratified CV)...")
    print("-" * 60)

    models = {}

    for idx, antibiotic in enumerate(ANTIBIOTIC_LIST):
        print(f"\n[{idx+1}/8] {antibiotic}")

        # Get non-NaN labels for this antibiotic
        label_mask = ~pd.isna(y_train[:, idx])
        n_labeled = label_mask.sum()
        n_missing = (~label_mask).sum()

        print(f"  Labeled: {n_labeled}, Missing: {n_missing}")

        if n_labeled < 100:
            print(f"  WARNING: Very few labeled samples ({n_labeled})")

        # Get labels for this antibiotic
        y = y_train[label_mask, idx]
        X = X_train_filtered[label_mask]
        species = species_train[label_mask]

        # Species-stratified 5-fold CV
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        fold_aucs = []
        fold_models = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, species)):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]

            # Compute sample weights for this fold
            species_fold_train = species[train_idx]
            sample_weights_fold = compute_sample_weights(species_fold_train)

            # Train LightGBM
            model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            )

            model.fit(
                X_fold_train,
                y_fold_train,
                sample_weight=sample_weights_fold,
                eval_set=[(X_fold_val, y_fold_val)],
                callbacks=[lgb.early_stopping(10, verbose=False)]
            )

            # Predict on validation fold
            y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
            fold_auc = roc_auc_score(y_fold_val, y_pred_proba)
            fold_aucs.append(fold_auc)
            fold_models.append(model)

        # Average fold AUCs
        mean_fold_auc = np.mean(fold_aucs)
        std_fold_auc = np.std(fold_aucs)

        print(f"  CV AUC: {mean_fold_auc:.4f} +/- {std_fold_auc:.4f}")

        # Retrain on full data for final model
        sample_weights_full = compute_sample_weights(species)
        final_model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        final_model.fit(X, y, sample_weight=sample_weights_full)
        models[antibiotic] = final_model

        # Generate OOF predictions using fold models
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, species)):
            fold_model = fold_models[fold_idx]
            X_fold_val = X[val_idx]

            # Map back to original indices
            original_indices = np.where(label_mask)[0][val_idx]
            oof_preds[original_indices, idx] = fold_model.predict_proba(X_fold_val)[:, 1]

    # Compute overall metrics
    print("\n" + "=" * 60)
    print("[Step 4] Overall OOF Metrics")
    print("-" * 60)

    metrics = mean_auc(y_train, oof_preds, ANTIBIOTIC_LIST)

    print("\nPer-Antibiotic AUC:")
    for antibiotic in ANTIBIOTIC_LIST:
        if antibiotic in metrics:
            print(f"  {antibiotic:30} {metrics[antibiotic]:.4f}")

    print(f"\nMean AUC: {metrics['mean_auc']:.4f}")

    # Per-species metrics (critical for detecting distribution shift)
    print("\n" + "=" * 60)
    print("[Step 5] Per-Species OOF Metrics")
    print("=" * 60)
    print("(Critical for detecting distribution shift - K.pneumoniae is 51% of test!)")

    species_metrics_all = {}
    for species_id, species_name in SPECIES_NAMES.items():
        species_mask = (species_train == species_id)
        n_samples = species_mask.sum()

        if n_samples > 0:
            # Filter y_train and oof_preds to this species
            y_species = y_train[species_mask]
            oof_species = oof_preds[species_mask]

            # Compute metrics for this species
            species_metrics = mean_auc(y_species, oof_species, ANTIBIOTIC_LIST)
            species_metrics_all[species_name] = species_metrics

            # Highlight K. pneumoniae (51% of test!)
            marker = "⭐ PRIMARY TARGET (51% of test)" if species_id == 1 else ""
            print(f"\n{species_name} (n={n_samples}) {marker}")
            print(f"  Mean AUC: {species_metrics['mean_auc']:.4f}")

            # Show per-antibiotic if it's the key species
            if species_id == 1:  # K. pneumoniae
                for antibiotic in ANTIBIOTIC_LIST:
                    if antibiotic in species_metrics:
                        print(f"    {antibiotic:28} {species_metrics[antibiotic]:.4f}")

    # Add species metrics to return dict
    metrics['species_metrics'] = species_metrics_all

    return models, oof_preds, feature_mask, X_test_filtered, metrics


if __name__ == "__main__":
    models, oof_preds, feature_mask, X_test_filtered, metrics = train_lightgbm_baseline()

    # Save models
    output_dir = Path("outputs/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    for antibiotic, model in models.items():
        model_path = output_dir / f"lgb_baseline_{antibiotic}.txt"
        model.booster_.save_model(str(model_path))
        print(f"Saved model for {antibiotic} to {model_path}")

    print("\nTraining complete!")
