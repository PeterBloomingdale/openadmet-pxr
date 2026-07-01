"""
Train both MMP-delta models: vanilla baseline + antisymmetric Siamese.

Prerequisite: scripts/08_build_mmps.py

Outputs:
- models/mmp_delta/baseline_booster.txt (LightGBM model)
- models/mmp_delta/baseline_oof.npy
- models/mmp_delta/baseline_test_preds.npy
- models/mmp_delta/siamese_model.pkl
- models/mmp_delta/siamese_oof.npy
- models/mmp_delta/siamese_test_preds.npy
- models/mmp_delta/alpha_weights.json (interpretable physics coefficients)

Runtime: baseline ~5 min; Siamese ~30-60 min on CPU (GPU faster)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np
import pandas as pd
import yaml
from loguru import logger

from openadmet.models.mmp_delta_baseline import train_mmp_delta_fold, predict_mmp_delta
from openadmet.models.delta_siamese import SiameseDeltaModel
from openadmet.data.splits import get_train_val_indices
from openadmet.cv.oof import evaluate_oof
from openadmet.utils.submission import format_submission


def train_baseline(train_df, test_df, config, smiles_col, pec50_col, out):
    """Train the vanilla MMP-delta baseline across all folds."""
    logger.info("Training MMP-delta baseline (vanilla LightGBM)...")
    n_folds = 5
    oof_preds = np.full(len(train_df), np.nan)
    boosters = []

    for fold in range(n_folds):
        train_idx, val_idx = get_train_val_indices(train_df, fold)
        fold_train = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val = train_df.iloc[val_idx].reset_index(drop=True)

        booster, fold_preds, _ = train_mmp_delta_fold(
            train_df=fold_train,
            val_df=fold_val,
            smiles_col=smiles_col,
            pec50_col=pec50_col,
            params=config["baseline"]["params"].copy(),
            fold_id=fold,
        )
        oof_preds[val_idx] = fold_preds
        boosters.append(booster)

    # Save last booster (full-data version for test prediction)
    # Re-train on full data
    booster_full, _, _ = train_mmp_delta_fold(
        train_df=train_df,
        val_df=train_df.head(50),  # Dummy val for API compatibility
        smiles_col=smiles_col,
        pec50_col=pec50_col,
        params=config["baseline"]["params"].copy(),
    )
    booster_full.save_model(str(out / "baseline_booster.txt"))

    # Test predictions
    test_preds = predict_mmp_delta(booster_full, test_df, train_df, smiles_col, pec50_col)

    np.save(out / "baseline_oof.npy", oof_preds)
    np.save(out / "baseline_test_preds.npy", test_preds)

    valid_mask = ~np.isnan(oof_preds)
    y_true = train_df[pec50_col].values
    metrics = evaluate_oof(y_true[valid_mask], oof_preds[valid_mask], train_df["fold"].values[valid_mask])
    logger.info(f"Baseline OOF MAE={metrics['mae']:.4f}, RAE={metrics['rae']:.4f}")
    return oof_preds, test_preds


def train_siamese(train_df, test_df, config, smiles_col, pec50_col, out):
    """Train the antisymmetric Siamese delta model."""
    logger.info("Training antisymmetric Siamese delta model...")
    scfg = config["siamese"]

    model = SiameseDeltaModel(
        fp_n_bits=scfg["fp_n_bits"],
        hidden_dim=scfg["hidden_dim"],
        n_hidden=scfg["n_hidden"],
        dropout=scfg["dropout"],
        lr=scfg["lr"],
        weight_decay=scfg["weight_decay"],
        n_epochs=scfg["n_epochs"],
        patience=scfg["patience"],
        batch_size=scfg["batch_size"],
    )

    model.fit(train_df, smiles_col=smiles_col, pec50_col=pec50_col)
    model.save(str(out / "siamese_model.pkl"))

    # Alpha weights (interpretable physics coefficients)
    alpha = model.get_alpha_weights()
    with open(out / "alpha_weights.json", "w") as f:
        json.dump(alpha, f, indent=2)
    logger.info(f"Siamese alpha weights: {alpha}")

    # OOF predictions (cross-fold)
    n_folds = 5
    oof_preds = np.full(len(train_df), np.nan)
    for fold in range(n_folds):
        train_idx, val_idx = get_train_val_indices(train_df, fold)
        fold_train = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val = train_df.iloc[val_idx].reset_index(drop=True)

        fold_model = SiameseDeltaModel(**{k: scfg[k] for k in scfg if k in SiameseDeltaModel.__init__.__code__.co_varnames})
        fold_model.fit(fold_train, smiles_col=smiles_col, pec50_col=pec50_col)
        preds = fold_model.predict(
            fold_val[smiles_col].tolist(), fold_train,
            smiles_col=smiles_col, pec50_col=pec50_col,
            n_neighbors=scfg["n_neighbors"],
        )
        oof_preds[val_idx] = preds

    test_preds = model.predict(
        test_df["smiles"].tolist() if "smiles" in test_df.columns else [],
        train_df, smiles_col=smiles_col, pec50_col=pec50_col,
        n_neighbors=scfg["n_neighbors"],
    )

    np.save(out / "siamese_oof.npy", oof_preds)
    np.save(out / "siamese_test_preds.npy", test_preds)

    valid_mask = ~np.isnan(oof_preds)
    y_true = train_df[pec50_col].values
    metrics = evaluate_oof(y_true[valid_mask], oof_preds[valid_mask], train_df["fold"].values[valid_mask])
    logger.info(f"Siamese OOF MAE={metrics['mae']:.4f}, RAE={metrics['rae']:.4f}")
    return oof_preds, test_preds


def main():
    with open("configs/mmp_delta.yaml") as f:
        config = yaml.safe_load(f)

    train_df = pd.read_parquet("data/splits/butina_folds.parquet")
    test_df = pd.read_parquet("data/raw/openadmet_test.parquet")
    out = Path("models/mmp_delta")
    out.mkdir(parents=True, exist_ok=True)

    smiles_col = "smiles_std" if "smiles_std" in train_df.columns else "smiles"
    pec50_col = "pec50_median" if "pec50_median" in train_df.columns else "pec50"

    # Train both models
    base_oof, base_test = train_baseline(train_df, test_df, config, smiles_col, pec50_col, out)
    siam_oof, siam_test = train_siamese(train_df, test_df, config, smiles_col, pec50_col, out)

    # Blend the two delta models per config
    w_base = config["ensemble_weight"]["baseline"]
    w_siam = config["ensemble_weight"]["siamese"]
    ensemble_oof = w_base * base_oof + w_siam * siam_oof
    ensemble_test = w_base * base_test + w_siam * siam_test

    np.save(out / "ensemble_oof.npy", ensemble_oof)
    np.save(out / "ensemble_test_preds.npy", ensemble_test)

    # Format submission
    compound_id_col = "compound_id" if "compound_id" in test_df.columns else test_df.columns[0]
    format_submission(
        test_df=test_df,
        predictions=ensemble_test,
        compound_id_col=compound_id_col,
        output_path="submissions/phase1/mmp_delta_ensemble.csv",
    )

    logger.info("MMP-delta ensemble complete. Next: python scripts/07_train_chemprop.py")


if __name__ == "__main__":
    main()
