"""
Phase 2 ensemble: SLSQP blend of available Phase 2 models.

Phase 2 models train on primary Octant-assay sources only
(4,827 compounds) or their active subset (4,718 non-NaN pEC50).
The OOF arrays have different lengths than the full training parquet (11,487).
This script handles alignment and runs the ensemble.

Usage:
  python scripts/39_ensemble_phase2.py
  python scripts/39_ensemble_phase2.py --submit    # also validates + copies to submissions/phase2/
"""

import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from openadmet.cv.oof import evaluate_oof
from openadmet.ensemble.blend import mean_blend, optimize_blend_weights, dynamic_recal
from openadmet.utils.submission import format_submission, validate_submission_format


PRIMARY_SOURCES = {"openadmet", "analog_set1", "htchem", "htchem_semi_pure"}


def load_primary_train(folds_path: str = "data/splits/butina_folds.parquet") -> pd.DataFrame:
    """Load training data filtered to primary Octant-assay sources."""
    df = pd.read_parquet(folds_path)
    if "source" in df.columns:
        df = df[df["source"].isin(PRIMARY_SOURCES)].reset_index(drop=True)
    logger.info(f"Primary sources: {len(df)} compounds")
    return df


def align_oof(oof: np.ndarray, train_df: pd.DataFrame, pec50_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align OOF to primary training DataFrame, returning (y_true, oof_aligned, folds).

    Handles two cases:
    - OOF length == len(train_df): direct 1:1 match (all primary compounds)
    - OOF length == non-NaN count: active-only OOF (catboost style)
    """
    y = train_df[pec50_col].values
    folds = train_df["fold"].values
    active_mask = ~np.isnan(y)

    if len(oof) == len(train_df):
        return y, oof, folds
    elif len(oof) == active_mask.sum():
        # Active-only OOF — expand to full primary length, NaN for censored
        oof_full = np.full(len(train_df), np.nan, dtype=np.float32)
        oof_full[active_mask] = oof
        return y, oof_full, folds
    else:
        raise ValueError(
            f"OOF length {len(oof)} doesn't match primary ({len(train_df)}) "
            f"or active ({active_mask.sum()}) count"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true", help="Validate and copy to submissions/phase2/")
    args = parser.parse_args()

    with open("configs/ensemble.yaml") as f:
        config = yaml.safe_load(f)

    train_df = load_primary_train()
    test_df = pd.read_parquet("data/raw/openadmet_test.parquet")
    pec50_col = "pec50_median" if "pec50_median" in train_df.columns else "pec50"

    out = Path("models/ensemble_phase2")
    out.mkdir(parents=True, exist_ok=True)

    # Model registry — add new Phase 2 models here
    registry = [
        ("lgbm",           "models/lgbm"),
        ("lgbm_optimal",   "models/lgbm_optimal"),
        ("catboost",       "models/catboost"),
        ("chemprop_4task", "models/chemprop_4task"),
        ("tabpfn",         "models/tabpfn"),
        ("tabicl",         "models/tabicl"),
        ("unimol2",        "models/unimol2"),
        ("unimol2_s3",     "models/unimol2_s3"),
        ("unimol2_s4",     "models/unimol2_s4"),
        # Phase 1 models that still have valid test predictions:
        ("lgbm_docking",   "models/lgbm_docking"),
    ]

    oof_preds: dict[str, np.ndarray] = {}
    test_preds: dict[str, np.ndarray] = {}
    y_aligned: np.ndarray | None = None
    folds_aligned: np.ndarray | None = None

    for name, model_dir in registry:
        oof_path = Path(model_dir) / "oof_predictions.npy"
        test_path = Path(model_dir) / "test_predictions.npy"

        if not oof_path.exists() or not test_path.exists():
            logger.debug(f"  {name}: missing — skipping")
            continue

        oof_raw = np.load(oof_path)
        test_raw = np.load(test_path)

        try:
            y, oof_aligned, folds = align_oof(oof_raw, train_df, pec50_col)
        except ValueError as e:
            logger.warning(f"  {name}: {e} — skipping")
            continue

        if y_aligned is None:
            y_aligned = y
            folds_aligned = folds

        if np.isnan(oof_aligned).mean() > 0.5:
            logger.warning(f"  {name}: >50% NaN OOF — skipping (likely stale Phase 1 OOF)")
            continue

        # Skip models whose test predictions are degenerate (all NaN or zero std)
        if np.isnan(test_raw).mean() > 0.5:
            logger.warning(f"  {name}: >50% NaN test predictions — skipping")
            continue
        if test_raw.std() < 0.01:
            logger.warning(f"  {name}: test std={test_raw.std():.4f} (constant/collapsed) — skipping")
            continue

        oof_preds[name] = np.where(np.isnan(oof_aligned), np.nanmean(oof_aligned), oof_aligned)
        test_preds[name] = test_raw
        logger.info(f"  {name}: OOF n={len(oof_raw)}, test n={len(test_raw)}")

    if not oof_preds:
        logger.error("No models loaded — train at least one Phase 2 model first.")
        sys.exit(1)

    logger.info(f"\nLoaded {len(oof_preds)} models: {list(oof_preds.keys())}")

    # SLSQP blend
    optimal_weights = optimize_blend_weights(oof_preds, y_aligned, metric="mae")
    with open(out / "blend_weights.json", "w") as f:
        json.dump(optimal_weights, f, indent=2)
    logger.info(f"SLSQP weights: {optimal_weights}")

    oof_blend = mean_blend(oof_preds, weights=optimal_weights)
    test_blend = mean_blend(test_preds, weights=optimal_weights)
    np.save(out / "oof_blend.npy", oof_blend)

    blend_metrics = evaluate_oof(y_aligned, oof_blend, folds_aligned)
    logger.info(
        f"Blend OOF: MAE={blend_metrics['mae']:.4f}, "
        f"RAE_test={blend_metrics['rae_test']:.4f}"
    )

    # Variance recalibration
    target_std = float(config.get("recal", {}).get("target_std", 0.70))
    test_cal, factor = dynamic_recal(test_blend, target_std=target_std)
    logger.info(f"Recalibration factor: {factor:.2f}× (target_std={target_std})")

    # Save submission
    Path("submissions/phase2").mkdir(parents=True, exist_ok=True)
    sub_path = "submissions/phase2/ensemble_raw.csv"
    compound_id_col = "compound_id" if "compound_id" in test_df.columns else test_df.columns[0]

    from openadmet.utils.submission import VALID_PEC50_RANGE
    lo, hi = VALID_PEC50_RANGE
    test_cal = np.clip(test_cal, lo, hi)

    format_submission(test_df, test_cal, compound_id_col=compound_id_col, output_path=sub_path)
    logger.info(f"\nSubmission saved: {sub_path}")
    logger.info(f"Test stats: mean={test_cal.mean():.3f}, std={test_cal.std():.3f}")

    if args.submit:
        result = validate_submission_format(sub_path, test_df[compound_id_col].astype(str).tolist())
        if result["valid"]:
            logger.info("✅ Submission validated — ready to upload")
        else:
            logger.error(f"Validation failed: {result['errors']}")

    logger.info(f"\nNext: python scripts/12_phase2_calibrate.py \\")
    logger.info(f"  --set1_labels data/raw/analog_set1_labels.csv \\")
    logger.info(f"  --set2_preds {sub_path}")


if __name__ == "__main__":
    main()
