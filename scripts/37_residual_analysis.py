"""
OOF residual diagnostics — answers: where does the model error live?

Reviewer question: "Are residuals concentrated in the weakly-active analogs or
the high-potency tail? A gap that's uniform across the range implies encoder quality;
a gap concentrated in the tails implies calibration or a representation blind spot."

This script:
1. Loads OOF ensemble blend predictions and true pEC50 labels
2. Splits compounds by pEC50 quartile and reports per-quartile MAE/bias/std
3. Checks for ensemble disagreement structure (high-std compounds = uncertain)
4. Identifies whether calibration error is uniform or concentrated
5. Saves residuals_by_quartile.json for reference in the decision log

Run before deciding whether to implement adaptive calibration in 11_ensemble.py.
If MAE is uniform across quartiles → global target_std=0.70 is fine.
If MAE is concentrated in a quartile → adaptive calibration is justified.

Outputs:
  models/ensemble/residuals_by_quartile.json
  Printed summary table
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import pearsonr, spearmanr


def main() -> None:
    logger.info("=== OOF Residual Analysis ===")

    # Load truth
    train_df = pd.read_parquet("data/splits/butina_folds.parquet")
    pec50_col = "pec50_median" if "pec50_median" in train_df.columns else "pec50"
    y_true = train_df[pec50_col].values.astype(np.float32)
    folds = train_df["fold"].values

    # Load OOF blend
    oof_blend_path = Path("models/ensemble/oof_blend.npy")
    if not oof_blend_path.exists():
        logger.error("Missing models/ensemble/oof_blend.npy — run scripts/11_ensemble.py first")
        sys.exit(1)
    oof_blend = np.load(oof_blend_path)

    residuals = oof_blend - y_true  # positive = over-prediction

    logger.info(f"\nOverall OOF MAE = {np.mean(np.abs(residuals)):.4f}")
    logger.info(f"Mean bias       = {np.mean(residuals):+.4f}  (+ = over-predict, - = under-predict)")
    logger.info(f"Residual std    = {np.std(residuals):.4f}")
    logger.info(f"Spearman ρ      = {spearmanr(y_true, oof_blend).statistic:.4f}")

    # Quartile analysis
    q1, q2, q3 = np.percentile(y_true, [25, 50, 75])
    logger.info(f"\npEC50 quartiles: Q1={q1:.2f}, Q2={q2:.2f}, Q3={q3:.2f}")

    quartile_labels = ["Q1 (weakly active)", "Q2 (moderate)", "Q3 (active)", "Q4 (potent)"]
    quartile_masks = [
        y_true <= q1,
        (y_true > q1) & (y_true <= q2),
        (y_true > q2) & (y_true <= q3),
        y_true > q3,
    ]

    results = {}
    logger.info(
        f"\n{'Quartile':<25} {'n':>5} {'MAE':>7} {'Bias':>8} {'Std':>7} {'Spearman':>10}"
    )
    logger.info("-" * 65)
    for label, mask in zip(quartile_labels, quartile_masks):
        if mask.sum() == 0:
            continue
        res = residuals[mask]
        mae = float(np.mean(np.abs(res)))
        bias = float(np.mean(res))
        std = float(np.std(res))
        sp = float(spearmanr(y_true[mask], oof_blend[mask]).statistic) if mask.sum() > 5 else float("nan")
        logger.info(f"{label:<25} {mask.sum():>5} {mae:>7.4f} {bias:>+8.4f} {std:>7.4f} {sp:>10.4f}")
        results[label] = {"n": int(mask.sum()), "mae": mae, "bias": bias, "std": std, "spearman": sp}

    # Load individual model OOF predictions to compute ensemble disagreement
    model_dirs = [
        "models/unimol2_s3", "models/lgbm_docking", "models/tabpfn",
        "models/unimol2", "models/tabicl",
    ]
    model_oofs = []
    for d in model_dirs:
        p = Path(d) / "oof_predictions.npy"
        if p.exists():
            model_oofs.append(np.load(p))

    if len(model_oofs) >= 2:
        ensemble_disagreement = np.std(np.stack(model_oofs), axis=0)
        logger.info(f"\nEnsemble disagreement (std across {len(model_oofs)} top models):")
        logger.info(f"  Mean = {ensemble_disagreement.mean():.4f}")
        logger.info(f"  Max  = {ensemble_disagreement.max():.4f}")

        # Correlation between disagreement and absolute residual
        r_disag_err = pearsonr(ensemble_disagreement, np.abs(residuals)).statistic
        logger.info(f"  Pearson r(disagreement, |residual|) = {r_disag_err:.4f}")
        logger.info(
            f"  {'Calibration interpretation:':<30} "
            f"{'Adaptive recal worthwhile' if r_disag_err > 0.2 else 'Global recal sufficient'}"
        )
        results["disagreement_residual_correlation"] = float(r_disag_err)

    # Identify most under- and over-predicted compounds
    worst_under = np.argsort(residuals)[:5]    # most under-predicted (too low)
    worst_over = np.argsort(residuals)[-5:][::-1]  # most over-predicted (too high)

    logger.info("\nTop-5 most under-predicted (true high, pred low):")
    for i in worst_under:
        logger.info(f"  idx={i}, pEC50={y_true[i]:.3f}, pred={oof_blend[i]:.3f}, resid={residuals[i]:+.3f}")

    logger.info("Top-5 most over-predicted (true low, pred high):")
    for i in worst_over:
        logger.info(f"  idx={i}, pEC50={y_true[i]:.3f}, pred={oof_blend[i]:.3f}, resid={residuals[i]:+.3f}")

    # Calibration diagnostics: are we systematically wrong at the tails?
    logger.info("\nCalibration check: predicted mean by true quartile")
    for label, mask in zip(quartile_labels, quartile_masks):
        if mask.sum() == 0:
            continue
        logger.info(
            f"  {label:<25}: true_mean={y_true[mask].mean():.3f}, "
            f"pred_mean={oof_blend[mask].mean():.3f}"
        )

    out_path = Path("models/ensemble/residuals_by_quartile.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved: {out_path}")
    logger.info(
        "\nNext: Review results above to decide on adaptive calibration in 11_ensemble.py.\n"
        "If MAE is concentrated in one quartile (>2× others), implement adaptive calibration.\n"
        "If disagreement-residual correlation > 0.20, per-compound scaling is justified."
    )


if __name__ == "__main__":
    main()
