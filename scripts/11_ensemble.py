"""
Blend all base models into a final ensemble prediction.

Base model registry (add new models by extending the list in main()):
  - lgbm         models/lgbm/         2D FP + Mordred + ECFP6 + FCFP4 (GBDT)
  - chemprop     models/chemprop/     graph MPNN, scratch-trained
  - chemprop_hts models/chemprop_hts/ graph MPNN, HTS pretrained (07b + 07 --pretrain-hts)
  - tabpfn       models/tabpfn/       CheMeleon + RDKit-2D → PCA → in-context transformer
  - knn          models/knn/          Tanimoto ECFP4 k-nearest-neighbour
  - unimol_lgbm  models/unimol_lgbm/  Uni-Mol 3D embeddings → GBDT

Prerequisites (in order):
  - scripts/06_train_lgbm.py
  - scripts/07_train_chemprop.py (and optionally --pretrain-hts variant)
  - scripts/10_train_tabpfn.py  (with CheMeleon embeddings preferred)
  - scripts/06d_train_knn.py
  - scripts/18_train_unimol_lgbm.py  (requires 17_extract_unimol_embeddings.py first)

Outputs:
  - submissions/phase1/ensemble_blend.csv    (SLSQP-optimized weighted blend)
  - submissions/phase1/ensemble_stacked.csv  (Ridge stacker — cross-validate before using)
  - submissions/phase1/ensemble_recal.csv     (canonical upload — **only if ≥2 base models**)
  - submissions/phase1/ensemble_lgbm_recovery.csv  (LGBM-only diagnostic; stale canonical removed)
  - submissions/phase1/lgbm_calibrated_ready.csv   (06_train_lgbm recal + clip, diagnostic)
  - models/ensemble/oof_blend.npy
  - models/ensemble/blend_weights.json
  - models/ensemble/ridge_stacker.pkl
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import pickle

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.isotonic import IsotonicRegression

from openadmet.ensemble.blend import mean_blend, optimize_blend_weights, dynamic_recal
from openadmet.ensemble.stack import (
    honest_oof_elasticnet,
    honest_oof_nnls,
    honest_oof_stack,
    predict_elasticnet_stack,
    predict_nnls_stack,
    predict_stack,
    train_elasticnet_stacker,
    train_nnls_stacker,
    train_ridge_stacker,
)
from openadmet.cv.oof import evaluate_oof
from openadmet.utils.submission import (
    PHASE1_CANONICAL_SUBMISSION,
    VALID_PEC50_RANGE,
    format_submission,
    validate_submission_format,
)


def load_model_outputs(model_dir: str, oof_file: str, test_file: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Loads OOF and test predictions, returns (oof, test) or (None, None) if missing."""
    oof_path = Path(model_dir) / oof_file
    test_path = Path(model_dir) / test_file
    if not oof_path.exists() or not test_path.exists():
        logger.warning(f"Missing predictions in {model_dir} — skipping")
        return None, None
    return np.load(oof_path), np.load(test_path)


def main():
    with open("configs/ensemble.yaml") as f:
        config = yaml.safe_load(f)

    train_df = pd.read_parquet("data/splits/butina_folds.parquet")
    test_df = pd.read_parquet("data/raw/openadmet_test.parquet")

    pec50_col = "pec50_median" if "pec50_median" in train_df.columns else "pec50"
    y_true = train_df[pec50_col].values

    out = Path("models/ensemble")
    out.mkdir(parents=True, exist_ok=True)

    # Load all available OOF + test predictions
    oof_preds: dict[str, np.ndarray] = {}
    test_preds: dict[str, np.ndarray] = {}

    expected_n = len(train_df)
    for name, model_dir, oof_f, test_f in [
        ("lgbm",           "models/lgbm",           "oof_predictions.npy", "test_predictions.npy"),
        ("lgbm_chemeleon", "models/lgbm_chemeleon", "oof_predictions.npy", "test_predictions.npy"),
        ("chemprop",       "models/chemprop",       "oof_predictions.npy", "test_predictions.npy"),
        ("chemprop_hts",   "models/chemprop_hts",   "oof_predictions.npy", "test_predictions.npy"),
        ("tabpfn",         "models/tabpfn",         "oof_predictions.npy", "test_predictions.npy"),
        ("knn",            "models/knn",            "oof_predictions.npy", "test_predictions.npy"),
        ("unimol_lgbm",    "models/unimol_lgbm",    "oof_predictions.npy", "test_predictions.npy"),
        # Binary CheMeleon: LGBMClassifier(pEC50>=5) → linear-calibrated pEC50. Orthogonal signal
        # to continuous regressors — optimized for activity classification, not pEC50 magnitude.
        ("chemeleon_binary", "models/chemeleon_binary", "oof_predictions.npy", "test_predictions.npy"),
        # lgbm_docking: docking (6 smina features) + classical (8262 2D) → LGBM. Physics-based prior
        # on PXR pocket filling. OOF MAE=0.499, Spearman=0.744 (Variant B, docking+classical wins).
        ("lgbm_docking",    "models/lgbm_docking",    "oof_predictions.npy", "test_predictions.npy"),
        # unimol2: UniMol v1 (84M params) fine-tuned end-to-end on PXR pEC50 (GPU). Task-specific
        # 3D transformer — unlike frozen embeddings, learns which 3D features matter for PXR LBD.
        # Rank-18 team used this approach (RAE=0.561 vs our 0.634). Run scripts/24_train_unimol2.py first.
        ("unimol2",         "models/unimol2",         "oof_predictions.npy", "test_predictions.npy"),
        # lgbm_optimal: LGB on CheMeleon + ECFP4(morgan_r2) + rdkit2d only — Jeremy's best LGB combo
        # (RMSE 0.531 in his study). Cleaner than lgbm_chemeleon which includes noisy Mordred/Avalon/ErG.
        # Run scripts/27_train_lgbm_optimal.py first.
        ("lgbm_optimal",    "models/lgbm_optimal",    "oof_predictions.npy", "test_predictions.npy"),
        # tabicl: TabICL (Tabular In-Context Learning v2, pretrained transformer) on
        # CheMeleon+ECFP4+rdkit2d → PCA-256. Jeremy's dominant model (RMSE 0.495 with HTS CheMeleon).
        # Run scripts/26_train_tabicl.py first.
        ("tabicl",          "models/tabicl",          "oof_predictions.npy", "test_predictions.npy"),
        # tabicl_3d: TabICL on CheMeleon + frozen UniMol(512d) + ECFP4 + rdkit2d → PCA-256.
        # Extends tabicl with 3D frozen embeddings for UniMol's conformer-derived structural signal.
        # Run scripts/29_train_tabicl_3d.py first.
        ("tabicl_3d",       "models/tabicl_3d",       "oof_predictions.npy", "test_predictions.npy"),
        # catboost: CatBoost ordered-boosting regressor on same CheMeleon+ECFP4+rdkit2d features as
        # lgbm_optimal. Adds GBDT diversity via different tree-growth strategy (symmetric oblivious trees).
        # Run scripts/28_train_catboost.py first.
        ("catboost",        "models/catboost",        "oof_predictions.npy", "test_predictions.npy"),
        # unimol2_s2: Second independent UniMol v1 fine-tuning run (different random init).
        # Averaging two stochastic fine-tuning runs reduces variance on our dominant model family.
        # Run scripts/25_train_unimol2_s2.py first.
        ("unimol2_s2",      "models/unimol2_s2",      "oof_predictions.npy", "test_predictions.npy"),
        # tabicl_chemprop: TabICL on CheMeleon + PXR-fine-tuned Chemprop(300d) + ECFP4 + rdkit2d → PCA-256.
        # Adds task-specific Chemprop embeddings to tabicl feature set for PXR-targeted representations.
        # Run scripts/31_train_tabicl_chemprop.py first (requires scripts/30_extract_... first).
        ("tabicl_chemprop", "models/tabicl_chemprop", "oof_predictions.npy", "test_predictions.npy"),
        # unimol2_s3: UniMol v1 fine-tuning with LR=2e-4 (4× higher than standard 5e-5).
        # Different LR gives different training dynamics and genuinely non-identical predictions.
        # Note: unimol_tools is deterministic — only changing a hyperparameter creates diversity.
        # Run scripts/32_train_unimol2_s3.py first.
        ("unimol2_s3", "models/unimol2_s3", "oof_predictions.npy", "test_predictions.npy"),
        # unimol2_s4: UniMol v1 fine-tuning with LR=5e-4 (10× higher than s1, 2.5× higher than s3).
        # Explores a more aggressive fine-tuning region; gated on pearsonr(s4,s1)<0.95 post-training.
        # Run scripts/35_train_unimol2_s4.py first.
        ("unimol2_s4", "models/unimol2_s4", "oof_predictions.npy", "test_predictions.npy"),
        # unimol2_s5: UniMol v1 fine-tuning with LR=1e-3, batch_size=4 (noisier gradients).
        # Two-lever diversity: higher LR + smaller batch → different optimization path from s1-s4.
        # Run scripts/36_train_unimol2_s5.py first.
        ("unimol2_s5", "models/unimol2_s5", "oof_predictions.npy", "test_predictions.npy"),
        # chemprop_4task: CheMeleon-pretrained init + 4-task multitask (pEC50 + counter + single-conc).
        # Phase 2 addition — discoverybytes found 4-task gives +0.070 RAE vs 2-task, and CheMeleon
        # init is the single largest gain (RAE 0.62→0.59). Run scripts/38_train_chemprop_4task.py.
        ("chemprop_4task", "models/chemprop_4task", "oof_predictions.npy", "test_predictions.npy"),
        # tabicl_hts: pearsonr(tabicl_hts, tabicl) = 1.000 — PCA collapses HTS embeddings to same
        # latent space as CheMeleon alone. Zero diversity gain vs tabicl. Do not add.
        # ("tabicl_hts", "models/tabicl_hts", "oof_predictions.npy", "test_predictions.npy"),
        # RF and XGB omitted — r > 0.95 with LGBM OOF, same fingerprint features, add noise not signal
        # MMP-delta omitted — Spearman=-0.071 leaderboard; 90.8% test compounds have no close training neighbor
    ]:
        oof, test = load_model_outputs(model_dir, oof_f, test_f)
        if oof is None:
            continue
        if len(oof) != expected_n:
            logger.warning(
                f"Skipping {name}: OOF length {len(oof)} != training rows {expected_n} "
                f"(re-train on current `data/splits/butina_folds.parquet` — e.g. "
                f"`07_train_chemprop.py` for chemprop, `10_train_tabpfn.py` for tabpfn)."
            )
            continue
        # Handle NaN in OOF (e.g., from failed folds)
        valid = ~np.isnan(oof)
        if valid.mean() < 0.9:
            logger.warning(f"{name}: {(~valid).sum()} NaN OOF predictions — model may have failed folds")
        oof_preds[name] = np.where(np.isnan(oof), np.nanmean(oof), oof)
        test_preds[name] = test
        logger.info(f"Loaded {name}: OOF mean={oof_preds[name].mean():.3f}")

    if not oof_preds:
        logger.error("No model predictions found! Train at least one model first.")
        sys.exit(1)

    lengths = {name: len(arr) for name, arr in oof_preds.items()}
    if len(set(lengths.values())) > 1:
        logger.error(f"Internal error: mismatched lengths after filter: {lengths}")
        sys.exit(1)

    logger.info(f"\nAvailable models: {list(oof_preds.keys())}")
    # Leaderboard slot: prefer ≥2 diverse towers (e.g. LGBM + Chemprop). Single-model
    # output is kept for diagnostics only — avoids overwriting the canonical CSV with
    # an LGBM-only file when skipping a weak submit (see README "Next submission").
    ready_for_canonical_hf = len(oof_preds) >= 2

    # 1. Optimize blend weights (SLSQP, MAE objective, non-negative summing to 1)
    logger.info("\n--- Optimizing blend weights ---")
    optimal_weights = optimize_blend_weights(oof_preds, y_true, metric="mae")

    with open(out / "blend_weights.json", "w") as f:
        json.dump(optimal_weights, f, indent=2)
    logger.info(f"Optimized weights saved to models/ensemble/blend_weights.json")

    # OOF blend
    oof_blend = mean_blend(oof_preds, weights=optimal_weights)
    test_blend = mean_blend(test_preds, weights=optimal_weights)
    np.save(out / "oof_blend.npy", oof_blend)

    # Evaluate blend
    blend_metrics = evaluate_oof(y_true, oof_blend, train_df["fold"].values)
    logger.info(f"Blend OOF: MAE={blend_metrics['mae']:.4f}, RAE={blend_metrics['rae']:.4f}, R²={blend_metrics['r2']:.4f}")

    # 2. Ridge stacking with honest leave-one-fold-out meta-CV.
    #    Previous behaviour: RidgeCV.fit(all_OOF, y_true) then predict on the
    #    SAME rows — that is a training-set fit and inflated the stacker score.
    if len(oof_preds) >= 2:
        logger.info("\n--- Training Ridge meta-learner (honest leave-one-fold-out OOF) ---")
        honest_stack_oof = honest_oof_stack(
            oof_preds,
            y_true,
            train_df["fold"].values,
            alphas=config["stack"]["alphas"],
        )
        stack_metrics = evaluate_oof(y_true, honest_stack_oof, train_df["fold"].values)
        logger.info(
            f"Honest Ridge stacker OOF: MAE={stack_metrics['mae']:.4f}, "
            f"RAE={stack_metrics['rae']:.4f}"
        )

        ridge_full = train_ridge_stacker(oof_preds, y_true, alphas=config["stack"]["alphas"])
        with open(out / "ridge_stacker.pkl", "wb") as f:
            pickle.dump(ridge_full, f)
        test_stacked = predict_stack(ridge_full, test_preds)

        use_elasticnet = bool(config.get("stack", {}).get("use_elasticnet", False))
        enet_metrics = None
        honest_enet_oof = None
        test_enet = None
        if use_elasticnet:
            l1_ratios = config["stack"].get("elasticnet_l1_ratios", [0.1, 0.5, 0.9])
            honest_enet_oof = honest_oof_elasticnet(
                oof_preds,
                y_true,
                train_df["fold"].values,
                l1_ratios=l1_ratios,
            )
            enet_metrics = evaluate_oof(y_true, honest_enet_oof, train_df["fold"].values)
            logger.info(
                f"Honest ElasticNet stacker OOF: MAE={enet_metrics['mae']:.4f}, "
                f"RAE={enet_metrics['rae']:.4f}"
            )
            enet_full = train_elasticnet_stacker(oof_preds, y_true, l1_ratios=l1_ratios)
            with open(out / "elasticnet_stacker.pkl", "wb") as f:
                pickle.dump(enet_full, f)
            test_enet = predict_elasticnet_stack(enet_full, test_preds)

        use_nnls = bool(config.get("stack", {}).get("use_nnls", False))
        nnls_metrics = None
        honest_nnls_oof = None
        test_nnls = None
        if use_nnls:
            logger.info("\n--- Training NNLS meta-learner (honest leave-one-fold-out OOF) ---")
            honest_nnls_oof = honest_oof_nnls(oof_preds, y_true, train_df["fold"].values)
            nnls_metrics = evaluate_oof(y_true, honest_nnls_oof, train_df["fold"].values)
            logger.info(
                f"Honest NNLS stacker OOF: MAE={nnls_metrics['mae']:.4f}, "
                f"RAE={nnls_metrics['rae']:.4f}"
            )
            nnls_full = train_nnls_stacker(oof_preds, y_true)
            test_nnls = predict_nnls_stack(nnls_full, test_preds)
            with open(out / "nnls_weights.json", "w") as f:
                json.dump(
                    {k: float(v) for k, v in zip(nnls_full["feature_names"], nnls_full["weights"])},
                    f, indent=2,
                )
            format_submission(
                test_df, test_nnls,
                compound_id_col="compound_id" if "compound_id" in test_df.columns else test_df.columns[0],
                output_path="submissions/phase1/ensemble_nnls.csv",
            )

        mae_blend = blend_metrics["mae"]
        stack_candidates: list[tuple[str, float, np.ndarray, np.ndarray]] = []
        if stack_metrics["mae"] < mae_blend - 0.005:
            stack_candidates.append(("ridge", stack_metrics["mae"], test_stacked, honest_stack_oof))
        if (
            use_elasticnet
            and enet_metrics is not None
            and enet_metrics["mae"] < mae_blend - 0.005
        ):
            stack_candidates.append(("elasticnet", enet_metrics["mae"], test_enet, honest_enet_oof))
        if (
            use_nnls
            and nnls_metrics is not None
            and nnls_metrics["mae"] < mae_blend - 0.005
        ):
            stack_candidates.append(("nnls", nnls_metrics["mae"], test_nnls, honest_nnls_oof))

        if stack_candidates:
            win_name, win_mae, best_test, oof_best = min(stack_candidates, key=lambda x: x[1])
            best_name = win_name
            logger.info(
                f"Picking {best_name} meta-learner (honest OOF MAE={win_mae:.4f} vs blend {mae_blend:.4f})"
            )
        else:
            logger.info(
                "No stacker (Ridge/ElasticNet/NNLS) beats blend by >0.005 MAE — using blended "
                "predictions (more robust to test-distribution shift)."
            )
            best_test = test_blend
            best_name = "blend"
            oof_best = oof_blend

        format_submission(
            test_df, test_stacked,
            compound_id_col="compound_id" if "compound_id" in test_df.columns else test_df.columns[0],
            output_path="submissions/phase1/ensemble_stacked.csv",
        )
        if use_elasticnet and test_enet is not None:
            format_submission(
                test_df,
                test_enet,
                compound_id_col="compound_id" if "compound_id" in test_df.columns else test_df.columns[0],
                output_path="submissions/phase1/ensemble_elasticnet.csv",
            )
    else:
        best_test = test_blend
        best_name = "blend"
        oof_best = oof_blend

    compound_id_col = "compound_id" if "compound_id" in test_df.columns else test_df.columns[0]
    sub_path = "submissions/phase1/ensemble_blend.csv"
    np.save(out / "test_blend_raw.npy", test_blend)
    format_submission(test_df, test_blend, compound_id_col=compound_id_col, output_path=sub_path)

    # Variance recalibration: expand `best_test` around its mean to match
    # target_std. Read from config (recal.target_std); falls back to training
    # pEC50 std if not set. Using training std (1.121) with a compressed blend
    # (raw std ~0.51) gives factor ~2.19× and clips ~46/513 predictions —
    # too aggressive for test compounds enriched in high-potency analogs.
    target_std = float(
        config.get("recal", {}).get("target_std", np.std(y_true))
    )
    best_test_cal, factor = dynamic_recal(best_test, target_std=target_std)
    oof_best_cal, _ = dynamic_recal(oof_best, target_std=target_std)
    recal_oof_metrics = evaluate_oof(y_true, oof_best_cal, train_df["fold"].values)
    logger.info(
        f"Best ({best_name}) raw: mean={best_test.mean():.3f}, std={best_test.std():.3f}"
    )
    logger.info(
        f"Best ({best_name}) cal ({factor:.2f}×): "
        f"mean={best_test_cal.mean():.3f}, std={best_test_cal.std():.3f}"
    )
    logger.info(
        f"Recalibrated OOF ({best_name}): MAE={recal_oof_metrics['mae']:.4f}, "
        f"RAE={recal_oof_metrics['rae']:.4f} — compare to uncorrected {best_name} OOF above"
    )
    lo, hi = VALID_PEC50_RANGE
    n_clip = int(((best_test_cal < lo) | (best_test_cal > hi)).sum())
    if n_clip:
        logger.warning(
            f"Clipping {n_clip} recalibrated test predictions to valid pEC50 range [{lo}, {hi}] "
            f"(spread scaling can push tails past assay-plausible bounds)."
        )
        best_test_cal = np.clip(best_test_cal, lo, hi)

    # Isotonic calibration (optional — enabled via configs/ensemble.yaml: recal.isotonic: true).
    # Fits a monotone function f(OOF_blend) → y_true on the 4,135-compound OOF predictions,
    # correcting systematic tail bias found in residual analysis (script 37):
    #   Q1 (pEC50 < 3.64): bias = +0.57 (over-prediction of inactives)
    #   Q4 (pEC50 > 5.13): bias = -0.47 (under-prediction of potents)
    # Since test analogs are from potent training hits, correcting Q4 under-prediction is high priority.
    # Applied AFTER variance recalibration so both spread and bias are corrected.
    if config.get("recal", {}).get("isotonic", False):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(oof_best_cal, y_true)
        best_test_cal = ir.predict(best_test_cal).astype(np.float32)
        oof_best_iso = ir.predict(oof_best_cal).astype(np.float32)
        iso_metrics = evaluate_oof(y_true, oof_best_iso, train_df["fold"].values)
        logger.info(
            f"Isotonic calibration applied: OOF MAE={iso_metrics['mae']:.4f} "
            f"(vs variance-recal OOF MAE={recal_oof_metrics['mae']:.4f})"
        )
        logger.info(
            f"Post-isotonic test: mean={best_test_cal.mean():.3f}, std={best_test_cal.std():.3f}"
        )
        best_test_cal = np.clip(best_test_cal, lo, hi)

    if ready_for_canonical_hf:
        recal_path = PHASE1_CANONICAL_SUBMISSION
    else:
        recal_path = "submissions/phase1/ensemble_lgbm_recovery.csv"

    format_submission(
        test_df, best_test_cal, compound_id_col=compound_id_col, output_path=recal_path
    )

    expected_ids = test_df[compound_id_col].astype(str).tolist()
    result = validate_submission_format(recal_path, expected_ids)
    if result["valid"]:
        logger.info(
            f"\nRecalibrated file ready: {recal_path} "
            f"(strategy: {best_name}, dynamic factor: {factor:.2f}×, "
            f"target_std: {target_std:.3f})"
        )
        if not ready_for_canonical_hf:
            stale = Path(PHASE1_CANONICAL_SUBMISSION)
            if stale.exists():
                stale.unlink()
                logger.warning(
                    f"Removed stale {PHASE1_CANONICAL_SUBMISSION} — single-model run is not "
                    f"the intended next leaderboard upload. Train Chemprop (+ optional TabPFN), "
                    f"re-run this script, then validate the regenerated canonical file."
                )
    else:
        logger.error(f"Submission validation failed: {result['errors']}")

    if ready_for_canonical_hf:
        logger.info(f"\nNext: python scripts/13_validate_submission.py {PHASE1_CANONICAL_SUBMISSION}")
    else:
        logger.info(
            f"\nSkipping canonical HF slot — only {len(oof_preds)} base model(s). "
            f"Next: GPU `07_train_chemprop.py` → `11_ensemble.py` → validate "
            f"{PHASE1_CANONICAL_SUBMISSION}. Recovery CSV (not default upload): {recal_path}"
        )

    # Alternate CSV from 06_train_lgbm variance recalibration (no ensemble dynamic_recal).
    lgbm_cal_path = Path("models/lgbm/test_predictions_calibrated.npy")
    if lgbm_cal_path.exists():
        lgbm_cal = np.clip(np.load(lgbm_cal_path), lo, hi)
        alt_path = "submissions/phase1/lgbm_calibrated_ready.csv"
        format_submission(test_df, lgbm_cal, compound_id_col=compound_id_col, output_path=alt_path)
        alt_result = validate_submission_format(alt_path, expected_ids)
        if alt_result["valid"]:
            logger.info(f"LGBM 06-calibrated + clip (diagnostic): {alt_path}")


if __name__ == "__main__":
    main()
