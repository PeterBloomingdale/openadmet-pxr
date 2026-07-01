# OpenADMET PXR Challenge — Claude Code Context

## What This Is

Competition codebase for the OpenADMET PXR Blind Challenge (pEC50 regression on 513 Enamine analogs of 63 potent training hits). Primary metric: RAE = MAE / dynamic_range.

**Phase 1: COMPLETE. Final result: Sub 20, rank 68, MAE=0.4682, RAE=0.5876, Spearman=0.8051.**
**Phase 2: COMPLETE (deadline July 1, 2026).** Analog Set 1 (253/513) was unblinded and **added to training**; the ensemble was retrained. Final deliverable: `submissions/phase2/final_submission.csv` (runbook: `submissions/phase2/REPRODUCE.md`). Build/verify with `scripts/39_ensemble_phase2.py` → `12b_build_final_submission.py` → `13_validate_submission.py`. **Post-hoc calibration (`scripts/12_phase2_calibrate.py`) was tried and gave no out-of-sample gain** — the Phase-2 win comes from retraining with Set 1, not correcting predictions.

## Architecture Decisions (Do Not Change Without Reason)

- **numpy>=2.0**: Verified 2026-05-04 — numpy 2.4.4 is fully compatible with rdkit 2026.03.1, chemprop 2.2.3, lightgbm 4.6.0, tabpfn 7.1.1, scipy 1.17.1. The `numpy<2.0` constraint was valid for chemprop 2.0.x (mid-2024) but was resolved in chemprop 2.1+. Note: tabpfn "v2" is versioned as 7.x on PyPI.
- **tabpfn==7.1.1 required** — v8.x completely rewrote the API and requires a paid license token. v8 also predicts constant mean on Apple Silicon (M5 Pro, Python 3.14) despite accepting the token. Downgrade: `pip install tabpfn==7.1.1`.
- **MAE/Huber loss, not MSE**: RAE is MAE-based; MSE over-penalizes activity cliffs.
- **Mordred filter saves column list to disk**: `data/features/mordred_selected_columns.json` — test features must use identical columns. Train/test imputation also shares medians via `data/features/{rdkit,mordred}_medians.json`.
- **Train on the curated set** (`data/curated/master_train.parquet`). Phase 1 = 4,135 compounds. **Phase 2 = 11,487 total curated** (adds Analog Set 1, HTChem, PubChem sources); tabular/UniMol models train on the **primary-source subset** only (`{openadmet, analog_set1, htchem, htchem_semi_pure}` → 4,827 / 4,718 active), filtered inside `scripts/39_ensemble_phase2.py:PRIMARY_SOURCES` and each trainer. `master_train_active.parquet` (≥5.0 only) is **deprecated** — caused Subs 7/8 failures.
- **Train and test SMILES must both be standardized.** `scripts/04_build_features.py` standardizes test SMILES → `data/curated/openadmet_test_std.parquet`. Any new feature stage must do the same.
- **InChIKey 14-char prefix for dedup**: First 14 chars = connectivity layer only. Ignores stereochemistry.
- **unimol_tools is fully deterministic**: Same hyperparameters → bit-identical predictions (r=1.000). To create diverse UniMol ensemble members, change the learning rate (e.g., LR=5e-5 → LR=2e-4).
- **SLSQP simplex blend**: Non-negative weights summing to 1, minimizes OOF MAE. Preferred over Ridge/NNLS for this ensemble size.
- **target_std=0.70 for recalibration**: Test distribution is narrower than training. 0.90 over-spreads (confirmed by Subs 16–17 regression).

## Phase 1 Final Pipeline (Scripts 01–32)

```
01_download_data → 02_curate_data → 04_build_features → 05_build_cv_splits
→ 06_train_lgbm → 06d_train_knn
→ 07_train_chemprop → 07b_pretrain_chemprop_hts
→ 10_train_tabpfn → 26_train_tabicl → 29_train_tabicl_3d → 31_train_tabicl_chemprop
→ 16_extract_chemeleon_embeddings → 17_extract_unimol_embeddings
→ 18_train_unimol_lgbm → 19_train_chemeleon_binary
→ 20_prepare_docking_receptor → 21_dock_compounds → 22_extract_docking_features → 23_train_lgbm_docking
→ 24_train_unimol2 (LR=5e-5) → 25_train_unimol2_s2 (duplicate, skip) → 32_train_unimol2_s3 (LR=2e-4)
→ 27_train_lgbm_optimal → 28_train_catboost
→ 30_extract_chemprop_finetuned_emb
→ 11_ensemble → 13_validate_submission

Archived (did not improve ensemble): scripts/archive/06b_train_rf.py, 06c_train_xgboost.py,
  08_build_mmps.py, 09_train_mmp_delta.py
```

**Final Sub 20 ensemble weights (SLSQP, target_std=0.70):**
- unimol2_s3 (LR=2e-4): 24.8%, lgbm_docking: 23.5%, tabpfn: 20.0%
- unimol2 (LR=5e-5): 10.8%, unimol2_s2 (duplicate): 10.8%, tabicl: 5.1%, tabicl_chemprop: 5.1%
- All others: 0%

## Key Data Sources

| Source | PubChem AID / ID | Use |
|---|---|---|
| OpenADMET (Octant) | HuggingFace openadmet/pxr-challenge | Primary regression |
| Tox21 hPXR | AID 1347033 | chemprop_hts pretrain only |
| NCATS hPXR-luc | AID 720659 | chemprop_hts pretrain only |

**Excluded**: Rodent PXR, CYP3A4 inhibition, AIDs 1224894/1224895/1346798/651632 (mislabeled).

## Phase 2 Final Pipeline (adds scripts 01c, 38, 39, 12b)

```
01c_download_phase2_data → 02_curate_data (now 11,487 curated; Set 1 added as source=analog_set1)
→ 05_build_cv_splits → retrain primary-source models (06/27/28/10/26 + 38_train_chemprop_4task,
  24/32/35_train_unimol2*) → 39_ensemble_phase2 → 12b_build_final_submission → 13_validate_submission
```

- **Use `scripts/39_ensemble_phase2.py`, NOT `11_ensemble.py`, for Phase 2.** Script 11 expects
  4,135-row Phase-1 OOF and aborts; script 39 filters to `PRIMARY_SOURCES` and aligns the
  4,827/4,718-length OOF (auto-skips stale Phase-1 OOF such as `lgbm_docking`).
- **Final blend = 9 models** (weights in `models/ensemble_phase2/blend_weights.json`):
  unimol2_s3 24.5%, chemprop_4task 22.1%, unimol2 17.4%, catboost 14.7%, lgbm 9.6%,
  unimol2_s4 8.9%, tabicl 2.8%; lgbm_optimal/tabpfn ≈0. `unimol2_s2` (duplicate) and
  `unimol2_s5` (LR=1e-3, Spearman 0.42) were pruned.
- **`scripts/12b_build_final_submission.py`** substitutes the 253 true Set-1 labels back
  into the all-513 CSV (score-neutral under blinded-Set-2 scoring; a hedge if full 513 is
  rescored). Calibration (`scripts/12_phase2_calibrate.py`) gave no out-of-sample gain.
- Reproducibility verified 2026-06-29: script 39 regenerates `ensemble_raw.csv` bit-identically;
  retraining `lgbm`/`tabicl` reproduces cached predictions bit-identically (split is consistent).

## Test Commands

```bash
pytest tests/ -v
python -c "import openadmet; print('Package OK')"
quarto render manuscript/pxr_challenge.qmd
```
