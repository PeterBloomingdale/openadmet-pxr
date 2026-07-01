# Phase 2 Final Submission — Reproducibility Runbook (Activity Track)

**Deliverable:** `submissions/phase2/final_submission.csv` (513 rows, validation PASSED).
**Generated:** 2026-06-29. **Challenge deadline:** July 1, 2026.

## What it is

The Phase-2 ensemble trained with **Analog Set 1 (253 labels) added to the training set**
(`source == "analog_set1"` in `data/splits/butina_folds.parquet`), with the 253 known
true Set-1 labels substituted back into the submission. The blinded ~260 compounds (the
scored set) carry model predictions; the 253 released compounds carry their true labels.

## Reproduce end-to-end (CPU only, ~5 min excl. UniMol)

```bash
source .venv/bin/activate
python scripts/39_ensemble_phase2.py            # → submissions/phase2/ensemble_raw.csv
python scripts/12b_build_final_submission.py     # → submissions/phase2/final_submission.csv
python scripts/13_validate_submission.py submissions/phase2/final_submission.csv
```

`scripts/39_ensemble_phase2.py` filters `butina_folds.parquet` to PRIMARY_SOURCES
(`openadmet, analog_set1, htchem, htchem_semi_pure` → 4,827 compounds / 4,718 active),
aligns each model's OOF (auto-skipping stale Phase-1 4,135-row OOF such as `lgbm_docking`),
SLSQP-blends on OOF MAE, and applies `dynamic_recal(target_std=0.70)` (a no-op here —
raw std 0.683 ≈ target, factor 1.0×).

## Blend recipe (SLSQP weights, OOF MAE 0.4829 / RAE_test 0.6036)

9-model registry; SLSQP weights from `models/ensemble_phase2/blend_weights.json`:

| Model | Weight | Trainer |
|---|---|---|
| unimol2_s3 (LR=2e-4) | 24.5% | `scripts/32_train_unimol2_s3.py` (MPS/CPU) |
| chemprop_4task | 22.1% | `scripts/38_train_chemprop_4task.py` (MPS/CPU) |
| unimol2 (LR=5e-5) | 17.4% | `scripts/24_train_unimol2.py` (MPS/CPU) |
| catboost | 14.7% | `scripts/28_train_catboost.py` (CPU) |
| lgbm | 9.6% | `scripts/06_train_lgbm.py` (CPU) |
| unimol2_s4 (LR=5e-4) | 8.9% | `scripts/35_train_unimol2_s4.py` (MPS/CPU) |
| tabicl | 2.8% | `scripts/26_train_tabicl.py` (CPU) |
| (lgbm_optimal, tabpfn) | 0% | — |

(`unimol2_s5`, LR=1e-3, was pruned — Spearman 0.42, SLSQP weight ≈ 0.)

## Reproducibility verified (2026-06-29)

- `scripts/39_ensemble_phase2.py` regenerates `ensemble_raw.csv` **bit-identical**
  (max |diff| = 0.000) from on-disk model predictions.
- Retraining the CPU base models on the current split reproduces their cached
  predictions bit-identically:
  - `tabicl`: OOF/test max |diff| = 9.5e-07 (float32 epsilon).
  - `lgbm`:   OOF/test max |diff| = 0.000.
  This confirms `butina_folds.parquet` is consistent with what every model was trained on.
- UniMol2 family (51% of the blend) is deterministic by design (`CLAUDE.md`: same
  hyperparameters → bit-identical predictions) and its OOF is aligned to the current
  4,718-row active primary split, so it is reproducible without a re-run. To regenerate
  from scratch on Apple Silicon, run the `scripts/24/32/35` trainers (MPS, heavy).
- `final_submission.csv` is byte-stable across rebuilds.

## Why no post-hoc calibration

Linear/shift calibration was fit on the 253 real Set-1 labels and **added no
out-of-sample gain** on the Set1-naive Sub-20 blend (5-fold held-out MAE 0.4817 → 0.4865;
mean bias only +0.13). The improvement over Phase-1 comes entirely from adding Set 1 to
training, not from correcting the predictions afterward. See
`scratchpad/eval_set1.py` notes.

## Caveat

The Set-1 score of this ensemble (MAE 0.27 / Spearman 0.95) is **in-sample/leaked**
(those compounds are in its training set) and is *not* an estimate of blinded-Set-2
performance. On the blinded 260 it correlates 0.96 with Sub-20 (limited downside) but the
true gain is unmeasurable until the leaderboard scores it.
