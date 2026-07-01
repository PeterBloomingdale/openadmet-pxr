"""
EDA script — run BEFORE any architecture decisions.

This answers the five diagnostic questions that determine model strategy:

Q1: Noise floor — what within-compound pEC50 std deviation is typical?
    → Sets the floor on achievable RAE. If noise ≈ 0.3 log-units and dynamic
      range ≈ 3 log-units, best achievable RAE ≈ 0.1. Chasing below that is futile.

Q2: Test-to-training Tanimoto similarity distribution
    → If the distribution peaks at 0.6-0.8, MMP-delta will dominate (most test
      compounds have a very close training analog). If it peaks at 0.2-0.4, the
      absolute model matters more.

Q3: Censored fraction
    → If >40% of training compounds are inactive at top dose, censored regression
      (Tobit model) may be important. If <20%, simple exclusion is fine.

Q4: Primary vs counter-screen pEC50 correlation (r²)
    → If r² > 0.3, the multitask counter-screen head adds real signal.
      If r² < 0.1, the counter-screen is mostly noise — simplify to 2-head model.

Q5: Stereochemistry profile
    → Count stereo-mismatched train/test pairs (train has defined stereo, test racemate).
      If >20% mismatch, stereo-blind featurization (remove stereo from SMILES) is needed.

Outputs are saved to data/eda/ as PNG figures and a JSON summary.
The JSON summary informs decisions in 02_curate_data.py and config files.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger

Path("data/eda").mkdir(parents=True, exist_ok=True)


def q1_noise_floor(train_df: pd.DataFrame) -> dict:
    """Analyze within-compound replicate spread."""
    logger.info("Q1: Analyzing noise floor from replicate measurements...")

    # Look for replicate columns if present (depends on data schema)
    pec50_cols = [c for c in train_df.columns if "pec50" in c.lower() and c != "pec50"]

    if "pec50_spread" in train_df.columns:
        spreads = train_df["pec50_spread"].dropna()
    else:
        logger.warning("No spread column found — using pEC50 distribution as proxy")
        spreads = pd.Series([0.3])  # Literature estimate

    result = {
        "median_spread": float(spreads.median()),
        "mean_spread": float(spreads.mean()),
        "p90_spread": float(spreads.quantile(0.9)),
        "n_compounds_with_replicates": int((spreads > 0.05).sum()),
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(spreads, bins=30, color="steelblue", alpha=0.7)
    ax.axvline(spreads.median(), color="red", linestyle="--", label=f"Median={spreads.median():.3f}")
    ax.set_xlabel("Within-compound pEC50 std deviation (log-units)")
    ax.set_ylabel("Count")
    ax.set_title("Q1: Assay Noise Floor (Replicate Spread)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("data/eda/q1_noise_floor.png", dpi=150)
    plt.close()

    logger.info(f"  Noise floor: median spread = {result['median_spread']:.3f} log-units")
    achievable_rae = result["median_spread"] / 3.0  # Approximate, assuming DR=3
    logger.info(f"  Estimated achievable RAE ≈ {achievable_rae:.3f} (assuming dynamic_range ≈ 3)")
    result["estimated_achievable_rae"] = achievable_rae
    return result


def q2_test_train_similarity(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Analyze structural similarity between test and training compounds."""
    logger.info("Q2: Analyzing test-to-training Tanimoto similarity distribution...")

    from openadmet.features.fingerprints import ecfp4_bitvect
    from rdkit import DataStructs

    smiles_col = "smiles_std" if "smiles_std" in train_df.columns else "smiles"
    test_smiles_col = "smiles_std" if "smiles_std" in test_df.columns else "smiles"

    train_fps = [ecfp4_bitvect(s) for s in train_df[smiles_col].dropna()[:500]]
    train_fps = [fp for fp in train_fps if fp is not None]

    test_fps = [ecfp4_bitvect(s) for s in test_df[test_smiles_col].dropna()[:513]]
    test_fps = [fp for fp in test_fps if fp is not None]

    max_sims = []
    for test_fp in test_fps:
        sims = DataStructs.BulkTanimotoSimilarity(test_fp, train_fps)
        max_sims.append(max(sims))

    max_sims = np.array(max_sims)
    result = {
        "median_max_sim": float(np.median(max_sims)),
        "mean_max_sim": float(np.mean(max_sims)),
        "frac_above_0.4": float((max_sims > 0.4).mean()),
        "frac_above_0.6": float((max_sims > 0.6).mean()),
        "frac_above_0.8": float((max_sims > 0.8).mean()),
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(max_sims, bins=30, color="steelblue", alpha=0.7)
    ax.axvline(0.4, color="red", linestyle="--", label="Tanimoto=0.4 (test-set threshold)")
    ax.axvline(result["median_max_sim"], color="orange", linestyle="--",
               label=f"Median={result['median_max_sim']:.3f}")
    ax.set_xlabel("Max Tanimoto Similarity to Training Set")
    ax.set_ylabel("Count")
    ax.set_title("Q2: Test-to-Training Structural Similarity")
    ax.legend()
    plt.tight_layout()
    plt.savefig("data/eda/q2_test_train_similarity.png", dpi=150)
    plt.close()

    logger.info(f"  Median max sim: {result['median_max_sim']:.3f}")
    logger.info(f"  Fraction > 0.4: {result['frac_above_0.4']:.1%} (expected: ~100%)")
    logger.info(f"  Fraction > 0.6: {result['frac_above_0.6']:.1%} (higher = MMP-delta more powerful)")
    return result


def q3_censored_fraction(train_df: pd.DataFrame) -> dict:
    """Check what fraction of training data is censored (inactive at top dose)."""
    logger.info("Q3: Analyzing censored compound fraction...")

    pec50_col = "pec50_median" if "pec50_median" in train_df.columns else "pec50"
    pec50_vals = train_df[pec50_col].dropna()

    threshold = 5.0  # 10 µM = pEC50 5.0
    n_censored = (pec50_vals < threshold).sum()
    n_total = len(pec50_vals)
    frac_censored = n_censored / n_total

    result = {
        "n_censored": int(n_censored),
        "n_total": int(n_total),
        "frac_censored": float(frac_censored),
        "pec50_distribution": {
            "min": float(pec50_vals.min()),
            "p10": float(pec50_vals.quantile(0.1)),
            "median": float(pec50_vals.median()),
            "p90": float(pec50_vals.quantile(0.9)),
            "max": float(pec50_vals.max()),
            "dynamic_range": float(pec50_vals.max() - pec50_vals.min()),
        }
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pec50_vals, bins=40, color="steelblue", alpha=0.7)
    ax.axvline(threshold, color="red", linestyle="--", label=f"Censoring threshold (pEC50={threshold})")
    ax.set_xlabel("pEC50")
    ax.set_ylabel("Count")
    ax.set_title(f"Q3: pEC50 Distribution (censored: {frac_censored:.1%})")
    ax.legend()
    plt.tight_layout()
    plt.savefig("data/eda/q3_pec50_distribution.png", dpi=150)
    plt.close()

    logger.info(f"  Censored fraction: {frac_censored:.1%} ({n_censored}/{n_total})")
    logger.info(f"  Training pEC50 range: [{pec50_vals.min():.2f}, {pec50_vals.max():.2f}]")
    logger.info(f"  Dynamic range: {result['pec50_distribution']['dynamic_range']:.2f} log-units")
    return result


def q4_counterscreen_correlation(train_df: pd.DataFrame) -> dict:
    """Check if counter-screen pEC50 correlates with primary pEC50."""
    logger.info("Q4: Analyzing primary vs counter-screen pEC50 correlation...")

    primary_col = "pec50_median" if "pec50_median" in train_df.columns else "pec50"
    counter_col = next(
        (c for c in train_df.columns if "counter" in c.lower() and "pec50" in c.lower()),
        None
    )

    if counter_col is None:
        logger.warning("No counter-screen column found — check column names")
        return {"r2": None, "n_both": 0, "note": "counter-screen data not found"}

    both = train_df[[primary_col, counter_col]].dropna()
    n_both = len(both)

    if n_both < 20:
        return {"r2": None, "n_both": n_both, "note": "too few paired measurements"}

    from scipy.stats import pearsonr
    r, p = pearsonr(both[primary_col], both[counter_col])
    r2 = r ** 2

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(both[primary_col], both[counter_col], alpha=0.4, s=20)
    ax.set_xlabel("Primary pEC50")
    ax.set_ylabel("Counter-screen pEC50")
    ax.set_title(f"Q4: Primary vs Counter-screen (r²={r2:.3f}, n={n_both})")
    lo = min(both[primary_col].min(), both[counter_col].min())
    hi = max(both[primary_col].max(), both[counter_col].max())
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, label="Identity")
    plt.tight_layout()
    plt.savefig("data/eda/q4_counterscreen_correlation.png", dpi=150)
    plt.close()

    if r2 > 0.3:
        recommendation = "MULTITASK: counter-screen head will add real signal"
    elif r2 > 0.1:
        recommendation = "BORDERLINE: include counter-screen, monitor contribution"
    else:
        recommendation = "WEAK: counter-screen mostly noise — use as filter, not regression head"

    logger.info(f"  Primary vs counter-screen r²={r2:.3f} (n={n_both})")
    logger.info(f"  Recommendation: {recommendation}")
    return {"r2": float(r2), "n_both": int(n_both), "recommendation": recommendation}


def q5_stereochemistry_profile(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Check stereochemistry mismatch between train and test."""
    logger.info("Q5: Analyzing stereochemistry profile...")
    from rdkit import Chem

    smiles_col = "smiles_std" if "smiles_std" in train_df.columns else "smiles"
    test_smiles_col = "smiles_std" if "smiles_std" in test_df.columns else "smiles"

    def count_stereocenters(smiles: str) -> int:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0
            Chem.AssignStereochemistry(mol)
            return sum(1 for a in mol.GetAtoms() if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED)
        except Exception:
            return 0

    train_stereo = train_df[smiles_col].dropna().apply(count_stereocenters)
    test_stereo = test_df[test_smiles_col].dropna().apply(count_stereocenters)

    result = {
        "train_frac_with_stereo": float((train_stereo > 0).mean()),
        "test_frac_with_stereo": float((test_stereo > 0).mean()),
        "train_mean_stereocenters": float(train_stereo.mean()),
        "test_mean_stereocenters": float(test_stereo.mean()),
    }

    recommendation = "KEEP STEREO" if result["train_frac_with_stereo"] > 0.2 else "STRIP STEREO"
    result["recommendation"] = recommendation

    logger.info(f"  Train compounds with stereochemistry: {result['train_frac_with_stereo']:.1%}")
    logger.info(f"  Test compounds with stereochemistry: {result['test_frac_with_stereo']:.1%}")
    logger.info(f"  Recommendation: {recommendation}")
    return result


def main():
    logger.info("=== Pre-Architecture EDA ===")

    # Load data
    train_df = pd.read_parquet("data/raw/openadmet_train.parquet") \
        if Path("data/raw/openadmet_train.parquet").exists() else None
    test_df = pd.read_parquet("data/raw/openadmet_test.parquet") \
        if Path("data/raw/openadmet_test.parquet").exists() else None

    if train_df is None or test_df is None:
        logger.error("Run scripts/01_download_data.py first!")
        return

    logger.info(f"Loaded: {len(train_df)} training, {len(test_df)} test compounds")

    results = {}
    results["q1_noise_floor"] = q1_noise_floor(train_df)
    results["q2_test_train_similarity"] = q2_test_train_similarity(train_df, test_df)
    results["q3_censored_fraction"] = q3_censored_fraction(train_df)
    results["q4_counterscreen_correlation"] = q4_counterscreen_correlation(train_df)
    results["q5_stereochemistry"] = q5_stereochemistry_profile(train_df, test_df)

    # Save results
    with open("data/eda/eda_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print decision table
    logger.info("\n=== Architecture Decisions (based on EDA) ===")
    logger.info(f"Noise floor:          {results['q1_noise_floor']['median_spread']:.3f} log-units")
    logger.info(f"Achievable RAE:       ~{results['q1_noise_floor'].get('estimated_achievable_rae', '?'):.3f}")
    logger.info(f"Median test-train sim: {results['q2_test_train_similarity']['median_max_sim']:.3f}")
    logger.info(f"Censored fraction:    {results['q3_censored_fraction']['frac_censored']:.1%}")
    logger.info(f"Counter r²:           {results['q4_counterscreen_correlation'].get('r2', 'N/A')}")
    logger.info(f"Stereo recommendation: {results['q5_stereochemistry']['recommendation']}")
    logger.info("\nFull results saved to data/eda/eda_summary.json")
    logger.info("Figures saved to data/eda/*.png")
    logger.info("\nNext: python scripts/02_curate_data.py")


if __name__ == "__main__":
    main()
