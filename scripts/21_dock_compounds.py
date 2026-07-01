"""
Dock all PXR training and test compounds against the prepared 2QD9 receptor.

Uses smina (CPU, Windows-compatible) or gnina (GPU, Linux/WSL2) to dock all
4135 training + 513 test compounds. Extracts top-pose docking scores.

Why docking for PXR?
PXR's large hydrophobic LBD (~1150-1600 Å³) makes shape complementarity the
dominant driver of binding. A compound that fits the pocket volume gains several
kcal/mol of hydrophobic burial; one that doesn't fit loses it entirely. This
shape-fitness signal is invisible to 2D fingerprints and only partially captured
by 3D embeddings (Uni-Mol) trained on single-molecule conformers without receptor
context. Explicit docking with a known crystal structure directly measures pocket
fit, giving the ensemble a qualitatively different signal.

Engine selection:
  gnina: deep-learning scoring (CNN + Vina), GPU-accelerated, best accuracy.
         Requires Linux or WSL2. Install: conda install -c conda-forge gnina
  smina: Vina fork with improved scoring, CPU-only, Windows-compatible.
         Install: conda install -c conda-forge smina

Runtime estimates (4648 compounds, exhaustiveness=8, 5 poses):
  gnina (GTX 1080): ~40-60 min
  smina (CPU, 8 cores): ~6-10 hours

=== INSTALLATION (Windows without conda) ===

Option A — Install WSL2 + gnina (recommended, free, GPU-accelerated):
  1. Run as Administrator in PowerShell: wsl --install
  2. Reboot
  3. Open Ubuntu from Start Menu, complete Ubuntu setup
  4. In Ubuntu: curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh | bash
  5. In Ubuntu: conda install -c conda-forge gnina
  6. Run docking from Windows: wsl gnina --receptor ... (or just run this script — it tries wsl gnina)

Option B — Install Miniconda for Windows then smina:
  1. Download Miniconda from anaconda.com/download
  2. In Miniconda Prompt: conda install -c conda-forge smina
  3. Add smina to PATH or copy smina.exe to project dir

Option C — Download smina.exe binary directly:
  1. Search "smina windows binary" on GitHub/SourceForge
  2. Copy smina.exe to scripts/ or anywhere on PATH

Prerequisites:
  - scripts/20_prepare_docking_receptor.py → data/docking/receptors/
  - smina or gnina on PATH (or via wsl)
  - pip install meeko  (ligand PDBQT preparation — already done)
  - pip install rdkit  (3D conformer generation; usually already installed)

Outputs:
  data/docking/train_docking_scores.parquet  — scores for 4135 training compounds
  data/docking/test_docking_scores.parquet   — scores for 513 test compounds

Next: python scripts/22_extract_docking_features.py
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ─── Conformer generation ────────────────────────────────────────────────────

def smiles_to_3d_sdf(smiles: str, compound_id: str, max_attempts: int = 5) -> str | None:
    """
    Generates a 3D conformer from SMILES using RDKit ETKDGv3 + MMFF94 optimization.

    ETKDGv3 (Experimental Torsion-Angle Knowledge Distance Geometry v3) uses
    torsion-angle statistics from the CSD to bias conformer sampling toward
    low-energy geometries before MMFF94 force-field optimization.

    Returns the SDF block string, or None if generation fails.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)  # explicit H required for MMFF94 geometry

    ps = AllChem.ETKDGv3()
    ps.randomSeed = 42
    ps.maxIterations = max_attempts * 200  # ETKDGv3 uses iteration count, not attempt count

    res = AllChem.EmbedMolecule(mol, ps)
    if res == -1:
        # Fallback: random conformer
        res = AllChem.EmbedMolecule(mol, AllChem.EmbedParameters())
    if res == -1:
        logger.debug(f"Conformer generation failed for {compound_id}")
        return None

    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    except Exception:
        pass  # use unoptimized conformer if MMFF fails (rare, usually macrocycles)

    mol.SetProp("_Name", compound_id)
    return Chem.MolToMolBlock(mol)


def sdf_to_pdbqt_meeko(sdf_block: str) -> str | None:
    """
    Converts an SDF block to PDBQT format using openbabel (preferred) or Meeko.

    openbabel-wheel is used as the primary method because meeko 0.5.x depends on
    rdkit.six which was removed from RDKit 2022.09+. openbabel applies Gasteiger
    charges and detects rotatable bonds — equivalent output for smina/gnina scoring.

    Returns the PDBQT string, or None if preparation fails.
    """
    # Primary: openbabel Python API (pip install openbabel-wheel)
    try:
        from openbabel import openbabel as ob
        conv = ob.OBConversion()
        conv.SetInAndOutFormats("sdf", "pdbqt")
        obmol = ob.OBMol()
        ok = conv.ReadString(obmol, sdf_block)
        if not ok or obmol.NumAtoms() == 0:
            raise ValueError("openbabel failed to read SDF block")
        obmol.AddHydrogens()
        charge_model = ob.OBChargeModel.FindType("gasteiger")
        if charge_model:
            charge_model.ComputeCharges(obmol)
        pdbqt_str = conv.WriteString(obmol)
        return pdbqt_str if pdbqt_str.strip() else None
    except ImportError:
        pass  # fall through to meeko
    except Exception as e:
        logger.debug(f"openbabel PDBQT failed: {e}")

    # Fallback: Meeko (works if meeko >=0.6 or rdkit has rdkit.six shim)
    try:
        from meeko import MoleculePreparation
        from rdkit import Chem

        mol = Chem.MolFromMolBlock(sdf_block, removeHs=False)
        if mol is None:
            return None

        prep = MoleculePreparation()
        mol_setups = prep.prepare(mol)
        if not mol_setups:
            return None

        setup = mol_setups[0]
        pdbqt_string, is_ok, error_msg = MoleculePreparation.write_pdbqt_string(setup, bad_charge=None)
        if not is_ok:
            logger.debug(f"Meeko PDBQT issue: {error_msg}")
        return pdbqt_string if pdbqt_string else None

    except Exception as e:
        logger.debug(f"Meeko failed: {e}")
        return None


# ─── Docking engine ──────────────────────────────────────────────────────────

def detect_docking_engine() -> str | None:
    """
    Detects available docking engine, checking native PATH, known conda env paths,
    then WSL2 fallback.
    Priority: gnina (native) > smina (native) > conda env smina > wsl gnina > wsl smina.
    """
    import os

    # Known conda environment paths to check for smina/gnina
    conda_candidates = [
        os.path.expandvars(r"%USERPROFILE%\miniconda3\envs\docking\Library\bin\smina.exe"),
        os.path.expandvars(r"%USERPROFILE%\miniconda3\envs\docking\Library\bin\gnina.exe"),
        os.path.expandvars(r"%USERPROFILE%\anaconda3\envs\docking\Library\bin\smina.exe"),
        os.path.expandvars(r"%USERPROFILE%\anaconda3\envs\docking\Library\bin\gnina.exe"),
    ]

    # Check PATH engines first
    for engine in ["gnina", "smina"]:
        try:
            result = subprocess.run([engine, "--version"], capture_output=True, timeout=10)
            out = result.stdout + result.stderr
            if result.returncode == 0 or b"smina" in out or b"gnina" in out:
                logger.info(f"Using docking engine (native PATH): {engine}")
                return engine
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    # Check conda env paths
    for path in conda_candidates:
        if os.path.exists(path):
            try:
                result = subprocess.run([path, "--version"], capture_output=True, timeout=10)
                out = result.stdout + result.stderr
                if b"smina" in out or b"gnina" in out or result.returncode == 0:
                    logger.info(f"Using docking engine (conda env): {path}")
                    return path  # full path works as the engine command
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

    # WSL2 fallback
    for engine in ["gnina", "smina"]:
        try:
            result = subprocess.run(["wsl", engine, "--version"], capture_output=True, timeout=15)
            out = result.stdout + result.stderr
            if result.returncode == 0 or engine.encode() in out:
                logger.info(f"Using docking engine (via WSL2): wsl {engine}")
                return f"wsl:{engine}"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    return None


def parse_vina_scores(pdbqt_output: str) -> list[float]:
    """
    Parses docking scores from smina/gnina PDBQT output.

    smina/gnina output format: lines starting with 'REMARK VINA RESULT' followed by
    the affinity score (kcal/mol, more negative = stronger binding).

    Returns list of affinity scores for each pose (sorted best-first by the engine).
    """
    scores = []
    for line in pdbqt_output.splitlines():
        if "REMARK VINA RESULT" in line or "REMARK minimizedAffinity" in line:
            parts = line.split()
            try:
                # smina: REMARK VINA RESULT  -8.5  0.000  0.000
                # gnina: REMARK minimizedAffinity -8.5
                for p in parts[2:]:
                    try:
                        scores.append(float(p))
                        break
                    except ValueError:
                        continue
            except (IndexError, ValueError):
                continue
    return scores


def parse_gnina_cnn_scores(pdbqt_output: str) -> list[tuple[float, float]]:
    """
    Parses gnina-specific CNN scores from output.

    gnina adds:
      REMARK CNNscore 0.85
      REMARK CNNaffinity -8.2

    Returns list of (cnn_score, cnn_affinity) tuples per pose.
    """
    cnn_scores = []
    cnn_affinities = []
    for line in pdbqt_output.splitlines():
        if "CNNscore" in line:
            try:
                cnn_scores.append(float(line.split()[-1]))
            except ValueError:
                pass
        elif "CNNaffinity" in line:
            try:
                cnn_affinities.append(float(line.split()[-1]))
            except ValueError:
                pass

    n = min(len(cnn_scores), len(cnn_affinities))
    return list(zip(cnn_scores[:n], cnn_affinities[:n]))


def dock_one(
    compound_id: str,
    smiles: str,
    receptor_pdbqt: str,
    box: dict,
    engine: str,
    tmpdir: Path,
    exhaustiveness: int = 8,
    num_modes: int = 5,
    poses_dir: Path | None = None,
) -> dict:
    """
    Docks a single compound and returns score dictionary.

    Returns dict with keys: compound_id, vina_score_1 (best pose, kcal/mol),
    vina_score_mean (mean of top-5 poses), cnn_score_1, cnn_affinity_1 (gnina only),
    n_poses, success.
    """
    result = {
        "compound_id": compound_id,
        "vina_score_1": np.nan,
        "vina_score_mean": np.nan,
        "cnn_score_1": np.nan,
        "cnn_affinity_1": np.nan,
        "n_poses": 0,
        "success": False,
    }

    # Generate 3D conformer
    sdf_block = smiles_to_3d_sdf(smiles, compound_id)
    if sdf_block is None:
        logger.debug(f"{compound_id}: conformer generation failed")
        return result

    # Convert to PDBQT
    pdbqt_str = sdf_to_pdbqt_meeko(sdf_block)
    if pdbqt_str is None:
        logger.debug(f"{compound_id}: PDBQT preparation failed")
        return result

    # Write ligand PDBQT to temp file
    lig_path = tmpdir / f"{compound_id}_lig.pdbqt"
    out_path = tmpdir / f"{compound_id}_out.pdbqt"
    lig_path.write_text(pdbqt_str)

    # Build docking command (handle WSL2 prefix)
    if engine.startswith("wsl:"):
        real_engine = engine[4:]
        # Convert Windows paths to WSL paths via wslpath
        def to_wsl(path: str) -> str:
            try:
                r = subprocess.run(["wsl", "wslpath", path.replace("\\", "/")], capture_output=True, text=True, timeout=5)
                return r.stdout.strip() if r.returncode == 0 else path
            except Exception:
                return path.replace("\\", "/")
        cmd = [
            "wsl", real_engine,
            "--receptor", to_wsl(receptor_pdbqt),
            "--ligand", to_wsl(str(lig_path)),
            "--center_x", str(box["center_x"]),
            "--center_y", str(box["center_y"]),
            "--center_z", str(box["center_z"]),
            "--size_x", str(box["size_x"]),
            "--size_y", str(box["size_y"]),
            "--size_z", str(box["size_z"]),
            "--exhaustiveness", str(exhaustiveness),
            "--num_modes", str(num_modes),
            "--out", to_wsl(str(out_path)),
        ]
    else:
        cmd = [
            engine,
            "--receptor", receptor_pdbqt,
            "--ligand", str(lig_path),
            "--center_x", str(box["center_x"]),
            "--center_y", str(box["center_y"]),
            "--center_z", str(box["center_z"]),
            "--size_x", str(box["size_x"]),
            "--size_y", str(box["size_y"]),
            "--size_z", str(box["size_z"]),
            "--exhaustiveness", str(exhaustiveness),
            "--num_modes", str(num_modes),
            "--out", str(out_path),
        ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        logger.debug(f"{compound_id}: docking timed out")
        return result

    if not out_path.exists():
        logger.debug(f"{compound_id}: docking produced no output — {proc.stderr[:200]}")
        return result

    output_text = out_path.read_text()
    vina_scores = parse_vina_scores(output_text)
    cnn_pairs = parse_gnina_cnn_scores(output_text)

    if not vina_scores:
        return result

    result["vina_score_1"] = vina_scores[0]
    result["vina_score_mean"] = float(np.mean(vina_scores[:num_modes]))
    result["n_poses"] = len(vina_scores)
    result["success"] = True

    if cnn_pairs:
        result["cnn_score_1"] = cnn_pairs[0][0]
        result["cnn_affinity_1"] = cnn_pairs[0][1]

    # Save pose to persistent directory for ProLIF IFP extraction (script 22)
    if poses_dir is not None and result["success"]:
        import shutil
        poses_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_path, poses_dir / f"{compound_id}_out.pdbqt")

    return result


# ─── Main ─────────────────────────────────────────────────────────────────────

def dock_batch(
    smiles_list: list[tuple[str, str]],
    receptor_pdbqt: str,
    box: dict,
    engine: str,
    exhaustiveness: int = 8,
    num_modes: int = 5,
    log_every: int = 50,
    poses_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Docks a batch of (compound_id, smiles) pairs and returns a DataFrame of scores.

    If poses_dir is provided, successfully docked pose PDBQT files are saved there
    for subsequent ProLIF interaction fingerprint extraction (script 22).
    """
    records = []
    n = len(smiles_list)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for i, (compound_id, smiles) in enumerate(smiles_list):
            if i % log_every == 0:
                n_ok = sum(1 for r in records if r.get("success"))
                logger.info(f"Docking {i}/{n} ({n_ok} successful so far)...")
            rec = dock_one(
                compound_id, smiles, receptor_pdbqt, box,
                engine=engine, tmpdir=tmpdir,
                exhaustiveness=exhaustiveness, num_modes=num_modes,
                poses_dir=poses_dir,
            )
            records.append(rec)

    df = pd.DataFrame(records)
    n_ok = df["success"].sum()
    logger.info(f"Docking complete: {n_ok}/{n} successful ({100*n_ok/n:.1f}%)")
    return df


def main() -> None:
    receptor_dir = Path("data/docking/receptors")
    out_dir = Path("data/docking")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load receptor and box
    box_path = receptor_dir / "docking_box.json"
    if not box_path.exists():
        logger.error("Docking box not found — run scripts/20_prepare_docking_receptor.py first.")
        sys.exit(1)
    with open(box_path) as f:
        box = json.load(f)
    logger.info(f"Docking box: center=({box['center_x']}, {box['center_y']}, {box['center_z']}), size={box['size_x']}³ Å³")

    # Prefer PDBQT, fall back to PDB (some gnina versions accept PDB directly)
    pdb_id = box["receptor_pdb_id"].lower()
    pdbqt_path = receptor_dir / f"{pdb_id}_protein.pdbqt"
    pdb_path = receptor_dir / f"{pdb_id}_protein.pdb"
    if pdbqt_path.exists():
        receptor = str(pdbqt_path)
        logger.info(f"Using PDBQT receptor: {pdbqt_path}")
    elif pdb_path.exists():
        receptor = str(pdb_path)
        logger.warning(
            f"PDBQT not found, using raw PDB: {pdb_path}\n"
            "Smina/gnina may add '--receptor_is_pdb' automatically, or may fail.\n"
            "Generate PDBQT with: obabel {pdb_path} -O {pdbqt_path} -xh"
        )
    else:
        logger.error("No receptor file found — run scripts/20_prepare_docking_receptor.py first.")
        sys.exit(1)

    # Detect engine
    engine = detect_docking_engine()
    if engine is None:
        logger.error(
            "No docking engine found on PATH.\n"
            "Install options:\n"
            "  Windows: conda install -c conda-forge smina\n"
            "  WSL2/Linux: conda install -c conda-forge gnina  (GPU-accelerated)\n"
            "Then re-run this script."
        )
        sys.exit(1)

    # Load training and test SMILES
    train_df = pd.read_parquet("data/curated/master_train.parquet")
    smiles_col = "smiles_std" if "smiles_std" in train_df.columns else "smiles"
    id_col = "compound_id" if "compound_id" in train_df.columns else train_df.columns[0]
    train_pairs = list(zip(train_df[id_col].astype(str), train_df[smiles_col]))

    test_std_path = Path("data/curated/openadmet_test_std.parquet")
    if test_std_path.exists():
        test_df = pd.read_parquet(test_std_path)
        test_smi_col = "smiles_std" if "smiles_std" in test_df.columns else "smiles"
    else:
        test_df = pd.read_parquet("data/raw/openadmet_test.parquet")
        test_smi_col = "smiles"
    test_id_col = "compound_id" if "compound_id" in test_df.columns else test_df.columns[0]
    test_pairs = list(zip(test_df[test_id_col].astype(str), test_df[test_smi_col]))

    logger.info(f"Compounds to dock: {len(train_pairs)} train + {len(test_pairs)} test = {len(train_pairs)+len(test_pairs)} total")

    # Pose directory — save best-pose PDBQT files for ProLIF IFP features (script 22)
    poses_dir = out_dir / "poses"
    poses_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving docked poses to: {poses_dir}/")

    # Dock training compounds
    train_out = out_dir / "train_docking_scores.parquet"
    if train_out.exists():
        logger.info(f"Training docking scores already exist: {train_out}")
        train_scores = pd.read_parquet(train_out)
    else:
        logger.info(f"\n--- Docking {len(train_pairs)} training compounds ---")
        train_scores = dock_batch(train_pairs, receptor, box, engine=engine, poses_dir=poses_dir)
        train_scores.to_parquet(train_out, index=False)
        logger.info(f"Training scores saved: {train_out}")

    # Dock test compounds
    test_out = out_dir / "test_docking_scores.parquet"
    if test_out.exists():
        logger.info(f"Test docking scores already exist: {test_out}")
        test_scores = pd.read_parquet(test_out)
    else:
        logger.info(f"\n--- Docking {len(test_pairs)} test compounds ---")
        test_scores = dock_batch(test_pairs, receptor, box, engine=engine, poses_dir=poses_dir)
        test_scores.to_parquet(test_out, index=False)
        logger.info(f"Test scores saved: {test_out}")

    # Summary stats
    for name, df in [("Train", train_scores), ("Test", test_scores)]:
        ok = df["success"].sum()
        logger.info(
            f"{name}: {ok}/{len(df)} docked | "
            f"vina_score_1 mean={df['vina_score_1'].mean():.2f} ± {df['vina_score_1'].std():.2f} kcal/mol"
        )

    # Sanity check: vina score should correlate with pEC50 for training set
    merged = train_scores.merge(
        train_df[[id_col, "pec50_median" if "pec50_median" in train_df.columns else "pec50"]].rename(
            columns={id_col: "compound_id"}
        ),
        on="compound_id", how="inner",
    )
    if len(merged) > 100 and merged["vina_score_1"].notna().sum() > 50:
        pec50_col = "pec50_median" if "pec50_median" in merged.columns else "pec50"
        valid = merged[merged["vina_score_1"].notna()]
        corr = float(np.corrcoef(valid["vina_score_1"], valid[pec50_col])[0, 1])
        logger.info(
            f"\nSanity check — Pearson r(vina_score_1, pEC50): {corr:.3f}\n"
            f"  Expected range: -0.4 to +0.1 (negative = more negative score → more potent)\n"
            f"  If |r| < 0.10: docking may have failed silently — check receptor and box.\n"
            f"  Note: r > 0 is possible if the box catches many inactive compounds binding weakly."
        )

    logger.info("\nNext: python scripts/22_extract_docking_features.py")


if __name__ == "__main__":
    main()
