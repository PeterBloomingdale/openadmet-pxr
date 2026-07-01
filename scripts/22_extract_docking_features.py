"""
Extract protein-ligand interaction fingerprint (IFP) features from docked poses.

Takes the raw docking scores from script 21 and augments them with:
  1. ProLIF interaction fingerprints — which residues/interaction types are engaged
  2. IFP Tanimoto similarity to top-10 training actives (known PXR binders)
  3. Aggregate contact counts by interaction type
  4. 3D molecular shape descriptors (RDKit Descriptors3D) from conformers

Why interaction fingerprints?
Docking scores alone conflate binding mode quality (does the compound form the
right contacts?) with overall binding strength. Two compounds with similar Vina
scores might have completely different binding modes — one forming the critical
contacts with F288 and W299 (correlated with PXR activity) and one stuck in a
sub-pocket with different contacts. IFPs disentangle these cases by encoding
*which* protein contacts are made, not just the overall score.

PXR key residues (from crystal structure analysis, Watkins 2003, Chrencik 2005):
  S247, F288, I237, L240, M243, W299, H407, F420 — the hydrophobic cage
  H163, S208 — polar contacts observed in some agonist structures
  The AF-2 helix (H12, L416, F429) — coactivator-recruiting surface

PDBQT → ProLIF conversion strategy:
  smina PDBQT poses contain Gasteiger partial charges that confuse RDKit's
  valence checker when MDAnalysis tries to convert them directly. The robust
  approach is: PDBQT → SDF (via openbabel, preserving 3D coords) → strip H →
  assign bond orders from SMILES template → build ProLIF Molecule. This avoids
  all valence errors caused by PDBQT atom type / charge encoding.

3D shape descriptors:
  PXR's large ellipsoidal LBD (1150–1540 Å³) binds ligands primarily through
  shape complementarity. NPR1/NPR2 (normalized PMI ratios) capture rod/disc/sphere
  shape — compounds that optimally fill the ellipsoidal cavity tend to be most potent.

Prerequisites:
  - scripts/21_dock_compounds.py → data/docking/train_docking_scores.parquet
  - pip install prolif  (also installs MDAnalysis)
  - data/docking/poses/ — saved PDBQT poses from script 21

Outputs:
  data/features/train_docking_feats.parquet  — per-compound feature matrix (train)
  data/features/test_docking_feats.parquet   — per-compound feature matrix (test)

Next: python scripts/23_train_lgbm_docking.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors3D

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Key PXR binding-pocket residues to track explicitly
# (from Watkins 2003, Chrencik 2005 crystal structure analyses)
PXR_KEY_RESIDUES = {
    "hydrophobic_core": ["PHE288", "TRP299", "LEU240", "ILE237", "MET243", "PHE420"],
    "polar_contacts":   ["SER247", "HIS163", "SER208", "HIS407"],
    "helix_h12":        ["LEU416", "PHE429", "LYS425"],
}
ALL_KEY_RESIDUES = [r for group in PXR_KEY_RESIDUES.values() for r in group]


def load_docking_scores(train_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_scores = pd.read_parquet(train_path)
    test_scores = pd.read_parquet(test_path)
    logger.info(f"Train docking: {len(train_scores)} compounds, {train_scores['success'].sum()} successful")
    logger.info(f"Test docking:  {len(test_scores)} compounds, {test_scores['success'].sum()} successful")
    return train_scores, test_scores


def add_hydrogens_to_receptor(receptor_pdb: str) -> str:
    """
    Add polar hydrogens to receptor PDB using openbabel (required for ProLIF H-bond detection).
    Returns path to H-added PDB file. Skips if already done.
    """
    out_path = receptor_pdb.replace(".pdb", "_H.pdb")
    if Path(out_path).exists():
        logger.info(f"Receptor with H already exists: {out_path}")
        return out_path
    try:
        from openbabel import openbabel as ob
        conv = ob.OBConversion()
        conv.SetInAndOutFormats("pdb", "pdb")
        obmol = ob.OBMol()
        if not conv.ReadFile(obmol, receptor_pdb):
            logger.warning(f"OpenBabel could not read {receptor_pdb}")
            return receptor_pdb
        obmol.AddHydrogens(False, True, 7.4)  # polar H only, pH 7.4
        conv.WriteFile(obmol, out_path)
        logger.info(f"Added polar H to receptor → {out_path}")
        return out_path
    except Exception as e:
        logger.warning(f"Could not add H to receptor ({e}) — using original PDB")
        return receptor_pdb


def pdbqt_to_rdkit_mol(pose_path: Path, smiles: str) -> "Chem.Mol | None":
    """
    Converts a smina PDBQT pose to an RDKit molecule with correct bond orders.

    Strategy:
      1. openbabel: PDBQT → SDF  (correctly handles Gasteiger charges & atom types)
      2. RDKit: read SDF without sanitization (avoids valence errors from charge encoding)
      3. Remove H atoms to match the SMILES template heavy-atom count
      4. AllChem.AssignBondOrdersFromTemplate: copy bond orders from SMILES → SDF coords
      5. Sanitize the final mol

    This avoids the "Explicit valence" errors that occur when MDAnalysis's PDBQT
    reader tries to convert Gasteiger-charged PDBQT atoms to RDKit formal charges.
    """
    try:
        from openbabel import openbabel as ob

        # Step 1: PDBQT → SDF string via openbabel
        conv = ob.OBConversion()
        conv.SetInAndOutFormats("pdbqt", "sdf")
        obmol = ob.OBMol()
        if not conv.ReadFile(obmol, str(pose_path)):
            return None
        sdf_block = conv.WriteString(obmol)
        if not sdf_block or not sdf_block.strip():
            return None

        # Step 2: Read SDF without sanitization
        mol_raw = Chem.MolFromMolBlock(sdf_block, sanitize=False, removeHs=False)
        if mol_raw is None:
            return None

        # Step 3: Remove hydrogens (template is typically a non-H SMILES)
        mol_noH = Chem.RemoveHs(mol_raw, sanitize=False)

        # Step 4: Get template from SMILES
        template = Chem.MolFromSmiles(smiles)
        if template is None:
            return None

        # Step 5: Assign bond orders from template to SDF coordinates
        mol_with_bonds = AllChem.AssignBondOrdersFromTemplate(template, mol_noH)
        Chem.SanitizeMol(mol_with_bonds)
        return mol_with_bonds

    except Exception as e:
        logger.debug(f"PDBQT→RDKit conversion failed for {pose_path.name}: {e}")
        return None


def compute_ifp_similarity(query_ifp: np.ndarray, reference_ifps: np.ndarray) -> float:
    """Maximum Tanimoto similarity between query IFP and a set of reference IFPs."""
    if reference_ifps.shape[0] == 0 or query_ifp.sum() == 0:
        return 0.0
    intersections = (query_ifp & reference_ifps).sum(axis=1)
    unions = (query_ifp | reference_ifps).sum(axis=1)
    similarities = np.where(unions > 0, intersections / unions, 0.0)
    return float(similarities.max())


def extract_ifp_features(
    pose_dir: Path,
    compound_ids: list[str],
    smiles_map: dict[str, str],
    receptor_pdb: str,
    reference_ifps: np.ndarray | None,
) -> pd.DataFrame:
    """
    Extracts ProLIF IFP features for all compounds with saved poses.

    Converts PDBQT → SDF (via openbabel) → RDKit mol with correct bond orders
    → ProLIF Fingerprint, avoiding valence errors from PDBQT Gasteiger charges.

    Returns DataFrame with one row per compound, columns = IFP features.
    Returns empty DataFrame if ProLIF unavailable or no poses exist.
    """
    try:
        import prolif as plf
        import MDAnalysis as mda
    except ImportError:
        logger.warning("prolif not installed — skipping IFP features. pip install prolif")
        return pd.DataFrame()

    if not pose_dir.exists():
        logger.warning(f"Pose directory not found: {pose_dir}")
        return pd.DataFrame()

    # Load receptor as ProLIF Molecule — select chain A only.
    # openbabel splits the original single-chain PDB into A/B/C/D when adding
    # polar hydrogens; B/C/D are openbabel artefacts, not real protein chains.
    # ProLIF fails with ResidueId errors on those spurious chains, so we restrict
    # to chain A (the PXR LBD monomer we actually docked into).
    try:
        u_prot = mda.Universe(receptor_pdb)
        chain_a = u_prot.select_atoms("chainID A")
        prot = plf.Molecule.from_mda(chain_a)
    except Exception as e:
        logger.warning(f"Failed to load receptor for ProLIF: {e}")
        return pd.DataFrame()

    records = []
    n_attempted = n_success = n_conv_fail = n_ifp_fail = 0

    for compound_id in compound_ids:
        pose_path = pose_dir / f"{compound_id}_out.pdbqt"
        if not pose_path.exists():
            continue
        smiles = smiles_map.get(compound_id)
        if not smiles:
            continue

        n_attempted += 1

        # Convert PDBQT → RDKit mol with correct bond orders
        rdmol = pdbqt_to_rdkit_mol(pose_path, smiles)
        if rdmol is None:
            n_conv_fail += 1
            continue

        try:
            lig = plf.Molecule(rdmol)
            fp = plf.Fingerprint(count=False)
            fp.run_from_iterable([lig], prot)
            ifp_df = fp.to_dataframe()

            if ifp_df.empty:
                n_ifp_fail += 1
                continue

            row_data = {"compound_id": compound_id}

            for col in ifp_df.columns:
                if isinstance(col, tuple) and len(col) >= 2:
                    res_name, interaction = str(col[0]), str(col[-1])
                    # Aggregate by interaction type
                    key = f"ifp_{interaction.lower()}_total"
                    row_data[key] = row_data.get(key, 0) + int(ifp_df[col].iloc[0])
                    # Track key PXR residues individually
                    res_upper = res_name.upper()
                    for key_res in ALL_KEY_RESIDUES:
                        if key_res in res_upper:
                            row_data[f"ifp_{key_res}_{interaction.lower()}"] = int(ifp_df[col].iloc[0])

            # IFP similarity to top actives
            if reference_ifps is not None and reference_ifps.shape[0] > 0:
                bool_cols = [c for c in ifp_df.columns if isinstance(c, tuple)]
                if bool_cols:
                    ifp_vec = ifp_df[bool_cols].iloc[0].values.astype(bool)
                    min_len = min(len(ifp_vec), reference_ifps.shape[1])
                    try:
                        row_data["ifp_tanimoto_to_actives"] = compute_ifp_similarity(
                            ifp_vec[:min_len], reference_ifps[:, :min_len]
                        )
                    except Exception:
                        pass

            records.append(row_data)
            n_success += 1

        except Exception as e:
            logger.debug(f"IFP fingerprint failed for {compound_id}: {e}")
            n_ifp_fail += 1
            continue

    logger.info(
        f"IFP extraction: {n_success} success, {n_conv_fail} conversion failures, "
        f"{n_ifp_fail} fingerprint failures out of {n_attempted} attempted"
    )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).fillna(0)
    logger.info(f"IFP features: {len(df)} compounds, {df.shape[1] - 1} features")
    return df


def compute_3d_shape_features(smiles: str) -> dict | None:
    """
    Computes 3D molecular shape descriptors from an ETKDGv3 conformer.

    NPR1/NPR2 (normalized PMI ratios) characterize molecular shape:
      NPR1 low, NPR2 ~0.5  → rod-like
      NPR1 ~0.5, NPR2 ~0.5 → disc-like
      NPR1 ~0.5, NPR2 high → sphere-like

    PXR's large ellipsoidal LBD preferentially accommodates disc/sphere-like
    ligands that fill the cavity efficiently.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        ps = AllChem.ETKDGv3()
        ps.randomSeed = 42
        ps.maxIterations = 2000
        if AllChem.EmbedMolecule(mol, ps) == -1:
            return None
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        return {
            "shape_asphericity":  Descriptors3D.Asphericity(mol, confId=0),
            "shape_eccentricity": Descriptors3D.Eccentricity(mol, confId=0),
            "shape_isf":          Descriptors3D.InertialShapeFactor(mol, confId=0),
            "shape_npr1":         Descriptors3D.NPR1(mol, confId=0),
            "shape_npr2":         Descriptors3D.NPR2(mol, confId=0),
            "shape_pmi1":         Descriptors3D.PMI1(mol, confId=0),
            "shape_pmi2":         Descriptors3D.PMI2(mol, confId=0),
            "shape_pmi3":         Descriptors3D.PMI3(mol, confId=0),
        }
    except Exception:
        return None


def compute_shape_batch(compound_ids: list[str], smiles_map: dict[str, str]) -> pd.DataFrame:
    records = []
    n_fail = 0
    for cid in compound_ids:
        smiles = smiles_map.get(cid)
        if not smiles:
            continue
        feats = compute_3d_shape_features(smiles)
        if feats is not None:
            feats["compound_id"] = cid
            records.append(feats)
        else:
            n_fail += 1
    if n_fail > 0:
        logger.warning(f"3D shape conformer failed for {n_fail}/{len(compound_ids)} compounds")
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    logger.info(f"3D shape features: {len(df)} compounds, {[c for c in df.columns if c != 'compound_id']}")
    return df


def build_feature_matrix(
    docking_scores: pd.DataFrame,
    ifp_features: pd.DataFrame,
    shape_features: pd.DataFrame,
    compound_id_col: str = "compound_id",
) -> pd.DataFrame:
    score_cols = ["vina_score_1", "vina_score_mean", "cnn_score_1", "cnn_affinity_1", "n_poses", "success"]
    feat_df = docking_scores[[compound_id_col] + [c for c in score_cols if c in docking_scores.columns]].copy()
    for col in score_cols:
        if col in feat_df.columns:
            feat_df[col] = feat_df[col].fillna(0.0)
    if "success" in feat_df.columns:
        feat_df["success"] = feat_df["success"].astype(int)
        feat_df = feat_df.rename(columns={"success": "docked_successfully"})
    if not ifp_features.empty:
        feat_df = feat_df.merge(ifp_features, on=compound_id_col, how="left")
        ifp_cols = [c for c in feat_df.columns if c.startswith("ifp_")]
        feat_df[ifp_cols] = feat_df[ifp_cols].fillna(0)
    if not shape_features.empty:
        feat_df = feat_df.merge(shape_features, on=compound_id_col, how="left")
        shape_cols = [c for c in feat_df.columns if c.startswith("shape_")]
        feat_df[shape_cols] = feat_df[shape_cols].fillna(feat_df[shape_cols].median())
    return feat_df


def main() -> None:
    out_feat_dir = Path("data/features")
    out_feat_dir.mkdir(parents=True, exist_ok=True)
    docking_dir = Path("data/docking")
    receptor_dir = Path("data/docking/receptors")
    pose_dir = docking_dir / "poses"

    train_scores_path = docking_dir / "train_docking_scores.parquet"
    test_scores_path = docking_dir / "test_docking_scores.parquet"
    if not train_scores_path.exists() or not test_scores_path.exists():
        logger.error("Docking scores not found — run scripts/21_dock_compounds.py first.")
        sys.exit(1)

    # Load data and build compound_id → SMILES lookup
    # compound_id in docking scores = inchikey_prefix (14-char InChIKey connectivity layer)
    train_df = pd.read_parquet("data/curated/master_train.parquet")
    test_std_path = Path("data/curated/openadmet_test_std.parquet")
    test_df = pd.read_parquet(
        str(test_std_path) if test_std_path.exists() else "data/raw/openadmet_test.parquet"
    )

    id_col_train  = "inchikey_prefix" if "inchikey_prefix" in train_df.columns else "compound_id"
    smiles_col_tr = "smiles_std" if "smiles_std" in train_df.columns else "smiles"
    id_col_test   = "compound_id" if "compound_id" in test_df.columns else test_df.columns[0]
    smiles_col_te = "smiles_std" if "smiles_std" in test_df.columns else "smiles"

    train_smiles_map = dict(zip(train_df[id_col_train], train_df[smiles_col_tr]))
    test_smiles_map  = dict(zip(test_df[id_col_test],   test_df[smiles_col_te]))
    logger.info(f"SMILES map: {len(train_smiles_map)} train, {len(test_smiles_map)} test")

    train_scores, test_scores = load_docking_scores(train_scores_path, test_scores_path)

    # Top-10 actives for IFP similarity reference
    pec50_col = "pec50_median" if "pec50_median" in train_df.columns else "pec50"
    top_actives = train_df.nlargest(10, pec50_col)[id_col_train].tolist()
    logger.info(f"Top-10 actives pEC50: {train_df.nlargest(10, pec50_col)[pec50_col].values.round(2)}")

    # Prepare receptor with polar H (required for ProLIF H-bond detection)
    receptor_pdb = None
    if (receptor_dir / "docking_box.json").exists():
        box = json.loads((receptor_dir / "docking_box.json").read_text())
        pdb_id = box["receptor_pdb_id"].lower()
        receptor_pdb = add_hydrogens_to_receptor(str(receptor_dir / f"{pdb_id}_protein.pdb"))

    # ─── IFP features ────────────────────────────────────────────────────────
    # ProLIF IFP extraction skipped: after three separate failure modes
    # (openbabel multi-chain split, chain B/C ResidueId errors, VAL26 chain A
    # error hitting 100% of compounds), the receptor preprocessing is
    # incompatible with this ProLIF version. The 3D shape features below
    # provide orthogonal 3D signal without requiring the receptor structure.
    train_ifp = test_ifp = pd.DataFrame()
    logger.info("IFP features skipped (ProLIF receptor compatibility issue) — using 3D shape features instead.")

    # ─── 3D shape features ───────────────────────────────────────────────────
    logger.info("\nComputing 3D molecular shape features (ETKDGv3)...")
    train_shape = compute_shape_batch(train_scores["compound_id"].tolist(), train_smiles_map)
    test_shape  = compute_shape_batch(test_scores["compound_id"].tolist(),  test_smiles_map)

    # ─── Assemble and save ────────────────────────────────────────────────────
    train_feats = build_feature_matrix(train_scores, train_ifp, train_shape)
    test_feats  = build_feature_matrix(test_scores,  test_ifp,  test_shape)

    train_feats.to_parquet(out_feat_dir / "train_docking_feats.parquet", index=False)
    test_feats.to_parquet(out_feat_dir  / "test_docking_feats.parquet",  index=False)

    feat_cols = [c for c in train_feats.columns if c != "compound_id"]
    logger.info(f"\nDocking features saved:")
    logger.info(f"  Train: {train_feats.shape} → data/features/train_docking_feats.parquet")
    logger.info(f"  Test:  {test_feats.shape}  → data/features/test_docking_feats.parquet")
    logger.info(f"  Vina:   {[c for c in feat_cols if not c.startswith(('ifp_','shape_'))]}")
    logger.info(f"  IFP:    {len([c for c in feat_cols if c.startswith('ifp_')])} features")
    logger.info(f"  Shape:  {len([c for c in feat_cols if c.startswith('shape_')])} features")

    # Sanity check: Vina score vs pEC50 correlation
    merged = train_scores.merge(
        train_df[[id_col_train, pec50_col]].rename(columns={id_col_train: "compound_id"}),
        on="compound_id", how="inner",
    )
    valid = merged[merged["vina_score_1"].notna() & merged[pec50_col].notna()]
    if len(valid) > 50:
        r = float(np.corrcoef(valid["vina_score_1"], valid[pec50_col])[0, 1])
        logger.info(f"\nVina score vs pEC50 Pearson r = {r:.3f} (n={len(valid)})")

    logger.info("\nNext: python scripts/23_train_lgbm_docking.py")


if __name__ == "__main__":
    main()
