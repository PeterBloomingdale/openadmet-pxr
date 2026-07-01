"""
Prepare the PXR crystal structure for molecular docking.

Downloads PDB 2QD9 (human PXR LBD bound to T0901317, 2.0 Å resolution) from RCSB,
strips the ligand and crystallographic waters, extracts the ligand centroid to define
the docking box, and writes the protein-only PDB for use with smina/gnina.

Why 2QD9?
PXR requires a full agonist complex as the receptor template. T0901317 is one of the
most potent known PXR agonists (EC50 ~60 nM in cell-based assay) and fully occupies
the hydrophobic core of the LBD. Crystal structures co-crystallized with weak or
partial agonists adopt slightly different helix-12 geometries that bias docking scores
toward the template chemotype. Using a potent full-agonist complex avoids this.

Why not multiple receptors (ensemble docking)?
Ensemble docking (averaging scores across 3-5 PXR crystal structures) reduces false
negatives for compounds that require induced-fit. For Phase 1 timelines, single-receptor
docking with a large box (28 Å) provides reasonable coverage while taking 6-8 hours
total docking time on CPU. Ensemble docking is a Phase 2 option if docking features
improve the ensemble MAE.

Outputs:
  data/docking/receptors/2qd9_protein.pdb        — protein-only PDB (no waters/ligand)
  data/docking/receptors/2qd9_protein.pdbqt      — PDBQT for smina/gnina (if openbabel available)
  data/docking/receptors/docking_box.json        — box center + dimensions

Prerequisites:
  pip install requests

For PDBQT conversion (needed for smina/gnina):
  conda install -c conda-forge openbabel
  OR: pip install meeko (for ligand PDBQT only; receptor needs obabel or prepare_receptor4.py)

Next: python scripts/21_dock_compounds.py
"""

import json
import sys
from pathlib import Path

import requests
from loguru import logger

# PDB IDs to try in order: 2QD9 is primary, others are fallbacks
RECEPTOR_PDBS = {
    "2QD9": {"ligand_resname": "LGF", "chain": "A", "description": "T0901317 analog (LGF), 2.0 Å, full agonist"},
    "1ILH": {"ligand_resname": "RFP", "chain": "A", "description": "Rifampicin, 2.5 Å"},
    "4J79": {"ligand_resname": "ACO", "chain": "A", "description": "Hyperforin analog, 2.0 Å"},
}
PRIMARY_PDB = "2QD9"

# Docking box dimensions — PXR LBD is large and irregular (~1150–1600 Å³).
# 28 Å on each side captures the full hydrophobic pocket plus some entrance channel.
# This is deliberately large to allow induced-fit exploration.
BOX_SIZE = 28.0  # Å per side


def download_pdb(pdb_id: str, out_dir: Path) -> Path:
    """Downloads a PDB file from RCSB."""
    out_path = out_dir / f"{pdb_id.lower()}.pdb"
    if out_path.exists():
        logger.info(f"{pdb_id} already downloaded: {out_path}")
        return out_path

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    logger.info(f"Downloading {pdb_id} from {url}...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    out_path.write_text(resp.text)
    logger.info(f"Saved: {out_path} ({len(resp.text):,} bytes)")
    return out_path


def parse_pdb(pdb_path: Path, ligand_resname: str, chain: str) -> tuple[list[str], list[str]]:
    """
    Splits a PDB file into protein ATOM records and ligand HETATM records.

    Returns (protein_lines, ligand_lines).
    Crystallographic waters (HOH/WAT) and small molecules that aren't the target
    ligand are excluded — they confuse docking programs and add no signal.
    """
    protein_lines = []
    ligand_lines = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                if line[21] == chain or chain == "*":
                    protein_lines.append(line)
            elif line.startswith("HETATM"):
                resname = line[17:20].strip()
                line_chain = line[21]
                if resname == ligand_resname and (line_chain == chain or chain == "*"):
                    ligand_lines.append(line)
                # Skip waters, buffers, glycerol, etc.
            elif line.startswith(("TER", "END")):
                protein_lines.append(line)

    logger.info(f"Protein: {len(protein_lines)} ATOM lines | Ligand ({ligand_resname}): {len(ligand_lines)} HETATM lines")
    if not ligand_lines:
        raise ValueError(
            f"Ligand {ligand_resname} not found in chain {chain}. "
            f"Check the PDB file manually: grep HETATM {pdb_path}"
        )
    return protein_lines, ligand_lines


def compute_ligand_centroid(ligand_lines: list[str]) -> tuple[float, float, float]:
    """Computes the geometric centroid of ligand heavy atoms from HETATM lines."""
    xs, ys, zs = [], [], []
    for line in ligand_lines:
        try:
            element = line[76:78].strip() if len(line) > 76 else ""
            if element == "H":
                continue  # skip hydrogens if present
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            xs.append(x)
            ys.append(y)
            zs.append(z)
        except (ValueError, IndexError):
            continue
    if not xs:
        raise ValueError("No coordinates found in ligand HETATM lines")
    cx, cy, cz = sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)
    logger.info(f"Ligand centroid: ({cx:.2f}, {cy:.2f}, {cz:.2f}) from {len(xs)} heavy atoms")
    return cx, cy, cz


def convert_to_pdbqt(pdb_path: Path, pdbqt_path: Path) -> bool:
    """
    Converts protein PDB to PDBQT format using the openbabel Python API.

    PDBQT is the format expected by smina/gnina. It adds partial charges
    and atom types (Gasteiger method) required for Vina-family scoring functions.
    Prefers openbabel-wheel (pip install openbabel-wheel) Python API over CLI.
    Returns True on success, False if openbabel is unavailable.
    """
    # Try Python API first (openbabel-wheel, no CLI needed)
    try:
        from openbabel import openbabel as ob
        conv = ob.OBConversion()
        conv.SetInAndOutFormats("pdb", "pdbqt")
        mol = ob.OBMol()
        ok = conv.ReadFile(mol, str(pdb_path))
        if not ok or mol.NumAtoms() == 0:
            raise ValueError(f"openbabel failed to read {pdb_path}")
        mol.AddHydrogens()
        ok2 = conv.WriteFile(mol, str(pdbqt_path))
        if ok2:
            logger.info(f"PDBQT saved: {pdbqt_path} ({mol.NumAtoms()} atoms, Python API)")
            return True
        raise ValueError("openbabel WriteFile returned False")
    except ImportError:
        pass  # fall through to CLI
    except Exception as e:
        logger.warning(f"openbabel Python API failed: {e}. Trying CLI...")

    # Fallback: obabel CLI
    import subprocess
    try:
        cmd = ["obabel", str(pdb_path), "-O", str(pdbqt_path), "-xh"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            logger.info(f"PDBQT saved: {pdbqt_path} (CLI)")
            return True
        logger.warning(f"obabel CLI failed: {result.stderr}")
    except FileNotFoundError:
        pass

    logger.warning(
        "openbabel not available — PDBQT not generated.\n"
        "Install: pip install openbabel-wheel\n"
        "Then re-run this script."
    )
    return False


def main() -> None:
    out_dir = Path("data/docking/receptors")
    out_dir.mkdir(parents=True, exist_ok=True)

    pdb_id = PRIMARY_PDB
    config = RECEPTOR_PDBS[pdb_id]

    # 1. Download
    pdb_path = download_pdb(pdb_id, out_dir)

    # 2. Parse protein + ligand
    try:
        protein_lines, ligand_lines = parse_pdb(
            pdb_path,
            ligand_resname=config["ligand_resname"],
            chain=config["chain"],
        )
    except ValueError as e:
        logger.error(str(e))
        # Try without chain filter as fallback
        logger.info("Retrying without chain filter...")
        protein_lines, ligand_lines = parse_pdb(pdb_path, ligand_resname=config["ligand_resname"], chain="*")

    # 3. Save protein-only PDB
    protein_pdb = out_dir / f"{pdb_id.lower()}_protein.pdb"
    with open(protein_pdb, "w") as f:
        f.writelines(protein_lines)
    logger.info(f"Protein-only PDB saved: {protein_pdb}")

    # 4. Compute docking box from ligand centroid
    cx, cy, cz = compute_ligand_centroid(ligand_lines)
    box = {
        "center_x": round(cx, 3),
        "center_y": round(cy, 3),
        "center_z": round(cz, 3),
        "size_x": BOX_SIZE,
        "size_y": BOX_SIZE,
        "size_z": BOX_SIZE,
        "receptor_pdb_id": pdb_id,
        "receptor_description": config["description"],
        "ligand_resname": config["ligand_resname"],
    }
    box_path = out_dir / "docking_box.json"
    with open(box_path, "w") as f:
        json.dump(box, f, indent=2)
    logger.info(f"Docking box saved: {box_path}")
    logger.info(f"  Center: ({cx:.2f}, {cy:.2f}, {cz:.2f}), Size: {BOX_SIZE}×{BOX_SIZE}×{BOX_SIZE} Å³")

    # 5. Convert to PDBQT (requires openbabel)
    pdbqt_path = out_dir / f"{pdb_id.lower()}_protein.pdbqt"
    has_pdbqt = convert_to_pdbqt(protein_pdb, pdbqt_path)

    # Summary
    logger.info("\n=== Receptor preparation complete ===")
    logger.info(f"  PDB: {protein_pdb}")
    logger.info(f"  PDBQT: {pdbqt_path if has_pdbqt else 'NOT GENERATED (install openbabel)'}")
    logger.info(f"  Box: {box_path}")

    if not has_pdbqt:
        logger.warning(
            "\nTo generate PDBQT manually:\n"
            f"  conda install -c conda-forge openbabel\n"
            f"  obabel {protein_pdb} -O {pdbqt_path} -xh\n"
            "OR use the raw PDB with gnina's --receptor flag (some versions accept PDB directly)."
        )

    logger.info("\nNext: python scripts/21_dock_compounds.py")


if __name__ == "__main__":
    main()
