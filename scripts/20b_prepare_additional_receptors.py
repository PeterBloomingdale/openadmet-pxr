"""
Prepare additional PXR crystal structures for ensemble docking (Phase 2).

Downloads 6TFIB and 4NY9, strips waters/ligands, converts to PDBQT, computes
docking boxes. Also creates the 6TFIB H12-free variant (residues 410-430 deleted)
which allows free conformational sampling of the activation helix.

Why these structures? (discoverybytes Phase 1 Structure Track analysis)
  6TFIB H12-free: H12 activation helix (residues 410-430) is the most
    conformationally variable region across 18 PXR crystal structures (1.5-2.2 Å
    mean Ca displacement). Anchoring H12 penalises ligands whose binding mode
    prefers an alternative H12 orientation. Removing it allows Boltz/gnina to
    sample freely. Best template for ~half the ligands.
  4NY9: Rifampicin-bound conformation — genuinely different pocket geometry
    from 6TFIB. Best template for the other ~half of ligands.

Outputs:
  data/docking/receptors/6tfib_protein.pdb
  data/docking/receptors/6tfib_protein.pdbqt
  data/docking/receptors/6tfib_h12free_protein.pdb     ← H12 deleted
  data/docking/receptors/6tfib_h12free_protein.pdbqt
  data/docking/receptors/6tfib_docking_box.json
  data/docking/receptors/4ny9_protein.pdb
  data/docking/receptors/4ny9_protein.pdbqt
  data/docking/receptors/4ny9_docking_box.json

Prerequisites:
  pip install requests openbabel-wheel
"""

import json
import subprocess
import sys
from pathlib import Path

import requests
from loguru import logger

# H12 activation helix residue range to delete for the H12-free variant
H12_START = 410
H12_END   = 430

BOX_SIZE  = 28.0  # Å per side — matches Phase 1 2QD9 box

STRUCTURES = {
    "6TFI": {
        "chain": "A",
        "ligand_resnames": ["SR1", "SRM", "LIG"],  # SR12813 analog (try common names)
        "description": "SR12813 analog, 2.1 Å, H12-free variant also generated",
    },
    "4NY9": {
        "chain": "A",
        "ligand_resnames": ["RIF", "RFP", "RF1"],  # rifampicin (try common residue names)
        "description": "Rifampicin-bound, 2.2 Å, distinct pocket conformation",
    },
}


def download_pdb(pdb_id: str, out_dir: Path) -> Path:
    out_path = out_dir / f"{pdb_id.lower()}.pdb"
    if out_path.exists():
        logger.info(f"{pdb_id} already cached: {out_path}")
        return out_path
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    logger.info(f"Downloading {pdb_id}...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    out_path.write_text(resp.text)
    logger.info(f"Saved {pdb_id}: {len(resp.text):,} bytes")
    return out_path


def parse_pdb(pdb_path: Path, ligand_resnames: list[str], chain: str):
    protein_lines, ligand_lines = [], []
    found_ligand = None
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                if line[21] == chain or chain == "*":
                    protein_lines.append(line)
            elif line.startswith("HETATM"):
                resname = line[17:20].strip()
                line_chain = line[21]
                if resname in ligand_resnames and (line_chain == chain or chain == "*"):
                    ligand_lines.append(line)
                    found_ligand = resname
            elif line.startswith(("TER", "END")):
                protein_lines.append(line)

    if not ligand_lines:
        # Fallback: scan all chains for any of the ligand resnames
        logger.warning(f"Ligand {ligand_resnames} not found in chain {chain}, scanning all chains...")
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("HETATM"):
                    resname = line[17:20].strip()
                    if resname in ligand_resnames:
                        ligand_lines.append(line)
                        found_ligand = resname

    if not ligand_lines:
        # Last resort: use first non-water HETATM as ligand proxy for box center
        logger.warning(f"No ligand found with names {ligand_resnames}. Using first HETATM for box center.")
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("HETATM") and line[17:20].strip() not in ("HOH", "WAT", "GOL", "EDO", "PEG"):
                    ligand_lines.append(line)
                    found_ligand = line[17:20].strip()
                    if len(ligand_lines) > 20:
                        break

    logger.info(f"Protein: {len(protein_lines)} ATOM lines | Ligand ({found_ligand}): {len(ligand_lines)} HETATM lines")
    return protein_lines, ligand_lines


def compute_centroid(ligand_lines: list[str]) -> tuple[float, float, float]:
    xs, ys, zs = [], [], []
    for line in ligand_lines:
        element = line[76:78].strip() if len(line) > 76 else ""
        if element == "H":
            continue
        try:
            xs.append(float(line[30:38]))
            ys.append(float(line[38:46]))
            zs.append(float(line[46:54]))
        except (ValueError, IndexError):
            continue
    if not xs:
        raise ValueError("No coordinates in ligand lines")
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    cz = sum(zs) / len(zs)
    logger.info(f"Ligand centroid: ({cx:.2f}, {cy:.2f}, {cz:.2f})")
    return cx, cy, cz


def delete_h12(protein_lines: list[str]) -> list[str]:
    """Remove activation helix H12 residues 410-430 from protein lines."""
    filtered = []
    removed = 0
    for line in protein_lines:
        if line.startswith("ATOM"):
            try:
                resnum = int(line[22:26].strip())
                if H12_START <= resnum <= H12_END:
                    removed += 1
                    continue
            except ValueError:
                pass
        filtered.append(line)
    logger.info(f"H12-free: removed {removed} ATOM lines (residues {H12_START}-{H12_END})")
    return filtered


def convert_to_pdbqt(pdb_path: Path, pdbqt_path: Path) -> bool:
    if pdbqt_path.exists():
        logger.info(f"PDBQT already exists: {pdbqt_path}")
        return True

    # Try openbabel Python API
    try:
        from openbabel import openbabel as ob
        conv = ob.OBConversion()
        conv.SetInAndOutFormats("pdb", "pdbqt")
        mol = ob.OBMol()
        if conv.ReadFile(mol, str(pdb_path)) and mol.NumAtoms() > 0:
            mol.AddHydrogens()
            if conv.WriteFile(mol, str(pdbqt_path)):
                logger.info(f"PDBQT saved: {pdbqt_path} (openbabel Python API)")
                return True
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"openbabel Python API error: {e}")

    # Fallback: obabel CLI
    try:
        result = subprocess.run(
            ["obabel", str(pdb_path), "-O", str(pdbqt_path), "-xh"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0 and pdbqt_path.exists():
            logger.info(f"PDBQT saved: {pdbqt_path} (obabel CLI)")
            return True
        logger.warning(f"obabel CLI failed: {result.stderr[:200]}")
    except FileNotFoundError:
        pass

    logger.warning(
        f"Could not convert {pdb_path.name} to PDBQT.\n"
        "Install: pip install openbabel-wheel\n"
        "PDBQT is required for gnina docking."
    )
    return False


def main() -> None:
    out_dir = Path("data/docking/receptors")
    out_dir.mkdir(parents=True, exist_ok=True)

    for pdb_id, config in STRUCTURES.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {pdb_id}: {config['description']}")

        pdb_path = download_pdb(pdb_id, out_dir)
        protein_lines, ligand_lines = parse_pdb(
            pdb_path, config["ligand_resnames"], config["chain"]
        )

        # Save protein-only PDB
        prefix = pdb_id.lower()
        protein_pdb = out_dir / f"{prefix}_protein.pdb"
        protein_pdb.write_text("".join(protein_lines))
        logger.info(f"Protein PDB: {protein_pdb}")

        # Compute docking box from ligand centroid
        cx, cy, cz = compute_centroid(ligand_lines)
        box = {"center_x": cx, "center_y": cy, "center_z": cz,
               "size_x": BOX_SIZE, "size_y": BOX_SIZE, "size_z": BOX_SIZE}
        box_path = out_dir / f"{prefix}_docking_box.json"
        box_path.write_text(json.dumps(box, indent=2))
        logger.info(f"Docking box: {box_path}")

        # Convert to PDBQT
        pdbqt_path = out_dir / f"{prefix}_protein.pdbqt"
        convert_to_pdbqt(protein_pdb, pdbqt_path)

        # For 6TFI: also create H12-free variant
        if pdb_id == "6TFI":
            logger.info(f"\n--- 6TFIB H12-free variant (residues {H12_START}-{H12_END} deleted) ---")
            h12free_lines = delete_h12(protein_lines)
            h12free_pdb  = out_dir / "6tfib_h12free_protein.pdb"
            h12free_pdb.write_text("".join(h12free_lines))
            logger.info(f"H12-free PDB: {h12free_pdb}")

            h12free_pdbqt = out_dir / "6tfib_h12free_protein.pdbqt"
            convert_to_pdbqt(h12free_pdb, h12free_pdbqt)

            # H12-free uses same docking box as full 6TFIB
            h12free_box = out_dir / "6tfib_h12free_docking_box.json"
            h12free_box.write_text(json.dumps(box, indent=2))

    logger.info("\n=== Receptor preparation complete ===")
    logger.info("Files in data/docking/receptors/:")
    for f in sorted(out_dir.iterdir()):
        logger.info(f"  {f.name} ({f.stat().st_size // 1024} KB)")
    logger.info("\nNext:")
    logger.info("  pip install openbabel-wheel  (if PDBQT conversion failed)")
    logger.info("  Upload *.pdbqt files to Google Drive for Colab docking")
    logger.info("  python scripts/40_prepare_boltz_inputs.py")


if __name__ == "__main__":
    main()
