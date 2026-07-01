"""scripts/14_compare_candidates.py merge logic."""

from pathlib import Path

import numpy as np
import pandas as pd


def test_compare_candidates_merge(tmp_path: Path) -> None:
    ids = [f"C{i}" for i in range(5)]
    yvals = [4.0, 5.0, 6.0, 7.0, 8.0]
    a = pd.DataFrame({"SMILES": ["x"] * 5, "Molecule Name": ids, "pEC50": yvals})
    # Same IDs with shuffled row order; pEC50 is +0.1 per compound
    b = pd.DataFrame(
        {
            "SMILES": ["y"] * 5,
            "Molecule Name": list(reversed(ids)),
            "pEC50": [v + 0.1 for v in reversed(yvals)],
        }
    )
    pa = Path(tmp_path / "a.csv")
    pb = Path(tmp_path / "b.csv")
    a.to_csv(pa, index=False)
    b.to_csv(pb, index=False)

    merged = a[["Molecule Name", "pEC50"]].merge(
        b[["Molecule Name", "pEC50"]],
        on="Molecule Name",
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    assert len(merged) == 5
    # B is +0.1 higher than A for every compound by construction
    assert np.allclose(merged["pEC50_b"] - merged["pEC50_a"], 0.1)
