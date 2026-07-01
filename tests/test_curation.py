"""Tests for the data curation pipeline."""

import pytest
import numpy as np
import pandas as pd

from openadmet.data.curation import (
    standardize_smiles,
    smiles_to_inchikey,
    inchikey_prefix,
    aggregate_replicates,
    deduplicate_train_test,
    handle_censored_values,
)


class TestStandardization:
    def test_valid_smiles_returns_canonical(self):
        result = standardize_smiles("c1ccccc1")  # Benzene
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_invalid_smiles_returns_none(self):
        assert standardize_smiles("NOT_A_SMILES") is None
        assert standardize_smiles("") is None
        assert standardize_smiles(None) is None

    def test_salt_stripped(self):
        # Sodium benzoate → benzoate (largest fragment)
        result = standardize_smiles("c1ccccc1C(=O)[O-].[Na+]")
        assert result is not None
        assert "Na" not in result

    def test_canonical_form_is_deterministic(self):
        smi1 = standardize_smiles("CC(O)=O")   # Acetic acid variant 1
        smi2 = standardize_smiles("OC(=O)C")   # Acetic acid variant 2
        assert smi1 == smi2


class TestInChIKey:
    def test_valid_smiles_returns_inchikey(self):
        ik = smiles_to_inchikey("c1ccccc1")
        assert ik is not None
        assert len(ik) == 27
        assert ik.count("-") == 2

    def test_inchikey_prefix_length(self):
        ik = smiles_to_inchikey("c1ccccc1")
        prefix = inchikey_prefix(ik)
        assert len(prefix) == 14

    def test_enantiomers_have_same_prefix(self):
        """Two enantiomers should share the same 14-char connectivity prefix."""
        ik_r = smiles_to_inchikey("[C@@H](F)(Cl)Br")
        ik_s = smiles_to_inchikey("[C@H](F)(Cl)Br")
        if ik_r and ik_s:
            assert inchikey_prefix(ik_r) == inchikey_prefix(ik_s)


class TestAggregation:
    def test_median_aggregation(self):
        df = pd.DataFrame({
            "inchikey_prefix": ["AAAA", "AAAA", "BBBB"],
            "pec50": [6.0, 7.0, 5.5],
        })
        result = aggregate_replicates(df, id_col="inchikey_prefix", value_col="pec50")
        aa_row = result[result["inchikey_prefix"] == "AAAA"].iloc[0]
        assert aa_row["pec50_median"] == pytest.approx(6.5)
        assert aa_row["n_replicates"] == 2

    def test_single_replicate(self):
        df = pd.DataFrame({
            "inchikey_prefix": ["AAAA"],
            "pec50": [6.5],
        })
        result = aggregate_replicates(df, id_col="inchikey_prefix", value_col="pec50")
        assert result["n_replicates"].iloc[0] == 1


class TestDeduplication:
    def test_removes_train_test_overlap(self):
        train = pd.DataFrame({"inchikey_prefix": ["AAAA", "BBBB", "CCCC"]})
        test = pd.DataFrame({"inchikey_prefix": ["BBBB", "DDDD"]})
        result = deduplicate_train_test(train, test)
        assert "BBBB" not in result["inchikey_prefix"].values
        assert "AAAA" in result["inchikey_prefix"].values
        assert len(result) == 2

    def test_no_overlap_unchanged(self):
        train = pd.DataFrame({"inchikey_prefix": ["AAAA", "BBBB"]})
        test = pd.DataFrame({"inchikey_prefix": ["CCCC", "DDDD"]})
        result = deduplicate_train_test(train, test)
        assert len(result) == 2


class TestCensoring:
    def test_marks_low_pec50_as_censored(self):
        df = pd.DataFrame({"pec50_median": [4.5, 6.0, 7.5, None]})
        result = handle_censored_values(df, inactive_pec50_threshold=5.0)
        assert result["is_censored"].iloc[0] == True   # 4.5 < 5.0
        assert result["is_censored"].iloc[1] == False  # 6.0 >= 5.0
        assert result["is_censored"].iloc[2] == False  # 7.5 >= 5.0
        assert result["is_censored"].iloc[3] == True   # None → censored

    def test_no_zero_imputation(self):
        """Censored compounds must NOT get a pEC50 value — they stay None."""
        df = pd.DataFrame({"pec50_median": [4.5, None]})
        result = handle_censored_values(df)
        # Values should be unchanged (not set to 5.0 or 0.0)
        assert pd.isna(result["pec50_median"].iloc[1])
