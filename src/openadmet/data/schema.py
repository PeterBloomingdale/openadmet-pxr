"""
Data contracts for the OpenADMET PXR pipeline.

These Pydantic models define the shape of data at each pipeline stage.
Everything downstream depends on these — change them carefully.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class RawRecord(BaseModel):
    """One compound measurement as downloaded from a source."""

    smiles: str
    pec50: Optional[float] = None       # None = not measured or inactive at top dose
    emax: Optional[float] = None        # Maximum efficacy (% of reference, e.g. rifampicin)
    hill_slope: Optional[float] = None  # Hill coefficient from dose-response curve fit
    counter_pec50: Optional[float] = None  # pEC50 in PXR-null counter-screen
    counter_emax: Optional[float] = None
    source: str                         # "openadmet" | "pubchem_1347033" | "chembl" | ...
    assay_id: str                       # Source-specific identifier
    compound_id: Optional[str] = None   # Source compound identifier

    @field_validator("pec50", "counter_pec50", mode="before")
    @classmethod
    def clamp_pec50(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return v
        # pEC50 outside 3-11 almost certainly a unit error or data entry problem
        if v < 3.0 or v > 11.0:
            return None
        return v


class CuratedRecord(BaseModel):
    """One compound after the full curation pipeline."""

    inchikey: str                       # 27-char full InChIKey
    inchikey_prefix: str                # First 14 chars (connectivity layer only)
    smiles_std: str                     # Standardized canonical SMILES
    pec50_median: Optional[float] = None
    counter_pec50_median: Optional[float] = None
    emax_median: Optional[float] = None
    hill_slope_median: Optional[float] = None
    n_replicates: int = 1
    pec50_spread: Optional[float] = None  # std across replicates; >0.5 warrants inspection
    is_censored: bool = False           # True = inactive at top dose (right-censored)
    source: str
    source_weight: float = Field(ge=0.0, le=1.0)  # 1.0 = primary Octant data
    split: Optional[str] = None         # "train" | "test" | assigned later
