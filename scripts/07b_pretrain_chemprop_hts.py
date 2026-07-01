"""
Stage 1: Pretrain Chemprop message-passing on PXR HTS data.

Pretraining on broad HTS datasets before fine-tuning on the challenge data
mirrors what the rank-42 team did in their Sub 5 (HTS-pretrained Chemprop).
The HTS data covers a wider chemical space (~10,000 compounds) than the 4,135
challenge training compounds, giving the backbone broader SAR generalization.

Data sources:
  - Tox21 hPXR agonist qHTS (PubChem AID 1347033, ~7,871 compounds)
    Highest transfer value: same assay type (luciferase reporter), human PXR.
  - NCATS hPXR-luc qHTS (PubChem AID 720659, ~2,800 compounds)
    Independent lab confirmation; adds chemical diversity.

Target: pEC50 derived from AC50 (µM) via pEC50 = 6 - log10(AC50_µM).
Inactive compounds without AC50 are assigned pEC50 = 3.0 (100 µM threshold).

Architecture: MUST match configs/chemprop.yaml (d_h=300, depth=3) so MP weights
transfer directly into the challenge fine-tuning without projection layers.

Output: models/chemprop_pretrained/hts_pretrain_mp.pt
  Contains: {'hyper_parameters': {...}, 'state_dict': OrderedDict}
  (same format as CheMeleon .pt, loadable by scripts/07_train_chemprop.py)

Next: python scripts/07_train_chemprop.py --pretrain-hts
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger
from sklearn.model_selection import train_test_split
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.models import MPNN
from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN
from chemprop.nn.metrics import MAE
from chemprop.nn.transforms import UnscaleTransform

from openadmet.data.curation import standardize_smiles
from openadmet.data.loaders import load_pubchem_assay

INACTIVE_PEC50 = 3.0       # 100 µM — proxy pEC50 for compounds with no AC50
HTS_WEIGHT = 0.5            # task weight for HTS pretraining (single task)
PRETRAIN_MAX_EPOCHS = 30
PRETRAIN_PATIENCE = 8
PRETRAIN_LR = 1e-3
PRETRAIN_BATCH_SIZE = 64
PRETRAIN_VAL_FRAC = 0.1


def _load_hts_data() -> pd.DataFrame:
    """
    Load and combine Tox21 (AID 1347033) + NCATS (AID 720659) HTS data.

    Returns DataFrame with columns: [smiles_std, pec50]
    - Active compounds with AC50: pEC50 = 6 - log10(AC50_µM)
    - Inactive compounds: pEC50 = INACTIVE_PEC50 (3.0)
    Compounds without valid SMILES are dropped.
    """
    dfs = []
    for aid in [1347033, 720659]:
        try:
            df = load_pubchem_assay(aid=aid, cache_dir="data/raw")
            logger.info(f"  AID {aid}: {len(df)} compounds loaded")
            dfs.append(df)
        except Exception as e:
            logger.warning(f"  AID {aid} failed to download: {e}. Skipping.")

    if not dfs:
        raise RuntimeError(
            "No HTS data available. Check network access or run scripts/01_download_data.py first."
        )

    hts = pd.concat(dfs, ignore_index=True)

    # Assign pEC50: use measured value or inactive proxy
    hts["pec50_hts"] = hts["pec50"].where(
        hts["pec50"].notna(), other=INACTIVE_PEC50
    )
    hts["pec50_hts"] = hts["pec50_hts"].clip(lower=2.0, upper=9.0)

    # Standardize SMILES to match training data preprocessing
    logger.info("Standardizing HTS SMILES...")
    hts["smiles_std"] = [standardize_smiles(s) for s in hts["smiles"].astype(str)]
    hts = hts[hts["smiles_std"].notna()].copy()

    # Deduplicate by InChIKey prefix (same as master_train curation)
    from rdkit.Chem import MolFromSmiles, inchi
    def _inchikey14(smi):
        mol = MolFromSmiles(smi)
        if mol is None:
            return None
        try:
            return inchi.MolToInchiKey(mol)[:14]
        except Exception:
            return None

    hts["ik14"] = hts["smiles_std"].map(_inchikey14)
    hts = hts[hts["ik14"].notna()]
    # Keep median pec50 for duplicates
    hts = hts.groupby("ik14").agg(
        smiles_std=("smiles_std", "first"),
        pec50_hts=("pec50_hts", "median"),
    ).reset_index(drop=True)

    logger.info(f"HTS dataset after dedup: {len(hts)} compounds")
    return hts[["smiles_std", "pec50_hts"]]


def main() -> None:
    with open("configs/chemprop.yaml") as f:
        config = yaml.safe_load(f)
    arch = config["architecture"]

    out = Path("models/chemprop_pretrained")
    out.mkdir(parents=True, exist_ok=True)
    mp_save_path = out / "hts_pretrain_mp.pt"

    if mp_save_path.exists():
        logger.info(f"HTS pretrained MP weights already exist: {mp_save_path}")
        logger.info("Delete the file to re-pretrain. Next: python scripts/07_train_chemprop.py --pretrain-hts")
        return

    logger.info("Loading HTS PXR data (Tox21 AID 1347033 + NCATS AID 720659)...")
    hts_df = _load_hts_data()

    smiles = hts_df["smiles_std"].tolist()
    targets = hts_df["pec50_hts"].values.astype(np.float32)

    # Stratified train/val split on the HTS data
    idx_tr, idx_va = train_test_split(
        np.arange(len(smiles)),
        test_size=PRETRAIN_VAL_FRAC,
        random_state=42,
    )
    logger.info(f"HTS pretrain split: {len(idx_tr)} train / {len(idx_va)} val")

    featurizer = SimpleMoleculeMolGraphFeaturizer()

    def _make_dps(idxs):
        return [MoleculeDatapoint.from_smi(smiles[i], np.array([targets[i]])) for i in idxs]

    train_dset = MoleculeDataset(_make_dps(idx_tr), featurizer)
    val_dset   = MoleculeDataset(_make_dps(idx_va),   featurizer)
    scaler = train_dset.normalize_targets()
    val_dset.normalize_targets(scaler)

    train_loader = build_dataloader(train_dset, batch_size=PRETRAIN_BATCH_SIZE, num_workers=0, shuffle=True)
    val_loader   = build_dataloader(val_dset,   batch_size=PRETRAIN_BATCH_SIZE, num_workers=0, shuffle=False)

    d_h = arch["hidden_size"]  # 300 — matches challenge fine-tune architecture
    mp  = BondMessagePassing(d_h=d_h, depth=arch["depth"], dropout=arch["dropout"])
    agg = MeanAggregation()
    tw  = torch.tensor([HTS_WEIGHT], dtype=torch.float)
    criterion = MAE(task_weights=tw)
    output_tf = UnscaleTransform.from_standard_scaler(scaler)
    ffn = RegressionFFN(n_tasks=1, input_dim=d_h, hidden_dim=d_h, n_layers=1,
                        dropout=arch["dropout"], criterion=criterion, output_transform=output_tf)

    model = MPNN(
        message_passing=mp,
        agg=agg,
        predictor=ffn,
        init_lr=PRETRAIN_LR / 100,
        max_lr=PRETRAIN_LR,
        final_lr=PRETRAIN_LR / 1000,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(out),
        filename="hts_pretrain_full",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_cb = EarlyStopping(monitor="val_loss", patience=PRETRAIN_PATIENCE, mode="min")

    trainer = pl.Trainer(
        max_epochs=PRETRAIN_MAX_EPOCHS,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False,
        callbacks=[ckpt_cb, early_cb],
    )
    logger.info(f"Pretraining Chemprop on {len(hts_df)} HTS compounds (d_h={d_h}, max_epochs={PRETRAIN_MAX_EPOCHS}) ...")
    trainer.fit(model, train_loader, val_loader)

    # Load best checkpoint and save only the message-passing weights
    best_model = MPNN.load_from_checkpoint(ckpt_cb.best_model_path)
    mp_hyper = {"d_h": d_h, "depth": arch["depth"], "dropout": arch["dropout"]}
    torch.save(
        {"hyper_parameters": mp_hyper, "state_dict": best_model.message_passing.state_dict()},
        mp_save_path,
    )
    logger.info(f"HTS pretrained MP weights saved: {mp_save_path}")
    logger.info("Next: python scripts/07_train_chemprop.py --pretrain-hts")


if __name__ == "__main__":
    main()
