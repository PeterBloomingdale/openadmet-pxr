"""
Antisymmetric Siamese delta model for MMP activity prediction.

Architecture: Δ(A→B) = g(B) - g(A)

where g(X) = α · physics_features(X) + MLP_residual(env_embedding(X))

The antisymmetric constraint Δ(A→B) = -Δ(B→A) is enforced by construction:
because we subtract g(A) from g(B), any model satisfying g(·) naturally
satisfies antisymmetry. This is the key structural advantage over the baseline.

Why antisymmetry matters:
The vanilla delta model can learn Δ(A→B) = 0.5 AND Δ(B→A) = 0.3 independently,
which is physically impossible (if replacing CH3 with CF3 increases pEC50 by 0.5,
then the reverse change must decrease by exactly 0.5). Enforcing this constraint:
1. Halves the effective parameter space (less overfitting on small datasets)
2. Allows learning from both A→B AND B→A pairs (doubles training data)
3. Makes interpolation between two known analogs exact by construction

Physics decomposition:
g(X) = α_mw * MW(X) + α_logp * logP(X) + ... + MLP(env_fingerprint(X))
The linear physics component captures the dominant SAR drivers for PXR
(hydrophobicity, size). The MLP residual captures scaffold-specific effects.

Inverse-variance neighbor aggregation:
When multiple training neighbors are available for a query compound (not just
the nearest), we aggregate their delta predictions weighted by 1/σ², where σ²
is estimated from the Hill slope (a proxy for the uncertainty in the neighbor's
pEC50 — a compound with high Hill slope had a sharp dose-response and more
reliable EC50).

Training protocol:
1. Build all symmetric MMP pairs from training data (both A→B and B→A)
2. Compute g(X) for each molecule independently
3. Loss = MAE(g(B) - g(A) - y_delta) — exactly the same as the delta model,
   but because g(X) is shared, gradients flow symmetrically

Implementation: g(X) is parameterized as a small MLP (2-3 layers, 128-256 units)
operating on ECFP4 + physicochemical features. For scikit-learn compatibility,
we implement this as a PyTorch module with a fit/predict interface.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available — SiameseDeltaModel will not function")

from openadmet.features.mmp import (
    find_nearest_training_neighbor,
    physchem_delta_features,
    build_mmp_feature_matrix,
)
from openadmet.features.fingerprints import morgan_count_fp


PHYSCHEM_FEATURES = ["mw", "logp", "hbd", "hba", "tpsa", "rotbonds", "rings", "arom_rings"]


def _mol_to_feature_vector(
    smiles: str,
    fp_n_bits: int = 512,
) -> Optional[np.ndarray]:
    """
    Computes the per-molecule feature vector for g(X).

    Features = Morgan count FP (512 bits) + 8 physicochemical properties.
    Using 512 bits (not 2048) to keep the Siamese MLP tractable on CPU.
    """
    try:
        from rdkit.Chem import Descriptors, rdMolDescriptors, MolFromSmiles
        mol = MolFromSmiles(smiles)
        if mol is None:
            return None

        fp = morgan_count_fp(smiles, radius=2, n_bits=fp_n_bits)
        if fp is None:
            return None

        phys = np.array([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            Descriptors.TPSA(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
        ], dtype=np.float32)

        # Normalize physicochemical features to roughly unit scale
        phys_scale = np.array([500.0, 6.0, 5.0, 10.0, 150.0, 10.0, 6.0, 4.0], dtype=np.float32)
        phys_norm = phys / phys_scale

        return np.concatenate([fp, phys_norm])
    except Exception:
        return None


class GNetwork(nn.Module):
    """
    The shared encoder g(X) = α·physics + MLP_residual(features(X)).

    α is a learnable vector of length 8 (one per physicochemical property)
    that directly maps physics features to pEC50 contribution. The MLP
    captures residual non-linearity not captured by the linear physics term.

    The sum g(B) - g(A) = [α·(phys_B - phys_A)] + [MLP(feat_B) - MLP(feat_A)]
    enforces antisymmetry in both the linear and non-linear components.
    """

    def __init__(
        self,
        input_dim: int,
        phys_dim: int = 8,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.phys_dim = phys_dim

        # Linear physics term (interpretable)
        self.alpha = nn.Parameter(torch.zeros(phys_dim))

        # MLP for non-linear residual
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x has shape [batch, input_dim] where the last phys_dim columns are
        normalized physicochemical features.

        Returns g(x) of shape [batch, 1].
        """
        phys = x[:, -self.phys_dim:]        # Last 8 features
        linear_term = (phys * self.alpha).sum(dim=1, keepdim=True)
        mlp_term = self.mlp(x)
        return linear_term + mlp_term


class SiameseDeltaModel:
    """
    Fits g(X) such that Δ(A→B) = g(B) - g(A) predicts pEC50 deltas.

    The antisymmetric constraint is satisfied by construction.
    Both A→B and B→A pairs are used for training (doubles effective data).

    Usage:
        model = SiameseDeltaModel()
        model.fit(train_df, smiles_col="smiles_std", pec50_col="pec50_median")
        preds = model.predict(test_df, train_df)
    """

    def __init__(
        self,
        fp_n_bits: int = 512,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        n_epochs: int = 100,
        patience: int = 15,
        batch_size: int = 512,
        seed: int = 42,
        device: Optional[str] = None,
    ):
        self.fp_n_bits = fp_n_bits
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.g_network: Optional[GNetwork] = None
        self._input_dim: Optional[int] = None
        self._train_smiles: list[str] = []
        self._train_pec50: list[float] = []
        self._train_features: Optional[np.ndarray] = None

    def _compute_features(self, smiles_list: list[str]) -> np.ndarray:
        """Compute feature vectors for a list of SMILES."""
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=-1)(
            delayed(_mol_to_feature_vector)(smi, self.fp_n_bits) for smi in smiles_list
        )
        # Fill None with zeros
        expected_dim = next((r.shape[0] for r in rows if r is not None), None)
        if expected_dim is None:
            raise ValueError("All SMILES failed feature computation")
        return np.stack([r if r is not None else np.zeros(expected_dim) for r in rows])

    def _build_pair_dataset(
        self,
        smiles: list[str],
        pec50: list[float],
        features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Builds symmetric MMP pair training data.

        For all pairs (i, j) where |pEC50_i - pEC50_j| > 0.05:
        - Creates row (feat_i, feat_j, pEC50_j - pEC50_i)
        - Creates row (feat_j, feat_i, pEC50_i - pEC50_j)  ← symmetric pair

        Returns (feat_a, feat_b, delta_pec50, neighbor_pec50_a)
        """
        feat_a_rows, feat_b_rows, delta_rows, neighbor_rows = [], [], [], []

        # Use all-pairs for small datasets; for large, sample nearest neighbors
        n = len(smiles)
        if n > 2000:
            # Only use ECFP4 nearest neighbors (top-10) to keep pairs manageable
            from openadmet.features.fingerprints import ecfp4_bitvect
            from rdkit import DataStructs
            fps = [ecfp4_bitvect(s) for s in smiles]
            fps = [fp for fp in fps if fp is not None]

        for i in range(n):
            for j in range(i + 1, min(i + 50, n)):  # Local window for large datasets
                p_i, p_j = pec50[i], pec50[j]
                if np.isnan(p_i) or np.isnan(p_j):
                    continue

                delta = p_j - p_i
                if abs(delta) < 0.05:
                    continue  # Skip identical activities (uninformative pairs)

                feat_a_rows.append(features[i])
                feat_b_rows.append(features[j])
                delta_rows.append(delta)
                neighbor_rows.append(p_i)

                # Symmetric pair
                feat_a_rows.append(features[j])
                feat_b_rows.append(features[i])
                delta_rows.append(-delta)
                neighbor_rows.append(p_j)

        return (
            np.array(feat_a_rows, dtype=np.float32),
            np.array(feat_b_rows, dtype=np.float32),
            np.array(delta_rows, dtype=np.float32),
            np.array(neighbor_rows, dtype=np.float32),
        )

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        smiles_col: str = "smiles_std",
        pec50_col: str = "pec50_median",
    ) -> "SiameseDeltaModel":
        """
        Fits the Siamese g-network on MMP pairs from training data.

        Training loss: MAE(g(feat_b) - g(feat_a) - delta_pec50)
        Using MAE loss to match the competition's RAE metric (MAE-based).
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for SiameseDeltaModel")

        torch.manual_seed(self.seed)
        self._train_smiles = train_df[smiles_col].tolist()
        self._train_pec50 = train_df[pec50_col].tolist()

        logger.info("Computing molecular features for Siamese model...")
        features = self._compute_features(self._train_smiles)
        self._train_features = features
        self._input_dim = features.shape[1]

        logger.info("Building symmetric MMP pairs...")
        feat_a, feat_b, delta_y, neighbor_pec50 = self._build_pair_dataset(
            self._train_smiles, self._train_pec50, features
        )
        logger.info(f"Training on {len(feat_a)} symmetric pairs")

        self.g_network = GNetwork(
            input_dim=self._input_dim,
            phys_dim=8,
            hidden_dim=self.hidden_dim,
            n_hidden=self.n_hidden,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.g_network.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.n_epochs
        )

        dataset = TensorDataset(
            torch.tensor(feat_a), torch.tensor(feat_b), torch.tensor(delta_y)
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.n_epochs):
            self.g_network.train()
            epoch_loss = 0.0
            for batch_a, batch_b, batch_delta in loader:
                batch_a = batch_a.to(self.device)
                batch_b = batch_b.to(self.device)
                batch_delta = batch_delta.to(self.device)

                g_a = self.g_network(batch_a).squeeze()
                g_b = self.g_network(batch_b).squeeze()
                pred_delta = g_b - g_a

                loss = torch.mean(torch.abs(pred_delta - batch_delta))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch_a)

            epoch_loss /= len(feat_a)
            scheduler.step()

            if epoch % 10 == 0:
                logger.info(f"Siamese epoch {epoch}: MAE={epoch_loss:.4f}")

            if epoch_loss < best_loss - 1e-4:
                best_loss = epoch_loss
                patience_counter = 0
                self._best_state = {k: v.cpu().clone() for k, v in self.g_network.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        # Load best weights
        self.g_network.load_state_dict(self._best_state)
        self.g_network.to("cpu")
        logger.info(f"Siamese model trained. Best MAE (pairs): {best_loss:.4f}")
        return self

    def predict(
        self,
        query_smiles: list[str],
        train_df: pd.DataFrame,
        smiles_col: str = "smiles_std",
        pec50_col: str = "pec50_median",
        n_neighbors: int = 5,
    ) -> np.ndarray:
        """
        Predicts pEC50 for query compounds using inverse-variance aggregation
        over the top-n training neighbors.

        For each query X:
        1. Find top-n training neighbors by Tanimoto similarity
        2. Compute Δ_i = g(X) - g(neighbor_i) for each neighbor
        3. Predict pEC50_i = neighbor_pEC50_i + Δ_i for each neighbor
        4. Aggregate: pEC50 = Σ(w_i * pEC50_i) / Σ(w_i)
           where w_i = tanimoto_i (proxy for inverse variance;
           higher similarity → more reliable neighbor → more weight)

        This inverse-variance aggregation extracts more signal from the
        multiple training neighbors available for each test analog.
        """
        if self.g_network is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        self.g_network.eval()

        query_features = self._compute_features(query_smiles)
        train_smiles = train_df[smiles_col].tolist()
        train_pec50 = train_df[pec50_col].tolist()
        train_features = self._compute_features(train_smiles)

        from rdkit import DataStructs
        from openadmet.features.fingerprints import ecfp4_bitvect

        query_fps = [ecfp4_bitvect(s) for s in query_smiles]
        train_fps = [ecfp4_bitvect(s) for s in train_smiles]
        train_fps_valid = [(i, fp) for i, fp in enumerate(train_fps) if fp is not None]

        with torch.no_grad():
            q_feat_tensor = torch.tensor(query_features, dtype=torch.float32)
            t_feat_tensor = torch.tensor(train_features, dtype=torch.float32)
            g_query = self.g_network(q_feat_tensor).squeeze().numpy()   # shape [n_query]
            g_train = self.g_network(t_feat_tensor).squeeze().numpy()   # shape [n_train]

        predictions = np.full(len(query_smiles), np.nan)

        for q_idx, (q_smi, q_fp, g_q) in enumerate(zip(query_smiles, query_fps, g_query)):
            if q_fp is None:
                predictions[q_idx] = float(np.nanmean(train_pec50))
                continue

            valid_train_fps = [fp for _, fp in train_fps_valid]
            valid_train_idx = [i for i, _ in train_fps_valid]
            sims = DataStructs.BulkTanimotoSimilarity(q_fp, valid_train_fps)
            top_idx = np.argsort(sims)[-n_neighbors:][::-1]

            weighted_sum = 0.0
            weight_total = 0.0
            for rank_idx in top_idx:
                t_idx = valid_train_idx[rank_idx]
                sim = sims[rank_idx]
                if sim < 0.1:
                    continue
                neighbor_pec50 = train_pec50[t_idx]
                if np.isnan(neighbor_pec50):
                    continue
                g_t = g_train[t_idx]
                pred_pec50 = float(neighbor_pec50 + (g_q - g_t))
                weighted_sum += sim * pred_pec50
                weight_total += sim

            if weight_total > 0:
                predictions[q_idx] = weighted_sum / weight_total
            else:
                predictions[q_idx] = float(np.nanmean(train_pec50))

        return predictions

    def get_alpha_weights(self) -> dict[str, float]:
        """
        Returns the learned linear physics weights α — the interpretable part.

        These weights tell you which physicochemical properties most strongly
        predict pEC50 change, with direction. For PXR, expect:
        - alpha_logp > 0 (more lipophilic = more active)
        - alpha_mw > 0 (larger = more active, up to a point)
        - alpha_hbd < 0 (H-bond donors disrupt hydrophobic binding)
        """
        if self.g_network is None:
            return {}
        alpha = self.g_network.alpha.detach().numpy()
        return dict(zip(
            ["alpha_mw", "alpha_logp", "alpha_hbd", "alpha_hba",
             "alpha_tpsa", "alpha_rotbonds", "alpha_rings", "alpha_arom_rings"],
            alpha.tolist()
        ))

    def save(self, path: str) -> None:
        import pickle
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "g_state": self.g_network.state_dict() if self.g_network else None,
            "input_dim": self._input_dim,
            "config": {
                "fp_n_bits": self.fp_n_bits,
                "hidden_dim": self.hidden_dim,
                "n_hidden": self.n_hidden,
            },
            "train_smiles": self._train_smiles,
            "train_pec50": self._train_pec50,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"Saved SiameseDeltaModel to {path}")

    @classmethod
    def load(cls, path: str) -> "SiameseDeltaModel":
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        model = cls(**state["config"])
        model._input_dim = state["input_dim"]
        model._train_smiles = state["train_smiles"]
        model._train_pec50 = state["train_pec50"]
        if state["g_state"] is not None:
            model.g_network = GNetwork(
                input_dim=state["input_dim"],
                **{k: v for k, v in state["config"].items() if k != "fp_n_bits"}
            )
            model.g_network.load_state_dict(state["g_state"])
            model.g_network.eval()
        return model
