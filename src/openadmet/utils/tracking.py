"""W&B experiment tracking helpers."""

import os
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def init_wandb_run(
    project: str = "openadmet-pxr",
    name: Optional[str] = None,
    config: Optional[dict] = None,
    tags: Optional[list[str]] = None,
):
    """Initializes a W&B run. Returns the run object, or None if W&B unavailable."""
    if not WANDB_AVAILABLE:
        logger.warning("W&B not available — experiment not tracked")
        return None

    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        logger.warning("WANDB_API_KEY not set — logging locally (offline mode)")
        os.environ["WANDB_MODE"] = "offline"

    try:
        run = wandb.init(
            project=project,
            name=name,
            config=config or {},
            tags=tags or [],
            reinit=True,
        )
    except Exception as e:
        logger.warning(f"W&B init failed ({e}) — continuing without tracking")
        return None
    return run


def log_fold_metrics(run, fold_id: int, train_mae: float, val_mae: float, val_rae: float) -> None:
    if run is None:
        return
    run.log({"fold": fold_id, "train_mae": train_mae, "val_mae": val_mae, "val_rae": val_rae})


def log_oof_summary(run, oof_metrics: dict, feature_importance: Optional[pd.DataFrame] = None) -> None:
    if run is None:
        return
    scalars = {k: v for k, v in oof_metrics.items() if not isinstance(v, (dict, list))}
    run.log({f"oof_{k}": v for k, v in scalars.items()})
    if feature_importance is not None:
        top20 = feature_importance.head(20)
        run.log({"feature_importance": wandb.Table(dataframe=top20)})
