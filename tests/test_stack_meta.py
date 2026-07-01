"""Sanity checks for Ridge / ElasticNet honest OOF meta-learners."""

import numpy as np

from openadmet.ensemble.stack import honest_oof_elasticnet, honest_oof_stack


def test_honest_oof_stack_matches_fold_assignments():
    rng = np.random.default_rng(0)
    n = 100
    folds = np.repeat(np.arange(5), 20)
    y = rng.normal(size=n)
    oof_preds = {
        "a": y + rng.normal(scale=0.1, size=n),
        "b": y + rng.normal(scale=0.15, size=n),
    }
    out = honest_oof_stack(oof_preds, y, folds, alphas=[1.0, 10.0])
    assert out.shape == (n,)
    assert np.isfinite(out).all()


def test_honest_oof_elasticnet_positive_coef_path():
    rng = np.random.default_rng(1)
    n = 80
    folds = np.repeat(np.arange(4), 20)
    y = rng.uniform(3.0, 8.0, size=n)
    oof_preds = {
        "a": y + rng.normal(scale=0.05, size=n),
        "b": y + rng.normal(scale=0.08, size=n),
    }
    out = honest_oof_elasticnet(oof_preds, y, folds, l1_ratios=[0.5])
    assert out.shape == (n,)
    assert np.isfinite(out).all()
