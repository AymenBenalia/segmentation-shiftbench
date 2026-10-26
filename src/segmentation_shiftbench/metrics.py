"""Foreground Dice and equal-specimen aggregation, with explicit missing-class handling."""
import numpy as np
import pandas as pd


def foreground_dice(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    prediction, truth = np.asarray(prediction), np.asarray(truth)
    if prediction.shape != truth.shape or prediction.ndim != 2:
        raise ValueError("Expected two 2D label maps with identical dimensions.")
    if not np.isin(prediction, (0, 1, 2)).all() or not np.isin(truth, (0, 1, 2)).all():
        raise ValueError("Label maps must contain only 0, 1 and 2.")
    scores = {}
    for label, name in ((1, "ring_dice"), (2, "core_dice")):
        actual, expected = prediction == label, truth == label
        denominator = actual.sum() + expected.sum()
        scores[name] = float(2 * (actual & expected).sum() / denominator) if denominator else float("nan")
    finite = [score for score in scores.values() if np.isfinite(score)]
    scores["foreground_dice"] = float(np.mean(finite)) if finite else float("nan")
    return scores


def summarize(rows: pd.DataFrame, *, seed: int, bootstrap: int = 1000) -> pd.DataFrame:
    """Aggregate views within specimens before resampling independent groups.

    Intervals are conditional on the synthetic test cohort and this trained
    model. They do not include training-seed or real-world acquisition uncertainty.
    """
    if bootstrap < 1:
        raise ValueError("bootstrap must be positive.")
    summary = []
    for (model, condition), subset in rows.groupby(["model", "condition"], sort=False):
        scores = subset.groupby("specimen")["foreground_dice"].mean().dropna().to_numpy()
        if not len(scores):
            raise ValueError("No defined foreground score is available.")
        # Same bootstrap seed means paired specimens get the same resampling draws.
        rng = np.random.default_rng(seed)
        means = scores[rng.integers(0, len(scores), size=(bootstrap, len(scores)))].mean(axis=1)
        low, high = np.quantile(means, [.025, .975]) if len(scores) > 1 else (np.nan, np.nan)
        summary.append({"model": model, "condition": condition, "family": subset.iloc[0]["family"],
                        "severity": int(subset.iloc[0]["severity"]), "mean_dice": float(scores.mean()),
                        "ci_low": float(low), "ci_high": float(high), "n_specimens": len(scores)})
    return pd.DataFrame(summary)
