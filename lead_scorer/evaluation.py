"""Evaluation metrics for the lead-ranking problem.

Top-decile precision is the headline metric — sales reps work from the top of
the ranked queue, so what matters is whether the highest-scored leads actually
converted. ROC-AUC and PR-AUC are reported for completeness.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


@dataclass
class RankingMetrics:
    """Container for the metrics emitted per model run."""

    roc_auc: float
    pr_auc: float
    top_decile_precision: float
    top_quintile_precision: float
    brier_score: float
    n_samples: int
    n_positives: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "top_decile_precision": self.top_decile_precision,
            "top_quintile_precision": self.top_quintile_precision,
            "brier_score": self.brier_score,
            "n_samples": self.n_samples,
            "n_positives": self.n_positives,
        }


def precision_at_top_k_fraction(y_true: np.ndarray, y_score: np.ndarray, fraction: float) -> float:
    """Precision computed over the top `fraction` of items by score.

    Args:
        y_true: 0/1 ground-truth labels.
        y_score: predicted scores (higher = more likely positive).
        fraction: in (0, 1]. e.g. 0.10 -> top decile.

    Returns:
        Precision in [0, 1]. Returns 0.0 if k rounds to zero.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    n = len(y_true)
    k = max(1, int(round(n * fraction)))
    # argsort descending, take top-k indices.
    top_idx = np.argsort(-y_score, kind="stable")[:k]
    return float(np.mean(y_true[top_idx]))


def evaluate(y_true: np.ndarray, y_score: np.ndarray) -> RankingMetrics:
    """Compute all reported metrics for a single model's predictions."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    return RankingMetrics(
        roc_auc=float(roc_auc_score(y_true, y_score)),
        pr_auc=float(average_precision_score(y_true, y_score)),
        top_decile_precision=precision_at_top_k_fraction(y_true, y_score, 0.10),
        top_quintile_precision=precision_at_top_k_fraction(y_true, y_score, 0.20),
        brier_score=float(brier_score_loss(y_true, y_score)),
        n_samples=int(len(y_true)),
        n_positives=int(y_true.sum()),
    )


def save_calibration_plot(
    y_true: np.ndarray,
    y_score: np.ndarray,
    path: str | Path,
    n_bins: int = 10,
    title: str = "Calibration",
) -> Path:
    """Save a reliability diagram comparing predicted vs observed frequency."""
    # Lazy import — keeps matplotlib optional for headless test runs.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.plot(prob_pred, prob_true, "o-", label="model")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
