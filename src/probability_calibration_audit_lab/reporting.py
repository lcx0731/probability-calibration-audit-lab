"""Artifact serialization and compact audit plots."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl-calibration-audit"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {
    "raw": "#4C78A8",
    "platt": "#F58518",
    "isotonic": "#54A24B",
    "oracle": "#B279A2",
}
METHODS = ["raw", "platt", "isotonic", "oracle"]


def write_json(payload: dict[str, object], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def plot_reliability(curves: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharex=True, sharey=True)
    for axis, split in zip(axes, ["iid_test", "shifted_test"], strict=True):
        axis.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="ideal")
        subset = curves[curves["split"] == split]
        for method in METHODS:
            values = subset[subset["method"] == method]
            axis.plot(
                values["mean_probability"],
                values["event_rate"],
                marker="o",
                linewidth=1.8,
                markersize=4,
                color=COLORS[method],
                label=method,
            )
        axis.set_title(split.replace("_", " ").title())
        axis.set_xlabel("Mean predicted probability")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Observed event rate")
    axes[1].legend(frameon=False, fontsize=8)
    figure.suptitle("Equal-frequency reliability audit")
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def plot_probability_metrics(metrics: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    width = 0.18
    positions = np.arange(2)
    for index, method in enumerate(METHODS):
        subset = metrics[metrics["method"] == method].set_index("split")
        ordered = subset.loc[["iid_test", "shifted_test"]]
        offset = (index - 1.5) * width
        axes[0].bar(
            positions + offset,
            ordered["brier_score"],
            width,
            label=method,
            color=COLORS[method],
        )
        axes[1].bar(
            positions + offset,
            ordered["ece"],
            width,
            label=method,
            color=COLORS[method],
        )
    for axis, title in zip(axes, ["Brier score", "Expected calibration error"], strict=True):
        axis.set_xticks(positions, ["IID", "Shifted"])
        axis.set_title(title + " (lower is better)")
        axis.grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def plot_decisions(decisions: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    width = 0.18
    positions = np.arange(2)
    for index, method in enumerate(METHODS):
        subset = decisions[decisions["method"] == method].set_index("split")
        ordered = subset.loc[["iid_test", "shifted_test"]]
        axes[0].bar(
            positions + (index - 1.5) * width,
            ordered["test_cost_per_observation"],
            width,
            label=method,
            color=COLORS[method],
        )
    thresholds = decisions.drop_duplicates("method").set_index("method").loc[METHODS]
    axes[1].bar(METHODS, thresholds["selected_threshold"], color=[COLORS[x] for x in METHODS])
    axes[0].set_xticks(positions, ["IID", "Shifted"])
    axes[0].set_title("Held-out decision cost (lower is better)")
    axes[0].set_ylabel("Cost per observation")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_title("Threshold selected on calibration split")
    axes[1].set_ylabel("Probability threshold")
    axes[1].tick_params(axis="x", rotation=20)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
