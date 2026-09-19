"""Command-line entry point for a reproducible calibration audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from .evaluation import run_audit
from .reporting import plot_decisions, plot_probability_metrics, plot_reliability, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit probability calibration under covariate shift.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--calibration-size", type=int, default=2200)
    parser.add_argument("--test-size", type=int, default=3000)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--shift", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260919)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = run_audit(
        n_train=args.train_size,
        n_calibration=args.calibration_size,
        n_test=args.test_size,
        n_bootstrap=args.bootstrap,
        shift=args.shift,
        seed=args.seed,
    )
    result.metrics.to_csv(args.output_dir / "probability_metrics.csv", index=False)
    result.calibration_curves.to_csv(args.output_dir / "calibration_curves.csv", index=False)
    result.decisions.to_csv(args.output_dir / "decision_metrics.csv", index=False)
    result.bootstrap.to_csv(args.output_dir / "bootstrap_metrics.csv", index=False)
    write_json(result.summary, args.output_dir / "run_summary.json")
    plot_reliability(result.calibration_curves, args.output_dir / "reliability_audit.png")
    plot_probability_metrics(result.metrics, args.output_dir / "probability_metrics.png")
    plot_decisions(result.decisions, args.output_dir / "decision_audit.png")
    print(result.metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nDecision audit")
    print(result.decisions.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nArtifacts written to {args.output_dir.resolve()}")
    return 0
