#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot average posture accuracy for each classifier."
    )
    parser.add_argument(
        "models",
        nargs="+",
        help="One or more .joblib model files saved by train_posture.py.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("posture_classifier_accuracy.png"),
        help="Output bar-chart image.",
    )
    return parser.parse_args()


def load_accuracy_records(model_paths: list[Path]) -> list[tuple[str, float]]:
    """Return (classifier_label, accuracy) records from saved posture models."""
    try:
        import joblib
    except ImportError as exc:
        raise SystemExit(f"Missing joblib/sklearn environment: {exc}") from exc

    records: list[tuple[str, float]] = []
    for path in model_paths:
        if not path.exists():
            print(f"Warning: model file not found: {path}")
            continue
        try:
            payload = joblib.load(path)
        except Exception as exc:
            print(f"Warning: could not load {path}: {exc}")
            continue

        label = str(payload.get("classifier_label") or payload.get("classifier", "unknown"))
        accuracy = float(payload.get("accuracy", 0.0))
        records.append((label, accuracy))

    if not records:
        raise SystemExit("No usable model files found.")
    return records


def average_accuracy_by_classifier(
    records: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Average repeated runs for each classifier."""
    from collections import defaultdict

    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)

    for label, accuracy in records:
        sums[label] += accuracy
        counts[label] += 1

    summary: list[tuple[str, float]] = []
    for label in sorted(sums):
        avg = sums[label] / counts[label]
        summary.append((label, avg))
    return summary


def plot_accuracy_bar_chart(summary: list[tuple[str, float]], out_path: Path) -> None:
    """Save a bar chart of average accuracy per classifier."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(f"Missing matplotlib: {exc}") from exc

    labels = [item[0] for item in summary]
    accuracies = [item[1] for item in summary]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.8), 5))
    colors = plt.cm.Blues([0.4 + 0.5 * (acc - min(accuracies)) / max(max(accuracies) - min(accuracies), 1e-9) for acc in accuracies])
    bars = ax.bar(labels, accuracies, color=colors, edgecolor="0.3", linewidth=0.8)

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Average Validation Accuracy")
    ax.set_title("Posture Classifier Comparison")
    ax.grid(axis="y", alpha=0.3)

    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.01,
            f"{acc:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Classifier accuracies: {dict(summary)}")


def main() -> int:
    args = parse_args()
    model_paths = [Path(item).expanduser().resolve() for item in args.models]
    records = load_accuracy_records(model_paths)
    summary = average_accuracy_by_classifier(records)
    plot_accuracy_bar_chart(summary, args.out)
    print(f"Saved plot: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
