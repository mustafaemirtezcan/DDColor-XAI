from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


PAIRS = {
    "energy_inside": ("gradcam_energy_inside", "rise_energy_inside"),
    "deletion_auc": ("gradcam_deletion_auc", "rise_deletion_auc"),
    "insertion_auc": ("gradcam_insertion_auc", "rise_insertion_auc"),
}


def finite_or_none(value: float):
    return float(value) if np.isfinite(value) else None


def paired_comparison(frame: pd.DataFrame, left: str, right: str) -> dict:
    paired = frame[[left, right]].dropna()
    if paired.empty:
        return {
            "count": 0,
            "gradcam_mean": None,
            "rise_mean": None,
            "mean_difference_gradcam_minus_rise": None,
            "wilcoxon_p": None,
        }

    differences = paired[left] - paired[right]
    try:
        p_value = wilcoxon(paired[left], paired[right], alternative="two-sided").pvalue
    except ValueError:
        p_value = float("nan")
    return {
        "count": int(len(paired)),
        "gradcam_mean": finite_or_none(paired[left].mean()),
        "rise_mean": finite_or_none(paired[right].mean()),
        "mean_difference_gradcam_minus_rise": finite_or_none(differences.mean()),
        "wilcoxon_p": finite_or_none(p_value),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Grad-CAM and RISE results")
    parser.add_argument("csv", nargs="+", help="One or more per_image.csv files")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frames = [pd.read_csv(path) for path in args.csv]
    frame = pd.concat(frames, ignore_index=True)
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {}
    groups = [("overall", frame)]
    groups.extend((str(name), group) for name, group in frame.groupby("dataset"))
    for group_name, group in groups:
        report[group_name] = {
            metric: paired_comparison(group, left, right)
            for metric, (left, right) in PAIRS.items()
        }
        report[group_name]["pointing_accuracy"] = {
            "count": int(len(group)),
            "gradcam_mean": finite_or_none(group["gradcam_pointing"].mean()),
            "rise_mean": finite_or_none(group["rise_pointing"].mean()),
        }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
