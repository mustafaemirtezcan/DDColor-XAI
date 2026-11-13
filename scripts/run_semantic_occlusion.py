from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from ddcolor_xai.datasets import read_manifest
from ddcolor_xai.image_io import mask_to_tensor, prepare_image, read_binary_mask
from ddcolor_xai.modeling import load_model
from ddcolor_xai.semantic_occlusion import (
    coarse_semantic_group,
    read_ade20k_labels,
    regional_output_change,
    select_semantic_regions,
)


GROUP_ORDER = ["sky", "road/path", "vegetation", "built surface", "object/other"]


def save_summary_figure(frame: pd.DataFrame, output_path: Path) -> None:
    matrix = frame.pivot_table(
        index="source_group",
        columns="target_group",
        values="influence",
        aggfunc="mean",
    ).reindex(index=GROUP_ORDER, columns=GROUP_ORDER)

    self_rows = frame[frame["is_self"]]
    context_rows = frame[~frame["is_self"]]
    summary = pd.DataFrame(
        {
            "Self region": self_rows.groupby("target_group")["influence"].mean(),
            "Context regions": context_rows.groupby("target_group")["influence"].mean(),
        }
    ).reindex(GROUP_ORDER)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    image = axes[0].imshow(matrix.values, cmap="magma", vmin=0)
    axes[0].set_xticks(range(len(GROUP_ORDER)), GROUP_ORDER, rotation=35, ha="right")
    axes[0].set_yticks(range(len(GROUP_ORDER)), GROUP_ORDER)
    axes[0].set_xlabel("Target color region")
    axes[0].set_ylabel("Occluded source region")
    axes[0].set_title("Mean semantic occlusion influence")
    for row in range(len(GROUP_ORDER)):
        for column in range(len(GROUP_ORDER)):
            value = matrix.values[row, column]
            if np.isfinite(value):
                axes[0].text(
                    column,
                    row,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color="white" if value > np.nanmean(matrix.values) else "black",
                    fontsize=8,
                )
    figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    summary.plot(kind="bar", ax=axes[1], color=["#2b6a9b", "#e67e22"])
    axes[1].set_title("Self versus contextual influence")
    axes[1].set_xlabel("Target semantic group")
    axes[1].set_ylabel("Bounded output change")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ADE20K semantic-region occlusion")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ddcolor-root", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="ddcolor_paper_tiny")
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-regions", type=int, default=8)
    parser.add_argument("--min-area", type=float, default=0.02)
    parser.add_argument("--device")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output).expanduser().resolve()
    annotations_root = Path(args.annotations).expanduser().resolve()
    labels = read_ade20k_labels(args.labels)
    model, device = load_model(args.ddcolor_root, args.model_name, args.device)
    rows = []

    samples = list(read_manifest(args.manifest, args.limit))
    for sample in tqdm(samples, desc="Semantic occlusion"):
        annotation_path = annotations_root / f"{sample.image_path.stem}.png"
        annotation = cv2.imread(str(annotation_path), cv2.IMREAD_GRAYSCALE)
        if annotation is None:
            continue
        prepared = prepare_image(sample.image_path, args.input_size, device)
        original_target = read_binary_mask(sample.mask_path)
        target_mask = mask_to_tensor(
            original_target, (args.input_size, args.input_size), device
        )
        resized_annotation = cv2.resize(
            annotation,
            (args.input_size, args.input_size),
            interpolation=cv2.INTER_NEAREST,
        )
        regions = select_semantic_regions(
            resized_annotation,
            labels,
            min_area_fraction=args.min_area,
            max_regions=args.max_regions,
            required_class=sample.category_id,
        )
        source_masks = [
            mask_to_tensor(region["mask"], (args.input_size, args.input_size), device)
            for region in regions
        ]
        influences = regional_output_change(
            model,
            prepared.model_input,
            target_mask,
            source_masks,
        )
        target_group = coarse_semantic_group(sample.label)
        for region, influence in zip(regions, influences):
            rows.append(
                {
                    "id": sample.sample_id,
                    "target_class_id": sample.category_id,
                    "target_label": sample.label,
                    "target_group": target_group,
                    "source_class_id": region["class_id"],
                    "source_label": region["label"],
                    "source_group": region["group"],
                    "source_area_fraction": region["area_fraction"],
                    "is_self": region["class_id"] == sample.category_id,
                    "influence": influence,
                }
            )

    frame = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_root / "region_influences.csv", index=False)

    per_image = []
    for sample_id, group in frame.groupby("id"):
        self_values = group.loc[group["is_self"], "influence"]
        context = group.loc[~group["is_self"]]
        strongest = group.loc[group["influence"].idxmax()]
        per_image.append(
            {
                "id": sample_id,
                "target_label": group["target_label"].iloc[0],
                "target_group": group["target_group"].iloc[0],
                "self_influence": (
                    float(self_values.max()) if not self_values.empty else np.nan
                ),
                "mean_context_influence": float(context["influence"].mean()),
                "max_context_influence": float(context["influence"].max()),
                "strongest_source_label": strongest["source_label"],
                "strongest_source_group": strongest["source_group"],
                "strongest_is_context": not bool(strongest["is_self"]),
            }
        )
    per_image_frame = pd.DataFrame(per_image)
    per_image_frame.to_csv(output_root / "per_image_summary.csv", index=False)

    pair_summary = (
        frame.groupby(["source_group", "target_group"])["influence"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    pair_summary.to_csv(output_root / "group_pair_summary.csv", index=False)
    save_summary_figure(frame, output_root / "semantic_occlusion.png")

    summary = {
        "images": int(per_image_frame.shape[0]),
        "region_interventions": int(frame.shape[0]),
        "mean_self_influence": float(per_image_frame["self_influence"].mean()),
        "mean_context_influence": float(per_image_frame["mean_context_influence"].mean()),
        "mean_max_context_influence": float(
            per_image_frame["max_context_influence"].mean()
        ),
        "strongest_is_context_fraction": float(
            per_image_frame["strongest_is_context"].mean()
        ),
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Results: {output_root}")


if __name__ == "__main__":
    main()
