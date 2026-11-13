from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ddcolor_xai.attention import QueryAttentionExplainer
from ddcolor_xai.datasets import read_manifest
from ddcolor_xai.faithfulness import deletion_insertion_auc
from ddcolor_xai.image_io import (
    grayscale_preview,
    mask_to_tensor,
    output_ab_to_bgr,
    prepare_image,
    read_binary_mask,
)
from ddcolor_xai.metrics import (
    energy_inside_mask,
    pointing_game,
    saliency_spearman,
)
from ddcolor_xai.modeling import load_model
from ddcolor_xai.visualization import overlay_heatmap


def normalized_entropy(saliency: np.ndarray) -> float:
    values = np.maximum(saliency.astype(np.float64), 0).reshape(-1)
    total = values.sum()
    if total <= 1e-12:
        return 1.0
    probabilities = values / total
    nonzero = probabilities > 0
    entropy = -(probabilities[nonzero] * np.log(probabilities[nonzero])).sum()
    return float(entropy / np.log(probabilities.size))


def save_map(root: Path, saliency: np.ndarray) -> None:
    root.parent.mkdir(parents=True, exist_ok=True)
    np.save(root.with_suffix(".npy"), saliency.astype(np.float32))
    cv2.imwrite(
        str(root.with_suffix(".png")),
        np.clip(np.rint(saliency * 255), 0, 255).astype(np.uint8),
    )


def save_panel(
    path: Path,
    original_bgr: np.ndarray,
    prediction_bgr: np.ndarray,
    mask: np.ndarray,
    attention: np.ndarray,
    gradcam: np.ndarray | None,
    rise: np.ndarray | None,
    title: str,
) -> None:
    panels = [
        ("Grayscale", grayscale_preview(original_bgr)),
        ("DDColor", prediction_bgr),
        ("Target region", cv2.cvtColor(mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)),
    ]
    if gradcam is not None:
        panels.append(("Region-Grad-CAM", overlay_heatmap(prediction_bgr, gradcam, mask)))
    if rise is not None:
        panels.append(("Region-RISE", overlay_heatmap(prediction_bgr, rise, mask)))
    panels.append(("Query attention", overlay_heatmap(prediction_bgr, attention, mask)))

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.4))
    for axis, (panel_title, image_bgr) in zip(axes, panels):
        axis.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        axis.set_title(panel_title)
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def summarize(frame: pd.DataFrame) -> dict:
    result = {}
    for column in frame.select_dtypes(include=[np.number]).columns:
        values = frame[column].dropna()
        result[column] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "count": int(values.count()),
        }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze DDColor query attention")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ddcolor-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--existing-results")
    parser.add_argument("--model-name", default="ddcolor_paper_tiny")
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--faithfulness-limit", type=int, default=30)
    parser.add_argument("--faithfulness-steps", type=int, default=10)
    parser.add_argument("--visual-limit", type=int, default=10)
    parser.add_argument("--device")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output).expanduser().resolve()
    existing_root = (
        Path(args.existing_results).expanduser().resolve()
        if args.existing_results
        else None
    )
    model, device = load_model(args.ddcolor_root, args.model_name, args.device)
    explainer = QueryAttentionExplainer(model)
    rows = []

    try:
        for index, sample in enumerate(
            tqdm(list(read_manifest(args.manifest, args.limit)), desc="Query attention")
        ):
            prepared = prepare_image(sample.image_path, args.input_size, device)
            original_mask = read_binary_mask(sample.mask_path)
            target_mask = mask_to_tensor(
                original_mask, (args.input_size, args.input_size), device
            )

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            result = explainer.explain(prepared.model_input, target_mask)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            saliency = result.saliency[0].cpu().numpy()
            resized_mask = cv2.resize(
                original_mask.astype(np.uint8),
                (args.input_size, args.input_size),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

            gradcam = rise = None
            if existing_root is not None:
                gradcam_path = existing_root / "maps" / sample.sample_id / "gradcam.npy"
                rise_path = existing_root / "maps" / sample.sample_id / "rise.npy"
                if gradcam_path.is_file():
                    gradcam = np.load(gradcam_path)
                if rise_path.is_file():
                    rise = np.load(rise_path)

            row = {
                "id": sample.sample_id,
                "dataset": sample.dataset,
                "label": sample.label,
                "attention_energy_inside": energy_inside_mask(saliency, resized_mask),
                "attention_pointing": pointing_game(saliency, resized_mask),
                "attention_entropy": normalized_entropy(saliency),
                "attention_seconds": elapsed,
                "active_queries_90pct": int(
                    np.searchsorted(
                        np.cumsum(
                            np.sort(result.query_importance.numpy())[::-1]
                            / max(float(result.query_importance.sum()), 1e-12)
                        ),
                        0.9,
                    )
                    + 1
                ),
            }
            if gradcam is not None:
                row["attention_gradcam_spearman"] = saliency_spearman(
                    saliency, gradcam
                )
            if rise is not None:
                row["attention_rise_spearman"] = saliency_spearman(saliency, rise)
            if index < args.faithfulness_limit:
                (
                    row["attention_deletion_auc"],
                    row["attention_insertion_auc"],
                    _,
                    _,
                ) = deletion_insertion_auc(
                    model,
                    prepared.model_input,
                    target_mask,
                    result.saliency[0],
                    steps=args.faithfulness_steps,
                )

            save_map(output_root / "maps" / sample.sample_id / "attention", saliency)
            if index < args.visual_limit:
                prediction = output_ab_to_bgr(
                    result.output_ab,
                    prepared.original_l,
                    prepared.original_bgr.shape[:2],
                )
                save_panel(
                    output_root / "figures" / f"{sample.sample_id}.png",
                    prepared.original_bgr,
                    prediction,
                    original_mask,
                    saliency,
                    gradcam,
                    rise,
                    f"{sample.dataset} | target: {sample.label}",
                )
            rows.append(row)
    finally:
        explainer.close()

    frame = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_root / "per_image.csv", index=False)
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summarize(frame), handle, indent=2)
    print(f"Processed {len(frame)} images")
    print(f"Results: {output_root}")


if __name__ == "__main__":
    main()
