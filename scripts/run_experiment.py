from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ddcolor_xai.datasets import read_manifest
from ddcolor_xai.explainers import RegionGradCAM, RegionRISE
from ddcolor_xai.faithfulness import deletion_insertion_auc
from ddcolor_xai.image_io import (
    mask_to_tensor,
    output_ab_to_bgr,
    prepare_image,
    read_binary_mask,
)
from ddcolor_xai.metrics import (
    OptionalLPIPS,
    energy_inside_mask,
    image_quality_metrics,
    pointing_game,
)
from ddcolor_xai.modeling import load_model, resolve_module
from ddcolor_xai.visualization import save_comparison_panel


def summarize(frame: pd.DataFrame) -> dict:
    numeric = frame.select_dtypes(include=[np.number])

    def finite_or_none(value):
        return float(value) if np.isfinite(value) else None

    return {
        column: {
            "mean": finite_or_none(numeric[column].mean()),
            "std": finite_or_none(numeric[column].std(ddof=1)),
            "count": int(numeric[column].count()),
        }
        for column in numeric.columns
    }


def save_saliency(path: Path, saliency: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix(".npy"), saliency.astype(np.float32))
    image = np.clip(np.rint(saliency * 255.0), 0, 255).astype(np.uint8)
    cv2.imwrite(str(path.with_suffix(".png")), image)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DDColor XAI experiments")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ddcolor-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="ddcolor_paper_tiny")
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--target-layer", default="decoder.layers.2")
    parser.add_argument("--target-mode", default="chroma", choices=["chroma", "a", "b"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rise-masks", type=int, default=256)
    parser.add_argument("--rise-grid", type=int, default=8)
    parser.add_argument("--rise-batch-size", type=int, default=8)
    parser.add_argument("--faithfulness-limit", type=int, default=30)
    parser.add_argument("--faithfulness-steps", type=int, default=10)
    parser.add_argument("--visual-limit", type=int, default=20)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-lpips", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_root = Path(args.output).expanduser().resolve()
    maps_root = output_root / "maps"
    figures_root = output_root / "figures"
    output_root.mkdir(parents=True, exist_ok=True)

    model, device = load_model(
        args.ddcolor_root,
        model_name=args.model_name,
        device=args.device,
    )
    target_layer = resolve_module(model, args.target_layer)
    gradcam = RegionGradCAM(model, target_layer)
    rise = RegionRISE(
        model,
        num_masks=args.rise_masks,
        grid_size=args.rise_grid,
        batch_size=args.rise_batch_size,
        seed=args.seed,
    )
    lpips_metric = None if args.disable_lpips else OptionalLPIPS(device)

    samples = list(read_manifest(args.manifest, limit=args.limit))
    if not samples:
        raise RuntimeError(f"No samples found in manifest: {args.manifest}")
    rows: list[dict] = []

    try:
        for index, sample in enumerate(tqdm(samples, desc="DDColor XAI")):
            prepared = prepare_image(sample.image_path, args.input_size, device)
            original_mask = read_binary_mask(sample.mask_path)
            region_mask = mask_to_tensor(
                original_mask,
                (args.input_size, args.input_size),
                device,
            )

            start = time.perf_counter()
            gradcam_result = gradcam.explain(
                prepared.model_input,
                region_mask,
                target_mode=args.target_mode,
            )
            gradcam_seconds = time.perf_counter() - start

            start = time.perf_counter()
            rise_result = rise.explain(
                prepared.model_input,
                region_mask,
                target_mode=args.target_mode,
            )
            rise_seconds = time.perf_counter() - start
            gradcam_map = gradcam_result.saliency[0].cpu().numpy()
            rise_map = rise_result.saliency[0].cpu().numpy()

            prediction = output_ab_to_bgr(
                gradcam_result.output_ab,
                prepared.original_l,
                prepared.original_bgr.shape[:2],
            )
            resized_mask = cv2.resize(
                original_mask.astype(np.uint8),
                (args.input_size, args.input_size),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

            quality = image_quality_metrics(prediction, prepared.original_bgr)
            if lpips_metric is not None:
                quality["lpips"] = lpips_metric(prediction, prepared.original_bgr)

            row = {
                "id": sample.sample_id,
                "dataset": sample.dataset,
                "label": sample.label,
                **quality,
                "target_score": gradcam_result.score,
                "gradcam_seconds": gradcam_seconds,
                "rise_seconds": rise_seconds,
                "gradcam_energy_inside": energy_inside_mask(gradcam_map, resized_mask),
                "gradcam_pointing": pointing_game(gradcam_map, resized_mask),
                "rise_energy_inside": energy_inside_mask(rise_map, resized_mask),
                "rise_pointing": pointing_game(rise_map, resized_mask),
            }

            if index < args.faithfulness_limit:
                (
                    row["gradcam_deletion_auc"],
                    row["gradcam_insertion_auc"],
                    _,
                    _,
                ) = deletion_insertion_auc(
                    model,
                    prepared.model_input,
                    region_mask,
                    gradcam_result.saliency[0],
                    target_mode=args.target_mode,
                    steps=args.faithfulness_steps,
                )
                (
                    row["rise_deletion_auc"],
                    row["rise_insertion_auc"],
                    _,
                    _,
                ) = deletion_insertion_auc(
                    model,
                    prepared.model_input,
                    region_mask,
                    rise_result.saliency[0],
                    target_mode=args.target_mode,
                    steps=args.faithfulness_steps,
                )

            sample_map_root = maps_root / sample.sample_id
            save_saliency(sample_map_root / "gradcam", gradcam_map)
            save_saliency(sample_map_root / "rise", rise_map)

            if index < args.visual_limit:
                save_comparison_panel(
                    figures_root / f"{sample.sample_id}.png",
                    prepared.original_bgr,
                    prediction,
                    original_mask,
                    gradcam_map,
                    rise_map,
                    title=f"{sample.dataset} | {sample.label}",
                )
            rows.append(row)
    finally:
        gradcam.close()

    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "per_image.csv", index=False)
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summarize(frame), handle, indent=2)

    print(f"Processed {len(frame)} images")
    print(f"Results: {output_root}")


if __name__ == "__main__":
    main()
