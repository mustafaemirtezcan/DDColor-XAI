from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ddcolor_xai.datasets import read_manifest
from ddcolor_xai.explainers import RegionGradCAM, RegionRISE
from ddcolor_xai.image_io import mask_to_tensor, prepare_image, read_binary_mask
from ddcolor_xai.metrics import saliency_spearman
from ddcolor_xai.modeling import (
    load_model,
    reset_module_parameters,
    resolve_module,
)


def explain_samples(args, model, device, samples) -> dict[str, dict[str, np.ndarray]]:
    gradcam = RegionGradCAM(model, resolve_module(model, args.target_layer))
    rise = RegionRISE(
        model,
        num_masks=args.rise_masks,
        grid_size=args.rise_grid,
        batch_size=args.rise_batch_size,
        seed=args.seed,
    )
    maps: dict[str, dict[str, np.ndarray]] = {}
    try:
        for sample in tqdm(samples, desc="Sanity explanations"):
            prepared = prepare_image(sample.image_path, args.input_size, device)
            mask = read_binary_mask(sample.mask_path)
            region_mask = mask_to_tensor(
                mask,
                (args.input_size, args.input_size),
                device,
            )
            gradcam_result = gradcam.explain(prepared.model_input, region_mask)
            rise_result = rise.explain(prepared.model_input, region_mask)
            maps[sample.sample_id] = {
                "gradcam": gradcam_result.saliency[0].cpu().numpy(),
                "rise": rise_result.saliency[0].cpu().numpy(),
            }
    finally:
        gradcam.close()
    return maps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parameter randomization sanity check")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ddcolor-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="ddcolor_paper_tiny")
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--target-layer", default="decoder.layers.2")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--rise-masks", type=int, default=256)
    parser.add_argument("--rise-grid", type=int, default=8)
    parser.add_argument("--rise-batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    samples = list(read_manifest(args.manifest, limit=args.limit))

    model, device = load_model(args.ddcolor_root, args.model_name, args.device)
    trained_maps = explain_samples(args, model, device, samples)

    # Compare against the same architecture with fully randomized parameters.
    reset_module_parameters(model)
    randomized_maps = explain_samples(args, model, device, samples)

    rows = []
    for sample in samples:
        trained = trained_maps[sample.sample_id]
        randomized = randomized_maps[sample.sample_id]
        rows.append(
            {
                "id": sample.sample_id,
                "dataset": sample.dataset,
                "label": sample.label,
                "gradcam_spearman": saliency_spearman(
                    trained["gradcam"],
                    randomized["gradcam"],
                ),
                "rise_spearman": saliency_spearman(
                    trained["rise"],
                    randomized["rise"],
                ),
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "sanity_check.csv", index=False)
    summary = {
        "gradcam_spearman_mean": float(frame["gradcam_spearman"].mean()),
        "rise_spearman_mean": float(frame["rise_spearman"].mean()),
        "count": len(frame),
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
