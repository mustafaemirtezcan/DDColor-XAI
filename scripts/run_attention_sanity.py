from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ddcolor_xai.attention import QueryAttentionExplainer
from ddcolor_xai.datasets import read_manifest
from ddcolor_xai.image_io import mask_to_tensor, prepare_image, read_binary_mask
from ddcolor_xai.metrics import saliency_spearman
from ddcolor_xai.modeling import load_model, reset_module_parameters


def explain_samples(model, device, samples, input_size: int) -> dict[str, np.ndarray]:
    explainer = QueryAttentionExplainer(model)
    maps = {}
    try:
        for sample in tqdm(samples, desc="Attention sanity maps"):
            prepared = prepare_image(sample.image_path, input_size, device)
            mask = mask_to_tensor(
                read_binary_mask(sample.mask_path),
                (input_size, input_size),
                device,
            )
            maps[sample.sample_id] = (
                explainer.explain(prepared.model_input, mask)
                .saliency[0]
                .cpu()
                .numpy()
            )
    finally:
        explainer.close()
    return maps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query-attention sanity check")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ddcolor-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="ddcolor_paper_tiny")
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    samples = list(read_manifest(args.manifest, args.limit))

    trained_model, device = load_model(
        args.ddcolor_root, args.model_name, args.device
    )
    trained_maps = explain_samples(trained_model, device, samples, args.input_size)
    del trained_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    randomized_model, device = load_model(
        args.ddcolor_root, args.model_name, args.device
    )
    reset_module_parameters(randomized_model)
    randomized_model.eval()
    randomized_maps = explain_samples(
        randomized_model, device, samples, args.input_size
    )

    correlations = [
        saliency_spearman(trained_maps[sample.sample_id], randomized_maps[sample.sample_id])
        for sample in samples
    ]
    summary = {
        "count": len(correlations),
        "mean_spearman": float(np.mean(correlations)),
        "std_spearman": float(np.std(correlations, ddof=1)),
        "per_image": {
            sample.sample_id: float(correlation)
            for sample, correlation in zip(samples, correlations)
        },
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
