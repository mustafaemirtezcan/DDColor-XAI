from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import normalized_auc


def _region_similarity(
    outputs: torch.Tensor,
    reference_ab: torch.Tensor,
    region_mask: torch.Tensor,
) -> torch.Tensor:
    if region_mask.ndim == 2:
        region_mask = region_mask[None, None]
    elif region_mask.ndim == 3:
        region_mask = region_mask[:, None]
    region_mask = region_mask.to(device=outputs.device, dtype=outputs.dtype)
    if region_mask.shape[-2:] != outputs.shape[-2:]:
        region_mask = F.interpolate(
            region_mask,
            size=outputs.shape[-2:],
            mode="nearest",
        )
    if region_mask.shape[0] == 1 and outputs.shape[0] > 1:
        region_mask = region_mask.expand(outputs.shape[0], -1, -1, -1)

    if reference_ab.shape[0] == 1 and outputs.shape[0] > 1:
        reference_ab = reference_ab.expand(outputs.shape[0], -1, -1, -1)

    denominator = region_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    difference = (outputs - reference_ab).abs().mean(dim=1, keepdim=True)
    mean_difference = (difference * region_mask).sum(dim=(1, 2, 3)) / denominator

    reference_chroma = torch.sqrt(
        reference_ab[:, 0:1].square() + reference_ab[:, 1:2].square() + 1e-8
    )
    scale = (reference_chroma * region_mask).sum(dim=(1, 2, 3)) / denominator
    scale = scale.clamp_min(1.0)
    return torch.exp(-mean_difference / scale)


def _similarity_batch(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    reference_ab: torch.Tensor,
    region_mask: torch.Tensor,
    batch_size: int,
) -> list[float]:
    scores: list[float] = []
    with torch.inference_mode():
        for start in range(0, inputs.shape[0], batch_size):
            output = model(inputs[start : start + batch_size])
            batch_scores = _region_similarity(output, reference_ab, region_mask)
            scores.extend(batch_scores.detach().cpu().tolist())
    return scores


def deletion_insertion_auc(
    model: torch.nn.Module,
    model_input: torch.Tensor,
    region_mask: torch.Tensor,
    saliency: torch.Tensor,
    target_mode: str = "chroma",
    steps: int = 10,
    batch_size: int = 4,
) -> tuple[float, float, list[float], list[float]]:
    """Compute deletion/insertion curves for a scalar colorization target."""
    if model_input.shape[0] != 1:
        raise ValueError("Faithfulness evaluation expects one image")

    _, _, height, width = model_input.shape
    flat_order = torch.argsort(saliency.reshape(-1), descending=True)
    baseline = model_input.mean(dim=(-2, -1), keepdim=True).expand_as(model_input)
    original = model_input
    with torch.inference_mode():
        reference_ab = model(original).detach()

    deletion_inputs = []
    insertion_inputs = []
    total = height * width

    for step in range(steps + 1):
        count = int(round(total * step / steps))
        selected = flat_order[:count]
        selector = torch.zeros(total, device=model_input.device, dtype=torch.bool)
        selector[selected] = True
        selector = selector.view(1, 1, height, width).expand_as(model_input)
        deletion_inputs.append(torch.where(selector, baseline, original))
        insertion_inputs.append(torch.where(selector, original, baseline))

    deletion_tensor = torch.cat(deletion_inputs, dim=0)
    insertion_tensor = torch.cat(insertion_inputs, dim=0)
    deletion_curve = _similarity_batch(
        model,
        deletion_tensor,
        reference_ab,
        region_mask,
        batch_size,
    )
    insertion_curve = _similarity_batch(
        model,
        insertion_tensor,
        reference_ab,
        region_mask,
        batch_size,
    )

    return (
        normalized_auc(deletion_curve),
        normalized_auc(insertion_curve),
        deletion_curve,
        insertion_curve,
    )
