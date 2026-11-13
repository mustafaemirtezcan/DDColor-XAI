from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def read_ade20k_labels(path: str | Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 5:
                labels[int(fields[0])] = fields[4].strip()
    return labels


def coarse_semantic_group(label: str) -> str:
    name = label.lower()
    if "sky" in name:
        return "sky"
    if any(word in name for word in ("road", "route", "sidewalk", "path")):
        return "road/path"
    if any(
        word in name
        for word in (
            "tree",
            "grass",
            "plant",
            "flower",
            "field",
            "palm",
            "bush",
            "vegetation",
        )
    ):
        return "vegetation"
    if any(
        word in name
        for word in (
            "wall",
            "building",
            "floor",
            "ceiling",
            "window",
            "door",
            "house",
            "skyscraper",
        )
    ):
        return "built surface"
    return "object/other"


def select_semantic_regions(
    annotation: np.ndarray,
    labels: dict[int, str],
    min_area_fraction: float = 0.02,
    max_regions: int = 8,
    required_class: int | None = None,
) -> list[dict]:
    total = annotation.size
    counts = Counter(annotation.reshape(-1).tolist())
    candidates = [
        (int(class_id), int(count))
        for class_id, count in counts.items()
        if class_id not in (0, 255) and count / max(total, 1) >= min_area_fraction
    ]
    candidates.sort(key=lambda item: item[1], reverse=True)
    selected = candidates[:max_regions]
    if required_class is not None and required_class not in {item[0] for item in selected}:
        required_count = counts.get(required_class, 0)
        if required_count:
            selected.append((required_class, required_count))

    return [
        {
            "class_id": class_id,
            "label": labels.get(class_id, f"class_{class_id}"),
            "group": coarse_semantic_group(labels.get(class_id, f"class_{class_id}")),
            "area_fraction": count / total,
            "mask": annotation == class_id,
        }
        for class_id, count in selected
    ]


def regional_output_change(
    model: torch.nn.Module,
    model_input: torch.Tensor,
    target_mask: torch.Tensor,
    source_masks: list[torch.Tensor],
    batch_size: int = 8,
) -> list[float]:
    """Measure target-region color change after occluding each source region."""
    if not source_masks:
        return []
    if target_mask.ndim == 2:
        target_mask = target_mask[None, None]
    elif target_mask.ndim == 3:
        target_mask = target_mask[:, None]
    target_mask = target_mask.to(device=model_input.device, dtype=model_input.dtype)

    baseline = model_input.mean(dim=(-2, -1), keepdim=True)
    occluded_inputs = []
    for mask in source_masks:
        if mask.ndim == 2:
            mask = mask[None, None]
        elif mask.ndim == 3:
            mask = mask[:, None]
        mask = mask.to(device=model_input.device, dtype=torch.bool)
        mask = mask.expand_as(model_input)
        occluded_inputs.append(torch.where(mask, baseline, model_input))
    occluded = torch.cat(occluded_inputs, dim=0)

    with torch.inference_mode():
        reference = model(model_input).detach()
        if target_mask.shape[-2:] != reference.shape[-2:]:
            target_mask = F.interpolate(
                target_mask,
                size=reference.shape[-2:],
                mode="nearest",
            )
        denominator = target_mask.sum().clamp_min(1.0)
        reference_chroma = torch.sqrt(
            reference[:, 0:1].square() + reference[:, 1:2].square() + 1e-8
        )
        scale = ((reference_chroma * target_mask).sum() / denominator).clamp_min(1.0)

        changes: list[float] = []
        for start in range(0, occluded.shape[0], batch_size):
            outputs = model(occluded[start : start + batch_size])
            difference = (outputs - reference).abs().mean(dim=1, keepdim=True)
            per_item = (
                (difference * target_mask).sum(dim=(1, 2, 3)) / denominator
            )
            bounded = 1.0 - torch.exp(-per_item / scale)
            changes.extend(bounded.cpu().tolist())
    return [float(value) for value in changes]
