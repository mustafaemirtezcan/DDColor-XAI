from __future__ import annotations

import torch
import torch.nn.functional as F


def _prepare_mask(mask: torch.Tensor, output_ab: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[:, None]
    mask = mask.to(device=output_ab.device, dtype=output_ab.dtype)
    if mask.shape[-2:] != output_ab.shape[-2:]:
        mask = F.interpolate(mask, size=output_ab.shape[-2:], mode="nearest")
    if mask.shape[0] == 1 and output_ab.shape[0] > 1:
        mask = mask.expand(output_ab.shape[0], -1, -1, -1)
    return mask


def region_score(
    output_ab: torch.Tensor,
    mask: torch.Tensor,
    mode: str = "chroma",
    target_ab: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return one scalar score per batch item for a selected output region."""
    region = _prepare_mask(mask, output_ab)
    denom = region.sum(dim=(1, 2, 3)).clamp_min(1.0)

    if mode == "chroma":
        values = torch.sqrt(output_ab[:, 0:1].square() + output_ab[:, 1:2].square() + 1e-8)
    elif mode == "a":
        values = output_ab[:, 0:1]
    elif mode == "b":
        values = output_ab[:, 1:2]
    elif mode == "negative_l1":
        if target_ab is None:
            raise ValueError("target_ab is required for negative_l1 mode")
        if target_ab.shape[-2:] != output_ab.shape[-2:]:
            target_ab = F.interpolate(
                target_ab,
                size=output_ab.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if target_ab.shape[0] == 1 and output_ab.shape[0] > 1:
            target_ab = target_ab.expand(output_ab.shape[0], -1, -1, -1)
        values = -(output_ab - target_ab).abs().mean(dim=1, keepdim=True)
    else:
        raise ValueError(f"Unsupported target mode: {mode}")

    return (values * region).sum(dim=(1, 2, 3)) / denom

