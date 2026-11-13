from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PreparedImage:
    original_bgr: np.ndarray
    original_l: np.ndarray
    model_input: torch.Tensor


def prepare_image(
    image_path: str | Path,
    input_size: int,
    device: torch.device,
) -> PreparedImage:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image_float = image.astype(np.float32) / 255.0
    original_l = cv2.cvtColor(image_float, cv2.COLOR_BGR2LAB)[..., :1]

    resized = cv2.resize(
        image_float,
        (input_size, input_size),
        interpolation=cv2.INTER_AREA,
    )
    resized_l = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)[..., :1]
    gray_lab = np.concatenate(
        [resized_l, np.zeros_like(resized_l), np.zeros_like(resized_l)],
        axis=-1,
    )
    gray_rgb = cv2.cvtColor(gray_lab, cv2.COLOR_LAB2RGB)
    tensor = torch.from_numpy(gray_rgb.transpose(2, 0, 1))
    tensor = tensor.float().unsqueeze(0).to(device)

    return PreparedImage(
        original_bgr=image,
        original_l=original_l,
        model_input=tensor,
    )


def read_binary_mask(mask_path: str | Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")
    return mask > 0


def mask_to_tensor(
    mask: np.ndarray,
    size: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    height, width = size
    resized = cv2.resize(
        mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    return torch.from_numpy(resized).to(device=device, dtype=torch.float32)


def output_ab_to_bgr(
    output_ab: torch.Tensor,
    original_l: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    height, width = output_shape
    resized_ab = F.interpolate(
        output_ab.detach(),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0]
    ab = resized_ab.float().cpu().numpy().transpose(1, 2, 0)
    output_lab = np.concatenate([original_l, ab], axis=-1).astype(np.float32)
    output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
    return np.clip(np.rint(output_bgr * 255.0), 0, 255).astype(np.uint8)


def grayscale_preview(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

