from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .image_io import grayscale_preview


def overlay_heatmap(
    image_bgr: np.ndarray,
    saliency: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    resized = cv2.resize(
        saliency.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    heatmap = cv2.applyColorMap(
        np.clip(np.rint(resized * 255.0), 0, 255).astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    overlay = cv2.addWeighted(image_bgr, 0.55, heatmap, 0.45, 0.0)
    if mask is not None:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
    return overlay


def save_comparison_panel(
    output_path: str | Path,
    original_bgr: np.ndarray,
    prediction_bgr: np.ndarray,
    mask: np.ndarray,
    gradcam: np.ndarray,
    rise: np.ndarray,
    title: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    panels = [
        ("Ground truth", original_bgr),
        ("Grayscale input", grayscale_preview(original_bgr)),
        ("DDColor", prediction_bgr),
        ("Region-Grad-CAM", overlay_heatmap(prediction_bgr, gradcam, mask)),
        ("Region-RISE", overlay_heatmap(prediction_bgr, rise, mask)),
    ]

    figure, axes = plt.subplots(1, len(panels), figsize=(17, 3.6))
    for axis, (panel_title, image_bgr) in zip(axes, panels):
        axis.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        axis.set_title(panel_title)
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
