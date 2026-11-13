from __future__ import annotations

import cv2
import numpy as np
import torch
from scipy.stats import spearmanr
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def colorfulness(image_bgr: np.ndarray) -> float:
    b, g, r = cv2.split(image_bgr.astype(np.float64))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    std_root = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean_root = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(std_root + 0.3 * mean_root)


def image_quality_metrics(
    prediction_bgr: np.ndarray,
    reference_bgr: np.ndarray,
) -> dict[str, float]:
    prediction_rgb = cv2.cvtColor(prediction_bgr, cv2.COLOR_BGR2RGB)
    reference_rgb = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2RGB)

    psnr = peak_signal_noise_ratio(
        reference_rgb,
        prediction_rgb,
        data_range=255,
    )
    ssim = structural_similarity(
        reference_rgb,
        prediction_rgb,
        channel_axis=2,
        data_range=255,
    )

    prediction_lab = rgb2lab(prediction_rgb.astype(np.float32) / 255.0)
    reference_lab = rgb2lab(reference_rgb.astype(np.float32) / 255.0)
    delta_e = deltaE_ciede2000(reference_lab, prediction_lab)

    prediction_cf = colorfulness(prediction_bgr)
    reference_cf = colorfulness(reference_bgr)
    return {
        "psnr": float(psnr),
        "ssim": float(ssim),
        "delta_e_00": float(delta_e.mean()),
        "colorfulness_pred": prediction_cf,
        "colorfulness_gt": reference_cf,
        "colorfulness_abs_error": abs(prediction_cf - reference_cf),
    }


def energy_inside_mask(saliency: np.ndarray, mask: np.ndarray) -> float:
    saliency = np.maximum(saliency.astype(np.float64), 0.0)
    mask = mask.astype(bool)
    total = saliency.sum()
    if total <= 1e-12:
        return 0.0
    return float(saliency[mask].sum() / total)


def pointing_game(saliency: np.ndarray, mask: np.ndarray) -> float:
    row, col = np.unravel_index(np.argmax(saliency), saliency.shape)
    return float(bool(mask[row, col]))


def normalized_auc(values: list[float] | np.ndarray) -> float:
    curve = np.asarray(values, dtype=np.float64)
    if curve.size < 2:
        return float("nan")
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(curve, dx=1.0 / (curve.size - 1)))
    return float(np.trapz(curve, dx=1.0 / (curve.size - 1)))


def saliency_spearman(first: np.ndarray, second: np.ndarray) -> float:
    correlation = spearmanr(first.reshape(-1), second.reshape(-1)).statistic
    if not np.isfinite(correlation):
        return 0.0
    return float(correlation)


class OptionalLPIPS:
    def __init__(self, device: torch.device):
        try:
            import lpips
        except ImportError:
            self.model = None
            self.device = device
            return
        self.model = lpips.LPIPS(net="alex").to(device).eval()
        self.device = device

    def __call__(
        self,
        prediction_bgr: np.ndarray,
        reference_bgr: np.ndarray,
    ) -> float:
        if self.model is None:
            return float("nan")

        def to_tensor(image_bgr: np.ndarray) -> torch.Tensor:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float()
            return (tensor.unsqueeze(0).to(self.device) / 127.5) - 1.0

        with torch.inference_mode():
            value = self.model(
                to_tensor(prediction_bgr),
                to_tensor(reference_bgr),
            )
        return float(value.item())
