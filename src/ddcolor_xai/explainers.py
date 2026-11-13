from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .targets import region_score


def normalize_map(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    values = values - values.amin(dim=(-2, -1), keepdim=True)
    scale = values.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return values / scale


@dataclass
class Explanation:
    saliency: torch.Tensor
    output_ab: torch.Tensor
    score: float


class RegionGradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activation: torch.Tensor | None = None
        self._handle = target_layer.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output):
        self._activation = output
        if output.requires_grad:
            output.retain_grad()

    def close(self) -> None:
        self._handle.remove()

    def explain(
        self,
        model_input: torch.Tensor,
        region_mask: torch.Tensor,
        target_mode: str = "chroma",
        target_ab: torch.Tensor | None = None,
    ) -> Explanation:
        self.model.zero_grad(set_to_none=True)
        self._activation = None

        with torch.enable_grad():
            output_ab = self.model(model_input)
            scores = region_score(
                output_ab,
                region_mask,
                mode=target_mode,
                target_ab=target_ab,
            )
            scores.sum().backward()

        activation = self._activation
        if activation is None or activation.grad is None:
            raise RuntimeError("Target layer did not retain activations and gradients")

        gradients = activation.grad
        weights = gradients.mean(dim=(-2, -1), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam,
            size=model_input.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        cam = normalize_map(cam).detach()

        return Explanation(
            saliency=cam,
            output_ab=output_ab.detach(),
            score=float(scores[0].detach().cpu()),
        )


class RegionRISE:
    def __init__(
        self,
        model: torch.nn.Module,
        num_masks: int = 256,
        grid_size: int = 8,
        probability: float = 0.5,
        batch_size: int = 8,
        seed: int = 0,
    ):
        self.model = model
        self.num_masks = int(num_masks)
        self.grid_size = int(grid_size)
        self.probability = float(probability)
        self.batch_size = int(batch_size)
        self.seed = int(seed)

    def _mask_batch(
        self,
        count: int,
        height: int,
        width: int,
        generator: torch.Generator,
        device: torch.device,
    ) -> torch.Tensor:
        coarse = torch.rand(
            (count, 1, self.grid_size, self.grid_size),
            generator=generator,
            device=device,
        )
        coarse = (coarse < self.probability).float()

        cell_h = max(1, (height + self.grid_size - 1) // self.grid_size)
        cell_w = max(1, (width + self.grid_size - 1) // self.grid_size)
        up_h = height + cell_h
        up_w = width + cell_w
        masks = F.interpolate(
            coarse,
            size=(up_h, up_w),
            mode="bilinear",
            align_corners=False,
        )

        cropped = []
        for index in range(count):
            shift_y = int(
                torch.randint(
                    0,
                    cell_h + 1,
                    (1,),
                    generator=generator,
                    device=device,
                ).item()
            )
            shift_x = int(
                torch.randint(
                    0,
                    cell_w + 1,
                    (1,),
                    generator=generator,
                    device=device,
                ).item()
            )
            cropped.append(
                masks[index : index + 1, :, shift_y : shift_y + height, shift_x : shift_x + width]
            )
        return torch.cat(cropped, dim=0)

    def explain(
        self,
        model_input: torch.Tensor,
        region_mask: torch.Tensor,
        target_mode: str = "chroma",
        target_ab: torch.Tensor | None = None,
    ) -> Explanation:
        if model_input.shape[0] != 1:
            raise ValueError("RegionRISE currently expects a single input image")

        device = model_input.device
        height, width = model_input.shape[-2:]
        generator = torch.Generator(device=device)
        generator.manual_seed(self.seed)

        baseline = model_input.mean(dim=(-2, -1), keepdim=True)
        weighted_masks = torch.zeros((height, width), device=device)
        mask_sum = torch.zeros((height, width), device=device)
        score_sum = torch.zeros((), device=device)
        processed = 0

        with torch.inference_mode():
            base_output = self.model(model_input)
            base_score = region_score(
                base_output,
                region_mask,
                mode=target_mode,
                target_ab=target_ab,
            )

            while processed < self.num_masks:
                count = min(self.batch_size, self.num_masks - processed)
                masks = self._mask_batch(
                    count,
                    height,
                    width,
                    generator,
                    device,
                )
                perturbed = model_input * masks + baseline * (1.0 - masks)
                outputs = self.model(perturbed)
                scores = region_score(
                    outputs,
                    region_mask,
                    mode=target_mode,
                    target_ab=target_ab,
                )
                weighted_masks += (scores[:, None, None, None] * masks).sum(dim=0)[0]
                mask_sum += masks.sum(dim=0)[0]
                score_sum += scores.sum()
                processed += count

        # Mean-centering removes the finite-sample mask prior. This is the
        # covariance between target scores and visible input locations.
        mean_weighted = weighted_masks / self.num_masks
        mean_mask = mask_sum / self.num_masks
        mean_score = score_sum / self.num_masks
        saliency = torch.relu(mean_weighted - mean_score * mean_mask)
        saliency = normalize_map(saliency[None]).detach()

        return Explanation(
            saliency=saliency,
            output_ab=base_output.detach(),
            score=float(base_score[0].cpu()),
        )
