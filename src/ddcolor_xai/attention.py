from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .explainers import normalize_map
from .targets import region_score


@dataclass
class AttentionExplanation:
    saliency: torch.Tensor
    output_ab: torch.Tensor
    score: float
    query_importance: torch.Tensor
    layer_maps: torch.Tensor


def _spatial_shape(token_count: int, target_aspect: float = 1.0) -> tuple[int, int]:
    """Find a factorization whose aspect ratio is closest to the input."""
    best = (1, token_count)
    best_error = float("inf")
    for height in range(1, int(math.sqrt(token_count)) + 1):
        if token_count % height:
            continue
        width = token_count // height
        error = abs((width / height) - target_aspect)
        if error < best_error:
            best = (height, width)
            best_error = error
    return best


def combine_query_attention(
    attention_weights: list[torch.Tensor],
    query_importance: torch.Tensor,
    output_size: tuple[int, int],
    layer_limit: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine cross-attention maps using region-sensitive query weights."""
    if not attention_weights:
        raise RuntimeError("No cross-attention weights were captured")

    height, width = output_size
    target_aspect = width / max(height, 1)
    query_weights = query_importance.float().clamp_min(0)
    if float(query_weights.sum()) <= 1e-12:
        query_weights = torch.ones_like(query_weights)
    query_weights = query_weights / query_weights.sum()

    layer_maps: list[torch.Tensor] = []
    for weights in attention_weights[-layer_limit:]:
        if weights.ndim == 4:
            weights = weights.mean(dim=1)
        if weights.ndim != 3:
            raise ValueError(f"Expected BxQxS attention, got {tuple(weights.shape)}")
        per_query = weights[0]
        if per_query.shape[0] != query_weights.numel():
            raise ValueError("Query count differs between attention and importance")

        flat = (per_query * query_weights[:, None]).sum(dim=0)
        source_h, source_w = _spatial_shape(flat.numel(), target_aspect)
        layer_map = flat.reshape(1, 1, source_h, source_w)
        layer_map = F.interpolate(
            layer_map,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        layer_maps.append(normalize_map(layer_map[None])[0])

    stacked = torch.stack(layer_maps)
    saliency = normalize_map(stacked.mean(dim=0, keepdim=True))
    return saliency, stacked


class QueryAttentionExplainer:
    """Region-conditioned analysis of DDColor decoder cross-attention."""

    def __init__(self, model: torch.nn.Module, layer_limit: int = 3):
        self.model = model
        self.layer_limit = int(layer_limit)
        self.decoder = model.decoder.color_decoder
        self._attention_weights: list[torch.Tensor] = []
        self._query_state: torch.Tensor | None = None
        self._handles = []

        for layer in self.decoder.transformer_cross_attention_layers:
            self._handles.append(
                layer.multihead_attn.register_forward_hook(self._capture_attention)
            )
        self._handles.append(
            self.decoder.decoder_norm.register_forward_hook(self._capture_queries)
        )

    def _capture_attention(self, _module, _inputs, output) -> None:
        weights = output[1]
        if weights is not None:
            self._attention_weights.append(weights.detach())

    def _capture_queries(self, _module, _inputs, output) -> None:
        self._query_state = output
        if output.requires_grad:
            output.retain_grad()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def explain(
        self,
        model_input: torch.Tensor,
        region_mask: torch.Tensor,
        target_mode: str = "chroma",
    ) -> AttentionExplanation:
        self.model.zero_grad(set_to_none=True)
        self._attention_weights = []
        self._query_state = None

        with torch.enable_grad():
            output_ab = self.model(model_input)
            scores = region_score(output_ab, region_mask, mode=target_mode)
            scores.sum().backward()

        query_state = self._query_state
        if query_state is None or query_state.grad is None:
            raise RuntimeError("Decoder query gradients were not captured")

        # The final decoder state is QxBxC. Absolute gradient-times-activation
        # gives a nonnegative sensitivity weight for each learned color query.
        query_importance = (
            (query_state.grad * query_state).abs().mean(dim=-1)[:, 0].detach()
        )
        saliency, layer_maps = combine_query_attention(
            self._attention_weights,
            query_importance,
            model_input.shape[-2:],
            layer_limit=self.layer_limit,
        )

        return AttentionExplanation(
            saliency=saliency.detach(),
            output_ab=output_ab.detach(),
            score=float(scores[0].detach().cpu()),
            query_importance=query_importance.cpu(),
            layer_maps=layer_maps.detach().cpu(),
        )
