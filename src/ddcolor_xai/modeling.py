from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch


def add_ddcolor_root(ddcolor_root: str | Path) -> Path:
    root = Path(ddcolor_root).expanduser().resolve()
    if not (root / "ddcolor" / "model.py").is_file():
        raise FileNotFoundError(
            f"DDColor repository not found at {root}. "
            "Clone https://github.com/piddnad/DDColor there first."
        )
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return root


def load_model(
    ddcolor_root: str | Path,
    model_name: str = "ddcolor_paper_tiny",
    device: str | torch.device | None = None,
):
    """Load an official DDColor Hugging Face checkpoint."""
    add_ddcolor_root(ddcolor_root)

    from ddcolor import DDColor
    from huggingface_hub import PyTorchModelHubMixin

    class DDColorHF(DDColor, PyTorchModelHubMixin):
        def __init__(self, config=None, **kwargs):
            if isinstance(config, dict):
                kwargs = {**config, **kwargs}
            super().__init__(**kwargs)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    local_model = Path(model_name).expanduser()
    if local_model.is_dir():
        with (local_model / "config.json").open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        model = DDColorHF(config=config)
        state_dict = torch.load(
            local_model / "pytorch_model.bin",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict)
    else:
        repo_id = model_name if "/" in model_name else f"piddnad/{model_name}"
        model = DDColorHF.from_pretrained(repo_id)
    model = model.to(device)
    model.eval()
    return model, device


def resolve_module(root: Any, dotted_path: str):
    """Resolve paths such as ``decoder.layers.2`` on a PyTorch model."""
    current = root
    for part in dotted_path.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def reset_module_parameters(module: torch.nn.Module) -> None:
    """Randomize all child modules that expose ``reset_parameters``."""
    # Hugging Face may materialize parameters as inference tensors.
    # In-place reinitialization of those tensors must stay in inference mode.
    with torch.inference_mode():
        for child in module.modules():
            reset = getattr(child, "reset_parameters", None)
            if callable(reset):
                reset()
