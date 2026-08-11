"""Safe checkpoint deserialization and normalized checkpoint payloads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .exceptions import UnsafeCheckpointError, UnsupportedCheckpointError


@dataclass(frozen=True)
class CheckpointPayload:
    path: Path
    kind: str
    state_dict: dict[str, torch.Tensor] | None
    serialized_model: nn.Module | None
    raw_metadata: dict[str, Any]
    warnings: tuple[str, ...] = ()


def load_checkpoint(path: str | Path, *, allow_unsafe_full_model: bool = False) -> CheckpointPayload:
    checkpoint = Path(path)
    if checkpoint.suffix.lower() not in {".pt", ".pth", ".ckpt"}:
        raise UnsupportedCheckpointError("Supported checkpoint extensions are .pt, .pth, and .ckpt.")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as error:
        if not allow_unsafe_full_model:
            raise UnsafeCheckpointError(
                "Checkpoint could not be safely read as weights/state_dict. "
                "Full serialized model loading requires allow_unsafe_full_model=True and a trusted source."
            ) from error
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(payload, nn.Module):
        if not allow_unsafe_full_model:
            raise UnsafeCheckpointError("Refusing serialized nn.Module without allow_unsafe_full_model=True.")
        return CheckpointPayload(checkpoint, "serialized_model", None, payload, {}, ("Loaded trusted serialized model using unsafe pickle deserialization.",))
    if not isinstance(payload, dict):
        raise UnsupportedCheckpointError(f"Unsupported checkpoint payload type: {type(payload).__name__}.")
    state_dict = payload.get("state_dict") or payload.get("model_state_dict")
    if state_dict is None and payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
        state_dict = payload
    if not isinstance(state_dict, dict) or not state_dict:
        raise UnsupportedCheckpointError("Checkpoint does not contain a non-empty state_dict or model_state_dict.")
    normalized = {str(key).removeprefix("module."): value for key, value in state_dict.items()}
    metadata = {key: value for key, value in payload.items() if key not in {"state_dict", "model_state_dict"}}
    kind = "state_dict" if payload is state_dict else "checkpoint_dict"
    return CheckpointPayload(checkpoint, kind, normalized, None, metadata)
