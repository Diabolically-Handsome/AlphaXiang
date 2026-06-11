"""Linearly interpolate two compatible Xiangqi checkpoints.

This is intended for low-risk adapter experiments: keep most of a known-good
checkpoint while mixing in a small fraction of a finetuned checkpoint.
Optimizer/scheduler state is kept from the base checkpoint for structural
compatibility, but interpolated checkpoints are meant for evaluation or fresh
optimizer resumes, not optimizer-state continuation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _interpolate_state_dict(
    base: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, base_value in base.items():
        target_value = target.get(key)
        if (
            target_value is not None
            and isinstance(base_value, torch.Tensor)
            and isinstance(target_value, torch.Tensor)
            and base_value.shape == target_value.shape
            and torch.is_floating_point(base_value)
            and torch.is_floating_point(target_value)
        ):
            out[key] = (base_value.float() * (1.0 - alpha) + target_value.float() * alpha).to(base_value.dtype)
        else:
            out[key] = base_value
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Interpolate two compatible model checkpoints.")
    parser.add_argument("--base", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--alpha", type=float, required=True, help="0=base, 1=target")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    alpha = float(args.alpha)
    if not (0.0 <= alpha <= 1.0):
        raise SystemExit("--alpha must be within [0, 1]")

    base_path = Path(args.base)
    target_path = Path(args.target)
    output_path = Path(args.output)
    base_state: dict[str, Any] = torch.load(base_path, map_location="cpu", weights_only=False)
    target_state: dict[str, Any] = torch.load(target_path, map_location="cpu", weights_only=False)

    base_model = base_state["model_state_dict"]
    target_model = target_state["model_state_dict"]
    out_state = dict(base_state)
    out_state["model_state_dict"] = _interpolate_state_dict(base_model, target_model, alpha)
    out_state["global_step"] = int(target_state.get("global_step", base_state.get("global_step", 0)))
    out_state["interpolation_meta"] = {
        "base": str(base_path.resolve()),
        "target": str(target_path.resolve()),
        "alpha": alpha,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_state, output_path)
    sidecar = output_path.with_suffix(output_path.suffix + ".meta.json")
    sidecar.write_text(
        json.dumps(out_state["interpolation_meta"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {output_path}")
    print(f"wrote {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
