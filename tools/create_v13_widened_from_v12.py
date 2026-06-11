"""Create a v13-sized checkpoint by structurally widening a v12 checkpoint.

This is a diagnostic/warm-start utility for the v13 failure mode where a large
scratch model lowers loss but has little arena strength. It copies the v12
subspace into the upper-left blocks of a v13 model, zeros newly introduced
cross-dimension connections where possible, and makes extra transformer layers
initially identity-like by zeroing their attention/MLP output projections.

The result is not mathematically identical to v12 because LayerNorm width and
the v13 value-pooling head differ, but it starts much closer to the known-good
policy/value function than a random 200M initialization.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from xiangqi_transformer_model import (  # noqa: E402
    XiangqiPVTransformer,
    XiangqiTransformerConfig,
    config_from_checkpoint_state,
    normalize_model_state_dict_keys,
)


def _v13_config(strategy: bool, *, value_token_pooling: bool = True) -> XiangqiTransformerConfig:
    return XiangqiTransformerConfig(
        d_model=896,
        num_layers=20,
        num_heads=14,
        ffn_dim=3584,
        policy_head_dim=384,
        use_2d_relative_attention_bias=True,
        use_line_of_sight_attention_bias=False,
        use_history_memory_attention=False,
        use_global_strategic_attention=False,
        use_trunk_global_strategy_tokens=bool(strategy),
        use_value_token_pooling=bool(value_token_pooling),
        num_global_strategy_tokens=8,
    )


def _zero_(tensor: torch.Tensor) -> None:
    if torch.is_floating_point(tensor):
        tensor.zero_()


def _copy_overlap_(dst: torch.Tensor, src: torch.Tensor, *, zero_dst: bool = True, scale: float = 1.0) -> None:
    if not torch.is_tensor(dst) or not torch.is_tensor(src):
        return
    if dst.ndim != src.ndim:
        return
    if zero_dst:
        _zero_(dst)
    slices = tuple(slice(0, min(int(a), int(b))) for a, b in zip(dst.shape, src.shape))
    dst[slices].copy_(src[slices].to(dtype=dst.dtype) * float(scale))


def _copy_layernorm_(dst_sd: dict[str, torch.Tensor], src_sd: dict[str, torch.Tensor], prefix: str) -> None:
    weight_key = f"{prefix}.weight"
    bias_key = f"{prefix}.bias"
    if weight_key in dst_sd:
        dst_sd[weight_key].fill_(1.0)
    if bias_key in dst_sd:
        dst_sd[bias_key].zero_()
    if weight_key in src_sd and weight_key in dst_sd:
        n = min(dst_sd[weight_key].numel(), src_sd[weight_key].numel())
        dst_sd[weight_key][:n].copy_(src_sd[weight_key][:n].to(dtype=dst_sd[weight_key].dtype))
    if bias_key in src_sd and bias_key in dst_sd:
        n = min(dst_sd[bias_key].numel(), src_sd[bias_key].numel())
        dst_sd[bias_key][:n].copy_(src_sd[bias_key][:n].to(dtype=dst_sd[bias_key].dtype))


def _copy_mha_(dst_sd: dict[str, torch.Tensor], src_sd: dict[str, torch.Tensor], prefix: str, old_d: int, new_d: int) -> None:
    src_w = src_sd.get(f"{prefix}.in_proj_weight")
    dst_w = dst_sd.get(f"{prefix}.in_proj_weight")
    if src_w is not None and dst_w is not None:
        dst_w.zero_()
        for part in range(3):
            src_r = slice(part * old_d, (part + 1) * old_d)
            dst_r = slice(part * new_d, part * new_d + old_d)
            dst_w[dst_r, :old_d].copy_(src_w[src_r, :old_d].to(dtype=dst_w.dtype))

    src_b = src_sd.get(f"{prefix}.in_proj_bias")
    dst_b = dst_sd.get(f"{prefix}.in_proj_bias")
    if src_b is not None and dst_b is not None:
        dst_b.zero_()
        for part in range(3):
            src_r = slice(part * old_d, (part + 1) * old_d)
            dst_r = slice(part * new_d, part * new_d + old_d)
            dst_b[dst_r].copy_(src_b[src_r].to(dtype=dst_b.dtype))

    _copy_overlap_(dst_sd[f"{prefix}.out_proj.weight"], src_sd[f"{prefix}.out_proj.weight"])
    _copy_overlap_(dst_sd[f"{prefix}.out_proj.bias"], src_sd[f"{prefix}.out_proj.bias"])


def _make_extra_block_identity_(sd: dict[str, torch.Tensor], block_idx: int) -> None:
    prefix = f"blocks.{block_idx}"
    for key in (
        f"{prefix}.attn.out_proj.weight",
        f"{prefix}.attn.out_proj.bias",
        f"{prefix}.mlp.3.weight",
        f"{prefix}.mlp.3.bias",
    ):
        if key in sd:
            sd[key].zero_()
    for norm in (f"{prefix}.norm1", f"{prefix}.norm2"):
        _copy_layernorm_(sd, {}, norm)


def widen_checkpoint(
    source: Path,
    output: Path,
    *,
    strategy: bool = False,
    value_token_pooling: bool = True,
) -> None:
    src_state: dict[str, Any] = torch.load(source, map_location="cpu", weights_only=False)
    src_cfg = config_from_checkpoint_state(src_state)
    src_sd = normalize_model_state_dict_keys(src_state["model_state_dict"])

    dst_cfg = _v13_config(strategy=strategy, value_token_pooling=bool(value_token_pooling))
    dst_model = XiangqiPVTransformer(dst_cfg)
    dst_sd = dst_model.state_dict()

    old_d = int(src_cfg.d_model)
    new_d = int(dst_cfg.d_model)
    old_head_dim = int(getattr(src_cfg, "policy_head_dim", 256))
    new_head_dim = int(dst_cfg.policy_head_dim)
    policy_repr_scale = math.pow(float(new_head_dim) / float(old_head_dim), 0.25)

    with torch.no_grad():
        for key in (
            "input_proj.weight",
            "input_proj.bias",
            "square_embedding.weight",
            "rank_embedding.weight",
            "file_embedding.weight",
            "material_mlp.0.weight",
            "material_mlp.0.bias",
            "material_mlp.2.weight",
            "material_mlp.2.bias",
        ):
            if key in src_sd and key in dst_sd:
                _copy_overlap_(dst_sd[key], src_sd[key])

        n_copy_layers = min(int(src_cfg.num_layers), int(dst_cfg.num_layers))
        for layer in range(n_copy_layers):
            src_prefix = f"blocks.{layer}"
            dst_prefix = f"blocks.{layer}"
            _copy_layernorm_(dst_sd, src_sd, f"{dst_prefix}.norm1")
            _copy_layernorm_(dst_sd, src_sd, f"{dst_prefix}.norm2")
            _copy_mha_(dst_sd, src_sd, f"{dst_prefix}.attn", old_d=old_d, new_d=new_d)
            for suffix in ("mlp.0.weight", "mlp.0.bias", "mlp.3.weight", "mlp.3.bias"):
                s_key = f"{src_prefix}.{suffix}"
                d_key = f"{dst_prefix}.{suffix}"
                if s_key in src_sd and d_key in dst_sd:
                    _copy_overlap_(dst_sd[d_key], src_sd[s_key])

        for layer in range(n_copy_layers, int(dst_cfg.num_layers)):
            _make_extra_block_identity_(dst_sd, layer)

        _copy_layernorm_(dst_sd, src_sd, "final_norm")

        for key in ("from_repr.weight", "from_repr.bias", "to_repr.weight", "to_repr.bias"):
            if key in src_sd and key in dst_sd:
                _copy_overlap_(dst_sd[key], src_sd[key], scale=policy_repr_scale)
        for key in ("from_bias.weight", "from_bias.bias", "to_bias.weight", "to_bias.bias"):
            if key in src_sd and key in dst_sd:
                _copy_overlap_(dst_sd[key], src_sd[key])

        _copy_layernorm_(dst_sd, src_sd, "value_shared.0")
        for key in (
            "value_shared.1.weight",
            "value_shared.1.bias",
            "wdl_head.weight",
            "wdl_head.bias",
            "scalar_head.weight",
            "scalar_head.bias",
        ):
            if key in src_sd and key in dst_sd:
                _copy_overlap_(dst_sd[key], src_sd[key])

    out_state = copy.deepcopy(src_state)
    out_state["model_state_dict"] = dict(dst_sd)
    out_state["model_config"] = dst_cfg.__dict__.copy()
    out_state["global_step"] = 0
    out_state.pop("optimizer_state_dict", None)
    out_state.pop("scheduler_state_dict", None)
    out_state["widened_from_v12_meta"] = {
        "source": str(source.resolve()),
        "source_global_step": int(src_state.get("global_step", 0)),
        "source_model_config": getattr(src_cfg, "__dict__", {}),
        "target_model_config": dst_cfg.__dict__.copy(),
        "value_token_pooling": bool(value_token_pooling),
        "copied_layers": min(int(src_cfg.num_layers), int(dst_cfg.num_layers)),
        "extra_layers_identity_initialized": True,
        "policy_repr_scale": policy_repr_scale,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_state, output)
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(out_state["widened_from_v12_meta"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {output}")
    print(f"wrote {output.with_suffix(output.suffix + '.meta.json')}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Widen a v12 checkpoint into a v13-sized warm-start checkpoint.")
    parser.add_argument("--source", required=True, help="Known-good v12/v12.6 checkpoint.")
    parser.add_argument("--output", required=True, help="Output v13 checkpoint path.")
    parser.add_argument("--strategy", action="store_true", help="Create the v13 strategy-token variant.")
    parser.add_argument(
        "--no-value-token-pooling",
        action="store_true",
        help="Disable v13 value-token pooling so the widened checkpoint keeps the v12 material-token value path.",
    )
    args = parser.parse_args()
    widen_checkpoint(
        Path(args.source),
        Path(args.output),
        strategy=bool(args.strategy),
        value_token_pooling=not bool(args.no_value_token_pooling),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
