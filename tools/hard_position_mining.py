"""Hard-position mining — flag training positions where model is most wrong.

Why
---
Active learning applied to value distillation.  For each shard with both
``oracle_value`` (Pikafish d=12 ground truth) and the current model's
prediction, compute disagreement |oracle_value - predicted_value| per position
and oversample the top-X% as "hard positions" for the next training cycle.

The model presumably already learned the easy positions well — its gradient
signal on those is small and noisy.  Hard positions are where the model and
the oracle disagree the most, which is where additional gradient updates have
the most leverage.

Output
------
Writes a ``sample_weight`` field (B,) float32 per shard:
- weight = ``light_weight`` (default 1.0) for non-hard positions
- weight = ``heavy_weight`` (default 3.0) for hard positions

Training reads ``sample_weight`` and multiplies the per-sample loss by it.
Positions with NaN oracle_value (oracle failed) get the light weight — we
have no disagreement signal for those.

Threshold mode
--------------
Default: per-shard top-X% (X=10).  Simple, no second pass.  In practice the
oracle disagreement distribution is similar shard-to-shard so the threshold
chosen per-shard tracks closely with a global percentile.

Usage
-----
    python tools/hard_position_mining.py \\
        --checkpoint /path/to/v11_step_NNNNN.pt \\
        --input-shard-dir /path/to/cycle_N/train/ \\
        --top-percent 10 --heavy-weight 3.0 \\
        --device cuda:0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from xiangqi_transformer_model import (  # noqa: E402
    XiangqiPVTransformer, build_model_from_checkpoint_state,
)


def _load_model(checkpoint_path: Path, device: torch.device) -> XiangqiPVTransformer:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "model_state_dict" not in state:
        raise RuntimeError(f"unsupported checkpoint format at {checkpoint_path}")
    model = build_model_from_checkpoint_state(state)
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_values_and_logits(
    model: XiangqiPVTransformer,
    state_tensor: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
    use_bfloat16: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run model on (N, 115, 10, 9) state, return (value_scalar (N,), policy_logits (N, 8100))."""
    n = int(state_tensor.shape[0])
    val_out = torch.empty((n,), dtype=torch.float32)
    pol_out = torch.empty((n, 8100), dtype=torch.float32)
    state_tensor = state_tensor.to(device=device, non_blocking=True)
    if state_tensor.dtype != torch.float32:
        state_tensor = state_tensor.to(torch.float32)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        x = state_tensor[start:stop]
        if use_bfloat16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(x)
        else:
            outputs = model(x)
        val_out[start:stop] = outputs["value_scalar"].float().squeeze(-1).cpu()
        pol_out[start:stop] = outputs["policy_logits"].float().cpu()
    return val_out, pol_out


def predict_values(
    model: XiangqiPVTransformer,
    state_tensor: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
    use_bfloat16: bool = True,
) -> torch.Tensor:
    """Backward-compat wrapper: just returns the value tensor."""
    val, _ = predict_values_and_logits(model, state_tensor, device, batch_size, use_bfloat16)
    return val


def mine_one_shard(
    shard_path: Path,
    output_path: Path,
    model: XiangqiPVTransformer,
    device: torch.device,
    top_percent: float,
    heavy_weight: float,
    light_weight: float,
    use_bfloat16: bool,
    policy_regret_weight: float = 0.0,
) -> dict:
    """Score one shard, write sample_weight, return stats.

    v12: optional policy-regret signal added to value-disagreement. When
    `policy_regret_weight > 0` and shard has oracle_policy_*, computes
    `regret_i = max(oracle_q) - oracle_q(model_top1_in_oracle_top_K)`,
    then combined_score = value_disagree + policy_regret_weight * regret.
    """
    t0 = time.monotonic()
    payload = torch.load(shard_path, map_location="cpu", weights_only=False)
    n = int(payload["state"].shape[0])

    if "oracle_value" not in payload:
        return {"samples": n, "marked_hard": 0, "skipped": True,
                "reason": "no oracle_value field", "duration_s": 0.0}

    oracle_v = payload["oracle_value"].float()
    valid_mask = ~torch.isnan(oracle_v)
    if int(valid_mask.sum()) == 0:
        weights = torch.full((n,), float(light_weight), dtype=torch.float32)
        payload["sample_weight"] = weights
        _atomic_save(payload, output_path)
        return {"samples": n, "marked_hard": 0, "skipped": False,
                "reason": "no valid oracle_value (all NaN)",
                "duration_s": time.monotonic() - t0}

    state_tensor = payload["state"]
    if state_tensor.dtype == torch.bfloat16:
        state_tensor = state_tensor.to(torch.float32)

    # v12: also need policy logits if computing policy regret
    use_regret = policy_regret_weight > 0.0 and "oracle_policy_offsets" in payload
    if use_regret:
        pred_v, pred_logits = predict_values_and_logits(
            model, state_tensor, device, use_bfloat16=use_bfloat16,
        )
    else:
        pred_v = predict_values(model, state_tensor, device, use_bfloat16=use_bfloat16)

    disagreement = torch.abs(oracle_v - pred_v)
    disagreement = torch.where(valid_mask, disagreement, torch.zeros_like(disagreement))

    # v12: policy regret. For each position with an oracle policy slice, look at
    # the top-K oracle moves' probs and find which one the model would pick (argmax
    # of model's logits restricted to oracle's top-K). Regret = best_oracle_prob -
    # oracle_prob[that_index]. Position where model picks the same as oracle's argmax
    # has regret 0; picking oracle's worst-of-top-K gives larger regret.
    if use_regret:
        op_offsets = payload["oracle_policy_offsets"].to(torch.int64)
        op_idxs = payload["oracle_policy_idxs"].to(torch.int64)
        op_probs = payload["oracle_policy_probs"].float()
        regret = torch.zeros(n, dtype=torch.float32)
        for i in range(n):
            s = int(op_offsets[i].item())
            e = int(op_offsets[i + 1].item())
            if e <= s:
                continue
            move_idxs = op_idxs[s:e]
            move_probs = op_probs[s:e]
            best_p = float(move_probs.max().item())
            # Restrict model logits to these oracle moves; pick argmax there
            sub_logits = pred_logits[i, move_idxs]
            chosen = int(sub_logits.argmax().item())
            regret[i] = best_p - float(move_probs[chosen].item())
        # Combine with value disagreement
        score = disagreement + float(policy_regret_weight) * regret
    else:
        score = disagreement

    n_valid = int(valid_mask.sum())
    n_hard = max(1, int(round(n_valid * float(top_percent) / 100.0)))
    sorted_score, _ = torch.sort(score[valid_mask], descending=True)
    threshold = float(sorted_score[min(n_hard - 1, n_valid - 1)])

    weights = torch.full((n,), float(light_weight), dtype=torch.float32)
    is_hard = valid_mask & (score >= threshold)
    weights[is_hard] = float(heavy_weight)
    n_marked = int(is_hard.sum())

    payload["sample_weight"] = weights
    payload["sample_weight_meta"] = {
        "top_percent": float(top_percent),
        "heavy_weight": float(heavy_weight),
        "light_weight": float(light_weight),
        "threshold": float(threshold),
        "n_valid": int(n_valid),
        "n_marked": int(n_marked),
        "labeled_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    _atomic_save(payload, output_path)
    return {
        "samples": n,
        "marked_hard": n_marked,
        "valid": n_valid,
        "threshold": threshold,
        "duration_s": time.monotonic() - t0,
        "skipped": False,
    }


def _atomic_save(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(output_path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Path to model checkpoint to use for value predictions.")
    p.add_argument("--input-shard-dir", required=True)
    p.add_argument("--output-shard-dir", default=None,
                   help="If omitted, overwrites in place.")
    p.add_argument("--top-percent", type=float, default=10.0,
                   help="Per-shard top-X%% by |oracle_v - predicted_v| disagreement "
                        "are flagged as hard. Default 10.")
    p.add_argument("--heavy-weight", type=float, default=3.0,
                   help="Sample weight for hard positions. Default 3.0.")
    p.add_argument("--light-weight", type=float, default=1.0,
                   help="Sample weight for non-hard positions. Default 1.0.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--shard-glob", default="shard_*.pt")
    p.add_argument("--disable-bf16", action="store_true")
    p.add_argument("--policy-regret-weight", type=float, default=0.0,
                   help="v12: when > 0, combine policy regret with value disagreement. "
                        "regret_i = max(oracle_q) - oracle_q(model_top1_in_oracle_top_K). "
                        "Default 0 (v11 behavior, value disagreement only). Try 1.0 for "
                        "equal-weight blend with value disagreement.")
    args = p.parse_args()

    in_dir = Path(args.input_shard_dir)
    out_dir = Path(args.output_shard_dir) if args.output_shard_dir else in_dir
    if not in_dir.is_dir():
        raise SystemExit(f"--input-shard-dir not found: {in_dir}")

    shards = sorted(in_dir.glob(args.shard_glob))
    if not shards:
        raise SystemExit(f"no shards matching {args.shard_glob!r} under {in_dir}")
    print(f"found {len(shards)} shard(s) under {in_dir}", flush=True)

    device = torch.device(args.device)
    model = _load_model(Path(args.checkpoint), device)
    print(f"loaded model from {args.checkpoint} on {device}", flush=True)

    use_bfloat16 = (not args.disable_bf16) and device.type == "cuda"

    total_samples = 0
    total_marked = 0
    t_start = time.monotonic()
    for i, shard_path in enumerate(shards):
        out_path = out_dir / shard_path.name
        stats = mine_one_shard(
            shard_path=shard_path,
            output_path=out_path,
            model=model,
            device=device,
            top_percent=float(args.top_percent),
            heavy_weight=float(args.heavy_weight),
            light_weight=float(args.light_weight),
            use_bfloat16=use_bfloat16,
            policy_regret_weight=float(args.policy_regret_weight),
        )
        total_samples += stats["samples"]
        total_marked += stats["marked_hard"]
        if stats["skipped"]:
            print(f"  shard {i+1}/{len(shards)} {shard_path.name}: SKIPPED "
                  f"({stats.get('reason', '')})", flush=True)
        else:
            valid = stats.get("valid", stats["samples"])
            thr = stats.get("threshold", float("nan"))
            print(f"  shard {i+1}/{len(shards)} {shard_path.name}: "
                  f"hard={stats['marked_hard']}/{valid} "
                  f"thr={thr:.3f} dt={stats['duration_s']:.1f}s", flush=True)

    dt_total = time.monotonic() - t_start
    print()
    print(f"DONE: marked {total_marked}/{total_samples} hard positions "
          f"in {dt_total:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
