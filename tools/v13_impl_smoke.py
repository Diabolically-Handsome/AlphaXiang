"""Implementation smoke for v13 model presets.

Checks both 200M arms without launching a long training run:
- instantiate and print parameter counts
- forward/backward one tiny batch
- checkpoint save/load roundtrip through build_model_from_checkpoint_state
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from xiangqi_transformer_model import (  # noqa: E402
    XiangqiPVTransformer,
    XiangqiTransformerConfig,
    build_model_from_checkpoint_state,
)


def _v13_config(strategy: bool) -> XiangqiTransformerConfig:
    return XiangqiTransformerConfig(
        d_model=896,
        num_layers=20,
        num_heads=14,
        ffn_dim=3584,
        policy_head_dim=384,
        use_2d_relative_attention_bias=True,
        use_trunk_global_strategy_tokens=bool(strategy),
        use_value_token_pooling=True,
        num_global_strategy_tokens=8,
    )


def _run_one(name: str, cfg: XiangqiTransformerConfig, device: torch.device, out_dir: Path) -> dict[str, object]:
    model = XiangqiPVTransformer(cfg).to(device)
    model.train()
    total_params = sum(p.numel() for p in model.parameters())
    batch = torch.randn(1, cfg.in_channels, cfg.board_h, cfg.board_w, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        outputs = model(batch)
        loss = (
            outputs["policy_logits"][:, :32].float().mean()
            + outputs["wdl_logits"].float().mean()
            + outputs["value_scalar"].float().mean()
        )
    loss.backward()
    nonzero_grad = sum(
        1 for p in model.parameters()
        if p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().sum().item()) > 0.0
    )
    ckpt = {
        "model_state_dict": model.state_dict(),
        "global_step": 0,
        "model_config": asdict(cfg),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{name}_roundtrip.pt"
    torch.save(ckpt, ckpt_path)
    reloaded = build_model_from_checkpoint_state(torch.load(ckpt_path, map_location="cpu", weights_only=False))
    reloaded.eval()
    shapes = {
        key: list(value.shape)
        for key, value in outputs.items()
    }
    result = {
        "name": name,
        "params": total_params,
        "params_m": round(total_params / 1_000_000, 3),
        "output_shapes": shapes,
        "nonzero_grad_tensors": nonzero_grad,
        "roundtrip_checkpoint": str(ckpt_path),
        "roundtrip_params": sum(p.numel() for p in reloaded.parameters()),
    }
    del model, reloaded, batch, outputs, loss
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test v13 200M model presets.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", default="/home/laure/alphaxiang/v13_impl_smoke")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    results = [
        _run_one("v13_200m_dense", _v13_config(strategy=False), device, out_dir),
        _run_one("v13_200m_strategy", _v13_config(strategy=True), device, out_dir),
    ]
    payload = {"device": str(device), "results": results}
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
