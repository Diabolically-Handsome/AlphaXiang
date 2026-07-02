"""Model spec tests: the numbers the README claims, verified mechanically."""

import torch

from xiangqi_transformer_model import (
    XiangqiPVTransformer,
    XiangqiTransformerConfig,
    load_xiangqi_model_state_dict,
)

INPUT_PLANES = 115
BOARD_H, BOARD_W = 10, 9
POLICY_SIZE = 8100


def _param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_default_config_parameter_count() -> None:
    model = XiangqiPVTransformer(XiangqiTransformerConfig())
    assert _param_count(model) == 38_610_182


def test_with_attention_biases_parameter_count() -> None:
    """The published 38.6M spec (README) measured with both optional
    attention biases enabled."""
    config = XiangqiTransformerConfig(
        use_2d_relative_attention_bias=True,
        use_line_of_sight_attention_bias=True,
    )
    model = XiangqiPVTransformer(config)
    assert _param_count(model) == 38_641_766


def test_forward_shapes_and_finiteness() -> None:
    model = XiangqiPVTransformer(XiangqiTransformerConfig())
    model.eval()
    batch = torch.zeros(2, INPUT_PLANES, BOARD_H, BOARD_W)
    with torch.inference_mode():
        out = model(batch)
    assert out["policy_logits"].shape == (2, POLICY_SIZE)
    assert out["wdl_logits"].shape == (2, 3)
    assert out["value_scalar"].shape == (2, 1)
    for key, tensor in out.items():
        assert torch.isfinite(tensor).all(), f"non-finite values in {key}"


def test_backward_produces_finite_gradients() -> None:
    model = XiangqiPVTransformer(XiangqiTransformerConfig())
    batch = torch.randn(2, INPUT_PLANES, BOARD_H, BOARD_W) * 0.1
    out = model(batch)
    loss = out["policy_logits"].logsumexp(dim=-1).mean() + out["value_scalar"].pow(2).mean()
    loss.backward()
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"non-finite grad in {name}"


def test_state_dict_round_trip() -> None:
    config = XiangqiTransformerConfig()
    model_a = XiangqiPVTransformer(config)
    model_b = XiangqiPVTransformer(config)
    load_xiangqi_model_state_dict(model_b, model_a.state_dict())
    batch = torch.randn(1, INPUT_PLANES, BOARD_H, BOARD_W)
    model_a.eval()
    model_b.eval()
    with torch.inference_mode():
        out_a = model_a(batch)
        out_b = model_b(batch)
    for key in out_a:
        assert torch.equal(out_a[key], out_b[key]), f"mismatch after round-trip: {key}"
