"""Step 3 of root-cause: side-by-side position evaluation between Transformer and CNN.

Loads both models, evaluates the same set of positions, prints:
  - scalar value
  - WDL probabilities (if available)
  - top 8 policy moves with probabilities

The key check: do both models agree on opening evaluation?  If Transformer's value
is biased (e.g. predicts red wins big from start) while CNN is balanced, that's a
training-signal symptom.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.pikafish_opponent import internal_move_to_uci, uci_move_to_internal  # noqa: E402
from xiangqi_mcts_ext import Board, canonical_action  # noqa: E402
from xiangqi_model_battle_gui import (  # noqa: E402
    LegacyCnnEvaluator,
    LegacyScalarCnnNet,
    load_module_from_path,
    normalize_state_dict_keys,
    pick_cnn_net_class,
    wdl_logits_to_value,
)
from xiangqi_transformer_model import build_model_from_checkpoint_state  # noqa: E402


def load_transformer(path: Path, device: str):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        model = build_model_from_checkpoint_state(state)
    else:
        state = {"model_state_dict": normalize_state_dict_keys(state)}
        model = build_model_from_checkpoint_state(state)
    return model.to(device).eval()


def load_cnn(engine_path: Path, weights_path: Path, device: str):
    module = load_module_from_path(engine_path)
    cfg = dict(getattr(module, "CFG", {})) if isinstance(getattr(module, "CFG", None), dict) else {}
    sd = torch.load(weights_path, map_location="cpu", weights_only=False)
    sd = normalize_state_dict_keys(sd)
    v_w = sd.get("v_fc2.weight")
    if torch.is_tensor(v_w) and v_w.ndim == 2 and v_w.shape[0] == 1 and "m_conv.weight" not in sd:
        model = LegacyScalarCnnNet(cfg).cpu()
    else:
        model = pick_cnn_net_class(module)(cfg).cpu()
    model.load_state_dict(sd, strict=True)
    return model.eval().to(device)


def eval_transformer(model, board: Board, device: str):
    state = board.to_tensor_canonical().to(torch.float32).to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(state)
    return {
        "policy_logits": out["policy_logits"].float().squeeze(0),
        "value": float(out["value_scalar"]),
        "wdl": F.softmax(out["wdl_logits"].float().squeeze(0), dim=0),
    }


def eval_cnn(model, board: Board, device: str):
    state = board.to_tensor_canonical().to(torch.float32).to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        out = model(state)
    if isinstance(out, tuple):
        if len(out) == 3:
            policy_logits, wdl_logits, _ = out
            value = float(wdl_logits_to_value(wdl_logits).squeeze())
            wdl = F.softmax(wdl_logits.float().squeeze(0), dim=0)
        elif len(out) == 2:
            policy_logits, value_scalar = out
            value = float(value_scalar.squeeze())
            wdl = None
        else:
            raise RuntimeError("unsupported CNN output")
    elif isinstance(out, dict):
        policy_logits = out["policy_logits"]
        value = float(out["value_scalar"])
        wdl_logits = out.get("wdl_logits")
        wdl = F.softmax(wdl_logits.float().squeeze(0), dim=0) if wdl_logits is not None else None
    else:
        raise RuntimeError("unsupported CNN output")
    return {
        "policy_logits": policy_logits.float().squeeze(0),
        "value": value,
        "wdl": wdl,
    }


def print_eval(label: str, e: dict, board: Board):
    legal = list(board.legal_moves())
    stm_is_black = bool(board.turn() == 1)
    canonical_idxs = torch.tensor(
        [canonical_action(int(mv), stm_is_black) for mv in legal],
        dtype=torch.int64, device=e["policy_logits"].device,
    )
    legal_logits = e["policy_logits"][canonical_idxs]
    legal_probs = F.softmax(legal_logits, dim=0)
    topk = torch.topk(legal_probs, k=min(8, len(legal)))
    entropy = float(-(legal_probs * torch.log(legal_probs.clamp(min=1e-12))).sum())

    print(f"\n  --- {label} ---")
    print(f"    value: {e['value']:+.4f}")
    if e["wdl"] is not None:
        w, d, l = e["wdl"].tolist()
        print(f"    WDL  : W={w:.3f}  D={d:.3f}  L={l:.3f}")
    print(f"    policy entropy ({len(legal)} legal moves): {entropy:.3f}")
    print(f"    top-8 moves:")
    for rank, (idx, prob) in enumerate(zip(topk.indices.tolist(), topk.values.tolist())):
        uci = internal_move_to_uci(int(legal[idx]))
        print(f"      {rank+1}. {uci}  prob={prob:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transformer-checkpoint", required=True)
    ap.add_argument("--cnn-engine", required=True)
    ap.add_argument("--cnn-weights", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--moves-prefix", default="",
                    help="UCI moves separated by spaces, applied to start position")
    args = ap.parse_args()

    print(f"loading Transformer: {args.transformer_checkpoint}")
    tr = load_transformer(Path(args.transformer_checkpoint), args.device)
    print(f"loading CNN: {args.cnn_weights}")
    cn = load_cnn(Path(args.cnn_engine), Path(args.cnn_weights), args.device)

    board = Board()
    moves = args.moves_prefix.split() if args.moves_prefix else []
    for u in moves:
        mv = uci_move_to_internal(u[:4])
        board.push(int(mv))

    stm = "BLACK" if int(board.turn()) == 1 else "RED"
    print(f"\n=== POSITION (after {len(moves)} moves; STM = {stm}) ===")
    print(f"FEN: {board.fen()}")

    e_tr = eval_transformer(tr, board, args.device)
    e_cn = eval_cnn(cn, board, args.device)
    print_eval("Transformer (Stage 1 final, step 181K)", e_tr, board)
    print_eval("CNN (best.pth)", e_cn, board)

    # value-disagreement summary
    print(f"\n  value gap: Transformer({e_tr['value']:+.3f}) - CNN({e_cn['value']:+.3f}) = {e_tr['value']-e_cn['value']:+.3f}")


if __name__ == "__main__":
    main()
