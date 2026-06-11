#!/usr/bin/env python3
"""Offline root-ranking evaluator for V13.4 repair checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))

from pikafish_opponent import uci_move_to_internal  # noqa: E402
from xiangqi_mcts_ext import Board, canonical_action  # noqa: E402
from xiangqi_transformer_model import build_model_from_checkpoint_state  # noqa: E402


def _pad_fen(fen: str) -> str:
    parts = str(fen).strip().split()
    while len(parts) < 6:
        if len(parts) in {2, 3}:
            parts.append("-")
        elif len(parts) == 4:
            parts.append("0")
        else:
            parts.append("1")
    return " ".join(parts)


def _root_key(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            str(row.get("audit_json", "")),
            str(row.get("fen", "")),
            str(row.get("game_index", "")),
            str(row.get("ply", "")),
            str(row.get("selected_move", "")),
        ]
    )


def _load_grouped(paths: list[Path]) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    grouped[_root_key(row)].append(row)
    return [grouped[key] for key in sorted(grouped)]


def _selected_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next((row for row in rows if bool(row.get("is_selected"))), rows[0])


def _teacher_best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next((row for row in rows if bool(row.get("is_teacher_best"))), min(rows, key=lambda row: float(row.get("regret_cp", 0.0))))


def _board_from_row(row: dict[str, Any]) -> Board:
    board = Board()
    board.set_fen(_pad_fen(str(row.get("fen", ""))))
    board.set_search_context(
        int(row.get("search_plies", row.get("ply", 0)) or 0),
        int(row.get("no_capture_count", 0) or 0),
        int(row.get("repetition_count_hint", 1) or 1),
    )
    return board


def _canonical_move(board: Board, move_uci: str) -> int | None:
    try:
        raw = int(uci_move_to_internal(str(move_uci)[:4]))
    except Exception:
        return None
    if not bool(board.is_legal(raw)):
        return None
    stm_black = bool(int(board.turn()) == 1)
    return int(canonical_action(raw, stm_black))


def _legal_idxs(board: Board) -> list[int]:
    stm_black = bool(int(board.turn()) == 1)
    return [int(canonical_action(int(move), stm_black)) for move in board.legal_moves()]


def _load_model(checkpoint: Path, device: torch.device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model_from_checkpoint_state(state)
    model.to(device)
    model.eval()
    return model, int(state.get("global_step", 0))


@torch.inference_mode()
def _forward(model, states: torch.Tensor, device: torch.device, use_bfloat16: bool, batch_size: int) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    autocast_enabled = bool(use_bfloat16 and device.type == "cuda")
    for start in range(0, int(states.shape[0]), max(1, int(batch_size))):
        stop = min(start + max(1, int(batch_size)), int(states.shape[0]))
        batch = states[start:stop].to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            out = model(batch)
        chunks.append(out["policy_logits"].detach().cpu().float())
    return torch.cat(chunks, dim=0)


def _masked_log_probs(logits: torch.Tensor, legal: list[int]) -> torch.Tensor:
    mask = torch.full_like(logits.float(), -1e9)
    idxs = torch.tensor(legal, dtype=torch.long)
    mask[idxs] = logits.float()[idxs]
    return F.log_softmax(mask, dim=0)


def _root_payload(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    selected = _selected_row(rows)
    teacher = _teacher_best_row(rows)
    board = _board_from_row(selected)
    candidates: list[dict[str, Any]] = []
    by_idx: dict[int, dict[str, Any]] = {}
    for row in rows:
        idx = _canonical_move(board, str(row.get("candidate_move", "")))
        if idx is None:
            continue
        move = str(row.get("candidate_move", ""))[:4]
        item = {
            "move": move,
            "idx": int(idx),
            "regret_cp": float(row.get("regret_cp", 0.0) or 0.0),
            "child_score_cp": float(row.get("child_d16_score_cp", row.get("teacher_child_q_root_pov_cp", 0.0))),
            "is_selected": bool(row.get("is_selected")),
            "is_teacher_best": bool(row.get("is_teacher_best")),
            "is_refuted": bool(row.get("is_refuted")),
        }
        prev = by_idx.get(int(idx))
        if prev is None or item["regret_cp"] < prev["regret_cp"]:
            by_idx[int(idx)] = item
        else:
            prev["is_selected"] = prev["is_selected"] or item["is_selected"]
            prev["is_teacher_best"] = prev["is_teacher_best"] or item["is_teacher_best"]
            prev["is_refuted"] = prev["is_refuted"] or item["is_refuted"]
    candidates = list(by_idx.values())
    if len(candidates) < 2:
        return None
    selected_idx = _canonical_move(board, str(selected.get("candidate_move", selected.get("selected_move", ""))))
    teacher_idx = _canonical_move(board, str(teacher.get("candidate_move", "")))
    if selected_idx is None or teacher_idx is None:
        return None
    selected_regret = float(selected.get("regret_cp", 0.0) or 0.0)
    return {
        "board": board,
        "state": board.to_tensor_canonical().to(torch.float32)[0].contiguous(),
        "legal_idxs": _legal_idxs(board),
        "fen": _pad_fen(str(selected.get("fen", ""))),
        "selected_move": str(selected.get("candidate_move", selected.get("selected_move", "")))[:4],
        "teacher_best_move": str(teacher.get("candidate_move", ""))[:4],
        "selected_idx": int(selected_idx),
        "teacher_idx": int(teacher_idx),
        "selected_regret_cp": selected_regret,
        "is_bad": bool(selected_regret >= 150.0),
        "is_catastrophic": bool(selected_regret >= 150.0 and (selected_regret >= 1000.0 or selected.get("pika_mate_in_child") is not None)),
        "termination": str(selected.get("termination", "") or ""),
        "game_index": selected.get("game_index"),
        "ply": selected.get("ply"),
        "candidates": candidates,
    }


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if math.isfinite(float(value)):
            return float(value)
    except Exception:
        return None
    return None


def _write_md(path: Path, summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    lines = [
        "# V13.4 Root Repair Offline Eval",
        "",
        f"- checkpoint: `{summary['checkpoint']}`",
        f"- roots: {summary['roots']}",
        f"- bad roots: {summary['bad_roots']}",
        f"- bad teacher-over-selected repairs: {summary['bad_teacher_margin_repairs']}/{summary['bad_roots']}",
        f"- bad safe-top repairs: {summary['bad_safe_top_repairs']}/{summary['bad_roots']}",
        f"- catastrophic repairs: {summary['catastrophic_repairs']}/{summary['catastrophic_roots']}",
        f"- non-bad regressions: {summary['non_bad_regressions']}/{summary['non_bad_roots']} ({summary['non_bad_regression_rate_pct']:.1f}%)",
        f"- new non-bad regressions vs anchor: {summary['new_non_bad_regressions_vs_anchor']}/{summary['non_bad_roots']} ({summary['new_non_bad_regression_vs_anchor_rate_pct']:.1f}%)",
        f"- median anchor KL: {summary.get('median_anchor_kl', 'n/a')}",
        "",
    ]
    bad = [rec for rec in records if rec["is_bad"]]
    if bad:
        lines += [
            "## Bad Roots",
            "",
            "| regret | selected | teacher | top | top regret | teacher-selected gap | fixed? |",
            "|---:|---|---|---|---:|---:|---|",
        ]
        for rec in sorted(bad, key=lambda row: float(row["selected_regret_cp"]), reverse=True):
            gap = rec.get("teacher_minus_selected_logp")
            gap_s = "n/a" if gap is None else f"{float(gap):.3f}"
            top_regret = rec.get("model_top_candidate_regret_cp")
            top_s = "n/a" if top_regret is None else f"{float(top_regret):.1f}"
            fixed = "yes" if rec.get("teacher_margin_repaired") or rec.get("safe_top_repaired") else "no"
            lines.append(
                f"| {rec['selected_regret_cp']:.1f} | {rec['selected_move']} | {rec['teacher_best_move']} | "
                f"{rec.get('model_top_candidate_move')} | {top_s} | {gap_s} | {fixed} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root_regret_jsonl", nargs="+")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--anchor-checkpoint", default="")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--disable-bf16", action="store_true")
    parser.add_argument("--repair-margin-logp", type=float, default=0.25)
    args = parser.parse_args()

    grouped = _load_grouped([Path(path) for path in args.root_regret_jsonl])
    roots = [payload for rows in grouped if (payload := _root_payload(rows)) is not None]
    if not roots:
        raise SystemExit("no evaluable roots")
    device = torch.device(args.device)
    use_bfloat16 = not bool(args.disable_bf16)
    model, step = _load_model(Path(args.checkpoint), device)
    states = torch.stack([root["state"] for root in roots], dim=0)
    logits = _forward(model, states, device, use_bfloat16, int(args.batch_size))

    anchor_logits = None
    if args.anchor_checkpoint:
        anchor_model, _anchor_step = _load_model(Path(args.anchor_checkpoint), device)
        anchor_logits = _forward(anchor_model, states, device, use_bfloat16, int(args.batch_size))

    records: list[dict[str, Any]] = []
    anchor_kls: list[float] = []
    for i, root in enumerate(roots):
        legal = list(root["legal_idxs"])
        logp = _masked_log_probs(logits[i], legal)
        anchor_kl = None
        anchor_logp = None
        anchor_top = None
        if anchor_logits is not None:
            anchor_logp = _masked_log_probs(anchor_logits[i], legal)
            idxs = torch.tensor(legal, dtype=torch.long)
            anchor_probs = torch.exp(anchor_logp[idxs])
            anchor_kl = float((anchor_probs * (anchor_logp[idxs] - logp[idxs])).sum().item())
            anchor_kls.append(anchor_kl)

        candidate_rows: list[dict[str, Any]] = []
        for cand in root["candidates"]:
            idx = int(cand["idx"])
            item = {
                **cand,
                "logp": float(logp[idx].item()),
                "prob": float(torch.exp(logp[idx]).item()),
            }
            if anchor_logp is not None:
                item["anchor_logp"] = float(anchor_logp[idx].item())
                item["anchor_prob"] = float(torch.exp(anchor_logp[idx]).item())
            candidate_rows.append(item)
        if anchor_logp is not None:
            anchor_order = sorted(candidate_rows, key=lambda row: float(row.get("anchor_logp", -1e9)), reverse=True)
            anchor_top = anchor_order[0] if anchor_order else None
        candidate_rows.sort(key=lambda row: float(row["logp"]), reverse=True)
        top = candidate_rows[0]
        selected = next((row for row in candidate_rows if row["idx"] == root["selected_idx"]), None)
        teacher = next((row for row in candidate_rows if row["idx"] == root["teacher_idx"]), None)
        teacher_gap = None
        teacher_margin_repaired = False
        if selected is not None and teacher is not None:
            teacher_gap = float(teacher["logp"]) - float(selected["logp"])
            teacher_margin_repaired = bool(root["is_bad"] and teacher_gap >= float(args.repair_margin_logp))
        safe_top_repaired = bool(
            root["is_bad"]
            and top.get("idx") != root["selected_idx"]
            and float(top.get("regret_cp", 1e9)) < 150.0
        )
        non_bad_regression = bool(
            not root["is_bad"]
            and float(top.get("regret_cp", 0.0)) > float(root["selected_regret_cp"]) + 100.0
            and float(top.get("regret_cp", 0.0)) >= 150.0
        )
        anchor_top_regret = None if anchor_top is None else float(anchor_top.get("regret_cp", 0.0))
        new_non_bad_regression_vs_anchor = bool(
            not root["is_bad"]
            and anchor_top_regret is not None
            and float(top.get("regret_cp", 0.0)) > float(anchor_top_regret) + 100.0
            and float(top.get("regret_cp", 0.0)) >= 150.0
        )
        records.append(
            {
                "fen": root["fen"],
                "game_index": root["game_index"],
                "ply": root["ply"],
                "termination": root["termination"],
                "selected_move": root["selected_move"],
                "teacher_best_move": root["teacher_best_move"],
                "selected_regret_cp": float(root["selected_regret_cp"]),
                "is_bad": bool(root["is_bad"]),
                "is_catastrophic": bool(root["is_catastrophic"]),
                "model_top_candidate_move": top["move"],
                "model_top_candidate_regret_cp": float(top["regret_cp"]),
                "anchor_top_candidate_move": None if anchor_top is None else anchor_top.get("move"),
                "anchor_top_candidate_regret_cp": anchor_top_regret,
                "selected_logp": None if selected is None else float(selected["logp"]),
                "teacher_best_logp": None if teacher is None else float(teacher["logp"]),
                "teacher_minus_selected_logp": teacher_gap,
                "teacher_margin_repaired": bool(teacher_margin_repaired),
                "safe_top_repaired": bool(safe_top_repaired),
                "non_bad_regression": bool(non_bad_regression),
                "new_non_bad_regression_vs_anchor": bool(new_non_bad_regression_vs_anchor),
                "anchor_kl": anchor_kl,
                "candidate_top5": candidate_rows[:5],
            }
        )

    bad = [rec for rec in records if rec["is_bad"]]
    non_bad = [rec for rec in records if not rec["is_bad"]]
    catastrophic = [rec for rec in records if rec["is_catastrophic"]]
    non_bad_regressions = sum(1 for rec in non_bad if rec["non_bad_regression"])
    new_non_bad_regressions_vs_anchor = sum(1 for rec in non_bad if rec.get("new_non_bad_regression_vs_anchor"))
    summary = {
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_step": int(step),
        "root_regret_jsonl": [str(Path(path)) for path in args.root_regret_jsonl],
        "roots": len(records),
        "bad_roots": len(bad),
        "non_bad_roots": len(non_bad),
        "catastrophic_roots": len(catastrophic),
        "bad_teacher_margin_repairs": sum(1 for rec in bad if rec["teacher_margin_repaired"]),
        "bad_safe_top_repairs": sum(1 for rec in bad if rec["safe_top_repaired"]),
        "catastrophic_repairs": sum(1 for rec in catastrophic if rec["teacher_margin_repaired"] or rec["safe_top_repaired"]),
        "non_bad_regressions": non_bad_regressions,
        "non_bad_regression_rate_pct": 0.0 if not non_bad else 100.0 * non_bad_regressions / float(len(non_bad)),
        "new_non_bad_regressions_vs_anchor": new_non_bad_regressions_vs_anchor,
        "new_non_bad_regression_vs_anchor_rate_pct": 0.0 if not non_bad else 100.0 * new_non_bad_regressions_vs_anchor / float(len(non_bad)),
        "mean_top_candidate_regret_cp": _safe_float(mean([float(rec["model_top_candidate_regret_cp"]) for rec in records])),
        "median_top_candidate_regret_cp": _safe_float(median([float(rec["model_top_candidate_regret_cp"]) for rec in records])),
        "median_anchor_kl": None if not anchor_kls else float(median(anchor_kls)),
        "mean_anchor_kl": None if not anchor_kls else float(mean(anchor_kls)),
    }
    payload = {"summary": summary, "records": records}
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.out_md:
        _write_md(Path(args.out_md), summary, records)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
