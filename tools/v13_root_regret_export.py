#!/usr/bin/env python3
"""Export root-decision audit JSON into candidate-regret JSONL.

The output is intentionally training-oriented but model-agnostic.  Each row is a
single (root position, candidate move) with V13 search/policy features and a
Pikafish child-eval regret target:

  regret_cp = teacher_best_q_root_pov_cp - candidate_q_root_pov_cp

This lets the next experiment train or analyze root-regret/ranking targets from
real V13 arena decisions instead of generic random-rollout distillation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any


def _detail(row: dict[str, Any], source: str) -> dict[str, Any]:
    details = row.get("source_details", {})
    return details.get(source, {}) if isinstance(details, dict) else {}


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_records(paths: list[Path]) -> list[tuple[Path, dict[str, Any], dict[str, Any]]]:
    out: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload.get("summary", {})
        summary = summary if isinstance(summary, dict) else {}
        for record in payload.get("records", []) or []:
            out.append((path, record, summary))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_json", nargs="+")
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-summary", default="")
    parser.add_argument("--regret-margin-cp", type=float, default=150.0)
    parser.add_argument("--max-candidates-per-position", type=int, default=0)
    parser.add_argument("--only-selected-or-refuted", action="store_true")
    args = parser.parse_args()

    records = _load_records([Path(path) for path in args.audit_json])
    rows: list[dict[str, Any]] = []
    position_count = 0
    refuted_count = 0
    selected_rows = 0
    regrets: list[float] = []

    for audit_path, record, audit_summary in records:
        candidates = [
            row for row in record.get("candidate_rows", []) or []
            if row.get("pika_q_root_pov_cp") is not None
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda row: float(row["pika_q_root_pov_cp"]), reverse=True)
        if int(args.max_candidates_per_position) > 0:
            candidates = candidates[: int(args.max_candidates_per_position)]
        teacher_best_q = float(candidates[0]["pika_q_root_pov_cp"])
        teacher_best_move = str(candidates[0]["move_uci"])
        position = record.get("position", {})
        mcts = record.get("mcts", {})
        mcts_q_order = [
            str(row.get("move_uci"))
            for row in sorted(
                candidates,
                key=lambda row: _float_or_none(_detail(row, "mcts_visit").get("q_root_pov")) or -999.0,
                reverse=True,
            )
            if "mcts_visit" in (row.get("sources", []) or [])
        ]
        position_count += 1

        for rank, row in enumerate(candidates, start=1):
            q_cp = float(row["pika_q_root_pov_cp"])
            regret_cp = teacher_best_q - q_cp
            is_refuted = regret_cp >= float(args.regret_margin_cp)
            is_selected = bool(row.get("is_selected"))
            if args.only_selected_or_refuted and not (is_selected or is_refuted):
                continue
            raw = _detail(row, "raw_policy")
            visit = _detail(row, "mcts_visit")
            pika_root = _detail(row, "pika_multipv")
            root_q_rank = None
            move_uci = str(row.get("move_uci"))
            if move_uci in mcts_q_order:
                root_q_rank = mcts_q_order.index(move_uci) + 1
            out_row = {
                "audit_json": str(audit_path),
                "pika_label_root_depth": _int_or_none(audit_summary.get("pika_root_depth")),
                "pika_label_child_depth": _int_or_none(audit_summary.get("pika_child_depth")),
                "pika_label_root_multipv": _int_or_none(audit_summary.get("pika_root_multipv")),
                "fen": position.get("fen"),
                "side_to_move": "black" if " b " in str(position.get("fen", "")) else "red",
                "our_side": position.get("our_side"),
                "opening_id": position.get("opening_id"),
                "opening_index": position.get("opening_index"),
                "game_index": position.get("game_index"),
                "ply": position.get("ply"),
                "search_plies": position.get("search_plies"),
                "no_capture_count": position.get("no_capture_count"),
                "repetition_count_hint": position.get("repetition_count_hint"),
                "termination": position.get("termination"),
                "selected_move": position.get("chosen_uci"),
                "mcts_best_move": mcts.get("best_move_uci"),
                "mcts_root_value": mcts.get("root_value"),
                "candidate_move": row.get("move_uci"),
                "candidate_rank_by_teacher_child": int(rank),
                "candidate_sources": row.get("sources", []),
                "is_selected": is_selected,
                "is_teacher_best": str(row.get("move_uci")) == teacher_best_move,
                "v13_prior": _float_or_none(raw.get("prob")),
                "v13_policy_rank": _int_or_none(raw.get("rank")),
                "v13_root_prior": _float_or_none(visit.get("prior")),
                "v13_visit_count": _int_or_none(visit.get("visit_count")),
                "v13_visit_prob": _float_or_none(visit.get("visit_prob")),
                "v13_visit_rank": _int_or_none(visit.get("rank")),
                "v13_root_q": _float_or_none(visit.get("q_root_pov")),
                "v13_root_q_rank": root_q_rank,
                "v13_root_ucb_score": _float_or_none(visit.get("ucb_score")),
                "v13_model_q_root_pov": _float_or_none(row.get("model_q_root_pov")),
                "pika_root_multipv_rank": _int_or_none(pika_root.get("rank")),
                "pika_root_score_cp": _float_or_none(pika_root.get("root_score_cp")),
                "pika_root_d14_score_cp": _float_or_none(pika_root.get("root_score_cp")),
                "pika_root_d20_score_cp": _float_or_none(pika_root.get("root_score_cp")),
                "teacher_child_q_root_pov_cp": q_cp,
                "child_d16_score_cp": q_cp,
                "child_d20_score_cp": q_cp,
                "teacher_best_move": teacher_best_move,
                "teacher_best_q_root_pov_cp": teacher_best_q,
                "regret_cp": regret_cp,
                "is_refuted": is_refuted,
                "is_capture": bool(row.get("is_capture")),
                "gives_check": bool(row.get("gives_check")),
                "pika_mate_in_child": row.get("pika_mate_in_child"),
            }
            rows.append(out_row)
            regrets.append(regret_cp)
            refuted_count += int(is_refuted)
            selected_rows += int(is_selected)

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "audit_json": [str(Path(path)) for path in args.audit_json],
        "positions": position_count,
        "rows": len(rows),
        "selected_rows": selected_rows,
        "refuted_rows": refuted_count,
        "regret_margin_cp": float(args.regret_margin_cp),
        "regret_cp": {
            "mean": None if not regrets else float(mean(regrets)),
            "median": None if not regrets else float(median(regrets)),
            "max": None if not regrets else float(max(regrets)),
        },
        "out_jsonl": str(out_jsonl),
    }
    if args.out_summary:
        out_summary = Path(args.out_summary)
        out_summary.parent.mkdir(parents=True, exist_ok=True)
        out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
