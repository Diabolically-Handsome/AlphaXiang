#!/usr/bin/env python3
"""Build candidate-level Root-Q error features from V13 root-regret audits."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


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


def _load_jsonl(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    grouped[_root_key(row)].append(row)
    return grouped


def _load_trajectory_classes(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for record in payload.get("records", []) or []:
        root = record.get("root", {})
        key = str(root.get("key", ""))
        if key:
            out[key] = str(record.get("classification", "unknown"))
    return out


def _float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _log_prob(value: Any) -> float:
    return math.log(max(_float(value, 0.0), 1e-8))


def _rank_feature(value: Any, missing_value: float = 99.0) -> tuple[float, float]:
    if value is None:
        return missing_value, 1.0
    try:
        return float(value), 0.0
    except (TypeError, ValueError):
        return missing_value, 1.0


def _source_flags(row: dict[str, Any]) -> dict[str, float]:
    sources = set(str(s) for s in (row.get("candidate_sources") or []))
    return {
        "source_mcts_visit": float("mcts_visit" in sources),
        "source_raw_policy": float("raw_policy" in sources),
        "source_selected": float("selected" in sources),
        "source_legal_check": float("legal_check" in sources),
        "source_legal_capture": float("legal_capture" in sources),
        "source_immediate_win": float("immediate_terminal_win" in sources),
    }


def _selected_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next((row for row in rows if bool(row.get("is_selected"))), rows[0])


def _teacher_best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next((row for row in rows if bool(row.get("is_teacher_best"))), min(rows, key=lambda row: _float(row.get("regret_cp"), 0.0)))


def _root_is_catastrophic(selected: dict[str, Any]) -> bool:
    regret = _float(selected.get("regret_cp"), 0.0)
    return bool(regret >= 150.0 and (regret >= 1000.0 or selected.get("pika_mate_in_child") is not None))


def _features_for_row(row: dict[str, Any], selected: dict[str, Any]) -> dict[str, float]:
    pol_rank, pol_missing = _rank_feature(row.get("v13_policy_rank"))
    visit_rank, visit_missing = _rank_feature(row.get("v13_visit_rank"))
    q_rank, q_rank_missing = _rank_feature(row.get("v13_root_q_rank"))
    selected_q = _float(selected.get("v13_root_q"), 0.0)
    selected_model_q = _float(selected.get("v13_model_q_root_pov"), 0.0)
    selected_visit = _float(selected.get("v13_visit_prob"), 0.0)
    selected_prior = _float(selected.get("v13_root_prior"), _float(selected.get("v13_prior"), 0.0))
    selected_ucb = _float(selected.get("v13_root_ucb_score"), 0.0)

    root_prior = _float(row.get("v13_root_prior"), _float(row.get("v13_prior"), 0.0))
    root_q = _float(row.get("v13_root_q"), 0.0)
    model_q = _float(row.get("v13_model_q_root_pov"), 0.0)
    visit_prob = _float(row.get("v13_visit_prob"), 0.0)
    ucb = _float(row.get("v13_root_ucb_score"), 0.0)

    feats = {
        "is_selected": float(bool(row.get("is_selected"))),
        "is_mcts_best": float(str(row.get("candidate_move")) == str(row.get("mcts_best_move"))),
        "is_capture": float(bool(row.get("is_capture"))),
        "gives_check": float(bool(row.get("gives_check"))),
        "policy_rank": pol_rank,
        "policy_rank_missing": pol_missing,
        "policy_prob_log": _log_prob(row.get("v13_prior")),
        "root_prior_log": _log_prob(row.get("v13_root_prior")),
        "root_prior": root_prior,
        "visit_rank": visit_rank,
        "visit_rank_missing": visit_missing,
        "visit_prob_log": _log_prob(row.get("v13_visit_prob")),
        "visit_prob": visit_prob,
        "visit_count_log": math.log1p(max(_float(row.get("v13_visit_count"), 0.0), 0.0)),
        "root_q": root_q,
        "root_q_missing": float(row.get("v13_root_q") is None),
        "root_q_rank": q_rank,
        "root_q_rank_missing": q_rank_missing,
        "root_ucb": ucb,
        "model_child_q": model_q,
        "model_child_q_missing": float(row.get("v13_model_q_root_pov") is None),
        "mcts_root_value": _float(row.get("mcts_root_value"), 0.0),
        "ply_norm": _float(row.get("ply"), 0.0) / 300.0,
        "search_plies_norm": _float(row.get("search_plies"), 0.0) / 300.0,
        "no_capture_norm": _float(row.get("no_capture_count"), 0.0) / 120.0,
        "repetition_hint": _float(row.get("repetition_count_hint"), 1.0),
        "q_minus_selected": root_q - selected_q,
        "model_q_minus_selected": model_q - selected_model_q,
        "visit_minus_selected": visit_prob - selected_visit,
        "prior_minus_selected": root_prior - selected_prior,
        "ucb_minus_selected": ucb - selected_ucb,
        "visit_ratio_selected": visit_prob / max(selected_visit, 1e-6),
        "prior_ratio_selected": root_prior / max(selected_prior, 1e-6),
    }
    feats.update(_source_flags(row))
    return feats


def _build_rows(grouped: dict[str, list[dict[str, Any]]], classes: dict[str, str], max_roots: int) -> list[dict[str, Any]]:
    keys = sorted(grouped)
    if max_roots > 0:
        keys = keys[:max_roots]
    out: list[dict[str, Any]] = []
    for key in keys:
        rows = grouped[key]
        selected = _selected_row(rows)
        teacher = _teacher_best_row(rows)
        selected_regret = _float(selected.get("regret_cp"), 0.0)
        root_class = classes.get(key, "unknown")
        root_bad = selected_regret >= 150.0
        root_cat = _root_is_catastrophic(selected)
        selected_child = _float(selected.get("child_d16_score_cp", selected.get("teacher_child_q_root_pov_cp")), 0.0)
        teacher_child = _float(teacher.get("child_d16_score_cp", teacher.get("teacher_child_q_root_pov_cp")), 0.0)
        for row in rows:
            move = str(row.get("candidate_move", ""))[:4]
            child = _float(row.get("child_d16_score_cp", row.get("teacher_child_q_root_pov_cp")), 0.0)
            regret = _float(row.get("regret_cp"), 0.0)
            item = {
                "root_key": key,
                "audit_json": row.get("audit_json"),
                "fen": row.get("fen"),
                "game_index": row.get("game_index"),
                "ply": row.get("ply"),
                "candidate_move": move,
                "selected_move": str(selected.get("candidate_move", selected.get("selected_move", "")))[:4],
                "teacher_best_move": str(teacher.get("candidate_move", ""))[:4],
                "mcts_best_move": row.get("mcts_best_move"),
                "root_class": root_class,
                "root_selected_bad": bool(root_bad),
                "root_catastrophic": bool(root_cat),
                "selected_regret_cp": selected_regret,
                "selected_child_d16_score_cp": selected_child,
                "teacher_best_child_d16_score_cp": teacher_child,
                "candidate_child_d16_score_cp": child,
                "candidate_regret_cp": regret,
                "candidate_is_refuted": bool(regret >= 150.0),
                "candidate_is_selected": bool(row.get("is_selected")),
                "candidate_is_teacher_best": bool(row.get("is_teacher_best")),
                "candidate_mate_in_child": row.get("pika_mate_in_child"),
                "v13_visit_rank": row.get("v13_visit_rank"),
                "v13_policy_rank": row.get("v13_policy_rank"),
                "v13_root_q": row.get("v13_root_q"),
                "v13_model_q_root_pov": row.get("v13_model_q_root_pov"),
                "v13_visit_prob": row.get("v13_visit_prob"),
                "features": _features_for_row(row, selected),
            }
            out.append(item)
    return out


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    roots = {row["root_key"] for row in rows}
    selected = [row for row in rows if row["candidate_is_selected"]]
    bad_selected = [row for row in selected if row["root_selected_bad"]]
    q_errors = []
    for row in bad_selected:
        q = row.get("v13_root_q")
        child = row.get("candidate_child_d16_score_cp")
        if q is not None and child is not None:
            q_errors.append(float(q) - float(child) / 1000.0)
    worst = sorted(bad_selected, key=lambda r: float(r["selected_regret_cp"]), reverse=True)[:12]
    return {
        "roots": len(roots),
        "candidate_rows": len(rows),
        "selected_rows": len(selected),
        "bad_selected_roots": len(bad_selected),
        "catastrophic_selected_roots": sum(1 for row in bad_selected if row["root_catastrophic"]),
        "root_class_counts": dict(Counter(row["root_class"] for row in selected)),
        "candidate_refuted_rows": sum(1 for row in rows if row["candidate_is_refuted"]),
        "selected_q_error_proxy": {
            "mean": None if not q_errors else mean(q_errors),
            "median": None if not q_errors else median(q_errors),
            "max": None if not q_errors else max(q_errors),
        },
        "worst_bad_roots": [
            {
                "regret_cp": row["selected_regret_cp"],
                "selected": row["selected_move"],
                "teacher_best": row["teacher_best_move"],
                "class": row["root_class"],
                "v13_root_q": row["v13_root_q"],
                "child_d16": row["candidate_child_d16_score_cp"],
            }
            for row in worst
        ],
    }


def _write_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# V13.5 Root-Q Error Audit",
        "",
        f"- roots: {summary['roots']}",
        f"- candidate rows: {summary['candidate_rows']}",
        f"- bad selected roots: {summary['bad_selected_roots']}",
        f"- catastrophic selected roots: {summary['catastrophic_selected_roots']}",
        f"- class counts: {summary['root_class_counts']}",
        "",
        "## Worst Bad Roots",
        "",
        "| regret | selected | teacher | class | V13 Q | child d16 |",
        "|---:|---|---|---|---:|---:|",
    ]
    for row in summary["worst_bad_roots"]:
        lines.append(
            f"| {row['regret_cp']:.1f} | {row['selected']} | {row['teacher_best']} | "
            f"{row['class']} | {row['v13_root_q']} | {row['child_d16']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root_regret_jsonl", nargs="+")
    parser.add_argument("--trajectory-json", default="")
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-summary", required=True)
    parser.add_argument("--out-md", default="")
    parser.add_argument("--max-roots", type=int, default=0)
    args = parser.parse_args()

    grouped = _load_jsonl([Path(path) for path in args.root_regret_jsonl])
    classes = _load_trajectory_classes(Path(args.trajectory_json) if args.trajectory_json else None)
    rows = _build_rows(grouped, classes, max_roots=int(args.max_roots))
    summary = _summary(rows)

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.out_md:
        _write_md(Path(args.out_md), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
