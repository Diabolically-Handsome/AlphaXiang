#!/usr/bin/env python3
"""Offline evaluator for root top-K child-verifier gates.

The input is a ``tools/v13_root_decision_audit.py`` JSON.  The tool replays the
same decision rule as ``external_arena.py --our-pikafish-verifier`` using the
already-recorded child evaluations:

* candidates are MCTS root top-K plus the selected move;
* lower child eval from the opponent POV is better for us;
* override only when the original child eval is above ``danger_threshold_cp``
  and the improvement reaches ``margin_cp``.

No engine is queried here.  This is a cheap gate-design pass before arena.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any


def _csv_ints(raw: str) -> list[int]:
    vals: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            vals.append(int(part))
    if not vals:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return vals


def _selected_row(record: dict[str, Any]) -> dict[str, Any] | None:
    for row in record.get("candidate_rows", []):
        if bool(row.get("is_selected")):
            return row
    return None


def _mcts_rank(row: dict[str, Any]) -> int | None:
    details = row.get("source_details")
    if not isinstance(details, dict):
        return None
    mcts = details.get("mcts_visit")
    if not isinstance(mcts, dict):
        return None
    try:
        return int(mcts.get("rank"))
    except Exception:
        return None


def _candidate_pool(record: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in record.get("candidate_rows", []):
        rank = _mcts_rank(row)
        if rank is None or rank > int(top_k):
            continue
        move = str(row.get("move_uci", ""))
        if move and move not in seen:
            rows.append(row)
            seen.add(move)
    selected = _selected_row(record)
    if selected is not None:
        move = str(selected.get("move_uci", ""))
        if move and move not in seen:
            rows.insert(0, selected)
    return rows


def _child_opp_cp(row: dict[str, Any]) -> int | None:
    value = row.get("pika_child_eval_opponent_pov_cp")
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _root_cp(row: dict[str, Any]) -> float | None:
    value = row.get("pika_q_root_pov_cp")
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _teacher_best_root_cp(record: dict[str, Any]) -> float | None:
    best = record.get("teacher_best")
    if not isinstance(best, dict):
        return None
    value = best.get("pika_q_root_pov_cp")
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _regret_cp(record: dict[str, Any], row: dict[str, Any]) -> float | None:
    best_cp = _teacher_best_root_cp(record)
    row_cp = _root_cp(row)
    if best_cp is None or row_cp is None:
        return None
    return max(0.0, best_cp - row_cp)


def _is_bad(regret_cp: float | None, margin_cp: float) -> bool:
    return regret_cp is not None and regret_cp >= float(margin_cp)


def _is_catastrophic(regret_cp: float | None) -> bool:
    return regret_cp is not None and regret_cp >= 1000.0


def _evaluate_config(
    records: list[dict[str, Any]],
    *,
    top_k: int,
    margin_cp: int,
    danger_threshold_cp: int,
    bad_margin_cp: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    selected_regrets: list[float] = []
    final_regrets: list[float] = []
    replacements = 0
    bad_before = 0
    bad_after = 0
    bad_fixed = 0
    clean_regressions = 0
    catastrophic_before = 0
    catastrophic_after = 0
    catastrophic_regressions = 0

    for idx, record in enumerate(records):
        selected = _selected_row(record)
        if selected is None:
            continue
        selected_opp_cp = _child_opp_cp(selected)
        selected_regret = _regret_cp(record, selected)
        if selected_opp_cp is None or selected_regret is None:
            continue
        pool = [row for row in _candidate_pool(record, top_k) if _child_opp_cp(row) is not None]
        if not pool:
            continue
        verifier_best = min(pool, key=lambda row: int(_child_opp_cp(row)))
        improvement_cp = int(selected_opp_cp) - int(_child_opp_cp(verifier_best))
        accepted = (
            int(selected_opp_cp) >= int(danger_threshold_cp)
            and int(improvement_cp) >= int(margin_cp)
            and str(verifier_best.get("move_uci", "")) != str(selected.get("move_uci", ""))
        )
        final = verifier_best if accepted else selected
        final_regret = _regret_cp(record, final)
        if final_regret is None:
            continue

        selected_bad = _is_bad(selected_regret, bad_margin_cp)
        final_bad = _is_bad(final_regret, bad_margin_cp)
        selected_cat = _is_catastrophic(selected_regret)
        final_cat = _is_catastrophic(final_regret)

        replacements += int(accepted)
        bad_before += int(selected_bad)
        bad_after += int(final_bad)
        bad_fixed += int(selected_bad and not final_bad)
        clean_regressions += int((not selected_bad) and final_bad)
        catastrophic_before += int(selected_cat)
        catastrophic_after += int(final_cat)
        catastrophic_regressions += int((not selected_cat) and final_cat)
        selected_regrets.append(float(selected_regret))
        final_regrets.append(float(final_regret))

        if accepted or selected_bad or final_bad:
            pos = record.get("position") if isinstance(record.get("position"), dict) else {}
            rows.append(
                {
                    "index": idx,
                    "game_index": pos.get("game_index"),
                    "ply": pos.get("ply"),
                    "selected_move": str(selected.get("move_uci", "")),
                    "final_move": str(final.get("move_uci", "")),
                    "verifier_best_move": str(verifier_best.get("move_uci", "")),
                    "accepted": bool(accepted),
                    "improvement_cp": int(improvement_cp),
                    "selected_child_eval_cp_opponent_pov": int(selected_opp_cp),
                    "final_child_eval_cp_opponent_pov": int(_child_opp_cp(final)),
                    "selected_regret_cp": float(selected_regret),
                    "final_regret_cp": float(final_regret),
                    "selected_bad": bool(selected_bad),
                    "final_bad": bool(final_bad),
                }
            )

    total = len(final_regrets)
    return {
        "top_k": int(top_k),
        "margin_cp": int(margin_cp),
        "danger_threshold_cp": int(danger_threshold_cp),
        "positions": int(total),
        "replacements": int(replacements),
        "bad_before": int(bad_before),
        "bad_after": int(bad_after),
        "bad_fixed": int(bad_fixed),
        "clean_regressions": int(clean_regressions),
        "catastrophic_before": int(catastrophic_before),
        "catastrophic_after": int(catastrophic_after),
        "catastrophic_regressions": int(catastrophic_regressions),
        "selected_mean_regret_cp": float(mean(selected_regrets)) if selected_regrets else None,
        "final_mean_regret_cp": float(mean(final_regrets)) if final_regrets else None,
        "selected_median_regret_cp": float(median(selected_regrets)) if selected_regrets else None,
        "final_median_regret_cp": float(median(final_regrets)) if final_regrets else None,
        "events": rows,
    }


def _score_config(row: dict[str, Any]) -> tuple[int, int, int, int, float]:
    # Higher is better.
    return (
        -int(row["clean_regressions"]),
        -int(row["catastrophic_regressions"]),
        int(row["bad_fixed"]),
        -int(row["bad_after"]),
        -float(row["final_mean_regret_cp"] or 0.0),
    )


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Root Top-K Verifier Offline Eval",
        "",
        f"- input: `{payload['input']}`",
        f"- configs: {len(payload['configs'])}",
        f"- bad margin: {payload['bad_margin_cp']}cp",
        "",
        "## Best Configs",
        "",
        "| topK | margin | danger | repl | bad before | bad after | fixed | clean reg | cat before | cat after | cat reg | final mean regret |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["best_configs"][:10]:
        mean_regret = row.get("final_mean_regret_cp")
        mean_s = "n/a" if mean_regret is None else f"{float(mean_regret):.1f}"
        lines.append(
            f"| {row['top_k']} | {row['margin_cp']} | {row['danger_threshold_cp']} | "
            f"{row['replacements']} | {row['bad_before']} | {row['bad_after']} | "
            f"{row['bad_fixed']} | {row['clean_regressions']} | "
            f"{row['catastrophic_before']} | {row['catastrophic_after']} | "
            f"{row['catastrophic_regressions']} | {mean_s} |"
        )
    best = payload["best_configs"][0] if payload["best_configs"] else None
    if best:
        lines += [
            "",
            "## Best Events",
            "",
            "| game | ply | selected | final | accepted | improvement | selected regret | final regret |",
            "|---:|---:|---|---|---|---:|---:|---:|",
        ]
        for event in best.get("events", [])[:30]:
            lines.append(
                f"| {event.get('game_index')} | {event.get('ply')} | "
                f"{event['selected_move']} | {event['final_move']} | "
                f"{'yes' if event['accepted'] else 'no'} | {event['improvement_cp']} | "
                f"{event['selected_regret_cp']:.1f} | {event['final_regret_cp']:.1f} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_json")
    parser.add_argument("--top-k", type=_csv_ints, default=[3, 4, 6, 8, 16])
    parser.add_argument("--margins", type=_csv_ints, default=[120, 200, 300, 500])
    parser.add_argument("--danger-thresholds", type=_csv_ints, default=[-20000, 0, 100, 300, 600, 19000])
    parser.add_argument("--bad-margin-cp", type=float, default=150.0)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", default="")
    args = parser.parse_args()

    audit_path = Path(args.audit_json)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    records = audit.get("records")
    if not isinstance(records, list):
        raise SystemExit(f"{audit_path} does not look like a root decision audit JSON")

    configs: list[dict[str, Any]] = []
    for top_k in args.top_k:
        for margin in args.margins:
            for danger in args.danger_thresholds:
                configs.append(
                    _evaluate_config(
                        records,
                        top_k=int(top_k),
                        margin_cp=int(margin),
                        danger_threshold_cp=int(danger),
                        bad_margin_cp=float(args.bad_margin_cp),
                    )
                )
    best_configs = sorted(configs, key=_score_config, reverse=True)
    payload = {
        "input": str(audit_path),
        "bad_margin_cp": float(args.bad_margin_cp),
        "configs": configs,
        "best_configs": best_configs,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.out_md:
        _write_md(Path(args.out_md), payload)
    print(json.dumps({k: payload[k] for k in ("input", "bad_margin_cp")}, ensure_ascii=False, indent=2))
    if best_configs:
        print(json.dumps(best_configs[0], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
