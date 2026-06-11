#!/usr/bin/env python3
"""Replay V13 root decisions across MCTS simulation milestones.

This tool is intentionally offline-only.  It reuses the frozen V13 checkpoint
and the d14/d16 root-regret rows to answer a narrower question than arena Elo:
when a bad root move is selected, did policy, Q, or visit allocation go wrong?
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))

from pikafish_opponent import internal_move_to_uci, uci_move_to_internal  # noqa: E402
from xiangqi_mcts_ext import Board, canonical_action, make_gpu_evaluator, mcts_search_with_root_stats  # noqa: E402
from xiangqi_transformer_model import build_model_from_checkpoint_state  # noqa: E402


@dataclass
class CandidateTarget:
    move_uci: str
    regret_cp: float
    child_score_cp: float
    is_selected: bool = False
    is_teacher_best: bool = False
    is_refuted: bool = False
    mate_in_child: Any = None


@dataclass
class RootTarget:
    key: str
    fen: str
    search_plies: int
    no_capture_count: int
    repetition_count_hint: int
    selected_move: str
    teacher_best_move: str
    selected_regret_cp: float
    termination: str
    opening_id: str
    opening_index: Any
    game_index: Any
    ply: int
    audit_json: str
    candidates: dict[str, CandidateTarget] = field(default_factory=dict)

    @property
    def is_bad(self) -> bool:
        return self.selected_regret_cp >= 150.0

    @property
    def is_near_bad(self) -> bool:
        return 80.0 <= self.selected_regret_cp < 150.0

    @property
    def is_catastrophic(self) -> bool:
        selected = self.candidates.get(self.selected_move)
        return bool(
            self.is_bad
            and (
                self.selected_regret_cp >= 1000.0
                or (selected is not None and selected.mate_in_child is not None)
            )
        )


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
    pieces = [
        str(row.get("audit_json", "")),
        str(row.get("fen", "")),
        str(row.get("game_index", "")),
        str(row.get("ply", "")),
        str(row.get("selected_move", "")),
    ]
    return "\n".join(pieces)


def _load_roots(jsonl_paths: list[Path]) -> list[RootTarget]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in jsonl_paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    grouped[_root_key(row)].append(row)

    roots: list[RootTarget] = []
    for key, rows in grouped.items():
        selected = next((row for row in rows if bool(row.get("is_selected"))), rows[0])
        teacher = next((row for row in rows if bool(row.get("is_teacher_best"))), None)
        if teacher is None:
            teacher = min(rows, key=lambda row: float(row.get("regret_cp", 0.0)))
        candidates: dict[str, CandidateTarget] = {}
        for row in rows:
            move = str(row.get("candidate_move", ""))[:4]
            if not move:
                continue
            prev = candidates.get(move)
            regret = float(row.get("regret_cp", 0.0))
            child_score = float(row.get("child_d16_score_cp", row.get("teacher_child_q_root_pov_cp", 0.0)))
            target = CandidateTarget(
                move_uci=move,
                regret_cp=regret,
                child_score_cp=child_score,
                is_selected=bool(row.get("is_selected")),
                is_teacher_best=bool(row.get("is_teacher_best")),
                is_refuted=bool(row.get("is_refuted")),
                mate_in_child=row.get("pika_mate_in_child"),
            )
            if prev is None or target.regret_cp < prev.regret_cp:
                candidates[move] = target
            else:
                prev.is_selected = prev.is_selected or target.is_selected
                prev.is_teacher_best = prev.is_teacher_best or target.is_teacher_best
                prev.is_refuted = prev.is_refuted or target.is_refuted
                if prev.mate_in_child is None:
                    prev.mate_in_child = target.mate_in_child

        selected_move = str(selected.get("candidate_move") or selected.get("selected_move", ""))[:4]
        teacher_best = str(teacher.get("candidate_move") or selected.get("teacher_best_move", ""))[:4]
        roots.append(
            RootTarget(
                key=key,
                fen=_pad_fen(str(selected.get("fen", ""))),
                search_plies=int(selected.get("search_plies", selected.get("ply", 0)) or 0),
                no_capture_count=int(selected.get("no_capture_count", 0) or 0),
                repetition_count_hint=int(selected.get("repetition_count_hint", 1) or 1),
                selected_move=selected_move,
                teacher_best_move=teacher_best,
                selected_regret_cp=float(selected.get("regret_cp", 0.0) or 0.0),
                termination=str(selected.get("termination", "") or ""),
                opening_id=str(selected.get("opening_id", "") or ""),
                opening_index=selected.get("opening_index"),
                game_index=selected.get("game_index"),
                ply=int(selected.get("ply", 0) or 0),
                audit_json=str(selected.get("audit_json", "")),
                candidates=candidates,
            )
        )
    roots.sort(key=lambda r: (not r.is_bad, -r.selected_regret_cp, r.audit_json, r.game_index or -1, r.ply))
    return roots


def _select_roots(
    roots: list[RootTarget],
    *,
    mode: str,
    max_roots: int,
    controls_per_bad: int,
) -> list[RootTarget]:
    if mode == "all":
        selected = roots
    elif mode == "bad":
        selected = [root for root in roots if root.is_bad]
    elif mode == "near":
        selected = [root for root in roots if root.is_near_bad]
    elif mode == "clean":
        selected = [root for root in roots if not root.is_bad and not root.is_near_bad]
    elif mode == "bad-control":
        bad = [root for root in roots if root.is_bad]
        clean = [root for root in roots if not root.is_bad and not root.is_near_bad]
        picked: list[RootTarget] = list(bad)
        used = {root.key for root in picked}
        for bad_root in bad:
            matches = [
                root for root in clean
                if root.key not in used
                and (root.termination == bad_root.termination or root.opening_id == bad_root.opening_id)
            ]
            if not matches:
                matches = [root for root in clean if root.key not in used]
            matches.sort(key=lambda r: (abs(int(r.ply) - int(bad_root.ply)), r.audit_json, r.game_index or -1))
            for ctrl in matches[: max(0, int(controls_per_bad))]:
                picked.append(ctrl)
                used.add(ctrl.key)
        selected = picked
    else:
        raise ValueError(f"unknown position mode: {mode}")
    if int(max_roots) > 0:
        selected = selected[: int(max_roots)]
    return selected


def _board_from_root(root: RootTarget) -> Board:
    board = Board()
    board.set_fen(root.fen)
    board.set_search_context(root.search_plies, root.no_capture_count, root.repetition_count_hint)
    return board


def _load_model_and_evaluator(checkpoint: Path, device: torch.device, use_bfloat16: bool):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model_from_checkpoint_state(state)
    model.to(device)
    model.eval()
    return model, make_gpu_evaluator(model, device=str(device), use_bfloat16=use_bfloat16), int(state.get("global_step", 0))


class _ValueSourceEvaluator:
    def __init__(self, base, value_source: str) -> None:
        self.base = base
        self.value_source = str(value_source)

    def __call__(self, batch_cpu: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.base(batch_cpu)
        if self.value_source == "scalar":
            return out
        if self.value_source != "wdl":
            raise ValueError(f"unknown value_source={self.value_source!r}")
        wdl_logits = out.get("wdl_logits")
        if wdl_logits is None:
            raise KeyError("--value-sources includes wdl but model output has no wdl_logits")
        probs = torch.softmax(wdl_logits.float(), dim=-1)
        out = dict(out)
        out["value_scalar"] = (probs[:, 0:1] - probs[:, 2:3]).to(dtype=torch.float32).contiguous()
        return out

    def stats(self):
        if hasattr(self.base, "stats"):
            return self.base.stats()
        raise AttributeError(f"{type(self.base).__name__!s} has no stats()")

    def close(self) -> None:
        if hasattr(self.base, "close"):
            self.base.close()


def _wrap_value_source(evaluator, value_source: str):
    value_source = str(value_source)
    if value_source == "scalar":
        return evaluator
    return _ValueSourceEvaluator(evaluator, value_source)


def _parse_value_sources(raw: str) -> list[str]:
    sources: list[str] = []
    for part in str(raw).split(","):
        value = part.strip()
        if not value:
            continue
        if value not in {"scalar", "wdl"}:
            raise ValueError(f"value source must be scalar or wdl, got {value!r}")
        if value not in sources:
            sources.append(value)
    if not sources:
        raise ValueError("at least one value source is required")
    return sources


@torch.inference_mode()
def _forward_policy(model, states: torch.Tensor, device: torch.device, use_bfloat16: bool, batch_size: int) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    autocast_enabled = bool(use_bfloat16 and device.type == "cuda")
    for start in range(0, int(states.shape[0]), max(1, int(batch_size))):
        stop = min(start + max(1, int(batch_size)), int(states.shape[0]))
        batch = states[start:stop].to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            out = model(batch)
        chunks.append(out["policy_logits"].detach().cpu().float())
    return torch.cat(chunks, dim=0)


def _legal_log_probs(logits: torch.Tensor, board: Board) -> dict[str, dict[str, Any]]:
    legal = [int(move) for move in board.legal_moves()]
    if not legal:
        return {}
    stm_black = bool(int(board.turn()) == 1)
    idxs = torch.tensor([int(canonical_action(move, stm_black)) for move in legal], dtype=torch.long)
    probs = torch.softmax(logits[idxs].float(), dim=0)
    order = torch.argsort(probs, descending=True)
    out: dict[str, dict[str, Any]] = {}
    for rank, pos in enumerate(order.tolist(), start=1):
        move = legal[int(pos)]
        out[internal_move_to_uci(move)] = {
            "rank": int(rank),
            "prob": float(probs[int(pos)].item()),
            "canonical_idx": int(idxs[int(pos)].item()),
        }
    return out


def _parse_search_configs(raw: str) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        fields = part.split(":")
        if len(fields) != 3:
            raise ValueError(f"search config must be name:c_puct:q_weight, got {part!r}")
        configs.append({"name": fields[0], "c_puct": float(fields[1]), "q_weight": float(fields[2])})
    if not configs:
        raise ValueError("at least one search config is required")
    return configs


def _parse_milestones(raw: str) -> list[int]:
    milestones = sorted({int(part.strip()) for part in str(raw).split(",") if part.strip()})
    if not milestones:
        raise ValueError("at least one milestone is required")
    return milestones


def _ranked_root_stats(root_stats: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for raw in root_stats:
        move_raw = int(raw.get("move_raw", -1))
        if move_raw < 0:
            continue
        row = {
            "move_uci": internal_move_to_uci(move_raw),
            "move_raw": move_raw,
            "canonical_idx": int(raw.get("canonical_idx", -1)),
            "visit_count": int(raw.get("visit_count", 0)),
            "visit_prob": float(raw.get("visit_prob", 0.0)),
            "prior": float(raw.get("prior", 0.0)),
            "q_root_pov": float(raw.get("q_root_pov", 0.0)),
            "q_child_pov": float(raw.get("q_child_pov", 0.0)),
            "ucb_score": float(raw.get("ucb_score", 0.0)),
            "selected": bool(raw.get("selected", False)),
        }
        rows.append(row)
    visit_order = sorted(rows, key=lambda row: row["visit_prob"], reverse=True)
    q_order = sorted(rows, key=lambda row: row["q_root_pov"], reverse=True)
    prior_order = sorted(rows, key=lambda row: row["prior"], reverse=True)
    ucb_order = sorted(rows, key=lambda row: row["ucb_score"], reverse=True)
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    for name, ordered in (("visit_rank", visit_order), ("q_rank", q_order), ("prior_rank", prior_order), ("ucb_rank", ucb_order)):
        for rank, row in enumerate(ordered, start=1):
            ranks[row["move_uci"]][name] = int(rank)
    by_move: dict[str, dict[str, Any]] = {}
    for row in rows:
        row.update(ranks.get(row["move_uci"], {}))
        by_move[row["move_uci"]] = row
    return visit_order, by_move


def _move_snapshot(move: str, by_move: dict[str, dict[str, Any]], raw_policy: dict[str, dict[str, Any]], target: CandidateTarget | None) -> dict[str, Any]:
    stat = by_move.get(move, {})
    pol = raw_policy.get(move, {})
    return {
        "move": move,
        "known_regret_cp": None if target is None else float(target.regret_cp),
        "child_score_cp": None if target is None else float(target.child_score_cp),
        "raw_policy_rank": pol.get("rank"),
        "raw_policy_prob": pol.get("prob"),
        "visit_rank": stat.get("visit_rank"),
        "visit_prob": stat.get("visit_prob"),
        "visit_count": stat.get("visit_count"),
        "q_rank": stat.get("q_rank"),
        "q_root_pov": stat.get("q_root_pov"),
        "prior_rank": stat.get("prior_rank"),
        "prior": stat.get("prior"),
        "ucb_rank": stat.get("ucb_rank"),
        "ucb_score": stat.get("ucb_score"),
    }


def _classify_root(root: RootTarget, milestones: list[dict[str, Any]], raw_policy: dict[str, dict[str, Any]]) -> str:
    if not root.is_bad:
        return "stable_correct"
    teacher_raw_rank = raw_policy.get(root.teacher_best_move, {}).get("rank")
    if teacher_raw_rank is None or int(teacher_raw_rank) > 16:
        return "policy_low"
    selected_target = root.candidates.get(root.selected_move)
    if root.is_catastrophic or (selected_target is not None and selected_target.mate_in_child is not None):
        return "horizon_mate"

    comparable = []
    for item in milestones:
        sel_q = item.get("selected", {}).get("q_root_pov")
        best_q = item.get("teacher_best", {}).get("q_root_pov")
        if sel_q is not None and best_q is not None:
            comparable.append((int(item["simulations"]), float(sel_q), float(best_q)))
    if comparable:
        first_sim, first_sel_q, first_best_q = comparable[0]
        last_sim, last_sel_q, last_best_q = comparable[-1]
        if first_best_q < first_sel_q:
            return "early_q_inversion"
        if first_best_q >= first_sel_q and last_best_q < last_sel_q:
            return "late_q_flip"
        last = milestones[-1]
        sel_v = last.get("selected", {}).get("visit_prob")
        best_v = last.get("teacher_best", {}).get("visit_prob")
        if last_best_q >= last_sel_q - 0.02 and sel_v is not None and best_v is not None and sel_v >= max(0.05, 2.0 * best_v):
            return "visit_lock_in"
    return "ranking_other"


def _known_regret(root: RootTarget, move_uci: str) -> float | None:
    target = root.candidates.get(str(move_uci)[:4])
    return None if target is None else float(target.regret_cp)


def _summarize(records: list[dict[str, Any]], search_configs: list[dict[str, Any]]) -> dict[str, Any]:
    root_count = len(records)
    bad_roots = sum(1 for rec in records if rec["root"]["is_bad"])
    out: dict[str, Any] = {
        "roots": root_count,
        "bad_roots": bad_roots,
        "catastrophic_roots": sum(1 for rec in records if rec["root"]["is_catastrophic"]),
        "classification_counts": dict(Counter(rec.get("classification", "unknown") for rec in records)),
        "configs": {},
    }
    for cfg in search_configs:
        name = str(cfg["name"])
        repaired = 0
        teacher_above_selected = 0
        regressions = 0
        judged_bad = 0
        judged_clean = 0
        top_regrets: list[float] = []
        for rec in records:
            root = rec["root"]
            cfg_rec = rec["search"].get(name, {})
            final = cfg_rec.get("final", {})
            top_regret = final.get("best_move_known_regret_cp")
            if top_regret is not None:
                top_regrets.append(float(top_regret))
            if root["is_bad"]:
                judged_bad += 1
                if top_regret is not None and float(top_regret) < 150.0 and final.get("best_move") != root["selected_move"]:
                    repaired += 1
                selected_rank = final.get("selected", {}).get("visit_rank")
                teacher_rank = final.get("teacher_best", {}).get("visit_rank")
                if selected_rank is not None and teacher_rank is not None and int(teacher_rank) < int(selected_rank):
                    teacher_above_selected += 1
            else:
                judged_clean += 1
                selected_regret = float(root.get("selected_regret_cp", 0.0))
                if top_regret is not None and float(top_regret) > selected_regret + 100.0 and float(top_regret) >= 150.0:
                    regressions += 1
        out["configs"][name] = {
            "value_source": str(cfg.get("value_source", "scalar")),
            "c_puct": float(cfg["c_puct"]),
            "q_weight": float(cfg["q_weight"]),
            "bad_repaired_known_top": repaired,
            "bad_teacher_visit_above_selected": teacher_above_selected,
            "bad_judged": judged_bad,
            "clean_regressions": regressions,
            "clean_judged": judged_clean,
            "clean_regression_rate": None if judged_clean == 0 else regressions / float(judged_clean),
            "mean_known_top_regret_cp": None if not top_regrets else mean(top_regrets),
        }
    return out


def _write_md(path: Path, summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    lines = [
        "# V13.4 Root Trajectory Audit",
        "",
        f"- roots: {summary['roots']}",
        f"- bad roots: {summary['bad_roots']}",
        f"- catastrophic roots: {summary['catastrophic_roots']}",
        f"- classification: {summary['classification_counts']}",
        "",
        "## Search Configs",
        "",
        "| config | bad repaired | teacher above selected | clean regressions | mean known top regret |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in summary["configs"].items():
        reg = row["clean_regression_rate"]
        reg_s = "n/a" if reg is None else f"{100.0 * reg:.1f}%"
        mean_s = "n/a" if row["mean_known_top_regret_cp"] is None else f"{row['mean_known_top_regret_cp']:.1f}"
        lines.append(
            f"| {name} | {row['bad_repaired_known_top']}/{row['bad_judged']} | "
            f"{row['bad_teacher_visit_above_selected']}/{row['bad_judged']} | "
            f"{row['clean_regressions']}/{row['clean_judged']} ({reg_s}) | {mean_s} |"
        )
    bad = [rec for rec in records if rec["root"]["is_bad"]]
    if bad:
        lines += [
            "",
            "## Bad Roots",
            "",
            "| regret | selected | teacher | classification | termination |",
            "|---:|---|---|---|---|",
        ]
        for rec in bad[:30]:
            root = rec["root"]
            lines.append(
                f"| {root['selected_regret_cp']:.1f} | {root['selected_move']} | "
                f"{root['teacher_best_move']} | {rec.get('classification')} | {root['termination']} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root_regret_jsonl", nargs="+")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", default="")
    parser.add_argument("--position-mode", choices=["all", "bad", "near", "clean", "bad-control"], default="all")
    parser.add_argument("--max-roots", type=int, default=0)
    parser.add_argument("--controls-per-bad", type=int, default=1)
    parser.add_argument("--milestones", default="0,64,256,1000,3000,8000")
    parser.add_argument("--search-configs", default="baseline:1.45:1.0")
    parser.add_argument("--value-sources", default="scalar",
                        help="Comma list of value sources consumed by MCTS: scalar,wdl. "
                             "With multiple sources, config names are prefixed with the source.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--disable-bf16", action="store_true")
    parser.add_argument("--q-clip", type=float, default=1.0)
    parser.add_argument("--temperature-move", type=float, default=0.02)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--repeat-limit", type=int, default=6)
    parser.add_argument("--repeat-min-ply", type=int, default=30)
    parser.add_argument("--no-capture-limit", type=int, default=60)
    parser.add_argument("--tactical-mate1-extension", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tactical-mate2-extension", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    t0 = time.monotonic()
    roots_all = _load_roots([Path(path) for path in args.root_regret_jsonl])
    roots = _select_roots(
        roots_all,
        mode=str(args.position_mode),
        max_roots=int(args.max_roots),
        controls_per_bad=int(args.controls_per_bad),
    )
    if not roots:
        raise SystemExit("no roots selected")
    milestones = _parse_milestones(str(args.milestones))
    value_sources = _parse_value_sources(str(args.value_sources))
    base_configs = _parse_search_configs(str(args.search_configs))
    configs: list[dict[str, Any]] = []
    multi_source = len(value_sources) > 1
    for source in value_sources:
        for cfg in base_configs:
            name = str(cfg["name"]) if not multi_source else f"{source}_{cfg['name']}"
            configs.append({
                "name": name,
                "c_puct": float(cfg["c_puct"]),
                "q_weight": float(cfg["q_weight"]),
                "value_source": source,
            })
    print(f"loaded roots={len(roots_all)} selected={len(roots)} milestones={milestones}", flush=True)

    device = torch.device(args.device)
    use_bfloat16 = not bool(args.disable_bf16)
    model, evaluator, checkpoint_step = _load_model_and_evaluator(Path(args.checkpoint), device, use_bfloat16)
    evaluators_by_source = {source: _wrap_value_source(evaluator, source) for source in value_sources}

    boards = [_board_from_root(root) for root in roots]
    states = torch.stack([board.to_tensor_canonical().to(torch.float32)[0].contiguous() for board in boards], dim=0)
    policy_logits = _forward_policy(model, states, device, use_bfloat16, int(args.batch_size))
    raw_policy_by_root = [_legal_log_probs(policy_logits[i], board) for i, board in enumerate(boards)]

    records: list[dict[str, Any]] = []
    for i, (root, board, raw_policy) in enumerate(zip(roots, boards, raw_policy_by_root)):
        root_payload = {
            "key": root.key,
            "fen": root.fen,
            "game_index": root.game_index,
            "ply": root.ply,
            "opening_id": root.opening_id,
            "opening_index": root.opening_index,
            "termination": root.termination,
            "selected_move": root.selected_move,
            "teacher_best_move": root.teacher_best_move,
            "selected_regret_cp": float(root.selected_regret_cp),
            "is_bad": bool(root.is_bad),
            "is_catastrophic": bool(root.is_catastrophic),
        }
        search_payload: dict[str, Any] = {}
        print(
            f"[{i+1}/{len(roots)}] root selected={root.selected_move} teacher={root.teacher_best_move} "
            f"regret={root.selected_regret_cp:.1f}",
            flush=True,
        )
        for cfg in configs:
            cfg_name = str(cfg["name"])
            cfg_milestones: list[dict[str, Any]] = []
            for sim in milestones:
                if int(sim) <= 0:
                    row = {
                        "simulations": 0,
                        "best_move": None,
                        "best_move_known_regret_cp": None,
                        "selected": _move_snapshot(root.selected_move, {}, raw_policy, root.candidates.get(root.selected_move)),
                        "teacher_best": _move_snapshot(root.teacher_best_move, {}, raw_policy, root.candidates.get(root.teacher_best_move)),
                        "top_moves": [],
                    }
                    cfg_milestones.append(row)
                    continue
                best_move, _idxs, _probs, root_v, root_stats = mcts_search_with_root_stats(
                    board=board,
                    net=evaluators_by_source[str(cfg.get("value_source", "scalar"))],
                    num_simulations=int(sim),
                    c_puct=float(cfg["c_puct"]),
                    q_weight=float(cfg["q_weight"]),
                    q_clip=float(args.q_clip),
                    add_root_noise=False,
                    dirichlet_alpha=0.3,
                    dirichlet_eps=0.0,
                    temperature_move=float(args.temperature_move),
                    temperature_target=1.0,
                    eval_batch_size=int(args.eval_batch_size),
                    seed=int(args.seed) + i * 1009 + int(sim),
                    canonical_input=True,
                    canonical_policy=True,
                    max_plies=int(args.max_plies),
                    repeat_limit=int(args.repeat_limit),
                    repeat_min_ply=int(args.repeat_min_ply),
                    no_capture_limit=int(args.no_capture_limit),
                    tactical_mate1_extension=bool(args.tactical_mate1_extension),
                    tactical_mate2_extension=bool(args.tactical_mate2_extension),
                    c_puct_base=1.0,
                    c_puct_factor=0.0,
                    fpu_reduction_root=-1.0,
                    fpu_reduction_tree=-1.0,
                )
                top_rows, by_move = _ranked_root_stats(list(root_stats))
                best_uci = internal_move_to_uci(int(best_move)) if int(best_move) >= 0 else ""
                row = {
                    "simulations": int(sim),
                    "root_value": float(root_v),
                    "best_move": best_uci,
                    "best_move_known_regret_cp": _known_regret(root, best_uci),
                    "selected": _move_snapshot(root.selected_move, by_move, raw_policy, root.candidates.get(root.selected_move)),
                    "teacher_best": _move_snapshot(root.teacher_best_move, by_move, raw_policy, root.candidates.get(root.teacher_best_move)),
                    "top_moves": [
                        {
                            **move_row,
                            "known_regret_cp": _known_regret(root, str(move_row["move_uci"])),
                        }
                        for move_row in top_rows[:16]
                    ],
                }
                cfg_milestones.append(row)
            search_payload[cfg_name] = {
                "c_puct": float(cfg["c_puct"]),
                "q_weight": float(cfg["q_weight"]),
                "milestones": cfg_milestones,
                "final": cfg_milestones[-1],
            }
        classification = _classify_root(root, search_payload[str(configs[0]["name"])]["milestones"], raw_policy)
        records.append(
            {
                "root": root_payload,
                "raw_policy_top16": sorted(
                    [{"move": mv, **info} for mv, info in raw_policy.items()],
                    key=lambda row: int(row["rank"]),
                )[:16],
                "classification": classification,
                "search": search_payload,
            }
        )

    summary = _summarize(records, configs)
    summary["checkpoint"] = str(Path(args.checkpoint))
    summary["checkpoint_step"] = int(checkpoint_step)
    summary["root_regret_jsonl"] = [str(Path(path)) for path in args.root_regret_jsonl]
    summary["milestones"] = milestones
    summary["elapsed_s"] = time.monotonic() - t0
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
