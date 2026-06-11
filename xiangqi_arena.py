from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from xiangqi_mcts_ext import Board, make_gpu_evaluator, mcts_search
from xiangqi_selfplay import (
    TERMINATION_CHECKMATE_OR_STALEMATE,
    TERMINATION_MAX_PLIES_DRAW,
    TERMINATION_NO_CAPTURE_DRAW,
    TERMINATION_PERPETUAL_CHECK_LOSS,
    TERMINATION_REPETITION_DRAW,
    ChimeraPolicyValueModel,
    _load_model_from_checkpoint,
)


TERMINAL_ONGOING = -1
_CANONICAL_PLANE_TO_FEN_CHAR = {
    0: "K",
    1: "A",
    2: "B",
    3: "N",
    4: "R",
    5: "C",
    6: "P",
    7: "k",
    8: "a",
    9: "b",
    10: "n",
    11: "r",
    12: "c",
    13: "p",
}


@dataclass
class ArenaConfig:
    games: int = 100
    games_per_opening: int = 2
    sims: int = 800
    c_puct: float = 1.25
    q_weight: float = 1.0
    q_clip: float = 1.0
    eval_batch_size: int = 16
    temperature_move: float = 1e-6
    # Arena needs SOME source of game-to-game variation; otherwise two near-identical
    # deterministic players from the standard start produce the same trajectory and loop
    # into a repetition draw.  Root Dirichlet noise at moderate strength diversifies the
    # opening of each game without overwhelming mid/endgame play.  If you want fully
    # reproducible games (e.g. opening-suite probes), pass --disable-arena-root-noise.
    add_root_noise: bool = True
    dirichlet_alpha: float = 0.30
    dirichlet_eps: float = 0.10
    max_plies: int = 240
    repeat_limit: int = 6
    repeat_min_ply: int = 30
    no_capture_limit: int = 60
    seed: int = 2026011530
    accept_threshold: float = 0.55
    min_non_draw_games: int = 10
    log_every_games: int = 10
    device: str = "cuda:0"
    use_bfloat16_eval: bool = True
    promote_on_pass: bool = True
    opening_suite_path: str | None = None
    # Frozen-evaluator (chimera) gate: if set, BOTH agents play with
    # value_scalar/wdl_logits from this frozen reference checkpoint while each
    # keeps its own policy. Used by the policy-only self-play loop.
    shared_value_checkpoint: str | None = None


@dataclass(frozen=True)
class ArenaAgent:
    checkpoint_path: Path
    label: str
    step: int | None
    evaluator: Any


def _update_arena_rate_metrics(summary: dict[str, Any]) -> None:
    games_played = int(summary["candidate_win"] + summary["champion_win"] + summary["draw"])
    if games_played <= 0:
        summary["candidate_score_rate"] = 0.0
        summary["candidate_win_rate"] = 0.0
        summary["candidate_red_win_rate"] = 0.0
        summary["candidate_black_win_rate"] = 0.0
        summary["draw_rate"] = 0.0
        return

    summary["candidate_score_rate"] = float(
        (summary["candidate_win"] + 0.5 * summary["draw"]) / games_played
    )
    summary["candidate_win_rate"] = float(summary["candidate_win"] / games_played)
    summary["candidate_red_win_rate"] = float(
        summary["candidate_red_wins"] / summary["candidate_as_red_games"]
    ) if int(summary["candidate_as_red_games"]) > 0 else 0.0
    summary["candidate_black_win_rate"] = float(
        summary["candidate_black_wins"] / summary["candidate_as_black_games"]
    ) if int(summary["candidate_as_black_games"]) > 0 else 0.0
    summary["draw_rate"] = float(summary["draw"] / games_played)


def _make_empty_result_counter() -> dict[str, Any]:
    return {
        "candidate_win": 0,
        "champion_win": 0,
        "draw": 0,
        "non_draw": 0,
        "candidate_score_rate": 0.0,
        "candidate_win_rate": 0.0,
        "candidate_red_win_rate": 0.0,
        "candidate_black_win_rate": 0.0,
        "draw_rate": 0.0,
        "avg_plies": 0.0,
        "candidate_as_red_games": 0,
        "candidate_as_black_games": 0,
        "candidate_red_wins": 0,
        "candidate_black_wins": 0,
        "termination_counts": {
            "mate": 0,
            "max": 0,
            "rep": 0,
            "longcheck": 0,
            "nocap": 0,
        },
        "errors": [],
    }


def _record_game_result(
    summary: dict[str, Any],
    result: dict[str, Any],
    candidate_is_red: bool,
    game_number: int,
    plies_total: int,
) -> int:
    if candidate_is_red:
        summary["candidate_as_red_games"] += 1
    else:
        summary["candidate_as_black_games"] += 1

    plies_total += int(result["plies"])
    term_label = _termination_label(int(result["termination_code"]))
    summary["termination_counts"][term_label] = int(summary["termination_counts"].get(term_label, 0)) + 1
    if "error" in result:
        summary["errors"].append(
            {
                "game": int(game_number),
                "message": result["error"],
            }
        )

    red_result = int(result["result_red_view"])
    if red_result == 0:
        summary["draw"] += 1
    else:
        winner_is_red = red_result > 0
        if winner_is_red == candidate_is_red:
            summary["candidate_win"] += 1
            if candidate_is_red:
                summary["candidate_red_wins"] += 1
            else:
                summary["candidate_black_wins"] += 1
        else:
            summary["champion_win"] += 1

    summary["non_draw"] = int(summary["candidate_win"] + summary["champion_win"])
    summary["avg_plies"] = float(plies_total / max(int(game_number), 1))
    _update_arena_rate_metrics(summary)
    return plies_total


def _expand_fen_row(row: str) -> str:
    expanded: list[str] = []
    for char in row:
        if char.isdigit():
            expanded.extend("1" for _ in range(int(char)))
        else:
            expanded.append(char)
    if len(expanded) != 9:
        raise ValueError(f"invalid fen row width: {row!r}")
    return "".join(expanded)


def _compress_fen_row(expanded_row: str) -> str:
    compressed: list[str] = []
    empty_run = 0
    for char in expanded_row:
        if char == "1":
            empty_run += 1
            continue
        if empty_run > 0:
            compressed.append(str(empty_run))
            empty_run = 0
        compressed.append(char)
    if empty_run > 0:
        compressed.append(str(empty_run))
    return "".join(compressed)


def _mirror_fen_horizontally(fen: str) -> str:
    rows_text, turn = fen.split()
    mirrored_rows = [
        _compress_fen_row(_expand_fen_row(row)[::-1])
        for row in rows_text.split("/")
    ]
    return "/".join(mirrored_rows) + f" {turn}"


def _opening_dedupe_key(fen: str) -> str:
    mirrored = _mirror_fen_horizontally(fen)
    return min(fen, mirrored)


def _state_plane_scalar(state: np.ndarray, plane_index: int) -> float:
    plane = np.asarray(state[plane_index], dtype=np.float32)
    return float(plane[0, 0])


def _infer_no_capture_count_from_state(state: np.ndarray) -> int:
    encoded = max(0.0, min(0.999999, _state_plane_scalar(state, 114)))
    if encoded <= 0.0:
        return 0
    return max(0, int(round(math.atanh(encoded) * 30.0)))


def _infer_repetition_count_hint_from_state(state: np.ndarray) -> int:
    encoded = max(0.0, min(1.0, _state_plane_scalar(state, 113)))
    repetitions_excluding_current = int(round(encoded * 2.0))
    return max(1, repetitions_excluding_current + 1)


def _canonical_state_to_fen(state: Any) -> str:
    state_np = np.asarray(state, dtype=np.float32)
    if state_np.shape != (115, 10, 9):
        raise ValueError(f"expected canonical state shape (115, 10, 9), got {tuple(state_np.shape)}")

    board_rows = [["1"] * 9 for _ in range(10)]
    for plane_index, piece_char in _CANONICAL_PLANE_TO_FEN_CHAR.items():
        occupied = np.argwhere(state_np[plane_index] > 0.5)
        for y_idx, x_idx in occupied:
            y = int(y_idx)
            x = int(x_idx)
            if board_rows[y][x] != "1":
                raise ValueError(
                    f"canonical state has overlapping pieces at y={y} x={x}"
                )
            board_rows[y][x] = piece_char

    rows = [_compress_fen_row("".join(row)) for row in board_rows]
    return "/".join(rows) + " w"


def _opening_entry_from_human_sample(
    sample: dict[str, Any],
    split: str,
    shard_name: str,
    local_index: int,
) -> dict[str, Any]:
    state_np = np.asarray(sample["state"], dtype=np.float32)
    fen = _canonical_state_to_fen(state_np)
    ply = max(0, int(sample.get("ply", 0)))
    inferred_no_capture = _infer_no_capture_count_from_state(state_np)
    return {
        "id": f"{split}_{Path(shard_name).stem}_{local_index:04d}",
        "fen": fen,
        "plies": ply,
        # Human opening samples are used as position seeds rather than exact resumed games.
        # Clamp search context so early-position suites do not start near an artificial draw threshold.
        "no_capture_count": min(inferred_no_capture, ply),
        "repetition_count_hint": 1,
        "source_split": split,
        "source_shard": shard_name,
        "source_local_index": int(local_index),
        "source_sample_id": int(sample.get("sample_id", local_index)),
        "game_len": int(sample.get("game_len", 0)),
        "is_draw": bool(sample.get("is_draw", False)),
        "dedupe_key": _opening_dedupe_key(fen),
    }


def _validate_opening_entry(entry: dict[str, Any], config: ArenaConfig) -> None:
    board = Board()
    board.set_fen(str(entry["fen"]))
    board.set_search_context(
        int(entry.get("plies", 0)),
        int(entry.get("no_capture_count", 0)),
        int(entry.get("repetition_count_hint", 1)),
    )
    if len(board.legal_moves()) <= 0:
        raise ValueError(f"opening entry has no legal moves: {entry['fen']}")
    terminal_code = int(
        board.terminal_code(
            int(config.max_plies),
            int(config.repeat_limit),
            int(config.repeat_min_ply),
            int(config.no_capture_limit),
        )
    )
    if terminal_code != TERMINAL_ONGOING:
        raise ValueError(
            f"opening entry is terminal under arena rules ({_termination_label(terminal_code)}): {entry['fen']}"
        )


def load_opening_suite(path: str | Path, config: ArenaConfig) -> list[dict[str, Any]]:
    suite_path = Path(path).resolve()
    payload = json.loads(suite_path.read_text(encoding="utf-8"))
    positions = payload["positions"] if isinstance(payload, dict) else payload
    if not isinstance(positions, list) or not positions:
        raise ValueError(f"opening suite is empty: {suite_path}")

    openings: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(positions):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"opening suite entry #{index + 1} must be an object")
        entry = {
            "id": str(raw_entry.get("id", f"opening_{index + 1:03d}")),
            "fen": str(raw_entry["fen"]),
            "plies": int(raw_entry.get("plies", 0)),
            "no_capture_count": int(raw_entry.get("no_capture_count", 0)),
            "repetition_count_hint": int(raw_entry.get("repetition_count_hint", 1)),
            "source_split": raw_entry.get("source_split"),
            "source_shard": raw_entry.get("source_shard"),
            "source_local_index": raw_entry.get("source_local_index"),
            "source_sample_id": raw_entry.get("source_sample_id"),
            "game_len": raw_entry.get("game_len"),
            "is_draw": raw_entry.get("is_draw"),
        }
        _validate_opening_entry(entry, config)
        openings.append(entry)
    return openings


def build_opening_suite_from_human_data(
    human_data_dir: str | Path,
    output_path: str | Path,
    *,
    split: str = "val",
    target_positions: int = 12,
    min_ply: int = 6,
    max_ply: int = 16,
    seed: int = 20260414,
    max_positions_per_shard: int = 2,
) -> dict[str, Any]:
    human_data_dir = Path(human_data_dir).resolve()
    output_path = Path(output_path).resolve()
    manifest_path = human_data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"human data manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_meta = manifest.get(split)
    if not isinstance(split_meta, dict):
        raise KeyError(f"manifest is missing split '{split}'")
    shard_specs = split_meta.get("shards", [])
    if not isinstance(shard_specs, list) or not shard_specs:
        raise ValueError(f"manifest split '{split}' has no shards")
    if target_positions < 1:
        raise ValueError("target_positions must be >= 1")
    if min_ply < 0 or max_ply < min_ply:
        raise ValueError("invalid min_ply/max_ply range")

    rng = random.Random(seed)
    shard_order = list(range(len(shard_specs)))
    rng.shuffle(shard_order)

    probe_config = ArenaConfig(promote_on_pass=False)
    selected: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    selected_by_shard: dict[str, int] = {}
    target_bucket_order = list(range(min_ply, max_ply + 1))
    rng.shuffle(target_bucket_order)
    bucket_targets = set(target_bucket_order[: min(len(target_bucket_order), target_positions)])

    def _consider_entry(entry: dict[str, Any]) -> bool:
        dedupe_key = str(entry["dedupe_key"])
        if dedupe_key in seen_keys:
            return False
        shard_key = str(entry["source_shard"])
        if selected_by_shard.get(shard_key, 0) >= max_positions_per_shard:
            return False
        _validate_opening_entry(entry, probe_config)
        seen_keys.add(dedupe_key)
        selected_by_shard[shard_key] = selected_by_shard.get(shard_key, 0) + 1
        selected.append(entry)
        return True

    for shard_index in shard_order:
        shard_spec = shard_specs[shard_index]
        shard_rel_path = str(shard_spec["path"])
        shard_path = human_data_dir / split / shard_rel_path
        if not shard_path.is_file():
            shard_path = human_data_dir / shard_rel_path
        if not shard_path.is_file():
            raise FileNotFoundError(f"human shard referenced by manifest not found: {shard_rel_path}")
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        samples = shard.get("samples")
        if not isinstance(samples, list):
            continue
        local_indices = list(range(len(samples)))
        rng.shuffle(local_indices)
        for local_index in local_indices:
            sample = samples[local_index]
            if not isinstance(sample, dict):
                continue
            ply = int(sample.get("ply", -1))
            if ply < min_ply or ply > max_ply:
                continue
            if bucket_targets and ply not in bucket_targets:
                continue
            entry = _opening_entry_from_human_sample(sample, split, shard_rel_path, local_index)
            if _consider_entry(entry):
                bucket_targets.discard(ply)
            if len(selected) >= target_positions and not bucket_targets:
                break
        if len(selected) >= target_positions and not bucket_targets:
            break

    if len(selected) < target_positions:
        for shard_index in shard_order:
            if len(selected) >= target_positions:
                break
            shard_spec = shard_specs[shard_index]
            shard_rel_path = str(shard_spec["path"])
            shard_path = human_data_dir / split / shard_rel_path
            if not shard_path.is_file():
                shard_path = human_data_dir / shard_rel_path
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            samples = shard.get("samples")
            if not isinstance(samples, list):
                continue
            local_indices = list(range(len(samples)))
            rng.shuffle(local_indices)
            for local_index in local_indices:
                if len(selected) >= target_positions:
                    break
                sample = samples[local_index]
                if not isinstance(sample, dict):
                    continue
                ply = int(sample.get("ply", -1))
                if ply < min_ply or ply > max_ply:
                    continue
                entry = _opening_entry_from_human_sample(sample, split, shard_rel_path, local_index)
                _consider_entry(entry)

    if len(selected) < min(8, target_positions):
        raise RuntimeError(
            f"only found {len(selected)} valid opening positions in ply range {min_ply}-{max_ply}; "
            "need at least 8 for the minimum suite"
        )

    selected = selected[:target_positions]
    for entry in selected:
        entry.pop("dedupe_key", None)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "xiangqi_arena_opening_suite_v1",
        "created_at": datetime.now().isoformat(),
        "human_data_dir": str(human_data_dir),
        "source_split": split,
        "target_positions": int(target_positions),
        "min_ply": int(min_ply),
        "max_ply": int(max_ply),
        "seed": int(seed),
        "max_positions_per_shard": int(max_positions_per_shard),
        "positions": selected,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _default_candidate_checkpoint() -> Path:
    root = _repo_root()
    candidates = [
        root / "training_runs" / "run_001" / "latest.pt",
        root / "training_runs" / "run_001" / "best.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def _default_champion_checkpoint() -> Path:
    root = _repo_root()
    candidates = [
        root / "training_runs" / "run_001" / "best.pt",
        root / "training_runs" / "run_001" / "latest.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def _load_checkpoint_step(checkpoint_path: Path) -> int | None:
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        step = raw.get("global_step")
        if step is not None:
            return int(step)
    return None


def _load_transformer_agent(
    checkpoint_path: Path,
    device: str,
    use_bfloat16_eval: bool,
    value_checkpoint_path: str | None = None,
) -> ArenaAgent:
    model, _ = _load_model_from_checkpoint(checkpoint_path)
    if value_checkpoint_path:
        value_model, _ = _load_model_from_checkpoint(Path(value_checkpoint_path))
        model = ChimeraPolicyValueModel(model, value_model)
    evaluator = make_gpu_evaluator(model, device=device, use_bfloat16=use_bfloat16_eval)
    step = _load_checkpoint_step(checkpoint_path)
    step_suffix = f" step {step}" if step is not None else ""
    label = f"{checkpoint_path.stem}{step_suffix}"
    return ArenaAgent(
        checkpoint_path=checkpoint_path,
        label=label,
        step=step,
        evaluator=evaluator,
    )


def _termination_label(code: int) -> str:
    mapping = {
        TERMINATION_CHECKMATE_OR_STALEMATE: "mate",
        TERMINATION_MAX_PLIES_DRAW: "max",
        TERMINATION_REPETITION_DRAW: "rep",
        TERMINATION_PERPETUAL_CHECK_LOSS: "longcheck",
        TERMINATION_NO_CAPTURE_DRAW: "nocap",
    }
    return mapping.get(int(code), f"code{int(code)}")


def _initialize_board_for_opening(board: Board, opening_entry: dict[str, Any] | None) -> None:
    if opening_entry is None:
        return
    board.set_fen(str(opening_entry["fen"]))
    board.set_search_context(
        int(opening_entry.get("plies", 0)),
        int(opening_entry.get("no_capture_count", 0)),
        int(opening_entry.get("repetition_count_hint", 1)),
    )


def _play_one_game(
    config: ArenaConfig,
    red_agent: ArenaAgent,
    black_agent: ArenaAgent,
    seed: int,
    opening_entry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    board = Board()
    _initialize_board_for_opening(board, opening_entry)

    while True:
        ply = int(board.plies_played())
        terminal_code = int(
            board.terminal_code(
                int(config.max_plies),
                int(config.repeat_limit),
                int(config.repeat_min_ply),
                int(config.no_capture_limit),
            )
        )
        if terminal_code != TERMINAL_ONGOING:
            return {
                "result_red_view": int(board.terminal_result_red_view(terminal_code)),
                "termination_code": int(terminal_code),
                "plies": ply,
            }

        side_to_move = int(board.turn())
        eval_net = red_agent.evaluator if side_to_move == 0 else black_agent.evaluator
        try:
            best_move, _idxs, _probs, _root_v = mcts_search(
                board=board,
                net=eval_net,
                num_simulations=config.sims,
                c_puct=config.c_puct,
                q_weight=config.q_weight,
                q_clip=config.q_clip,
                add_root_noise=config.add_root_noise,
                dirichlet_alpha=config.dirichlet_alpha,
                dirichlet_eps=config.dirichlet_eps,
                temperature_move=config.temperature_move,
                temperature_target=1.0,
                eval_batch_size=config.eval_batch_size,
                seed=int((seed + ply * 10007) & 0x7FFFFFFF),
                canonical_input=True,
                canonical_policy=True,
                max_plies=config.max_plies,
                repeat_limit=config.repeat_limit,
                repeat_min_ply=config.repeat_min_ply,
                no_capture_limit=config.no_capture_limit,
            )
        except Exception as exc:
            loser = "red" if side_to_move == 0 else "black"
            winner_result = -1 if side_to_move == 0 else 1
            return {
                "result_red_view": winner_result,
                "termination_code": TERMINATION_CHECKMATE_OR_STALEMATE,
                "plies": ply,
                "error": f"{loser} search failed: {exc}",
            }

        if int(best_move) < 0:
            terminal_code = int(
                board.terminal_code(
                    int(config.max_plies),
                    int(config.repeat_limit),
                    int(config.repeat_min_ply),
                    int(config.no_capture_limit),
                )
            )
            return {
                "result_red_view": int(
                    board.terminal_result_red_view(
                        int(terminal_code) if terminal_code != TERMINAL_ONGOING else TERMINATION_CHECKMATE_OR_STALEMATE
                    )
                ),
                "termination_code": (int(terminal_code) if terminal_code != TERMINAL_ONGOING else TERMINATION_CHECKMATE_OR_STALEMATE),
                "plies": ply,
            }

        board.push(int(best_move))


def run_arena(
    candidate_checkpoint: str | Path,
    champion_checkpoint: str | Path,
    output_root: str | Path,
    config: ArenaConfig,
) -> dict[str, Any]:
    candidate_checkpoint = Path(candidate_checkpoint).resolve()
    champion_checkpoint = Path(champion_checkpoint).resolve()
    output_root = Path(output_root).resolve()

    if not candidate_checkpoint.is_file():
        raise FileNotFoundError(f"candidate checkpoint not found: {candidate_checkpoint}")
    if not champion_checkpoint.is_file():
        raise FileNotFoundError(f"champion checkpoint not found: {champion_checkpoint}")
    if config.games < 1:
        raise ValueError("games must be >= 1")
    if config.games_per_opening < 1:
        raise ValueError("games_per_opening must be >= 1")
    if config.log_every_games < 1:
        raise ValueError("log_every_games must be >= 1")
    if config.min_non_draw_games < 0:
        raise ValueError("min_non_draw_games must be >= 0")
    if config.opening_suite_path is not None and (config.games_per_opening % 2) != 0:
        raise ValueError("games_per_opening must be even when using an opening suite")

    if str(config.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested for arena, but CUDA is not available")

    output_root.mkdir(parents=True, exist_ok=True)

    candidate_agent = _load_transformer_agent(
        candidate_checkpoint, config.device, config.use_bfloat16_eval,
        value_checkpoint_path=config.shared_value_checkpoint,
    )
    champion_agent = _load_transformer_agent(
        champion_checkpoint, config.device, config.use_bfloat16_eval,
        value_checkpoint_path=config.shared_value_checkpoint,
    )
    if config.shared_value_checkpoint:
        print(
            f"[ARENA] CHIMERA gate: both sides use value from {config.shared_value_checkpoint}",
            flush=True,
        )
    openings = (
        load_opening_suite(config.opening_suite_path, config)
        if config.opening_suite_path is not None
        else []
    )
    total_games = int(len(openings) * config.games_per_opening) if openings else int(config.games)
    if total_games < 1:
        raise ValueError("arena requires at least one game")

    summary: dict[str, Any] = {
        "candidate_checkpoint": str(candidate_checkpoint),
        "champion_checkpoint": str(champion_checkpoint),
        "candidate_label": candidate_agent.label,
        "champion_label": champion_agent.label,
        "candidate_step": candidate_agent.step,
        "champion_step": champion_agent.step,
        "games": total_games,
        "winrate": 0.0,
        "accepted": False,
        "promoted": False,
        "accept_threshold": float(config.accept_threshold),
        "min_non_draw_games": int(config.min_non_draw_games),
        "config": asdict(config),
        "opening_suite_path": None if config.opening_suite_path is None else str(Path(config.opening_suite_path).resolve()),
        "opening_suite_size": int(len(openings)),
        "games_per_opening": int(config.games_per_opening) if openings else None,
        "per_opening_results": [],
    }
    summary.update(_make_empty_result_counter())

    plies_total = 0
    completed_games = 0
    opening_schedule = openings if openings else [None]

    for opening_index, opening_entry in enumerate(opening_schedule):
        games_for_opening = int(config.games_per_opening) if opening_entry is not None else int(config.games)
        opening_summary = _make_empty_result_counter()
        opening_plies_total = 0

        for local_game_index in range(games_for_opening):
            candidate_is_red = (local_game_index % 2 == 0)
            if candidate_is_red:
                red_agent = candidate_agent
                black_agent = champion_agent
            else:
                red_agent = champion_agent
                black_agent = candidate_agent

            seed = int(config.seed + opening_index * 1_000_003 + local_game_index * 9_973)
            result = _play_one_game(config, red_agent, black_agent, seed, opening_entry)
            completed_games += 1
            plies_total = _record_game_result(summary, result, candidate_is_red, completed_games, plies_total)
            opening_plies_total = _record_game_result(
                opening_summary,
                result,
                candidate_is_red,
                local_game_index + 1,
                opening_plies_total,
            )
            summary["winrate"] = float(
                (summary["candidate_win"] + 0.5 * summary["draw"]) / float(total_games)
            )

            if completed_games % config.log_every_games == 0 or completed_games == total_games:
                print(
                    f"[ARENA] {completed_games}/{total_games} "
                    f"cand_win={summary['candidate_win']} champ_win={summary['champion_win']} draw={summary['draw']} "
                    f"score={summary['candidate_score_rate'] * 100:.1f}% "
                    f"red={summary['candidate_red_win_rate'] * 100:.1f}% "
                    f"black={summary['candidate_black_win_rate'] * 100:.1f}% "
                    f"draw={summary['draw_rate'] * 100:.1f}% avg_plies={summary['avg_plies']:.1f} "
                    f"term={summary['termination_counts']}",
                    flush=True,
                )

        if opening_entry is not None:
            opening_summary["opening_id"] = str(opening_entry["id"])
            opening_summary["fen"] = str(opening_entry["fen"])
            opening_summary["plies"] = int(opening_entry.get("plies", 0))
            opening_summary["no_capture_count"] = int(opening_entry.get("no_capture_count", 0))
            opening_summary["repetition_count_hint"] = int(opening_entry.get("repetition_count_hint", 1))
            opening_summary["games"] = int(games_for_opening)
            opening_summary["source_split"] = opening_entry.get("source_split")
            opening_summary["source_shard"] = opening_entry.get("source_shard")
            opening_summary["source_local_index"] = opening_entry.get("source_local_index")
            summary["per_opening_results"].append(opening_summary)

    # Dual-criterion acceptance:
    #   (A) Score-rate criterion (original): winrate >= threshold AND enough non-draw games.
    #       Works well when draw rate is moderate.
    #   (B) Decisive-winrate criterion (added 2026-04-18): when draws dominate (e.g. 80%+),
    #       the score rate is pulled toward 0.5 even when the candidate is clearly beating
    #       the champion on decisive games. Accept if decisive winrate W/(W+L) is strong
    #       AND we have at least 10 decisive games for stability.
    decisive_total = int(summary["candidate_win"]) + int(summary["champion_win"])
    if decisive_total > 0:
        decisive_winrate = float(summary["candidate_win"]) / float(decisive_total)
    else:
        decisive_winrate = 0.5
    summary["decisive_winrate"] = float(decisive_winrate)
    summary["decisive_games"] = int(decisive_total)
    accept_by_score = bool(
        summary["winrate"] >= float(config.accept_threshold)
        and summary["non_draw"] >= int(config.min_non_draw_games)
    )
    accept_by_decisive = bool(
        decisive_winrate >= 0.60 and decisive_total >= 10
    )
    summary["accepted"] = bool(accept_by_score or accept_by_decisive)
    summary["accept_reason"] = (
        "score_rate" if accept_by_score else ("decisive_winrate" if accept_by_decisive else "none")
    )
    if champion_checkpoint.stem == "best":
        summary["arena_win_rate_vs_best"] = float(summary["candidate_score_rate"])
    elif champion_checkpoint.stem == "latest":
        summary["arena_win_rate_vs_latest_baseline"] = float(summary["candidate_score_rate"])
    summary["arena_red_win_rate"] = float(summary["candidate_red_win_rate"])
    summary["arena_black_win_rate"] = float(summary["candidate_black_win_rate"])
    summary["arena_draw_rate"] = float(summary["draw_rate"])

    if config.promote_on_pass and summary["accepted"]:
        if candidate_checkpoint != champion_checkpoint:
            shutil.copy2(candidate_checkpoint, champion_checkpoint)
            summary["promoted"] = True
            step_suffix = (
                f"best_step{int(candidate_agent.step):07d}.pt"
                if candidate_agent.step is not None
                else f"best_promoted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            )
            snapshot_path = champion_checkpoint.with_name(step_suffix)
            shutil.copy2(candidate_checkpoint, snapshot_path)
            summary["promoted_to"] = str(champion_checkpoint)
            summary["promotion_snapshot"] = str(snapshot_path)
        else:
            summary["promoted"] = True
            summary["promoted_to"] = str(champion_checkpoint)
            summary["promotion_snapshot"] = None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    latest_path = output_root / "latest_arena_summary.json"
    dated_path = output_root / f"arena_{timestamp}.json"
    payload = json.dumps(summary, indent=2, ensure_ascii=False)
    latest_path.write_text(payload, encoding="utf-8")
    dated_path.write_text(payload, encoding="utf-8")
    summary["summary_path"] = str(dated_path)
    summary["latest_summary_path"] = str(latest_path)
    return summary


def _parse_args() -> tuple[Path, Path, Path, ArenaConfig]:
    default_config = ArenaConfig()
    parser = argparse.ArgumentParser(description="Evaluate a challenger Transformer checkpoint against the current best.")
    parser.add_argument("--candidate-checkpoint", default=str(_default_candidate_checkpoint()))
    parser.add_argument("--champion-checkpoint", default=str(_default_champion_checkpoint()))
    parser.add_argument("--output-root", default="arena_runs")
    parser.add_argument("--games", type=int, default=default_config.games)
    parser.add_argument("--games-per-opening", type=int, default=default_config.games_per_opening)
    parser.add_argument("--sims", type=int, default=default_config.sims)
    parser.add_argument("--c-puct", type=float, default=default_config.c_puct)
    parser.add_argument("--q-weight", type=float, default=default_config.q_weight)
    parser.add_argument("--q-clip", type=float, default=default_config.q_clip)
    parser.add_argument("--eval-batch-size", type=int, default=default_config.eval_batch_size)
    parser.add_argument("--temperature-move", type=float, default=default_config.temperature_move)
    parser.add_argument("--disable-arena-root-noise", action="store_true",
                        help="Disable Dirichlet root noise in arena MCTS (fully deterministic).")
    parser.add_argument("--dirichlet-alpha", type=float, default=default_config.dirichlet_alpha)
    parser.add_argument("--dirichlet-eps", type=float, default=default_config.dirichlet_eps)
    parser.add_argument("--max-plies", type=int, default=default_config.max_plies)
    parser.add_argument("--repeat-limit", type=int, default=default_config.repeat_limit)
    parser.add_argument("--repeat-min-ply", type=int, default=default_config.repeat_min_ply)
    parser.add_argument("--no-capture-limit", type=int, default=default_config.no_capture_limit)
    parser.add_argument("--seed", type=int, default=default_config.seed)
    parser.add_argument("--accept-threshold", type=float, default=default_config.accept_threshold)
    parser.add_argument("--min-non-draw-games", type=int, default=default_config.min_non_draw_games)
    parser.add_argument("--log-every-games", type=int, default=default_config.log_every_games)
    parser.add_argument("--opening-suite-path", default=default_config.opening_suite_path)
    parser.add_argument("--device", default=default_config.device)
    parser.add_argument("--disable-bfloat16-eval", action="store_true")
    parser.add_argument("--disable-promote-on-pass", action="store_true")
    args = parser.parse_args()

    config = ArenaConfig(
        games=args.games,
        games_per_opening=args.games_per_opening,
        sims=args.sims,
        c_puct=args.c_puct,
        q_weight=args.q_weight,
        q_clip=args.q_clip,
        eval_batch_size=args.eval_batch_size,
        temperature_move=args.temperature_move,
        add_root_noise=not bool(args.disable_arena_root_noise),
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_eps=args.dirichlet_eps,
        max_plies=args.max_plies,
        repeat_limit=args.repeat_limit,
        repeat_min_ply=args.repeat_min_ply,
        no_capture_limit=args.no_capture_limit,
        seed=args.seed,
        accept_threshold=args.accept_threshold,
        min_non_draw_games=args.min_non_draw_games,
        log_every_games=args.log_every_games,
        device=args.device,
        use_bfloat16_eval=not bool(args.disable_bfloat16_eval),
        promote_on_pass=not bool(args.disable_promote_on_pass),
        opening_suite_path=args.opening_suite_path,
    )
    return (
        Path(args.candidate_checkpoint).resolve(),
        Path(args.champion_checkpoint).resolve(),
        Path(args.output_root).resolve(),
        config,
    )


def main() -> None:
    candidate_checkpoint, champion_checkpoint, output_root, config = _parse_args()
    print(f"[ARENA] candidate: {candidate_checkpoint}", flush=True)
    print(f"[ARENA] champion: {champion_checkpoint}", flush=True)
    print(
        f"[ARENA] config: games={config.games} suite={config.opening_suite_path or 'standard'} "
        f"games_per_opening={config.games_per_opening} sims={config.sims} "
        f"threshold={config.accept_threshold:.2f} min_non_draw={config.min_non_draw_games} "
        f"device={config.device} promote={config.promote_on_pass}",
        flush=True,
    )
    summary = run_arena(candidate_checkpoint, champion_checkpoint, output_root, config)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
