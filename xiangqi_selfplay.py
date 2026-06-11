from __future__ import annotations

import argparse
import io
import json
import math
import random
import queue as queue_module
import time
import traceback
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import torch
import torch.multiprocessing as tmp
from torch import Tensor, nn

from xiangqi_transformer_model import (
    XiangqiPVTransformer,
    XiangqiTransformerConfig,
    load_xiangqi_model_state_dict,
)


TERMINAL_ONGOING = -1
TERMINATION_CHECKMATE_OR_STALEMATE = 0
TERMINATION_MAX_PLIES_DRAW = 1
TERMINATION_REPETITION_DRAW = 2
TERMINATION_NO_CAPTURE_DRAW = 3
TERMINATION_PERPETUAL_CHECK_LOSS = 4

SHUTDOWN_KIND = "shutdown"
STATUS_HEARTBEAT = "heartbeat"
STATUS_FATAL = "fatal"
STATUS_STARTED = "started"
STATUS_EXITING = "exiting"
STATUS_GAME_COMPLETE = "game_complete"
STATUS_WORKER_DONE = "worker_done"
STATUS_SHARD_WRITTEN = "shard_written"
STATUS_WRITER_DONE = "writer_done"

_STATUS_PUT_TIMEOUT_S = 0.2
_HEARTBEAT_LAST_SENT: dict[tuple[str, int], float] = {}


class StopRequested(RuntimeError):
    pass


class QueueTimeoutError(RuntimeError):
    pass


class QueueTransportError(RuntimeError):
    pass


@dataclass
class SelfPlayConfig:
    target_games: int | None = None
    target_samples: int | None = 300_000

    # Frozen-evaluator (chimera) mode: if set, the GPU evaluator serves
    # policy_logits from the play checkpoint but value_scalar/wdl_logits from
    # this frozen reference checkpoint (e.g. geo).
    frozen_value_checkpoint: str | None = None

    num_workers: int = 8
    max_states_per_batch: int = 128
    max_wait_ms: int = 2

    queue_put_timeout_s: float = 1.0
    queue_get_timeout_s: float = 1.0
    queue_retry_deadline_s: float = 30.0
    worker_eval_timeout_s: float = 60.0
    writer_flush_timeout_s: float = 30.0
    heartbeat_interval_s: float = 2.0
    heartbeat_timeout_s: float = 90.0
    shutdown_join_timeout_s: float = 10.0
    shutdown_kill_timeout_s: float = 5.0

    num_simulations: int = 256
    c_puct: float = 1.25
    q_weight: float = 1.0
    q_clip: float = 1.0
    add_root_noise: bool = True
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    eval_batch_size: int = 16
    temperature_target: float = 1.0
    move_temperature_schedule: tuple[tuple[int, float], ...] = (
        (16, 1.0),
        (60, 0.7),
        (10**9, 0.2),
    )
    root_noise_end_ply: int = 60
    max_plies: int = 256
    repeat_limit: int = 6
    repeat_min_ply: int = 30
    no_capture_limit: int = 60
    progress_log_games: int = 20
    progress_log_shards: int = 5
    bootstrap_quality_rep_draw_threshold: float = 65.0
    bootstrap_quality_decisive_threshold: float = 20.0
    bootstrap_quality_nocap_draw_threshold: float = 55.0
    bootstrap_quality_warn_rep_draw_threshold: float = 55.0
    bootstrap_quality_warn_decisive_threshold: float = 30.0
    bootstrap_quality_warn_nocap_draw_threshold: float = 45.0
    avoid_immediate_draw_moves: bool = True
    immediate_draw_candidate_scan: int = 64
    immediate_draw_capture_injection_threshold: int = 12
    immediate_draw_capture_priority_threshold: int = 18
    immediate_draw_nocap_pressure_threshold: int = 25
    immediate_draw_nocap_pressure_scan: int = 128
    start_position_mode: str = "human_positions"
    human_position_source_dir: str | Path = "human_bootstrap_data_elite_wdl/train"
    human_position_mix_ratio: float = 0.8
    opening_like_material_min_ratio: float = 0.7
    opening_like_min_piece_count: int = 24
    pause_at_local_time: str | None = None

    shard_size: int = 4096
    device: str = "cuda:0"
    use_bfloat16_eval: bool = True
    seed: int = 0

    eval_request_queue_maxsize: int = 128
    sample_queue_maxsize: int = 64
    reply_queue_maxsize: int = 8
    status_queue_maxsize: int = 8192

    def __post_init__(self) -> None:
        if self.target_games is None and self.target_samples is None:
            raise ValueError("SelfPlayConfig requires at least one of target_games or target_samples")
        if self.num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        if self.max_states_per_batch < 1:
            raise ValueError("max_states_per_batch must be >= 1")
        if self.max_wait_ms < 0:
            raise ValueError("max_wait_ms must be >= 0")
        if self.shard_size < 1:
            raise ValueError("shard_size must be >= 1")
        if self.root_noise_end_ply < 0:
            raise ValueError("root_noise_end_ply must be >= 0")
        if self.max_plies < 1:
            raise ValueError("max_plies must be >= 1")
        if self.repeat_limit < 1:
            raise ValueError("repeat_limit must be >= 1")
        if self.repeat_min_ply < 0:
            raise ValueError("repeat_min_ply must be >= 0")
        if self.no_capture_limit < 1:
            raise ValueError("no_capture_limit must be >= 1")
        if self.progress_log_games < 1:
            raise ValueError("progress_log_games must be >= 1")
        if self.progress_log_shards < 1:
            raise ValueError("progress_log_shards must be >= 1")
        if self.bootstrap_quality_rep_draw_threshold < 0:
            raise ValueError("bootstrap_quality_rep_draw_threshold must be >= 0")
        if self.bootstrap_quality_decisive_threshold < 0:
            raise ValueError("bootstrap_quality_decisive_threshold must be >= 0")
        if self.bootstrap_quality_nocap_draw_threshold < 0:
            raise ValueError("bootstrap_quality_nocap_draw_threshold must be >= 0")
        if self.bootstrap_quality_warn_rep_draw_threshold < 0:
            raise ValueError("bootstrap_quality_warn_rep_draw_threshold must be >= 0")
        if self.bootstrap_quality_warn_decisive_threshold < 0:
            raise ValueError("bootstrap_quality_warn_decisive_threshold must be >= 0")
        if self.bootstrap_quality_warn_nocap_draw_threshold < 0:
            raise ValueError("bootstrap_quality_warn_nocap_draw_threshold must be >= 0")
        if self.immediate_draw_candidate_scan < 1:
            raise ValueError("immediate_draw_candidate_scan must be >= 1")
        if self.immediate_draw_capture_injection_threshold < 0:
            raise ValueError("immediate_draw_capture_injection_threshold must be >= 0")
        if self.immediate_draw_capture_priority_threshold < 0:
            raise ValueError("immediate_draw_capture_priority_threshold must be >= 0")
        if self.immediate_draw_nocap_pressure_threshold < 0:
            raise ValueError("immediate_draw_nocap_pressure_threshold must be >= 0")
        if self.immediate_draw_nocap_pressure_scan < 1:
            raise ValueError("immediate_draw_nocap_pressure_scan must be >= 1")
        if self.immediate_draw_capture_injection_threshold > self.immediate_draw_capture_priority_threshold:
            raise ValueError(
                "immediate_draw_capture_injection_threshold must be <= immediate_draw_capture_priority_threshold"
            )
        if self.immediate_draw_capture_priority_threshold > self.immediate_draw_nocap_pressure_threshold:
            raise ValueError(
                "immediate_draw_capture_priority_threshold must be <= immediate_draw_nocap_pressure_threshold"
            )
        if self.start_position_mode not in {"human_positions", "standard_start"}:
            raise ValueError("start_position_mode must be one of: human_positions, standard_start")
        if not (0.0 <= self.human_position_mix_ratio <= 1.0):
            raise ValueError("human_position_mix_ratio must be within [0, 1]")
        if not (0.0 <= self.opening_like_material_min_ratio <= 1.0):
            raise ValueError("opening_like_material_min_ratio must be within [0, 1]")
        if self.opening_like_min_piece_count < 2:
            raise ValueError("opening_like_min_piece_count must be >= 2")
        _parse_pause_local_time(self.pause_at_local_time)
        if not self.move_temperature_schedule:
            raise ValueError("move_temperature_schedule must not be empty")
        normalized_schedule: list[tuple[int, float]] = []
        last_limit = -1
        for limit, temp in self.move_temperature_schedule:
            limit = int(limit)
            temp = float(temp)
            if limit < 0:
                raise ValueError("move_temperature_schedule limits must be >= 0")
            if temp < 0.0:
                raise ValueError("move_temperature_schedule temperatures must be >= 0")
            if limit < last_limit:
                raise ValueError("move_temperature_schedule limits must be non-decreasing")
            normalized_schedule.append((limit, temp))
            last_limit = limit
        self.move_temperature_schedule = tuple(normalized_schedule)


def _parse_pause_local_time(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"off", "none", "disable", "disabled"}:
        return None
    try:
        hour_text, minute_text = text.split(":", maxsplit=1)
        hour = int(hour_text)
        minute = int(minute_text)
    except Exception as exc:
        raise ValueError("pause_at_local_time must use HH:MM, e.g. 06:59") from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError("pause_at_local_time must use HH:MM, e.g. 06:59")
    return hour, minute


def _next_pause_local_deadline(pause_local_time: tuple[int, int] | None) -> datetime | None:
    if pause_local_time is None:
        return None
    now_local = datetime.now().astimezone()
    deadline = now_local.replace(
        hour=int(pause_local_time[0]),
        minute=int(pause_local_time[1]),
        second=0,
        microsecond=0,
    )
    if deadline <= now_local:
        deadline += timedelta(days=1)
    return deadline


def _pause_deadline_reached(pause_deadline: datetime | None) -> bool:
    if pause_deadline is None:
        return False
    return datetime.now().astimezone() >= pause_deadline


class _RemoteEvaluatorProxy:
    def __init__(
        self,
        worker_id: int,
        eval_request_queue: Any,
        reply_queue: Any,
        status_queue: Any,
        stop_event: Any,
        config: SelfPlayConfig,
    ) -> None:
        self.worker_id = worker_id
        self.eval_request_queue = eval_request_queue
        self.reply_queue = reply_queue
        self.status_queue = status_queue
        self.stop_event = stop_event
        self.config = config
        self._request_id = 0
        self.proc_name = f"worker-{worker_id}"

    def __call__(self, batch_cpu: Tensor) -> dict[str, Tensor]:
        if not isinstance(batch_cpu, torch.Tensor):
            raise TypeError("Remote evaluator expects a torch.Tensor")

        batch_cpu = batch_cpu.detach().to(device="cpu", dtype=torch.float32).contiguous()
        request_id = self._request_id
        self._request_id += 1

        request = {
            "kind": "eval",
            "worker_id": self.worker_id,
            "request_id": request_id,
            "timestamp": time.time(),
            "states": batch_cpu,
        }
        safe_put(
            self.eval_request_queue,
            request,
            timeout_s=self.config.queue_put_timeout_s,
            retry_deadline_s=self.config.queue_retry_deadline_s,
            stop_event=self.stop_event,
            proc_name=self.proc_name,
            status_queue=self.status_queue,
            worker_id=self.worker_id,
            heartbeat_interval_s=self.config.heartbeat_interval_s,
            queue_name="eval_request_queue",
        )

        reply = safe_get(
            self.reply_queue,
            timeout_s=self.config.queue_get_timeout_s,
            retry_deadline_s=self.config.worker_eval_timeout_s,
            stop_event=self.stop_event,
            proc_name=self.proc_name,
            status_queue=self.status_queue,
            worker_id=self.worker_id,
            heartbeat_interval_s=self.config.heartbeat_interval_s,
            queue_name=f"worker_reply_queue[{self.worker_id}]",
            on_timeout="raise",
        )

        if reply is None:
            raise QueueTimeoutError(f"{self.proc_name}: evaluator reply unexpectedly missing")
        if reply.get("kind") == SHUTDOWN_KIND:
            raise StopRequested(f"{self.proc_name}: received shutdown while waiting for evaluator reply")
        if reply.get("kind") == STATUS_FATAL:
            raise RuntimeError(f"{self.proc_name}: evaluator fatal: {reply.get('error', 'unknown error')}")
        if reply.get("kind") != "eval_result":
            raise RuntimeError(f"{self.proc_name}: unexpected evaluator reply kind {reply.get('kind')!r}")
        if int(reply.get("request_id", -1)) != request_id:
            raise RuntimeError(
                f"{self.proc_name}: evaluator reply request_id mismatch, "
                f"expected {request_id}, got {reply.get('request_id')}"
            )
        return _validate_evaluator_result(reply, batch_cpu.shape[0], source=self.proc_name)


def _get_move_temperature(config: SelfPlayConfig, ply: int) -> float:
    for ply_limit, temperature in config.move_temperature_schedule:
        if ply <= int(ply_limit):
            return float(temperature)
    return float(config.move_temperature_schedule[-1][1])


_PIECE_CHARS_BY_PLANE = ("K", "A", "B", "N", "R", "C", "P", "k", "a", "b", "n", "r", "c", "p")
_PIECE_VALUES_BY_TYPE = (0.0, 2.0, 2.0, 4.0, 9.0, 4.5, 1.0)
_INITIAL_NONKING_MATERIAL_TOTAL = 96.0


@dataclass(frozen=True)
class _HumanShardSpec:
    path: Path
    sample_count: int


class _HumanShardCache:
    def __init__(self, max_size: int = 4) -> None:
        self.max_size = max(1, int(max_size))
        self._cache: OrderedDict[str, Any] = OrderedDict()

    def get(self, path: Path) -> Any:
        key = str(path.resolve())
        if key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value
            return value

        value = torch.load(path, map_location="cpu", weights_only=False)
        self._cache[key] = value
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
        return value


class _HumanPositionSampler:
    def __init__(self, config: SelfPlayConfig, rng: random.Random) -> None:
        self.config = config
        self.rng = rng
        self.train_dir, self.shards = _load_human_position_shard_specs(Path(config.human_position_source_dir))
        self.cache = _HumanShardCache(max_size=4)

    def maybe_create_start_board(self) -> tuple[Any, dict[str, int]]:
        stats = {
            "attempts": 0,
            "opening_accepts": 0,
            "human_start": 0,
            "standard_start": 0,
            "fallback_to_startpos": 0,
            "context_restored": 0,
            "context_restore_failed": 0,
        }
        if self.config.start_position_mode != "human_positions":
            stats["standard_start"] = 1
            return None, stats
        if self.rng.random() > float(self.config.human_position_mix_ratio):
            stats["standard_start"] = 1
            return None, stats

        from xiangqi_mcts_ext import Board

        max_attempts = 32
        for _ in range(max_attempts):
            stats["attempts"] += 1
            sample = self._sample_legacy_sample()
            state = _legacy_sample_state(sample)
            if not _is_opening_like_state(state, self.config):
                continue
            stats["opening_accepts"] += 1
            try:
                fen = _state_to_fen(state)
            except Exception:
                continue
            board = Board()
            try:
                board.set_fen(fen)
            except Exception:
                continue
            try:
                plies_played, no_capture_count, repetition_count_hint = _infer_start_context_from_sample(sample, state)
                board.set_search_context(plies_played, no_capture_count, repetition_count_hint)
                terminal_code = int(
                    board.terminal_code(
                        int(self.config.max_plies),
                        int(self.config.repeat_limit),
                        int(self.config.repeat_min_ply),
                        int(self.config.no_capture_limit),
                    )
                )
            except Exception:
                stats["context_restore_failed"] += 1
                continue
            if terminal_code != TERMINAL_ONGOING:
                stats["context_restore_failed"] += 1
                continue
            stats["context_restored"] = 1
            stats["human_start"] = 1
            return board, stats

        stats["fallback_to_startpos"] = 1
        stats["standard_start"] = 1
        return None, stats

    def _sample_legacy_sample(self) -> dict[str, Any]:
        spec = _weighted_choice_by_sample_count(self.shards, self.rng)
        shard = self.cache.get(spec.path)
        if not isinstance(shard, dict) or "samples" not in shard or not isinstance(shard["samples"], list):
            raise RuntimeError(f"legacy human shard at {spec.path} is missing 'samples'")
        samples = shard["samples"]
        if not samples:
            raise RuntimeError(f"legacy human shard at {spec.path} is empty")
        local_index = self.rng.randrange(min(spec.sample_count, len(samples)))
        return samples[local_index]


def _weighted_choice_by_sample_count(specs: list[_HumanShardSpec], rng: random.Random) -> _HumanShardSpec:
    total = sum(max(spec.sample_count, 0) for spec in specs)
    if total <= 0:
        return specs[0]
    threshold = rng.randrange(total)
    cumulative = 0
    for spec in specs:
        cumulative += max(spec.sample_count, 0)
        if cumulative > threshold:
            return spec
    return specs[-1]


def _load_human_position_shard_specs(source_dir: Path) -> tuple[Path, list[_HumanShardSpec]]:
    source_dir = source_dir.resolve()
    train_dir = (source_dir / "train") if (source_dir / "train").is_dir() else source_dir
    if not train_dir.is_dir():
        raise FileNotFoundError(f"human position source dir not found: {source_dir}")

    manifest_candidates = [
        source_dir / "manifest.json",
        train_dir.parent / "manifest.json",
    ]
    manifest_path = next((path for path in manifest_candidates if path.is_file()), None)
    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        train_meta = manifest.get("train")
        if isinstance(train_meta, dict) and isinstance(train_meta.get("shards"), list):
            specs = []
            for shard_meta in train_meta["shards"]:
                shard_path = train_dir / str(shard_meta["path"])
                specs.append(
                    _HumanShardSpec(
                        path=shard_path.resolve(),
                        sample_count=max(int(shard_meta.get("samples", 0)), 1),
                    )
                )
            if specs:
                return train_dir, specs

    shard_paths = sorted(train_dir.glob("shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"no human training shards found under {train_dir}")
    return train_dir, [_HumanShardSpec(path=path.resolve(), sample_count=1) for path in shard_paths]


def _legacy_sample_state(sample: dict[str, Any]) -> Tensor:
    state = torch.as_tensor(sample["state"], dtype=torch.float32).contiguous()
    if state.ndim == 4:
        if int(state.shape[0]) != 1:
            raise RuntimeError(f"expected legacy sample state batch dimension 1, got {tuple(state.shape)}")
        state = state[0]
    if state.ndim != 3 or tuple(state.shape[1:]) != (10, 9) or int(state.shape[0]) < 14:
        raise RuntimeError(f"legacy sample state has invalid shape {tuple(state.shape)}")
    return state


def _current_position_planes(state: Tensor) -> Tensor:
    state = _legacy_sample_state({"state": state}) if not isinstance(state, torch.Tensor) else state
    if state.ndim == 4:
        state = state[0]
    if state.ndim != 3:
        raise RuntimeError(f"expected state tensor with 3 dims, got {tuple(state.shape)}")
    return state[:14].to(torch.float32).contiguous()


def _state_is_canonical(state: Tensor) -> bool:
    if state.ndim == 4:
        state = state[0]
    if state.shape[0] <= 112:
        return True
    side_plane = state[112].to(torch.float32)
    return float(side_plane.abs().max().item()) < 0.5


def _opening_like_metrics_from_state(state: Tensor) -> dict[str, float]:
    planes = _current_position_planes(state)
    occupancy = (planes > 0.5).to(torch.int64).view(14, -1).sum(dim=1)
    piece_count = int(occupancy.sum().item())
    nonking_material = 0.0
    for plane_index in range(14):
        piece_type_index = plane_index % 7
        if piece_type_index == 0:
            continue
        nonking_material += float(occupancy[plane_index].item()) * _PIECE_VALUES_BY_TYPE[piece_type_index]
    material_ratio = nonking_material / _INITIAL_NONKING_MATERIAL_TOTAL
    return {
        "piece_count": float(piece_count),
        "material_ratio": float(material_ratio),
        "current_king_count": float(occupancy[0].item()),
        "opponent_king_count": float(occupancy[7].item()),
    }


def _is_opening_like_state(state: Tensor, config: SelfPlayConfig) -> bool:
    metrics = _opening_like_metrics_from_state(state)
    if int(metrics["current_king_count"]) != 1 or int(metrics["opponent_king_count"]) != 1:
        return False
    if metrics["material_ratio"] < float(config.opening_like_material_min_ratio):
        return False
    if int(metrics["piece_count"]) < int(config.opening_like_min_piece_count):
        return False
    return True


def _state_to_fen(state: Tensor) -> str:
    if state.ndim == 4:
        state = state[0]
    if state.ndim != 3:
        raise RuntimeError(f"expected state tensor with 3 dims, got {tuple(state.shape)}")
    planes = _current_position_planes(state)
    is_canonical = _state_is_canonical(state)
    rows: list[str] = []
    for y in range(10):
        empties = 0
        row_parts: list[str] = []
        for x in range(9):
            active_planes = torch.nonzero(planes[:, y, x] > 0.5, as_tuple=False).view(-1)
            if active_planes.numel() == 0:
                empties += 1
                continue
            if active_planes.numel() > 1:
                raise RuntimeError(f"multiple active piece planes at square ({x}, {y})")
            if empties > 0:
                row_parts.append(str(empties))
                empties = 0
            row_parts.append(_PIECE_CHARS_BY_PLANE[int(active_planes[0].item())])
        if empties > 0:
            row_parts.append(str(empties))
        rows.append("".join(row_parts))
    turn = "w" if is_canonical else ("b" if float(state[112].mean().item()) > 0.5 else "w")
    return f"{'/'.join(rows)} {turn}"


def _infer_no_capture_from_state(state: Tensor) -> int:
    state = _legacy_sample_state({"state": state}) if not isinstance(state, torch.Tensor) else state
    if state.ndim == 4:
        state = state[0]
    if state.shape[0] <= 114:
        return 0
    plane_value = float(state[114].to(torch.float32).mean().item())
    if not math.isfinite(plane_value):
        return 0
    plane_value = max(0.0, min(0.999999, plane_value))
    if plane_value <= 0.0:
        return 0
    return max(0, int(round(30.0 * math.atanh(plane_value))))


def _infer_repetition_hint_from_state(state: Tensor) -> int:
    state = _legacy_sample_state({"state": state}) if not isinstance(state, torch.Tensor) else state
    if state.ndim == 4:
        state = state[0]
    if state.shape[0] <= 113:
        return 1
    plane_value = float(state[113].to(torch.float32).mean().item())
    if not math.isfinite(plane_value):
        return 1
    hint = int(round(plane_value / 0.5)) + 1
    return max(1, min(3, hint))


def _infer_start_context_from_sample(sample: dict[str, Any], state: Tensor) -> tuple[int, int, int]:
    ply_value = sample.get("ply", 0)
    try:
        plies_played = max(int(ply_value), 0)
    except Exception:
        plies_played = 0
    return (
        plies_played,
        _infer_no_capture_from_state(state),
        _infer_repetition_hint_from_state(state),
    )


def _build_start_position_metrics(summary: dict[str, Any]) -> dict[str, float]:
    games_completed = max(int(summary.get("games_completed", 0)), 0)
    human_starts = max(int(summary.get("_human_start_games", 0)), 0)
    standard_starts = max(int(summary.get("_standard_start_games", 0)), 0)
    fallbacks = max(int(summary.get("_fallback_to_startpos_games", 0)), 0)
    attempts = max(int(summary.get("_human_position_attempts", 0)), 0)
    opening_accepts = max(int(summary.get("_opening_like_accepts", 0)), 0)
    context_restored = max(int(summary.get("_context_restored_games", 0)), 0)
    context_restore_failed = max(int(summary.get("_context_restore_failed", 0)), 0)
    return {
        "human_start_rate": (100.0 * human_starts / games_completed) if games_completed > 0 else 0.0,
        "standard_start_rate": (100.0 * standard_starts / games_completed) if games_completed > 0 else 0.0,
        "fallback_to_startpos_rate": (100.0 * fallbacks / games_completed) if games_completed > 0 else 0.0,
        "opening_like_accept_rate": (100.0 * opening_accepts / attempts) if attempts > 0 else 0.0,
        "context_restore_rate": (100.0 * context_restored / games_completed) if games_completed > 0 else 0.0,
        "context_restore_fail_rate": (100.0 * context_restore_failed / attempts) if attempts > 0 else 0.0,
    }


def _select_bootstrap_move(
    *,
    board: Any,
    best_move: int,
    policy_idxs: Tensor,
    policy_probs: Tensor,
    stm_is_black: bool,
    config: SelfPlayConfig,
    canonical_action_fn: Any,
) -> tuple[int, dict[str, int]]:
    default_stats = {
        "move_decision": 1,
        "override": 0,
        "forced_draw": 0,
        "avoided_repeat": 0,
        "avoided_nocap": 0,
        "capture_pressure": 0,
        "capture_priority": 0,
        "capture_injection": 0,
        "capture_selected": 0,
    }
    best_move = int(best_move)
    if not config.avoid_immediate_draw_moves:
        return best_move, default_stats

    current_no_capture_count = 0
    no_capture_getter = getattr(board, "no_capture_count", None)
    if callable(no_capture_getter):
        try:
            current_no_capture_count = max(int(no_capture_getter()), 0)
        except Exception:
            current_no_capture_count = 0
    candidate_scan = max(1, int(config.immediate_draw_candidate_scan))
    capture_injection_threshold = max(int(config.immediate_draw_capture_injection_threshold), 0)
    capture_priority_threshold = max(int(config.immediate_draw_capture_priority_threshold), 0)
    nocap_pressure_threshold = max(int(config.immediate_draw_nocap_pressure_threshold), 0)
    capture_pressure_active = current_no_capture_count >= capture_injection_threshold
    capture_priority_active = current_no_capture_count >= capture_priority_threshold
    if current_no_capture_count >= nocap_pressure_threshold:
        candidate_scan = max(candidate_scan, int(config.immediate_draw_nocap_pressure_scan))
    candidate_moves: OrderedDict[int, float] = OrderedDict()
    best_move_policy_prob = 0.0
    if int(policy_idxs.numel()) > 0 and int(policy_probs.numel()) == int(policy_idxs.numel()):
        idxs_cpu = policy_idxs.detach().to(device="cpu", dtype=torch.int64).view(-1)
        probs_cpu = policy_probs.detach().to(device="cpu", dtype=torch.float32).view(-1)
        top_count = min(candidate_scan, int(idxs_cpu.numel()))
        sorted_positions = torch.argsort(probs_cpu, descending=True)[:top_count].tolist()
        for pos in sorted_positions:
            candidate_policy_idx = int(idxs_cpu[pos].item())
            candidate_move = int(canonical_action_fn(candidate_policy_idx, stm_is_black))
            candidate_prob = float(probs_cpu[pos].item())
            previous_prob = candidate_moves.get(candidate_move)
            if previous_prob is None or candidate_prob > previous_prob:
                candidate_moves[candidate_move] = candidate_prob
            if candidate_move == best_move:
                best_move_policy_prob = max(best_move_policy_prob, candidate_prob)
    if best_move not in candidate_moves:
        candidate_moves[best_move] = best_move_policy_prob

    injected_capture_moves: set[int] = set()
    if capture_pressure_active:
        legal_moves_getter = getattr(board, "legal_moves", None)
        legal_moves: list[int] = []
        if callable(legal_moves_getter):
            try:
                legal_moves = [int(move) for move in legal_moves_getter()]
            except Exception:
                legal_moves = []
        for legal_move in legal_moves:
            if legal_move in candidate_moves:
                continue
            try:
                is_capture = bool(board.is_capture(int(legal_move)))
            except Exception:
                is_capture = False
            if not is_capture:
                continue
            candidate_moves[int(legal_move)] = 0.0
            injected_capture_moves.add(int(legal_move))

    evaluated_candidates: list[dict[str, Any]] = []
    for candidate_move, candidate_prob in candidate_moves.items():
        is_capture = bool(board.is_capture(int(candidate_move)))
        board.push(int(candidate_move))
        try:
            resulting_no_capture_count = current_no_capture_count
            resulting_no_capture_getter = getattr(board, "no_capture_count", None)
            if callable(resulting_no_capture_getter):
                try:
                    resulting_no_capture_count = max(int(resulting_no_capture_getter()), 0)
                except Exception:
                    resulting_no_capture_count = current_no_capture_count
            terminal_code = int(
                board.terminal_code(
                    int(config.max_plies),
                    int(config.repeat_limit),
                    int(config.repeat_min_ply),
                    int(config.no_capture_limit),
                )
            )
            result_after_move = (
                int(board.terminal_result_red_view(terminal_code))
                if terminal_code != TERMINAL_ONGOING
                else 0
            )
            mover_wins = (
                result_after_move != 0
                and ((not stm_is_black and result_after_move > 0) or (stm_is_black and result_after_move < 0))
            )
            mover_loses = (
                result_after_move != 0
                and ((not stm_is_black and result_after_move < 0) or (stm_is_black and result_after_move > 0))
            )
        finally:
            board.pop()
        would_repeat_draw = terminal_code == TERMINATION_REPETITION_DRAW
        would_nocap_draw = terminal_code == TERMINATION_NO_CAPTURE_DRAW
        would_max_draw = terminal_code == TERMINATION_MAX_PLIES_DRAW
        would_immediate_draw = would_repeat_draw or would_nocap_draw or would_max_draw
        improves_nocap_pressure = resulting_no_capture_count < current_no_capture_count
        evaluated_candidates.append(
            {
                "move": int(candidate_move),
                "prob": float(candidate_prob),
                "is_capture": is_capture,
                "injected_capture": bool(int(candidate_move) in injected_capture_moves),
                "resulting_no_capture_count": int(resulting_no_capture_count),
                "improves_nocap_pressure": bool(improves_nocap_pressure),
                "terminal_code": int(terminal_code),
                "mover_wins": bool(mover_wins),
                "mover_loses": bool(mover_loses),
                "would_repeat_draw": bool(would_repeat_draw),
                "would_nocap_draw": bool(would_nocap_draw),
                "would_max_draw": bool(would_max_draw),
                "would_immediate_draw": bool(would_immediate_draw),
            }
        )

    if not evaluated_candidates:
        return best_move, default_stats

    def _is_safe_capture(item: dict[str, Any]) -> bool:
        return bool(item["is_capture"]) and not bool(item["would_immediate_draw"]) and not bool(item["mover_loses"])

    safe_capture_available = any(_is_safe_capture(item) for item in evaluated_candidates)
    prioritize_safe_capture = capture_priority_active and safe_capture_available
    prioritize_injected_capture = capture_priority_active and any(
        bool(item["injected_capture"]) and _is_safe_capture(item) for item in evaluated_candidates
    )

    def _rank_candidate(item: dict[str, Any]) -> tuple[Any, ...]:
        return (
            1 if item["would_immediate_draw"] else 0,
            0 if item["mover_wins"] else 1,
            1 if item["mover_loses"] else 0,
            0 if prioritize_safe_capture and _is_safe_capture(item) else (1 if prioritize_safe_capture else 0),
            0 if prioritize_injected_capture and item["injected_capture"] else (1 if prioritize_injected_capture else 0),
            0 if item["is_capture"] else 1,
            0 if item["improves_nocap_pressure"] else 1,
            item["resulting_no_capture_count"],
            0 if int(item["move"]) == best_move else 1,
            -float(item["prob"]),
        )

    selected = min(evaluated_candidates, key=_rank_candidate)
    best_eval = next((item for item in evaluated_candidates if item["move"] == best_move), selected)
    stats = {
        "move_decision": 1,
        "override": int(selected["move"] != best_move),
        "forced_draw": int(selected["would_immediate_draw"]),
        "avoided_repeat": int(best_eval["would_repeat_draw"] and not selected["would_repeat_draw"]),
        "avoided_nocap": int(best_eval["would_nocap_draw"] and not selected["would_nocap_draw"]),
        "capture_pressure": int(capture_pressure_active),
        "capture_priority": int(capture_priority_active),
        "capture_available": int(safe_capture_available),
        "capture_injection": int(bool(injected_capture_moves)),
        "capture_selected": int(bool(selected.get("injected_capture", False))),
        "capture_any_selected": int(bool(selected.get("is_capture", False))),
    }
    return int(selected["move"]), stats


class _TensorizedShardBuffer:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.state_chunks: list[Tensor] = []
        self.policy_idxs_chunks: list[Tensor] = []
        self.policy_probs_chunks: list[Tensor] = []
        self.policy_offsets: list[int] = [0]
        self.sample_fields: dict[str, list[Tensor]] = {
            "z": [],
            "wdl_target": [],
            "root_value": [],
            "root_wdl_value": [],
            "chosen_move": [],
            "num_legal_moves": [],
            "ply": [],
            "game_id": [],
            "stm_is_black": [],
            "is_draw": [],
            "termination_code": [],
        }
        self.sample_count = 0

    def add_payload(self, payload: dict[str, Tensor]) -> None:
        count = int(payload["state"].shape[0])
        if count == 0:
            return

        self.state_chunks.append(payload["state"].contiguous())
        self.policy_idxs_chunks.append(payload["policy_idxs"].contiguous())
        self.policy_probs_chunks.append(payload["policy_probs"].contiguous())

        base = self.policy_offsets[-1]
        offsets = payload["policy_offsets"].to(dtype=torch.int64).cpu()
        self.policy_offsets.extend((offsets[1:] + base).tolist())

        for key in self.sample_fields:
            self.sample_fields[key].append(payload[key].contiguous())

        self.sample_count += count

    def flush(self, path: Path) -> int:
        if self.sample_count == 0:
            return 0

        data = {
            "state": torch.cat(self.state_chunks, dim=0).contiguous(),
            "policy_offsets": torch.tensor(self.policy_offsets, dtype=torch.int64),
            "policy_idxs": torch.cat(self.policy_idxs_chunks, dim=0).contiguous(),
            "policy_probs": torch.cat(self.policy_probs_chunks, dim=0).contiguous(),
        }
        for key, chunks in self.sample_fields.items():
            data[key] = torch.cat(chunks, dim=0).contiguous()

        torch.save(data, path)
        flushed = self.sample_count
        self.reset()
        return flushed


def run_selfplay(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    config: SelfPlayConfig,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path).resolve()
    output_dir = Path(output_dir).resolve()
    _validate_run_inputs(checkpoint_path, output_dir)

    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested for self-play evaluator, but CUDA is not available")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train").mkdir(parents=True, exist_ok=True)
    pause_local_time = _parse_pause_local_time(config.pause_at_local_time)
    pause_deadline = _next_pause_local_deadline(pause_local_time)
    pause_reached_logged = False

    ctx = tmp.get_context("spawn")
    stop_event = ctx.Event()

    eval_request_queue = ctx.Queue(maxsize=config.eval_request_queue_maxsize)
    sample_queue = ctx.Queue(maxsize=config.sample_queue_maxsize)
    status_queue = ctx.Queue(maxsize=config.status_queue_maxsize)
    reply_queues = [ctx.Queue(maxsize=config.reply_queue_maxsize) for _ in range(config.num_workers)]

    evaluator_proc = ctx.Process(
        target=_gpu_evaluator_main,
        name="gpu-evaluator",
        args=(
            checkpoint_path,
            eval_request_queue,
            reply_queues,
            status_queue,
            stop_event,
            config,
        ),
    )
    writer_proc = ctx.Process(
        target=_writer_main,
        name="writer",
        args=(
            output_dir,
            sample_queue,
            status_queue,
            stop_event,
            config,
        ),
    )
    worker_procs = [
        ctx.Process(
            target=_worker_main,
            name=f"worker-{worker_id}",
            args=(
                worker_id,
                eval_request_queue,
                reply_queues[worker_id],
                sample_queue,
                status_queue,
                stop_event,
                config,
            ),
        )
        for worker_id in range(config.num_workers)
    ]
    all_procs = [evaluator_proc, writer_proc, *worker_procs]

    summary = {
        "games_completed": 0,
        "samples_written": 0,
        "shards_written": 0,
        "terminated_cleanly": False,
        "fatal_error": None,
        "_generated_samples": 0,
        "_draw_games": 0,
        "_plies_total": 0,
        "_root_value_sum": 0.0,
        "_root_value_count": 0,
        "_root_abs_value_sum": 0.0,
        "_root_wdl_value_sum": 0.0,
        "_root_wdl_value_count": 0,
        "_root_abs_wdl_value_sum": 0.0,
        "_root_value_gap_sum": 0.0,
        "_root_value_gap_count": 0,
        "_stm_black_samples": 0,
        "_stm_total_samples": 0,
        "_human_position_attempts": 0,
        "_opening_like_accepts": 0,
        "_human_start_games": 0,
        "_standard_start_games": 0,
        "_fallback_to_startpos_games": 0,
        "_context_restored_games": 0,
        "_context_restore_failed": 0,
        "_move_decision_count": 0,
        "_anti_draw_override_count": 0,
        "_anti_draw_forced_draw_count": 0,
        "_anti_draw_avoided_repeat_count": 0,
        "_anti_draw_avoided_nocap_count": 0,
        "_anti_draw_capture_pressure_count": 0,
        "_anti_draw_capture_priority_count": 0,
        "_anti_draw_capture_available_count": 0,
        "_anti_draw_capture_injection_count": 0,
        "_anti_draw_capture_selected_count": 0,
        "_anti_draw_capture_any_selected_count": 0,
        "scheduled_pause_requested": False,
        "_termination_counts": {
            TERMINATION_CHECKMATE_OR_STALEMATE: 0,
            TERMINATION_MAX_PLIES_DRAW: 0,
            TERMINATION_REPETITION_DRAW: 0,
            TERMINATION_NO_CAPTURE_DRAW: 0,
            TERMINATION_PERPETUAL_CHECK_LOSS: 0,
        },
        "_next_progress_log_games": config.progress_log_games,
        "_next_progress_log_shards": config.progress_log_shards,
    }
    _write_manifest(
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        config=config,
        summary=summary,
        quality_info={"quality": "pending", "metrics": _build_quality_metrics(summary)},
        manifest_state="in_progress",
    )

    last_heartbeat: dict[str, float] = {}
    expected_exits: set[str] = set()
    writer_done = False

    try:
        for proc in all_procs:
            proc.start()
            last_heartbeat[proc.name] = time.monotonic()

        while True:
            message = None
            try:
                message = status_queue.get(timeout=config.queue_get_timeout_s)
            except queue_module.Empty:
                message = None
            except (EOFError, BrokenPipeError, ConnectionResetError, OSError) as exc:
                summary["fatal_error"] = f"supervisor lost status_queue connection: {exc}"
                stop_event.set()
                break

            if message is not None:
                proc_name = str(message.get("proc_name", "unknown"))
                last_heartbeat[proc_name] = time.monotonic()
                kind = message.get("kind")

                if kind == STATUS_FATAL:
                    summary["fatal_error"] = f"{proc_name}: {message.get('error', 'unknown fatal error')}"
                    stop_event.set()
                    break
                writer_done = _apply_status_message(summary, message) or writer_done
                _maybe_log_progress(summary, config)
                _maybe_log_shard_progress(summary, config)
                if kind == STATUS_GAME_COMPLETE:
                    if config.target_games is not None and summary["games_completed"] >= config.target_games:
                        stop_event.set()
                        break
                    if (
                        config.target_samples is not None
                        and int(summary["_generated_samples"]) >= config.target_samples
                    ):
                        stop_event.set()
                        break
                if kind == STATUS_EXITING:
                    expected_exits.add(proc_name)

            now = time.monotonic()
            for proc in all_procs:
                if proc.name in expected_exits and not proc.is_alive():
                    continue
                if proc.exitcode is not None and proc.exitcode != 0 and summary["fatal_error"] is None:
                    summary["fatal_error"] = f"{proc.name} exited with code {proc.exitcode}"
                    stop_event.set()
                    break
                if proc.is_alive():
                    last = last_heartbeat.get(proc.name, now)
                    if (now - last) > config.heartbeat_timeout_s:
                        summary["fatal_error"] = (
                            f"{proc.name} heartbeat timed out after {config.heartbeat_timeout_s} seconds"
                        )
                        stop_event.set()
                        break
            if summary["fatal_error"] is not None:
                break
            if _pause_deadline_reached(pause_deadline):
                summary["scheduled_pause_requested"] = True
                if not pause_reached_logged:
                    pause_label = pause_deadline.astimezone().strftime("%Y-%m-%d %H:%M") if pause_deadline else "unknown"
                    print(
                        f"self-play scheduled pause reached at local {pause_label}; stopping gracefully.",
                        flush=True,
                    )
                    pause_reached_logged = True
                stop_event.set()
                break

        stop_event.set()
    finally:
        _send_shutdown_to_queue(
            eval_request_queue,
            stop_event,
            proc_name="supervisor",
            status_queue=None,
            queue_name="eval_request_queue",
            repeats=config.num_workers + 1,
        )
        for worker_id, reply_queue in enumerate(reply_queues):
            _send_shutdown_to_queue(
                reply_queue,
                stop_event,
                proc_name="supervisor",
                status_queue=None,
                queue_name=f"worker_reply_queue[{worker_id}]",
                repeats=1,
            )

        _join_or_kill_processes(worker_procs + [evaluator_proc], config)

        writer_proc.join(timeout=config.shutdown_join_timeout_s)
        if writer_proc.is_alive():
            _send_shutdown_to_queue(
                sample_queue,
                stop_event,
                proc_name="supervisor",
                status_queue=None,
                queue_name="sample_queue",
                repeats=1,
            )
            _join_or_kill_processes([writer_proc], config)

        while True:
            try:
                late_message = status_queue.get(timeout=0.05)
            except queue_module.Empty:
                break
            except (EOFError, BrokenPipeError, ConnectionResetError, OSError):
                break
            writer_done = _apply_status_message(summary, late_message) or writer_done
            _maybe_log_progress(summary, config)
            _maybe_log_shard_progress(summary, config)

        if summary["fatal_error"] is None:
            for proc in all_procs:
                if proc.exitcode not in (None, 0):
                    summary["fatal_error"] = f"{proc.name} exited with code {proc.exitcode}"
                    break
        if summary["fatal_error"] is None and not writer_done:
            summary["fatal_error"] = "writer did not report completion before shutdown"

        _drain_and_close_queue(eval_request_queue)
        _drain_and_close_queue(sample_queue)
        _drain_and_close_queue(status_queue)
        for reply_queue in reply_queues:
            _drain_and_close_queue(reply_queue)

    quality, metrics = _classify_bootstrap_quality(summary, config)
    quality_info = {"quality": quality, "metrics": metrics}
    start_position_metrics = _build_start_position_metrics(summary)
    for key in (
        "_generated_samples",
        "_draw_games",
        "_plies_total",
        "_root_value_sum",
        "_root_value_count",
        "_root_abs_value_sum",
        "_root_wdl_value_sum",
        "_root_wdl_value_count",
        "_root_abs_wdl_value_sum",
        "_root_value_gap_sum",
        "_root_value_gap_count",
        "_stm_black_samples",
        "_stm_total_samples",
        "_human_position_attempts",
        "_opening_like_accepts",
        "_human_start_games",
        "_standard_start_games",
        "_fallback_to_startpos_games",
        "_move_decision_count",
        "_anti_draw_override_count",
        "_anti_draw_forced_draw_count",
        "_anti_draw_avoided_repeat_count",
        "_anti_draw_avoided_nocap_count",
        "_anti_draw_capture_pressure_count",
        "_anti_draw_capture_priority_count",
        "_anti_draw_capture_available_count",
        "_anti_draw_capture_injection_count",
        "_anti_draw_capture_selected_count",
        "_anti_draw_capture_any_selected_count",
        "_termination_counts",
        "_next_progress_log_games",
        "_next_progress_log_shards",
    ):
        summary.pop(key, None)
    summary["terminated_cleanly"] = summary["fatal_error"] is None and writer_done
    summary["quality"] = quality_info["quality"]
    summary["quality_metrics"] = quality_info["metrics"]
    summary["start_position_metrics"] = start_position_metrics
    _write_manifest(output_dir, checkpoint_path, config, summary, quality_info, manifest_state="complete")
    return summary


def safe_put(
    queue: Any,
    item: Any,
    timeout_s: float,
    retry_deadline_s: float,
    stop_event: Any,
    proc_name: str,
    status_queue: Any,
    worker_id: int = -1,
    heartbeat_interval_s: float = 2.0,
    queue_name: str = "queue",
    on_timeout: str = "raise",
    abort_on_stop_event: bool = True,
) -> bool:
    deadline = time.monotonic() + retry_deadline_s
    while True:
        if abort_on_stop_event and stop_event.is_set():
            raise StopRequested(f"{proc_name}: stop requested during put to {queue_name}")
        try:
            queue.put(item, timeout=timeout_s)
            _maybe_emit_heartbeat(status_queue, stop_event, proc_name, worker_id, heartbeat_interval_s)
            return True
        except queue_module.Full as exc:
            _maybe_emit_heartbeat(
                status_queue,
                stop_event,
                proc_name,
                worker_id,
                heartbeat_interval_s,
                extra={"queue_name": queue_name, "wait": "put"},
            )
            if time.monotonic() >= deadline:
                message = f"{proc_name}: timed out putting to {queue_name} after {retry_deadline_s:.1f}s"
                if on_timeout == "drop":
                    return False
                raise QueueTimeoutError(message) from exc
        except (EOFError, BrokenPipeError, ConnectionResetError, OSError) as exc:
            raise QueueTransportError(f"{proc_name}: transport failure putting to {queue_name}: {exc}") from exc


def safe_get(
    queue: Any,
    timeout_s: float,
    retry_deadline_s: float,
    stop_event: Any,
    proc_name: str,
    status_queue: Any,
    worker_id: int = -1,
    heartbeat_interval_s: float = 2.0,
    queue_name: str = "queue",
    on_timeout: str = "raise",
    abort_on_stop_event: bool = True,
) -> Any | None:
    deadline = time.monotonic() + retry_deadline_s
    while True:
        if abort_on_stop_event and stop_event.is_set():
            raise StopRequested(f"{proc_name}: stop requested during get from {queue_name}")
        try:
            item = queue.get(timeout=timeout_s)
            _maybe_emit_heartbeat(status_queue, stop_event, proc_name, worker_id, heartbeat_interval_s)
            return item
        except queue_module.Empty as exc:
            _maybe_emit_heartbeat(
                status_queue,
                stop_event,
                proc_name,
                worker_id,
                heartbeat_interval_s,
                extra={"queue_name": queue_name, "wait": "get"},
            )
            if time.monotonic() >= deadline:
                if on_timeout == "return_none":
                    return None
                message = f"{proc_name}: timed out getting from {queue_name} after {retry_deadline_s:.1f}s"
                raise QueueTimeoutError(message) from exc
        except (EOFError, BrokenPipeError, ConnectionResetError, OSError) as exc:
            raise QueueTransportError(f"{proc_name}: transport failure getting from {queue_name}: {exc}") from exc


def _worker_main(
    worker_id: int,
    eval_request_queue: Any,
    reply_queue: Any,
    sample_queue: Any,
    status_queue: Any,
    stop_event: Any,
    config: SelfPlayConfig,
) -> None:
    proc_name = f"worker-{worker_id}"
    _emit_status(status_queue, stop_event, proc_name, worker_id, STATUS_STARTED)
    proxy = _RemoteEvaluatorProxy(worker_id, eval_request_queue, reply_queue, status_queue, stop_event, config)

    try:
        from xiangqi_mcts_ext import Board, canonical_action, mcts_search

        local_game_index = 0
        seed_offset = config.seed + worker_id * 100_003
        position_rng = random.Random(seed_offset + 17)
        human_position_sampler = (
            _HumanPositionSampler(config, position_rng)
            if config.start_position_mode == "human_positions"
            else None
        )

        while not stop_event.is_set():
            game_id = worker_id * 1_000_000_000 + local_game_index
            board = Board()
            start_stats = {
                "attempts": 0,
                "opening_accepts": 0,
                "human_start": 0,
                "standard_start": 1,
                "fallback_to_startpos": 0,
                "context_restored": 0,
                "context_restore_failed": 0,
            }
            if human_position_sampler is not None:
                sampled_board, start_stats = human_position_sampler.maybe_create_start_board()
                if sampled_board is not None:
                    board = sampled_board
            game_records: list[dict[str, Any]] = []
            termination_code: int | None = None
            final_red_result = 0
            is_draw = False
            game_move_decisions = {
                "move_decision": 0,
                "override": 0,
                "forced_draw": 0,
                "avoided_repeat": 0,
                "avoided_nocap": 0,
                "capture_pressure": 0,
                "capture_priority": 0,
                "capture_available": 0,
                "capture_injection": 0,
                "capture_selected": 0,
                "capture_any_selected": 0,
            }

            while True:
                if stop_event.is_set():
                    raise StopRequested(f"{proc_name}: stop requested while playing game {game_id}")

                _maybe_emit_heartbeat(status_queue, stop_event, proc_name, worker_id, config.heartbeat_interval_s)

                search_ply = int(board.plies_played())
                terminal_code = int(
                    board.terminal_code(
                        int(config.max_plies),
                        int(config.repeat_limit),
                        int(config.repeat_min_ply),
                        int(config.no_capture_limit),
                    )
                )
                if terminal_code != TERMINAL_ONGOING:
                    termination_code = terminal_code
                    final_red_result = int(board.terminal_result_red_view(termination_code))
                    is_draw = final_red_result == 0
                    break

                state_cpu = board.to_tensor_canonical().to(torch.float32).contiguous()
                eval_out = proxy(state_cpu)
                root_wdl_value = float("nan")
                if "wdl_logits" in eval_out:
                    wdl_probs = torch.softmax(eval_out["wdl_logits"][0], dim=0)
                    root_wdl_value = float((wdl_probs[0] - wdl_probs[2]).item())

                stm_is_black = bool(board.turn() == 1)
                temperature_move = _get_move_temperature(config, search_ply)
                best_move, policy_idxs, policy_probs, root_value = mcts_search(
                    board=board,
                    net=proxy,
                    num_simulations=config.num_simulations,
                    c_puct=config.c_puct,
                    q_weight=config.q_weight,
                    q_clip=config.q_clip,
                    add_root_noise=(config.add_root_noise and search_ply < config.root_noise_end_ply),
                    dirichlet_alpha=config.dirichlet_alpha,
                    dirichlet_eps=config.dirichlet_eps,
                    temperature_move=temperature_move,
                    temperature_target=config.temperature_target,
                    eval_batch_size=config.eval_batch_size,
                    seed=seed_offset + local_game_index * 1_009 + search_ply,
                    canonical_input=True,
                    canonical_policy=True,
                    max_plies=config.max_plies,
                    repeat_limit=config.repeat_limit,
                    repeat_min_ply=config.repeat_min_ply,
                    no_capture_limit=config.no_capture_limit,
                )

                if int(best_move) < 0:
                    termination_code = int(
                        board.terminal_code(
                            int(config.max_plies),
                            int(config.repeat_limit),
                            int(config.repeat_min_ply),
                            int(config.no_capture_limit),
                        )
                    )
                    if termination_code == TERMINAL_ONGOING:
                        termination_code = TERMINATION_CHECKMATE_OR_STALEMATE
                    final_red_result = int(board.terminal_result_red_view(termination_code))
                    is_draw = final_red_result == 0
                    break

                selected_move, move_decision_stats = _select_bootstrap_move(
                    board=board,
                    best_move=int(best_move),
                    policy_idxs=policy_idxs,
                    policy_probs=policy_probs,
                    stm_is_black=stm_is_black,
                    config=config,
                    canonical_action_fn=canonical_action,
                )
                for stat_key in game_move_decisions:
                    game_move_decisions[stat_key] += int(move_decision_stats.get(stat_key, 0))

                chosen_move_canonical = int(canonical_action(int(selected_move), stm_is_black))
                game_records.append(
                    {
                        "state": state_cpu[0].to(torch.bfloat16).contiguous().clone(),
                        "policy_idxs": policy_idxs.to(torch.int64).contiguous().clone(),
                        "policy_probs": policy_probs.to(torch.float32).contiguous().clone(),
                        "root_value": float(root_value),
                        "root_wdl_value": root_wdl_value,
                        "chosen_move": chosen_move_canonical,
                        "num_legal_moves": int(policy_idxs.numel()),
                        "ply": int(search_ply),
                        "game_id": int(game_id),
                        "stm_is_black": stm_is_black,
                    }
                )
                board.push(int(selected_move))

            if termination_code is None:
                termination_code = TERMINATION_MAX_PLIES_DRAW
                final_red_result = 0
                is_draw = True
            else:
                is_draw = final_red_result == 0

            if stop_event.is_set():
                raise StopRequested(f"{proc_name}: stop requested before emitting game {game_id}")

            if game_records:
                payload = _finalize_game_records(
                    game_records=game_records,
                    final_red_result=final_red_result,
                    termination_code=termination_code,
                    is_draw=is_draw,
                )
                serialized_payload = _serialize_payload(payload)
                payload_samples = int(payload["state"].shape[0])
                safe_put(
                    sample_queue,
                    {
                        "kind": "sample_chunk",
                        "worker_id": worker_id,
                        "game_id": int(game_id),
                        "timestamp": time.time(),
                        "payload_bytes": serialized_payload,
                        "payload_num_samples": payload_samples,
                    },
                    timeout_s=config.queue_put_timeout_s,
                    retry_deadline_s=config.queue_retry_deadline_s,
                    stop_event=stop_event,
                    proc_name=proc_name,
                    status_queue=status_queue,
                    worker_id=worker_id,
                    heartbeat_interval_s=config.heartbeat_interval_s,
                    queue_name="sample_queue",
                )
                _emit_status(
                    status_queue,
                    stop_event,
                    proc_name,
                    worker_id,
                    STATUS_GAME_COMPLETE,
                    games_completed=1,
                    samples_generated=payload_samples,
                    game_id=int(game_id),
                    termination_code=int(termination_code),
                    is_draw=bool(is_draw),
                    plies_in_game=int(len(game_records)),
                    human_position_attempts=int(start_stats["attempts"]),
                    opening_like_accepts=int(start_stats["opening_accepts"]),
                    human_start_games=int(start_stats["human_start"]),
                    standard_start_games=int(start_stats["standard_start"]),
                    fallback_to_startpos_games=int(start_stats["fallback_to_startpos"]),
                    context_restored_games=int(start_stats["context_restored"]),
                    context_restore_failed=int(start_stats["context_restore_failed"]),
                    move_decision_count=int(game_move_decisions["move_decision"]),
                    anti_draw_override_count=int(game_move_decisions["override"]),
                    anti_draw_forced_draw_count=int(game_move_decisions["forced_draw"]),
                    anti_draw_avoided_repeat_count=int(game_move_decisions["avoided_repeat"]),
                    anti_draw_avoided_nocap_count=int(game_move_decisions["avoided_nocap"]),
                    anti_draw_capture_pressure_count=int(game_move_decisions["capture_pressure"]),
                    anti_draw_capture_priority_count=int(game_move_decisions["capture_priority"]),
                    anti_draw_capture_available_count=int(game_move_decisions["capture_available"]),
                    anti_draw_capture_injection_count=int(game_move_decisions["capture_injection"]),
                    anti_draw_capture_selected_count=int(game_move_decisions["capture_selected"]),
                    anti_draw_capture_any_selected_count=int(game_move_decisions["capture_any_selected"]),
                    root_value_sum=float(sum(float(record["root_value"]) for record in game_records)),
                    root_value_count=int(len(game_records)),
                    root_abs_value_sum=float(sum(abs(float(record["root_value"])) for record in game_records)),
                    root_wdl_value_sum=float(
                        sum(
                            float(record["root_wdl_value"])
                            for record in game_records
                            if math.isfinite(float(record["root_wdl_value"]))
                        )
                    ),
                    root_wdl_value_count=int(
                        sum(
                            1
                            for record in game_records
                            if math.isfinite(float(record["root_wdl_value"]))
                        )
                    ),
                    root_abs_wdl_value_sum=float(
                        sum(
                            abs(float(record["root_wdl_value"]))
                            for record in game_records
                            if math.isfinite(float(record["root_wdl_value"]))
                        )
                    ),
                    root_value_gap_sum=float(
                        sum(
                            abs(float(record["root_value"]) - float(record["root_wdl_value"]))
                            for record in game_records
                            if math.isfinite(float(record["root_wdl_value"]))
                        )
                    ),
                    root_value_gap_count=int(
                        sum(
                            1
                            for record in game_records
                            if math.isfinite(float(record["root_wdl_value"]))
                        )
                    ),
                    stm_black_samples=int(sum(1 for record in game_records if bool(record["stm_is_black"]))),
                    stm_total_samples=int(len(game_records)),
                )

            local_game_index += 1

    except StopRequested:
        pass
    except Exception as exc:
        stop_event.set()
        _emit_status(
            status_queue,
            stop_event,
            proc_name,
            worker_id,
            STATUS_FATAL,
            error=_format_exception(exc),
            critical=True,
        )
    finally:
        try:
            safe_put(
                sample_queue,
                {
                    "kind": STATUS_WORKER_DONE,
                    "worker_id": worker_id,
                    "timestamp": time.time(),
                },
                timeout_s=config.queue_put_timeout_s,
                retry_deadline_s=config.queue_retry_deadline_s,
                stop_event=stop_event,
                proc_name=proc_name,
                status_queue=status_queue,
                worker_id=worker_id,
                heartbeat_interval_s=config.heartbeat_interval_s,
                queue_name="sample_queue",
                on_timeout="drop",
                abort_on_stop_event=False,
            )
        except Exception:
            pass
        _emit_status(status_queue, stop_event, proc_name, worker_id, STATUS_EXITING)
        _close_queue_endpoint(sample_queue)
        _close_queue_endpoint(eval_request_queue)
        _close_queue_endpoint(reply_queue)
        _close_queue_endpoint(status_queue)


def _gpu_evaluator_main(
    checkpoint_path: Path,
    eval_request_queue: Any,
    reply_queues: list[Any],
    status_queue: Any,
    stop_event: Any,
    config: SelfPlayConfig,
) -> None:
    proc_name = "gpu-evaluator"
    _emit_status(status_queue, stop_event, proc_name, -1, STATUS_STARTED)

    try:
        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available inside evaluator process")

        model, _ = _load_model_from_checkpoint(checkpoint_path)
        if config.frozen_value_checkpoint:
            value_model, _ = _load_model_from_checkpoint(Path(config.frozen_value_checkpoint))
            model = ChimeraPolicyValueModel(model, value_model)
            print(
                f"[selfplay] CHIMERA evaluator: policy={checkpoint_path} "
                f"value={config.frozen_value_checkpoint}",
                flush=True,
            )
        model = model.eval().to(config.device)

        pending: list[dict[str, Any]] = []
        pending_states = 0
        shutdown_requested = False
        first_pending_at = 0.0

        while True:
            if stop_event.is_set():
                shutdown_requested = True
            _maybe_emit_heartbeat(status_queue, stop_event, proc_name, -1, config.heartbeat_interval_s)

            if pending_states < config.max_states_per_batch:
                message = safe_get(
                    eval_request_queue,
                    timeout_s=config.queue_get_timeout_s,
                    retry_deadline_s=config.queue_get_timeout_s,
                    stop_event=stop_event,
                    proc_name=proc_name,
                    status_queue=status_queue,
                    worker_id=-1,
                    heartbeat_interval_s=config.heartbeat_interval_s,
                    queue_name="eval_request_queue",
                    on_timeout="return_none",
                    abort_on_stop_event=False,
                )
                if message is None and shutdown_requested and not pending:
                    break
                if message is not None:
                    kind = message.get("kind")
                    if kind == SHUTDOWN_KIND:
                        shutdown_requested = True
                    elif kind == "eval":
                        states = message.get("states")
                        if not isinstance(states, torch.Tensor):
                            raise TypeError("evaluator received non-tensor states payload")
                        states = states.to(device="cpu", dtype=torch.float32).contiguous()
                        message["states"] = states
                        pending.append(message)
                        pending_states += int(states.shape[0])
                        if first_pending_at == 0.0:
                            first_pending_at = time.monotonic()
                    else:
                        raise RuntimeError(f"evaluator received unexpected message kind {kind!r}")

            should_flush = False
            if pending:
                if pending_states >= config.max_states_per_batch:
                    should_flush = True
                else:
                    elapsed_ms = (time.monotonic() - first_pending_at) * 1000.0
                    if elapsed_ms >= config.max_wait_ms or shutdown_requested or stop_event.is_set():
                        should_flush = True

            if should_flush:
                batch = torch.cat([item["states"] for item in pending], dim=0).contiguous()
                outputs = _run_model_batch(model, batch, config)

                cursor = 0
                for item in pending:
                    count = int(item["states"].shape[0])
                    worker_id = int(item["worker_id"])
                    request_id = int(item["request_id"])
                    reply = {
                        "kind": "eval_result",
                        "worker_id": worker_id,
                        "request_id": request_id,
                        "timestamp": time.time(),
                        "policy_logits": outputs["policy_logits"][cursor:cursor + count].contiguous(),
                        "value_scalar": outputs["value_scalar"][cursor:cursor + count].contiguous(),
                    }
                    if "wdl_logits" in outputs:
                        reply["wdl_logits"] = outputs["wdl_logits"][cursor:cursor + count].contiguous()

                    safe_put(
                        reply_queues[worker_id],
                        reply,
                        timeout_s=config.queue_put_timeout_s,
                        retry_deadline_s=config.queue_retry_deadline_s,
                        stop_event=stop_event,
                        proc_name=proc_name,
                        status_queue=status_queue,
                        worker_id=-1,
                        heartbeat_interval_s=config.heartbeat_interval_s,
                        queue_name=f"worker_reply_queue[{worker_id}]",
                    )
                    cursor += count

                pending.clear()
                pending_states = 0
                first_pending_at = 0.0
                if shutdown_requested:
                    break

            if shutdown_requested and not pending:
                break

    except StopRequested:
        pass
    except Exception as exc:
        stop_event.set()
        error_message = _format_exception(exc)
        for worker_id, reply_queue in enumerate(reply_queues):
            try:
                safe_put(
                    reply_queue,
                    {
                        "kind": STATUS_FATAL,
                        "worker_id": worker_id,
                        "request_id": -1,
                        "timestamp": time.time(),
                        "error": error_message,
                    },
                    timeout_s=0.2,
                    retry_deadline_s=1.0,
                    stop_event=stop_event,
                    proc_name=proc_name,
                    status_queue=None,
                    queue_name=f"worker_reply_queue[{worker_id}]",
                    on_timeout="drop",
                    abort_on_stop_event=False,
                )
            except Exception:
                pass

        _emit_status(
            status_queue,
            stop_event,
            proc_name,
            -1,
            STATUS_FATAL,
            error=error_message,
            critical=True,
        )
    finally:
        _emit_status(status_queue, stop_event, proc_name, -1, STATUS_EXITING)
        _close_queue_endpoint(eval_request_queue)
        for reply_queue in reply_queues:
            _close_queue_endpoint(reply_queue)
        _close_queue_endpoint(status_queue)


def _writer_main(
    output_dir: Path,
    sample_queue: Any,
    status_queue: Any,
    stop_event: Any,
    config: SelfPlayConfig,
) -> None:
    proc_name = "writer"
    _emit_status(status_queue, stop_event, proc_name, -1, STATUS_STARTED)

    train_dir = output_dir / "train"
    buffer = _TensorizedShardBuffer()
    shard_index = 0
    samples_written_total = 0
    shards_written_total = 0
    last_flush_at = time.monotonic()
    workers_done: set[int] = set()

    try:
        while True:
            _maybe_emit_heartbeat(status_queue, stop_event, proc_name, -1, config.heartbeat_interval_s)
            message = safe_get(
                sample_queue,
                timeout_s=config.queue_get_timeout_s,
                retry_deadline_s=config.queue_get_timeout_s,
                stop_event=stop_event,
                proc_name=proc_name,
                status_queue=status_queue,
                worker_id=-1,
                heartbeat_interval_s=config.heartbeat_interval_s,
                queue_name="sample_queue",
                on_timeout="return_none",
                abort_on_stop_event=False,
            )

            if message is None:
                if buffer.sample_count > 0 and (time.monotonic() - last_flush_at) >= config.writer_flush_timeout_s:
                    flushed = buffer.flush(train_dir / f"shard_{shard_index:05d}.pt")
                    if flushed > 0:
                        shard_index += 1
                        shards_written_total += 1
                        samples_written_total += flushed
                        last_flush_at = time.monotonic()
                        _emit_status(
                            status_queue,
                            stop_event,
                            proc_name,
                            -1,
                            STATUS_SHARD_WRITTEN,
                            shard_index=shard_index - 1,
                            shard_samples=flushed,
                            samples_written_total=samples_written_total,
                            shards_written_total=shards_written_total,
                        )
                continue

            kind = message.get("kind")
            if kind == SHUTDOWN_KIND:
                break
            if kind == STATUS_WORKER_DONE:
                workers_done.add(int(message.get("worker_id", -1)))
                if len(workers_done) >= config.num_workers:
                    break
                continue
            if kind != "sample_chunk":
                raise RuntimeError(f"writer received unexpected message kind {kind!r}")

            payload = message.get("payload")
            payload_bytes = message.get("payload_bytes")
            if payload is None:
                if not isinstance(payload_bytes, (bytes, bytearray, memoryview)):
                    raise TypeError("writer expected payload bytes in sample_chunk")
                payload = _deserialize_payload(payload_bytes)
            if not isinstance(payload, dict):
                raise TypeError("writer expected a payload dict in sample_chunk")

            cursor = 0
            total_samples = int(payload["state"].shape[0])
            while cursor < total_samples:
                space = config.shard_size - buffer.sample_count
                take = min(space, total_samples - cursor)
                part = _slice_payload(payload, cursor, cursor + take)
                buffer.add_payload(part)
                cursor += take

                if buffer.sample_count >= config.shard_size:
                    flushed = buffer.flush(train_dir / f"shard_{shard_index:05d}.pt")
                    if flushed > 0:
                        shard_index += 1
                        shards_written_total += 1
                        samples_written_total += flushed
                        last_flush_at = time.monotonic()
                        _emit_status(
                            status_queue,
                            stop_event,
                            proc_name,
                            -1,
                            STATUS_SHARD_WRITTEN,
                            shard_index=shard_index - 1,
                            shard_samples=flushed,
                            samples_written_total=samples_written_total,
                            shards_written_total=shards_written_total,
                        )

        if buffer.sample_count > 0:
            flushed = buffer.flush(train_dir / f"shard_{shard_index:05d}.pt")
            if flushed > 0:
                shard_index += 1
                shards_written_total += 1
                samples_written_total += flushed
                _emit_status(
                    status_queue,
                    stop_event,
                    proc_name,
                    -1,
                    STATUS_SHARD_WRITTEN,
                    shard_index=shard_index - 1,
                    shard_samples=flushed,
                    samples_written_total=samples_written_total,
                    shards_written_total=shards_written_total,
                )

        _emit_status(
            status_queue,
            stop_event,
            proc_name,
            -1,
            STATUS_WRITER_DONE,
            samples_written_total=samples_written_total,
            shards_written_total=shards_written_total,
        )

    except StopRequested:
        pass
    except Exception as exc:
        stop_event.set()
        _emit_status(
            status_queue,
            stop_event,
            proc_name,
            -1,
            STATUS_FATAL,
            error=_format_exception(exc),
            critical=True,
        )
    finally:
        _emit_status(status_queue, stop_event, proc_name, -1, STATUS_EXITING)
        _close_queue_endpoint(sample_queue)
        _close_queue_endpoint(status_queue)


def _run_model_batch(model: nn.Module, batch_cpu: Tensor, config: SelfPlayConfig) -> dict[str, Tensor]:
    batch_gpu = batch_cpu.to(config.device, non_blocking=config.device.startswith("cuda"))
    with torch.inference_mode():
        if config.use_bfloat16_eval and str(config.device).startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(batch_gpu)
        else:
            outputs = model(batch_gpu)

    if not isinstance(outputs, dict):
        raise TypeError("model(batch) must return a dict")

    result: dict[str, Tensor] = {}
    for key in ("policy_logits", "value_scalar", "wdl_logits"):
        tensor = outputs.get(key)
        if tensor is None:
            continue
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"model(batch)['{key}'] must be a torch.Tensor")
        cpu_tensor = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if not torch.isfinite(cpu_tensor).all():
            raise RuntimeError(f"model(batch)['{key}'] contains non-finite values")
        result[key] = cpu_tensor

    if "policy_logits" not in result or "value_scalar" not in result:
        raise KeyError("model(batch) must return 'policy_logits' and 'value_scalar'")
    return result


def _validate_evaluator_result(result: dict[str, Tensor], expected_batch: int, source: str) -> dict[str, Tensor]:
    policy = result.get("policy_logits")
    value = result.get("value_scalar")
    if policy is None or value is None:
        raise KeyError(f"{source}: missing required evaluator result keys")
    if tuple(policy.shape) != (expected_batch, 8100):
        raise RuntimeError(f"{source}: policy_logits must have shape [{expected_batch},8100], got {tuple(policy.shape)}")
    if tuple(value.shape) != (expected_batch, 1):
        raise RuntimeError(f"{source}: value_scalar must have shape [{expected_batch},1], got {tuple(value.shape)}")
    if policy.dtype != torch.float32 or value.dtype != torch.float32:
        raise RuntimeError(f"{source}: evaluator outputs must be float32 on CPU")
    if policy.device.type != "cpu" or value.device.type != "cpu":
        raise RuntimeError(f"{source}: evaluator outputs must be CPU tensors")
    if "wdl_logits" in result:
        wdl = result["wdl_logits"]
        if tuple(wdl.shape) != (expected_batch, 3):
            raise RuntimeError(f"{source}: wdl_logits must have shape [{expected_batch},3], got {tuple(wdl.shape)}")
        if wdl.dtype != torch.float32 or wdl.device.type != "cpu":
            raise RuntimeError(f"{source}: wdl_logits must be a float32 CPU tensor")
    return result


def _finalize_game_records(
    game_records: list[dict[str, Any]],
    final_red_result: int,
    termination_code: int,
    is_draw: bool,
) -> dict[str, Tensor]:
    states = torch.stack([record["state"] for record in game_records], dim=0).to(torch.bfloat16).contiguous()

    policy_offsets = [0]
    policy_idxs_chunks: list[Tensor] = []
    policy_probs_chunks: list[Tensor] = []
    z_values = []
    wdl_targets = []
    root_values = []
    root_wdl_values = []
    chosen_moves = []
    num_legal_moves = []
    plies = []
    game_ids = []
    stm_is_black_list = []
    is_draw_list = []
    termination_codes = []

    for record in game_records:
        idxs = record["policy_idxs"].to(torch.int64).contiguous()
        probs = record["policy_probs"].to(torch.float32).contiguous()
        policy_offsets.append(policy_offsets[-1] + int(idxs.numel()))
        policy_idxs_chunks.append(idxs)
        policy_probs_chunks.append(probs)

        stm_is_black = bool(record["stm_is_black"])
        z_value = float(final_red_result if not stm_is_black else -final_red_result)
        z_values.append(z_value)
        if z_value > 0:
            wdl_targets.append([1.0, 0.0, 0.0])
        elif z_value < 0:
            wdl_targets.append([0.0, 0.0, 1.0])
        else:
            wdl_targets.append([0.0, 1.0, 0.0])

        root_values.append(float(record["root_value"]))
        root_wdl_values.append(float(record["root_wdl_value"]))
        chosen_moves.append(int(record["chosen_move"]))
        num_legal_moves.append(int(record["num_legal_moves"]))
        plies.append(int(record["ply"]))
        game_ids.append(int(record["game_id"]))
        stm_is_black_list.append(stm_is_black)
        is_draw_list.append(bool(is_draw))
        termination_codes.append(int(termination_code))

    return {
        "state": states,
        "policy_offsets": torch.tensor(policy_offsets, dtype=torch.int64),
        "policy_idxs": torch.cat(policy_idxs_chunks, dim=0).to(torch.int64).contiguous(),
        "policy_probs": torch.cat(policy_probs_chunks, dim=0).to(torch.float32).contiguous(),
        "z": torch.tensor(z_values, dtype=torch.float32),
        "wdl_target": torch.tensor(wdl_targets, dtype=torch.float32),
        "root_value": torch.tensor(root_values, dtype=torch.float32),
        "root_wdl_value": torch.tensor(root_wdl_values, dtype=torch.float32),
        "chosen_move": torch.tensor(chosen_moves, dtype=torch.int64),
        "num_legal_moves": torch.tensor(num_legal_moves, dtype=torch.int32),
        "ply": torch.tensor(plies, dtype=torch.int16),
        "game_id": torch.tensor(game_ids, dtype=torch.int64),
        "stm_is_black": torch.tensor(stm_is_black_list, dtype=torch.bool),
        "is_draw": torch.tensor(is_draw_list, dtype=torch.bool),
        "termination_code": torch.tensor(termination_codes, dtype=torch.int8),
    }


def _slice_payload(payload: dict[str, Tensor], start: int, end: int) -> dict[str, Tensor]:
    local_offsets = payload["policy_offsets"][start:end + 1].to(torch.int64).contiguous()
    base_offset = int(local_offsets[0].item())
    policy_slice = slice(base_offset, int(local_offsets[-1].item()))

    result = {
        "state": payload["state"][start:end].contiguous(),
        "policy_offsets": (local_offsets - base_offset).contiguous(),
        "policy_idxs": payload["policy_idxs"][policy_slice].contiguous(),
        "policy_probs": payload["policy_probs"][policy_slice].contiguous(),
    }
    for key in (
        "z",
        "wdl_target",
        "root_value",
        "root_wdl_value",
        "chosen_move",
        "num_legal_moves",
        "ply",
        "game_id",
        "stm_is_black",
        "is_draw",
        "termination_code",
    ):
        result[key] = payload[key][start:end].contiguous()
    return result


def _serialize_payload(payload: dict[str, Tensor]) -> bytes:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _deserialize_payload(payload_bytes: bytes | bytearray | memoryview) -> dict[str, Tensor]:
    buffer = io.BytesIO(bytes(payload_bytes))
    payload = torch.load(buffer, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("deserialized sample payload must be a dict")
    return payload


def _termination_label(termination_code: int) -> str:
    mapping = {
        TERMINATION_CHECKMATE_OR_STALEMATE: "mate",
        TERMINATION_MAX_PLIES_DRAW: "max",
        TERMINATION_REPETITION_DRAW: "rep",
        TERMINATION_NO_CAPTURE_DRAW: "nocap",
        TERMINATION_PERPETUAL_CHECK_LOSS: "longcheck",
    }
    return mapping.get(int(termination_code), f"code{int(termination_code)}")


def _format_progress_value(total: float, count: int, digits: int = 3) -> str:
    if count <= 0:
        return "n/a"
    return f"{(total / count):.{digits}f}"


def _build_quality_metrics(summary: dict[str, Any]) -> dict[str, float]:
    games_completed = max(int(summary.get("games_completed", 0)), 0)
    draw_games = max(int(summary.get("_draw_games", 0)), 0)
    plies_total = max(int(summary.get("_plies_total", 0)), 0)
    move_decision_count = max(int(summary.get("_move_decision_count", 0)), 0)
    termination_counts = summary.get("_termination_counts", {})
    decisive_rate = (100.0 * (games_completed - draw_games) / games_completed) if games_completed > 0 else 0.0
    draw_rate = (100.0 * draw_games / games_completed) if games_completed > 0 else 0.0
    rep_draw_rate = (
        100.0 * int(termination_counts.get(TERMINATION_REPETITION_DRAW, 0)) / games_completed
        if games_completed > 0
        else 0.0
    )
    long_check_loss_rate = (
        100.0 * int(termination_counts.get(TERMINATION_PERPETUAL_CHECK_LOSS, 0)) / games_completed
        if games_completed > 0
        else 0.0
    )
    nocap_draw_rate = (
        100.0 * int(termination_counts.get(TERMINATION_NO_CAPTURE_DRAW, 0)) / games_completed
        if games_completed > 0
        else 0.0
    )
    avg_plies = (plies_total / games_completed) if games_completed > 0 else 0.0
    avg_root_v = (
        float(summary.get("_root_value_sum", 0.0)) / int(summary.get("_root_value_count", 0))
        if int(summary.get("_root_value_count", 0)) > 0
        else 0.0
    )
    avg_abs_root_v = (
        float(summary.get("_root_abs_value_sum", 0.0)) / int(summary.get("_root_value_count", 0))
        if int(summary.get("_root_value_count", 0)) > 0
        else 0.0
    )
    avg_root_wdl = (
        float(summary.get("_root_wdl_value_sum", 0.0)) / int(summary.get("_root_wdl_value_count", 0))
        if int(summary.get("_root_wdl_value_count", 0)) > 0
        else 0.0
    )
    avg_abs_root_wdl = (
        float(summary.get("_root_abs_wdl_value_sum", 0.0)) / int(summary.get("_root_wdl_value_count", 0))
        if int(summary.get("_root_wdl_value_count", 0)) > 0
        else 0.0
    )
    avg_v_gap = (
        float(summary.get("_root_value_gap_sum", 0.0)) / int(summary.get("_root_value_gap_count", 0))
        if int(summary.get("_root_value_gap_count", 0)) > 0
        else 0.0
    )
    anti_draw_override_rate = (
        100.0 * int(summary.get("_anti_draw_override_count", 0)) / move_decision_count
        if move_decision_count > 0
        else 0.0
    )
    anti_draw_forced_draw_rate = (
        100.0 * int(summary.get("_anti_draw_forced_draw_count", 0)) / move_decision_count
        if move_decision_count > 0
        else 0.0
    )
    anti_draw_avoided_repeat_rate = (
        100.0 * int(summary.get("_anti_draw_avoided_repeat_count", 0)) / move_decision_count
        if move_decision_count > 0
        else 0.0
    )
    anti_draw_avoided_nocap_rate = (
        100.0 * int(summary.get("_anti_draw_avoided_nocap_count", 0)) / move_decision_count
        if move_decision_count > 0
        else 0.0
    )
    anti_draw_capture_pressure_rate = (
        100.0 * int(summary.get("_anti_draw_capture_pressure_count", 0)) / move_decision_count
        if move_decision_count > 0
        else 0.0
    )
    anti_draw_capture_priority_rate = (
        100.0 * int(summary.get("_anti_draw_capture_priority_count", 0)) / move_decision_count
        if move_decision_count > 0
        else 0.0
    )
    anti_draw_capture_available_rate = (
        100.0 * int(summary.get("_anti_draw_capture_available_count", 0)) / move_decision_count
        if move_decision_count > 0
        else 0.0
    )
    anti_draw_capture_injection_rate = (
        100.0 * int(summary.get("_anti_draw_capture_injection_count", 0)) / move_decision_count
        if move_decision_count > 0
        else 0.0
    )
    anti_draw_capture_selected_rate = (
        100.0 * int(summary.get("_anti_draw_capture_selected_count", 0)) / move_decision_count
        if move_decision_count > 0
        else 0.0
    )
    anti_draw_capture_any_selected_rate = (
        100.0 * int(summary.get("_anti_draw_capture_any_selected_count", 0)) / move_decision_count
        if move_decision_count > 0
        else 0.0
    )
    stm_is_black_rate = (
        100.0 * int(summary.get("_stm_black_samples", 0)) / int(summary.get("_stm_total_samples", 0))
        if int(summary.get("_stm_total_samples", 0)) > 0
        else 0.0
    )
    return {
        "decisive_rate": decisive_rate,
        "draw_rate": draw_rate,
        "rep_draw_rate": rep_draw_rate,
        "long_check_loss_rate": long_check_loss_rate,
        "nocap_draw_rate": nocap_draw_rate,
        "avg_plies": avg_plies,
        "avg_root_v": avg_root_v,
        "avg_abs_root_v": avg_abs_root_v,
        "avg_root_wdl": avg_root_wdl,
        "avg_abs_root_wdl": avg_abs_root_wdl,
        "avg_v_gap": avg_v_gap,
        "anti_draw_override_rate": anti_draw_override_rate,
        "anti_draw_forced_draw_rate": anti_draw_forced_draw_rate,
        "anti_draw_avoided_repeat_rate": anti_draw_avoided_repeat_rate,
        "anti_draw_avoided_nocap_rate": anti_draw_avoided_nocap_rate,
        "anti_draw_capture_pressure_rate": anti_draw_capture_pressure_rate,
        "anti_draw_capture_priority_rate": anti_draw_capture_priority_rate,
        "anti_draw_capture_available_rate": anti_draw_capture_available_rate,
        "anti_draw_capture_injection_rate": anti_draw_capture_injection_rate,
        "anti_draw_capture_selected_rate": anti_draw_capture_selected_rate,
        "anti_draw_capture_any_selected_rate": anti_draw_capture_any_selected_rate,
        "stm_is_black_rate": stm_is_black_rate,
    }


def _classify_bootstrap_quality(summary: dict[str, Any], config: SelfPlayConfig) -> tuple[str, dict[str, float]]:
    metrics = _build_quality_metrics(summary)
    if (
        metrics["rep_draw_rate"] >= float(config.bootstrap_quality_rep_draw_threshold)
        or metrics["decisive_rate"] < float(config.bootstrap_quality_decisive_threshold)
        or metrics["nocap_draw_rate"] >= float(config.bootstrap_quality_nocap_draw_threshold)
    ):
        return "stuck", metrics
    if (
        metrics["rep_draw_rate"] >= float(config.bootstrap_quality_warn_rep_draw_threshold)
        or metrics["decisive_rate"] < float(config.bootstrap_quality_warn_decisive_threshold)
        or metrics["nocap_draw_rate"] >= float(config.bootstrap_quality_warn_nocap_draw_threshold)
    ):
        return "warn", metrics
    return "ok", metrics


def _maybe_log_progress(summary: dict[str, Any], config: SelfPlayConfig) -> None:
    next_games = int(summary.get("_next_progress_log_games", config.progress_log_games))
    while summary["games_completed"] >= next_games:
        games_completed = int(summary["games_completed"])
        quality, metrics = _classify_bootstrap_quality(summary, config)
        start_metrics = _build_start_position_metrics(summary)
        termination_counts = summary.get("_termination_counts", {})
        term_text = ", ".join(
            f"{_termination_label(code)}:{int(termination_counts.get(code, 0))}"
            for code in (
                TERMINATION_CHECKMATE_OR_STALEMATE,
                TERMINATION_MAX_PLIES_DRAW,
                TERMINATION_REPETITION_DRAW,
                TERMINATION_PERPETUAL_CHECK_LOSS,
                TERMINATION_NO_CAPTURE_DRAW,
            )
        )
        print(
            "selfplay "
            f"games={games_completed} "
            f"samples={int(summary['samples_written'])} "
            f"generated={int(summary.get('_generated_samples', 0))} "
            f"shards={int(summary['shards_written'])} "
            f"quality={quality} "
            f"decisive={metrics['decisive_rate']:.1f}% "
            f"draw_rate={metrics['draw_rate']:.1f}% "
            f"rep_draw={metrics['rep_draw_rate']:.1f}% "
            f"long_check_loss={metrics['long_check_loss_rate']:.1f}% "
            f"nocap_draw={metrics['nocap_draw_rate']:.1f}% "
            f"avg_plies={metrics['avg_plies']:.1f} "
            f"avg_abs_root_v={metrics['avg_abs_root_v']:.3f} "
            f"avg_v_gap={metrics['avg_v_gap']:.3f} "
            f"anti_draw={metrics['anti_draw_override_rate']:.1f}% "
            f"forced_draw={metrics['anti_draw_forced_draw_rate']:.1f}% "
            f"anti_draw_avoided_repeat={metrics['anti_draw_avoided_repeat_rate']:.1f}% "
            f"anti_draw_avoided_nocap={metrics['anti_draw_avoided_nocap_rate']:.1f}% "
            f"anti_draw_capture_pressure={metrics['anti_draw_capture_pressure_rate']:.1f}% "
            f"anti_draw_capture_priority={metrics['anti_draw_capture_priority_rate']:.1f}% "
            f"anti_draw_capture_available={metrics['anti_draw_capture_available_rate']:.1f}% "
            f"anti_draw_capture_inject={metrics['anti_draw_capture_injection_rate']:.1f}% "
            f"anti_draw_capture_select={metrics['anti_draw_capture_selected_rate']:.1f}% "
            f"anti_draw_capture_any_select={metrics['anti_draw_capture_any_selected_rate']:.1f}% "
            f"stm_black={metrics['stm_is_black_rate']:.1f}% "
            f"human_start={start_metrics['human_start_rate']:.1f}% "
            f"fallback={start_metrics['fallback_to_startpos_rate']:.1f}% "
            f"ctx_restore={start_metrics['context_restore_rate']:.1f}% "
            f"opening_accept={start_metrics['opening_like_accept_rate']:.1f}% "
            f"wdl_diag={metrics['avg_root_wdl']:.3f}/{metrics['avg_abs_root_wdl']:.3f} "
            f"term={{{term_text}}}",
            flush=True,
        )
        summary["_next_progress_log_games"] = next_games + config.progress_log_games
        next_games = int(summary["_next_progress_log_games"])


def _maybe_log_shard_progress(summary: dict[str, Any], config: SelfPlayConfig) -> None:
    next_shards = int(summary.get("_next_progress_log_shards", config.progress_log_shards))
    while int(summary.get("shards_written", 0)) >= next_shards:
        games_completed = int(summary.get("games_completed", 0))
        start_metrics = _build_start_position_metrics(summary)
        quality, metrics = _classify_bootstrap_quality(summary, config)
        print(
            "selfplay-shards "
            f"shards={int(summary.get('shards_written', 0))} "
            f"written={int(summary.get('samples_written', 0))} "
            f"generated={int(summary.get('_generated_samples', 0))} "
            f"games={games_completed} "
            f"quality={quality} "
            f"decisive={metrics['decisive_rate']:.1f}% "
            f"rep_draw={metrics['rep_draw_rate']:.1f}% "
            f"long_check_loss={metrics['long_check_loss_rate']:.1f}% "
            f"nocap_draw={metrics['nocap_draw_rate']:.1f}% "
            f"anti_draw={metrics['anti_draw_override_rate']:.1f}% "
            f"anti_draw_avoided_repeat={metrics['anti_draw_avoided_repeat_rate']:.1f}% "
            f"anti_draw_avoided_nocap={metrics['anti_draw_avoided_nocap_rate']:.1f}% "
            f"anti_draw_capture_pressure={metrics['anti_draw_capture_pressure_rate']:.1f}% "
            f"anti_draw_capture_priority={metrics['anti_draw_capture_priority_rate']:.1f}% "
            f"anti_draw_capture_available={metrics['anti_draw_capture_available_rate']:.1f}% "
            f"anti_draw_capture_inject={metrics['anti_draw_capture_injection_rate']:.1f}% "
            f"anti_draw_capture_select={metrics['anti_draw_capture_selected_rate']:.1f}% "
            f"anti_draw_capture_any_select={metrics['anti_draw_capture_any_selected_rate']:.1f}% "
            f"stm_black={metrics['stm_is_black_rate']:.1f}% "
            f"human_start={start_metrics['human_start_rate']:.1f}% "
            f"ctx_restore={start_metrics['context_restore_rate']:.1f}% "
            f"fallback={start_metrics['fallback_to_startpos_rate']:.1f}%",
            flush=True,
        )
        summary["_next_progress_log_shards"] = next_shards + config.progress_log_shards
        next_shards = int(summary["_next_progress_log_shards"])


def _apply_status_message(summary: dict[str, Any], message: dict[str, Any]) -> bool:
    kind = message.get("kind")
    writer_done = False

    if kind == STATUS_GAME_COMPLETE:
        summary["games_completed"] += int(message.get("games_completed", 1))
        produced_samples = int(message.get("samples_generated", 0))
        if produced_samples > 0:
            summary["_generated_samples"] += produced_samples
        if bool(message.get("is_draw", False)):
            summary["_draw_games"] += 1
        summary["_plies_total"] += int(message.get("plies_in_game", 0))
        summary["_root_value_sum"] += float(message.get("root_value_sum", 0.0))
        summary["_root_value_count"] += int(message.get("root_value_count", 0))
        summary["_root_abs_value_sum"] += float(message.get("root_abs_value_sum", 0.0))
        summary["_root_wdl_value_sum"] += float(message.get("root_wdl_value_sum", 0.0))
        summary["_root_wdl_value_count"] += int(message.get("root_wdl_value_count", 0))
        summary["_root_abs_wdl_value_sum"] += float(message.get("root_abs_wdl_value_sum", 0.0))
        summary["_root_value_gap_sum"] += float(message.get("root_value_gap_sum", 0.0))
        summary["_root_value_gap_count"] += int(message.get("root_value_gap_count", 0))
        summary["_stm_black_samples"] += int(message.get("stm_black_samples", 0))
        summary["_stm_total_samples"] += int(message.get("stm_total_samples", 0))
        summary["_human_position_attempts"] += int(message.get("human_position_attempts", 0))
        summary["_opening_like_accepts"] += int(message.get("opening_like_accepts", 0))
        summary["_human_start_games"] += int(message.get("human_start_games", 0))
        summary["_standard_start_games"] += int(message.get("standard_start_games", 0))
        summary["_fallback_to_startpos_games"] += int(message.get("fallback_to_startpos_games", 0))
        summary["_context_restored_games"] += int(message.get("context_restored_games", 0))
        summary["_context_restore_failed"] += int(message.get("context_restore_failed", 0))
        summary["_move_decision_count"] += int(message.get("move_decision_count", 0))
        summary["_anti_draw_override_count"] += int(message.get("anti_draw_override_count", 0))
        summary["_anti_draw_forced_draw_count"] += int(message.get("anti_draw_forced_draw_count", 0))
        summary["_anti_draw_avoided_repeat_count"] += int(message.get("anti_draw_avoided_repeat_count", 0))
        summary["_anti_draw_avoided_nocap_count"] += int(message.get("anti_draw_avoided_nocap_count", 0))
        summary["_anti_draw_capture_pressure_count"] += int(message.get("anti_draw_capture_pressure_count", 0))
        summary["_anti_draw_capture_priority_count"] += int(message.get("anti_draw_capture_priority_count", 0))
        summary["_anti_draw_capture_available_count"] += int(message.get("anti_draw_capture_available_count", 0))
        summary["_anti_draw_capture_injection_count"] += int(message.get("anti_draw_capture_injection_count", 0))
        summary["_anti_draw_capture_selected_count"] += int(message.get("anti_draw_capture_selected_count", 0))
        summary["_anti_draw_capture_any_selected_count"] += int(message.get("anti_draw_capture_any_selected_count", 0))
        termination_code = int(message.get("termination_code", -1))
        termination_counts = summary["_termination_counts"]
        termination_counts[termination_code] = int(termination_counts.get(termination_code, 0)) + 1

    if kind == STATUS_SHARD_WRITTEN:
        summary["samples_written"] = int(message.get("samples_written_total", summary["samples_written"]))
        summary["shards_written"] = int(message.get("shards_written_total", summary["shards_written"]))

    if kind == STATUS_WRITER_DONE:
        writer_done = True
        summary["samples_written"] = int(message.get("samples_written_total", summary["samples_written"]))
        summary["shards_written"] = int(message.get("shards_written_total", summary["shards_written"]))

    return writer_done


def _validate_run_inputs(checkpoint_path: Path, output_dir: Path) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint file not found: {checkpoint_path}")

    train_dir = output_dir / "train"
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {manifest_path}")
    if train_dir.exists() and any(train_dir.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty train directory: {train_dir}")


def _load_model_from_checkpoint(checkpoint_path: Path) -> tuple[nn.Module, XiangqiTransformerConfig]:
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict: dict[str, Tensor] | None = None
    raw_config: Any = None

    if isinstance(raw, dict):
        if "model_state_dict" in raw:
            state_dict = raw["model_state_dict"]
            raw_config = raw.get("model_config") or raw.get("config")
        elif "state_dict" in raw:
            state_dict = raw["state_dict"]
            raw_config = raw.get("model_config") or raw.get("config")
        elif raw and all(isinstance(v, torch.Tensor) for v in raw.values()):
            state_dict = raw
            raw_config = None
    if state_dict is None:
        raise RuntimeError(
            "checkpoint must be a state_dict or contain 'model_state_dict' / 'state_dict'"
        )

    config = _coerce_model_config(raw_config)
    model = XiangqiPVTransformer(config)
    load_xiangqi_model_state_dict(model, state_dict)
    return model, config


def _coerce_model_config(raw_config: Any) -> XiangqiTransformerConfig:
    if raw_config is None:
        return XiangqiTransformerConfig()
    if isinstance(raw_config, XiangqiTransformerConfig):
        return raw_config
    if isinstance(raw_config, dict):
        allowed = set(XiangqiTransformerConfig.__dataclass_fields__.keys())
        filtered = {k: v for k, v in raw_config.items() if k in allowed}
        return XiangqiTransformerConfig(**filtered)
    raise TypeError(f"unsupported model_config type: {type(raw_config).__name__}")


class ChimeraPolicyValueModel(nn.Module):
    """Policy head from one full network, value (and wdl) head from another.

    Used by the frozen-evaluator self-play mode: the play checkpoint supplies
    policy_logits while a frozen reference (e.g. geo) supplies value_scalar /
    wdl_logits, so search always runs on a healthy, calibrated evaluator while
    only the policy learns. Both submodels run a full forward; only the head
    outputs are recombined (a weight-level splice would break the value head,
    which is calibrated to its own trunk's features).
    """

    def __init__(self, policy_model: nn.Module, value_model: nn.Module) -> None:
        super().__init__()
        self.policy_model = policy_model
        self.value_model = value_model

    def forward(self, x):
        out_p = self.policy_model(x)
        out_v = self.value_model(x)
        out = dict(out_p)
        out["value_scalar"] = out_v["value_scalar"]
        if "wdl_logits" in out_v:
            out["wdl_logits"] = out_v["wdl_logits"]
        return out


def _emit_status(
    status_queue: Any,
    stop_event: Any,
    proc_name: str,
    worker_id: int,
    kind: str,
    critical: bool = False,
    **fields: Any,
) -> None:
    if status_queue is None:
        return

    message = {
        "kind": kind,
        "proc_name": proc_name,
        "worker_id": worker_id,
        "timestamp": time.time(),
    }
    message.update(fields)

    deadline = time.monotonic() + (2.0 if critical else _STATUS_PUT_TIMEOUT_S)
    while True:
        try:
            status_queue.put(message, timeout=_STATUS_PUT_TIMEOUT_S)
            return
        except queue_module.Full:
            if time.monotonic() >= deadline or stop_event.is_set():
                return
        except (EOFError, BrokenPipeError, ConnectionResetError, OSError):
            return


def _maybe_emit_heartbeat(
    status_queue: Any,
    stop_event: Any,
    proc_name: str,
    worker_id: int,
    interval_s: float,
    extra: dict[str, Any] | None = None,
) -> None:
    key = (proc_name, worker_id)
    now = time.monotonic()
    last = _HEARTBEAT_LAST_SENT.get(key, 0.0)
    if (now - last) < interval_s:
        return
    _HEARTBEAT_LAST_SENT[key] = now
    payload = extra.copy() if extra else {}
    _emit_status(status_queue, stop_event, proc_name, worker_id, STATUS_HEARTBEAT, **payload)


def _join_or_kill_processes(processes: list[Any], config: SelfPlayConfig) -> None:
    for proc in processes:
        proc.join(timeout=config.shutdown_join_timeout_s)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=config.shutdown_kill_timeout_s)
        if proc.is_alive():
            try:
                proc.kill()
            except Exception:
                proc.terminate()
            proc.join(timeout=config.shutdown_kill_timeout_s)


def _close_queue_endpoint(queue: Any) -> None:
    try:
        queue.close()
    except Exception:
        return

    try:
        queue.join_thread()
    except Exception:
        pass


def _send_shutdown_to_queue(
    queue: Any,
    stop_event: Any,
    proc_name: str,
    status_queue: Any,
    queue_name: str,
    repeats: int,
) -> None:
    message = {"kind": SHUTDOWN_KIND, "worker_id": -1, "timestamp": time.time()}
    for _ in range(repeats):
        deadline = time.monotonic() + 1.0
        while True:
            try:
                queue.put(message, timeout=0.2)
                break
            except queue_module.Full:
                if time.monotonic() >= deadline:
                    break
                _maybe_emit_heartbeat(
                    status_queue,
                    stop_event,
                    proc_name,
                    -1,
                    0.5,
                    extra={"queue_name": queue_name, "wait": "put_shutdown"},
                )
            except (EOFError, BrokenPipeError, ConnectionResetError, OSError):
                return


def _drain_and_close_queue(queue: Any) -> None:
    try:
        while True:
            try:
                queue.get(timeout=0.01)
            except queue_module.Empty:
                break
            except (EOFError, BrokenPipeError, ConnectionResetError, OSError):
                break
        queue.close()
        queue.join_thread()
    except Exception:
        pass


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(path)


def _build_manifest_payload(
    checkpoint_path: Path,
    config: SelfPlayConfig,
    summary: dict[str, Any],
    quality_info: dict[str, Any],
    manifest_state: str,
) -> dict[str, Any]:
    start_position_metrics = summary.get("start_position_metrics")
    if not isinstance(start_position_metrics, dict):
        start_position_metrics = _build_start_position_metrics(summary)
    return {
        "format": "xiangqi_selfplay_v1_bf16_sparse",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest_state": manifest_state,
        "checkpoint_path": str(checkpoint_path),
        "state_dtype": "bfloat16",
        "policy_layout": "csr_sparse",
        "canonical_input": True,
        "canonical_policy": True,
        "split": "train_only",
        "termination_codes": {
            "CHECKMATE_OR_STALEMATE": TERMINATION_CHECKMATE_OR_STALEMATE,
            "MAX_PLIES_DRAW": TERMINATION_MAX_PLIES_DRAW,
            "REPETITION_DRAW": TERMINATION_REPETITION_DRAW,
            "NO_CAPTURE_DRAW": TERMINATION_NO_CAPTURE_DRAW,
            "PERPETUAL_CHECK_LOSS": TERMINATION_PERPETUAL_CHECK_LOSS,
        },
        "search_defaults": {
            "num_simulations": config.num_simulations,
            "c_puct": config.c_puct,
            "q_weight": config.q_weight,
            "q_clip": config.q_clip,
            "add_root_noise": config.add_root_noise,
            "dirichlet_alpha": config.dirichlet_alpha,
            "dirichlet_eps": config.dirichlet_eps,
            "eval_batch_size": config.eval_batch_size,
            "temperature_target": config.temperature_target,
            "move_temperature_schedule": [list(item) for item in config.move_temperature_schedule],
            "root_noise_end_ply": config.root_noise_end_ply,
            "max_plies": config.max_plies,
            "repeat_limit": config.repeat_limit,
            "repeat_min_ply": config.repeat_min_ply,
            "no_capture_limit": config.no_capture_limit,
            "immediate_draw_candidate_scan": config.immediate_draw_candidate_scan,
            "immediate_draw_capture_injection_threshold": config.immediate_draw_capture_injection_threshold,
            "immediate_draw_capture_priority_threshold": config.immediate_draw_capture_priority_threshold,
            "immediate_draw_nocap_pressure_threshold": config.immediate_draw_nocap_pressure_threshold,
            "immediate_draw_nocap_pressure_scan": config.immediate_draw_nocap_pressure_scan,
            "start_position_mode": config.start_position_mode,
            "human_position_source_dir": str(config.human_position_source_dir),
            "human_position_mix_ratio": config.human_position_mix_ratio,
            "bootstrap_quality_nocap_draw_threshold": config.bootstrap_quality_nocap_draw_threshold,
            "bootstrap_quality_warn_nocap_draw_threshold": config.bootstrap_quality_warn_nocap_draw_threshold,
        },
        "runtime": {
            "num_workers": config.num_workers,
            "max_states_per_batch": config.max_states_per_batch,
            "max_wait_ms": config.max_wait_ms,
            "device": config.device,
            "use_bfloat16_eval": config.use_bfloat16_eval,
            "progress_log_games": config.progress_log_games,
            "progress_log_shards": config.progress_log_shards,
        },
        "counts": {
            "games_completed": int(summary.get("games_completed", 0)),
            "samples_written": int(summary.get("samples_written", 0)),
            "shards_written": int(summary.get("shards_written", 0)),
        },
        "start_position_mode": config.start_position_mode,
        "human_position_source_dir": str(config.human_position_source_dir),
        "human_position_mix_ratio": config.human_position_mix_ratio,
        "quality": quality_info["quality"],
        "quality_metrics": quality_info["metrics"],
        "start_position_metrics": start_position_metrics,
        "opening_like_filters": {
            "material_min_ratio": config.opening_like_material_min_ratio,
            "min_piece_count": config.opening_like_min_piece_count,
        },
        "scheduled_pause_requested": bool(summary.get("scheduled_pause_requested", False)),
        "terminated_cleanly": bool(summary.get("terminated_cleanly", False)),
        "fatal_error": summary.get("fatal_error"),
        "config": asdict(config),
    }


def _write_manifest(
    output_dir: Path,
    checkpoint_path: Path,
    config: SelfPlayConfig,
    summary: dict[str, Any],
    quality_info: dict[str, Any],
    manifest_state: str = "complete",
) -> None:
    manifest = _build_manifest_payload(
        checkpoint_path=checkpoint_path,
        config=config,
        summary=summary,
        quality_info=quality_info,
        manifest_state=manifest_state,
    )
    _write_json_atomic(output_dir / "manifest.json", manifest)


def _format_exception(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()


def _default_checkpoint_path() -> Path:
    repo_root = Path(__file__).resolve().parent
    candidates = [
        repo_root / "training_runs" / "run_001" / "best.pt",
        repo_root / "training_runs" / "run_001" / "latest.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def _next_selfplay_output_dir(output_root: Path) -> Path:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for run_index in range(1, 10_000):
        candidate = output_root / f"run_{run_index:03d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"unable to allocate a self-play run directory under {output_root}")


def _parse_args() -> tuple[Path, Path, SelfPlayConfig]:
    default_config = SelfPlayConfig()
    parser = argparse.ArgumentParser(description="Run Xiangqi self-play data generation.")
    parser.add_argument("--checkpoint-path", default=str(_default_checkpoint_path()))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--output-root", default="selfplay_runs_bootstrap")
    parser.add_argument("--target-games", type=int, default=default_config.target_games)
    parser.add_argument("--target-samples", type=int, default=default_config.target_samples)
    parser.add_argument("--num-workers", type=int, default=default_config.num_workers)
    parser.add_argument("--max-states-per-batch", type=int, default=default_config.max_states_per_batch)
    parser.add_argument("--max-wait-ms", type=int, default=default_config.max_wait_ms)
    parser.add_argument("--num-simulations", type=int, default=default_config.num_simulations)
    parser.add_argument("--eval-batch-size", type=int, default=default_config.eval_batch_size)
    parser.add_argument("--temperature-target", type=float, default=default_config.temperature_target)
    parser.add_argument("--root-noise-end-ply", type=int, default=default_config.root_noise_end_ply)
    parser.add_argument("--max-plies", type=int, default=default_config.max_plies)
    parser.add_argument("--repeat-limit", type=int, default=default_config.repeat_limit)
    parser.add_argument("--repeat-min-ply", type=int, default=default_config.repeat_min_ply)
    parser.add_argument("--no-capture-limit", type=int, default=default_config.no_capture_limit)
    parser.add_argument("--progress-log-games", type=int, default=default_config.progress_log_games)
    parser.add_argument("--progress-log-shards", type=int, default=default_config.progress_log_shards)
    parser.add_argument(
        "--bootstrap-quality-rep-draw-threshold",
        type=float,
        default=default_config.bootstrap_quality_rep_draw_threshold,
    )
    parser.add_argument(
        "--bootstrap-quality-decisive-threshold",
        type=float,
        default=default_config.bootstrap_quality_decisive_threshold,
    )
    parser.add_argument(
        "--bootstrap-quality-warn-rep-draw-threshold",
        type=float,
        default=default_config.bootstrap_quality_warn_rep_draw_threshold,
    )
    parser.add_argument(
        "--bootstrap-quality-warn-decisive-threshold",
        type=float,
        default=default_config.bootstrap_quality_warn_decisive_threshold,
    )
    parser.add_argument(
        "--bootstrap-quality-nocap-draw-threshold",
        type=float,
        default=default_config.bootstrap_quality_nocap_draw_threshold,
    )
    parser.add_argument(
        "--bootstrap-quality-warn-nocap-draw-threshold",
        type=float,
        default=default_config.bootstrap_quality_warn_nocap_draw_threshold,
    )
    parser.add_argument(
        "--immediate-draw-candidate-scan",
        type=int,
        default=default_config.immediate_draw_candidate_scan,
    )
    parser.add_argument(
        "--immediate-draw-nocap-pressure-threshold",
        type=int,
        default=default_config.immediate_draw_nocap_pressure_threshold,
    )
    parser.add_argument(
        "--immediate-draw-capture-injection-threshold",
        type=int,
        default=default_config.immediate_draw_capture_injection_threshold,
    )
    parser.add_argument(
        "--immediate-draw-capture-priority-threshold",
        type=int,
        default=default_config.immediate_draw_capture_priority_threshold,
    )
    parser.add_argument(
        "--immediate-draw-nocap-pressure-scan",
        type=int,
        default=default_config.immediate_draw_nocap_pressure_scan,
    )
    parser.add_argument(
        "--start-position-mode",
        choices=["human_positions", "standard_start"],
        default=default_config.start_position_mode,
    )
    parser.add_argument("--human-position-source-dir", default=default_config.human_position_source_dir)
    parser.add_argument("--human-position-mix-ratio", type=float, default=default_config.human_position_mix_ratio)
    parser.add_argument(
        "--opening-like-material-min-ratio",
        type=float,
        default=default_config.opening_like_material_min_ratio,
    )
    parser.add_argument(
        "--opening-like-min-piece-count",
        type=int,
        default=default_config.opening_like_min_piece_count,
    )
    parser.add_argument("--pause-at-local-time", default=default_config.pause_at_local_time or "")
    parser.add_argument("--device", default=default_config.device)
    parser.add_argument("--seed", type=int, default=default_config.seed)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else _next_selfplay_output_dir(Path(args.output_root))
    config = SelfPlayConfig(
        target_games=args.target_games,
        target_samples=args.target_samples,
        num_workers=args.num_workers,
        max_states_per_batch=args.max_states_per_batch,
        max_wait_ms=args.max_wait_ms,
        num_simulations=args.num_simulations,
        eval_batch_size=args.eval_batch_size,
        temperature_target=args.temperature_target,
        root_noise_end_ply=args.root_noise_end_ply,
        max_plies=args.max_plies,
        repeat_limit=args.repeat_limit,
        repeat_min_ply=args.repeat_min_ply,
        no_capture_limit=args.no_capture_limit,
        progress_log_games=args.progress_log_games,
        progress_log_shards=args.progress_log_shards,
        bootstrap_quality_rep_draw_threshold=args.bootstrap_quality_rep_draw_threshold,
        bootstrap_quality_decisive_threshold=args.bootstrap_quality_decisive_threshold,
        bootstrap_quality_nocap_draw_threshold=args.bootstrap_quality_nocap_draw_threshold,
        bootstrap_quality_warn_rep_draw_threshold=args.bootstrap_quality_warn_rep_draw_threshold,
        bootstrap_quality_warn_decisive_threshold=args.bootstrap_quality_warn_decisive_threshold,
        bootstrap_quality_warn_nocap_draw_threshold=args.bootstrap_quality_warn_nocap_draw_threshold,
        immediate_draw_candidate_scan=args.immediate_draw_candidate_scan,
        immediate_draw_capture_injection_threshold=args.immediate_draw_capture_injection_threshold,
        immediate_draw_capture_priority_threshold=args.immediate_draw_capture_priority_threshold,
        immediate_draw_nocap_pressure_threshold=args.immediate_draw_nocap_pressure_threshold,
        immediate_draw_nocap_pressure_scan=args.immediate_draw_nocap_pressure_scan,
        start_position_mode=args.start_position_mode,
        human_position_source_dir=args.human_position_source_dir,
        human_position_mix_ratio=args.human_position_mix_ratio,
        opening_like_material_min_ratio=args.opening_like_material_min_ratio,
        opening_like_min_piece_count=args.opening_like_min_piece_count,
        pause_at_local_time=args.pause_at_local_time,
        device=args.device,
        seed=args.seed,
    )
    return checkpoint_path, output_dir, config


def main() -> None:
    checkpoint_path, output_dir, config = _parse_args()
    print(f"self-play checkpoint: {checkpoint_path}", flush=True)
    print(f"self-play output_dir: {output_dir}", flush=True)
    print(
        f"self-play config: workers={config.num_workers} sims={config.num_simulations} "
        f"target_samples={config.target_samples} device={config.device} "
        f"temp_sched={list(config.move_temperature_schedule)} noise_end={config.root_noise_end_ply} "
        f"repeat={config.repeat_limit}@{config.repeat_min_ply} nocap={config.no_capture_limit} "
        f"quality_nocap=({config.bootstrap_quality_warn_nocap_draw_threshold:.1f},{config.bootstrap_quality_nocap_draw_threshold:.1f}) "
        f"draw_scan={config.immediate_draw_candidate_scan}/{config.immediate_draw_nocap_pressure_scan} "
        f"capture_thresholds=({config.immediate_draw_capture_injection_threshold},{config.immediate_draw_capture_priority_threshold}) "
        f"nocap_pressure={config.immediate_draw_nocap_pressure_threshold} "
        f"start_mode={config.start_position_mode} human_mix={config.human_position_mix_ratio:.2f} "
        f"opening_filter=({config.opening_like_material_min_ratio:.2f},{config.opening_like_min_piece_count}) "
        f"pause_at={config.pause_at_local_time or 'off'} "
        f"log_games={config.progress_log_games} log_shards={config.progress_log_shards}",
        flush=True,
    )
    summary = run_selfplay(checkpoint_path=checkpoint_path, output_dir=output_dir, config=config)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


__all__ = ["SelfPlayConfig", "run_selfplay"]


if __name__ == "__main__":
    main()
