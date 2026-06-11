from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import queue as queue_module
import shutil
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import torch

from xiangqi_arena import ArenaConfig, run_arena
from xiangqi_selfplay import SelfPlayConfig, run_selfplay
from xiangqi_train import TrainingConfig, run_training
from xiangqi_transformer_model import XiangqiTransformerConfig


@dataclass
class ClosedLoopConfig:
    human_data_dir: str | Path = "human_bootstrap_data_elite_wdl"
    training_output_dir: str | Path = "training_runs/run_001"
    selfplay_output_root: str | Path = "selfplay_runs_bootstrap_clean"
    arena_output_root: str | Path = "arena_runs"
    device: str = "cuda:0"
    pause_at_local_time: str | None = "06:59"
    selfplay_checkpoint_source: str = "best"
    execution_mode: str = "sequential"

    cycles: int = 0
    sleep_between_cycles_s: float = 2.0
    seed: int = 2026041101

    selfplay_target_samples_per_cycle: int = 12_000
    selfplay_num_workers: int = 8
    selfplay_num_simulations: int = 256
    selfplay_max_states_per_batch: int = 128
    selfplay_eval_batch_size: int = 16
    selfplay_progress_log_games: int = 20
    selfplay_progress_log_shards: int = 5
    selfplay_human_position_mix_ratio: float = 0.5
    selfplay_opening_like_material_min_ratio: float = 0.85
    selfplay_opening_like_min_piece_count: int = 28
    selfplay_max_plies: int = 256
    selfplay_repeat_limit: int = 6
    selfplay_repeat_min_ply: int = 30
    selfplay_no_capture_limit: int = 60
    selfplay_immediate_draw_candidate_scan: int = 64
    selfplay_immediate_draw_capture_injection_threshold: int = 12
    selfplay_immediate_draw_capture_priority_threshold: int = 18
    selfplay_immediate_draw_nocap_pressure_threshold: int = 25
    selfplay_immediate_draw_nocap_pressure_scan: int = 128
    selfplay_nocap_warn_threshold: float = 45.0
    selfplay_nocap_stuck_threshold: float = 55.0
    overlap_selfplay_num_workers: int | None = None
    overlap_keep_free_cpu_cores: int = 4

    train_steps_per_cycle: int = 0
    skip_train: bool = False
    train_lr_schedule_max_steps: int = 200_000
    train_bootstrap_human_floor: float = 0.20
    train_reset_selfplay_ingest_state_on_resume: bool = True
    train_micro_batch_size: int = 1024
    train_grad_accum_steps: int = 1
    train_cpu_sampler_workers: int = 12
    train_cpu_prefetch_batches: int = 8
    train_shard_cache_size: int = 16
    min_selfplay_quality_for_training: str = "ok"
    snapshot_keep_latest_count: int = 3
    snapshot_keep_best_count: int = 3

    arena_games: int = 100
    arena_sims: int = 800
    arena_accept_threshold: float = 0.55
    arena_min_non_draw_games: int = 10
    arena_log_every_games: int = 10
    arena_games_per_opening: int = 2
    arena_max_plies: int = 240
    arena_repeat_limit: int = 6
    arena_repeat_min_ply: int = 30
    arena_no_capture_limit: int = 60
    arena_opening_suite_path: str | Path | None = None
    arena_enabled: bool = True
    arena_every_n_cycles: int = 1
    # B-experiment: policy anchor to a frozen reference (geo) to curb self-play forgetting.
    rl_anchor_checkpoint: str | Path | None = None
    rl_anchor_policy_kl_weight: float = 0.0
    rl_anchor_anneal_steps: int = 0
    # v16 "frozen evaluator, evolving policy": if set, self-play AND gates run
    # policy/value chimeras with value from this frozen reference (e.g. geo),
    # the trainer gets value/wdl loss weight 0 (policy-only learning), and the
    # optimizer/scheduler are RESET on resume so the configured LR actually
    # applies (run_070..074 silently resumed geo's 2e-4 — verified bug).
    rl_frozen_value_checkpoint: str | Path | None = None
    arena_promote_on_pass: bool = True
    arena_gates_best_update: bool = True
    arena_add_root_noise: bool = True
    arena_dirichlet_alpha: float = 0.30
    arena_dirichlet_eps: float = 0.10
    arena_temperature_move: float = 1e-6

    def __post_init__(self) -> None:
        if self.cycles < 0:
            raise ValueError("cycles must be >= 0")
        if self.sleep_between_cycles_s < 0.0:
            raise ValueError("sleep_between_cycles_s must be >= 0")
        _parse_pause_local_time(self.pause_at_local_time)
        if self.selfplay_checkpoint_source not in {"best", "latest", "auto"}:
            raise ValueError("selfplay_checkpoint_source must be one of: best, latest, auto")
        if self.execution_mode not in {"sequential", "overlap"}:
            raise ValueError("execution_mode must be one of: sequential, overlap")
        if self.selfplay_target_samples_per_cycle < 1:
            raise ValueError("selfplay_target_samples_per_cycle must be >= 1")
        if self.selfplay_num_workers < 1:
            raise ValueError("selfplay_num_workers must be >= 1")
        if self.selfplay_num_simulations < 1:
            raise ValueError("selfplay_num_simulations must be >= 1")
        if self.selfplay_max_states_per_batch < 1:
            raise ValueError("selfplay_max_states_per_batch must be >= 1")
        if self.selfplay_eval_batch_size < 1:
            raise ValueError("selfplay_eval_batch_size must be >= 1")
        if self.selfplay_progress_log_games < 1:
            raise ValueError("selfplay_progress_log_games must be >= 1")
        if self.selfplay_progress_log_shards < 1:
            raise ValueError("selfplay_progress_log_shards must be >= 1")
        if not (0.0 <= self.selfplay_human_position_mix_ratio <= 1.0):
            raise ValueError("selfplay_human_position_mix_ratio must be within [0, 1]")
        if not (0.0 <= self.selfplay_opening_like_material_min_ratio <= 1.0):
            raise ValueError("selfplay_opening_like_material_min_ratio must be within [0, 1]")
        if self.selfplay_opening_like_min_piece_count < 2:
            raise ValueError("selfplay_opening_like_min_piece_count must be >= 2")
        if self.selfplay_max_plies < 1:
            raise ValueError("selfplay_max_plies must be >= 1")
        if self.selfplay_repeat_limit < 1:
            raise ValueError("selfplay_repeat_limit must be >= 1")
        if self.selfplay_repeat_min_ply < 0:
            raise ValueError("selfplay_repeat_min_ply must be >= 0")
        if self.selfplay_no_capture_limit < 1:
            raise ValueError("selfplay_no_capture_limit must be >= 1")
        if self.selfplay_immediate_draw_candidate_scan < 1:
            raise ValueError("selfplay_immediate_draw_candidate_scan must be >= 1")
        if self.selfplay_immediate_draw_capture_injection_threshold < 0:
            raise ValueError("selfplay_immediate_draw_capture_injection_threshold must be >= 0")
        if self.selfplay_immediate_draw_capture_priority_threshold < 0:
            raise ValueError("selfplay_immediate_draw_capture_priority_threshold must be >= 0")
        if self.selfplay_immediate_draw_nocap_pressure_threshold < 0:
            raise ValueError("selfplay_immediate_draw_nocap_pressure_threshold must be >= 0")
        if self.selfplay_immediate_draw_nocap_pressure_scan < 1:
            raise ValueError("selfplay_immediate_draw_nocap_pressure_scan must be >= 1")
        if self.selfplay_immediate_draw_capture_injection_threshold > self.selfplay_immediate_draw_capture_priority_threshold:
            raise ValueError(
                "selfplay_immediate_draw_capture_injection_threshold must be <= selfplay_immediate_draw_capture_priority_threshold"
            )
        if self.selfplay_immediate_draw_capture_priority_threshold > self.selfplay_immediate_draw_nocap_pressure_threshold:
            raise ValueError(
                "selfplay_immediate_draw_capture_priority_threshold must be <= selfplay_immediate_draw_nocap_pressure_threshold"
            )
        if self.selfplay_nocap_warn_threshold < 0.0:
            raise ValueError("selfplay_nocap_warn_threshold must be >= 0")
        if self.selfplay_nocap_stuck_threshold < 0.0:
            raise ValueError("selfplay_nocap_stuck_threshold must be >= 0")
        if self.overlap_selfplay_num_workers is not None and self.overlap_selfplay_num_workers < 1:
            raise ValueError("overlap_selfplay_num_workers must be >= 1 when provided")
        if self.overlap_keep_free_cpu_cores < 0:
            raise ValueError("overlap_keep_free_cpu_cores must be >= 0")
        if self.train_steps_per_cycle < 0:
            raise ValueError("train_steps_per_cycle must be >= 0")
        if self.train_lr_schedule_max_steps < 1:
            raise ValueError("train_lr_schedule_max_steps must be >= 1")
        if not (0.0 <= self.train_bootstrap_human_floor <= 1.0):
            raise ValueError("train_bootstrap_human_floor must be within [0, 1]")
        if self.train_micro_batch_size < 1:
            raise ValueError("train_micro_batch_size must be >= 1")
        if self.train_grad_accum_steps < 1:
            raise ValueError("train_grad_accum_steps must be >= 1")
        if self.train_cpu_sampler_workers < 0:
            raise ValueError("train_cpu_sampler_workers must be >= 0")
        if self.train_cpu_prefetch_batches < 1:
            raise ValueError("train_cpu_prefetch_batches must be >= 1")
        if self.train_shard_cache_size < 1:
            raise ValueError("train_shard_cache_size must be >= 1")
        if self.min_selfplay_quality_for_training not in {"stuck", "warn", "ok"}:
            raise ValueError("min_selfplay_quality_for_training must be one of: stuck, warn, ok")
        if self.snapshot_keep_latest_count < 1:
            raise ValueError("snapshot_keep_latest_count must be >= 1")
        if self.snapshot_keep_best_count < 1:
            raise ValueError("snapshot_keep_best_count must be >= 1")
        if self.arena_games < 1:
            raise ValueError("arena_games must be >= 1")
        if self.arena_sims < 1:
            raise ValueError("arena_sims must be >= 1")
        if not (0.0 <= self.arena_accept_threshold <= 1.0):
            raise ValueError("arena_accept_threshold must be within [0, 1]")
        if self.arena_min_non_draw_games < 0:
            raise ValueError("arena_min_non_draw_games must be >= 0")
        if self.arena_log_every_games < 1:
            raise ValueError("arena_log_every_games must be >= 1")
        if self.arena_games_per_opening < 1:
            raise ValueError("arena_games_per_opening must be >= 1")
        if self.arena_max_plies < 1:
            raise ValueError("arena_max_plies must be >= 1")
        if self.arena_repeat_limit < 1:
            raise ValueError("arena_repeat_limit must be >= 1")
        if self.arena_repeat_min_ply < 0:
            raise ValueError("arena_repeat_min_ply must be >= 0")
        if self.arena_no_capture_limit < 1:
            raise ValueError("arena_no_capture_limit must be >= 1")

def _quality_rank(label: str) -> int:
    order = {"stuck": 0, "warn": 1, "ok": 2}
    return order.get(str(label), -1)


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


def _next_pause_local_deadline(value: str | None) -> datetime | None:
    pause_local_time = _parse_pause_local_time(value)
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


_DEFAULT_CONFIG = ClosedLoopConfig()
_STOP_REQUESTED = False


def _configured_overlap_selfplay_workers(config: ClosedLoopConfig) -> int:
    if config.overlap_selfplay_num_workers is not None:
        return int(config.overlap_selfplay_num_workers)
    if config.execution_mode == "overlap":
        return 6
    return int(config.selfplay_num_workers)


def _training_probe_mode(config: ClosedLoopConfig) -> bool:
    return bool(config.skip_train or config.train_steps_per_cycle <= 0)


def _install_signal_handlers() -> dict[int, Any]:
    previous: dict[int, Any] = {}

    def _handler(signum: int, _frame: Any) -> None:
        global _STOP_REQUESTED
        signal_name = signal.Signals(signum).name if signum in signal.Signals._value2member_map_ else str(signum)
        print(f"[LOOP] received {signal_name}; will stop after the current phase", flush=True)
        _STOP_REQUESTED = True

    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig is None:
            continue
        try:
            previous[int(sig)] = signal.getsignal(sig)
            signal.signal(sig, _handler)
        except Exception:
            continue
    return previous


def _restore_signal_handlers(previous: dict[int, Any]) -> None:
    for signum, handler in previous.items():
        try:
            signal.signal(signum, handler)
        except Exception:
            continue


def _normalize_config(config: ClosedLoopConfig) -> ClosedLoopConfig:
    config.human_data_dir = str(Path(config.human_data_dir).resolve())
    config.training_output_dir = str(Path(config.training_output_dir).resolve())
    config.selfplay_output_root = str(Path(config.selfplay_output_root).resolve())
    config.arena_output_root = str(Path(config.arena_output_root).resolve())
    if config.arena_opening_suite_path is not None:
        config.arena_opening_suite_path = str(Path(config.arena_opening_suite_path).resolve())
    return config


def _current_latest_checkpoint(training_output_dir: Path) -> Path:
    return (training_output_dir / "latest.pt").resolve()


def _current_best_checkpoint(training_output_dir: Path) -> Path:
    return (training_output_dir / "best.pt").resolve()


def _resolve_selfplay_checkpoint(training_output_dir: Path, source: str = "best") -> Path:
    source = str(source).strip().lower() or "best"
    latest_checkpoint = _current_latest_checkpoint(training_output_dir)
    best_checkpoint = _current_best_checkpoint(training_output_dir)
    if source == "latest":
        candidates = [latest_checkpoint, best_checkpoint]
    elif source == "auto":
        latest_step = _load_checkpoint_step(latest_checkpoint)
        best_step = _load_checkpoint_step(best_checkpoint)
        if latest_step is not None and best_step is not None:
            candidates = (
                [latest_checkpoint, best_checkpoint]
                if latest_step >= best_step
                else [best_checkpoint, latest_checkpoint]
            )
        else:
            candidates = [latest_checkpoint, best_checkpoint]
    else:
        candidates = [best_checkpoint, latest_checkpoint]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "no checkpoint available for self-play; expected one of "
        f"{', '.join(str(path) for path in candidates)}"
    )


def _load_checkpoint_step(checkpoint_path: Path) -> int | None:
    if not checkpoint_path.is_file():
        return None
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and raw.get("global_step") is not None:
        return int(raw["global_step"])
    return None


def _next_selfplay_output_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    for run_index in range(1, 10_000):
        candidate = output_root / f"run_{run_index:03d}"
        if not candidate.exists():
            return candidate.resolve()
    raise RuntimeError(f"unable to allocate a self-play run directory under {output_root}")


def _selfplay_root_has_completed_runs(output_root: Path) -> bool:
    if not output_root.exists():
        return False
    for run_dir in sorted(output_root.glob("run_*")):
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        manifest_state = str(manifest.get("manifest_state", "complete")).strip().lower()
        if manifest_state == "complete":
            return True
    return False


def _snapshot_dir(training_output_dir: Path) -> Path:
    return (training_output_dir / "snapshots").resolve()


def _snapshot_step_from_name(path: Path, family_prefix: str) -> int:
    stem = path.stem
    if not stem.startswith(family_prefix):
        return -1
    try:
        return int(stem[len(family_prefix) :])
    except ValueError:
        return -1


def _prune_snapshot_family(snapshot_dir: Path, family_prefix: str, keep_count: int) -> None:
    snapshots: list[tuple[int, float, Path]] = []
    for path in snapshot_dir.glob(f"{family_prefix}*.pt"):
        snapshots.append((_snapshot_step_from_name(path, family_prefix), float(path.stat().st_mtime), path))
    snapshots.sort(key=lambda item: (item[0], item[1]))
    for _, _, stale_path in snapshots[:-keep_count]:
        stale_path.unlink(missing_ok=True)


def _snapshot_checkpoint(
    checkpoint_path: Path,
    training_output_dir: Path,
    *,
    family: str,
    step: int | None,
    keep_count: int,
) -> Path | None:
    if step is None or not checkpoint_path.is_file():
        return None
    snapshot_dir = _snapshot_dir(training_output_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / f"{family}_step{int(step)}.pt"
    shutil.copy2(checkpoint_path, snapshot_path)
    _prune_snapshot_family(snapshot_dir, f"{family}_step", keep_count)
    return snapshot_path.resolve()


def _detect_legacy_detached_trainer(training_output_dir: Path) -> int | None:
    try:
        process_table = subprocess.check_output(
            ["ps", "-eo", "pid,args"],
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        process_table = ""
    for raw_line in process_table.splitlines():
        line = raw_line.strip()
        if not line or "xiangqi_train.py" not in line:
            continue
        parts = line.split(maxsplit=1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except Exception:
            continue
        if pid == os.getpid():
            continue
        if "grep xiangqi_train.py" in line:
            continue
        return pid

    pid_path = training_output_dir / "train_pid.txt"
    if not pid_path.is_file():
        return None
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except Exception:
        try:
            pid_path.unlink()
        except Exception:
            pass
        return None
    if pid <= 0:
        try:
            pid_path.unlink()
        except Exception:
            pass
        return None

    try:
        os.kill(pid, 0)
    except OSError:
        try:
            pid_path.unlink()
        except Exception:
            pass
        return None

    cmdline_path = Path("/proc") / str(pid) / "cmdline"
    try:
        cmdline = cmdline_path.read_text(encoding="utf-8", errors="ignore").replace("\x00", " ")
    except Exception:
        cmdline = ""
    if "xiangqi_train.py" not in cmdline:
        return None
    return pid


def _ensure_candidate_and_champion(training_output_dir: Path) -> tuple[Path, Path]:
    latest_checkpoint = _current_latest_checkpoint(training_output_dir)
    best_checkpoint = _current_best_checkpoint(training_output_dir)

    if not latest_checkpoint.is_file() and not best_checkpoint.is_file():
        raise FileNotFoundError(
            f"no training checkpoint found under {training_output_dir}; expected latest.pt or best.pt"
        )

    if latest_checkpoint.is_file() and not best_checkpoint.is_file():
        shutil.copy2(latest_checkpoint, best_checkpoint)

    candidate_checkpoint = latest_checkpoint if latest_checkpoint.is_file() else best_checkpoint
    champion_checkpoint = best_checkpoint if best_checkpoint.is_file() else candidate_checkpoint
    return candidate_checkpoint.resolve(), champion_checkpoint.resolve()


def _release_device_memory(device: str) -> None:
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()


def _read_selfplay_manifest(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _print_cycle_header(cycle_index: int, config: ClosedLoopConfig) -> None:
    cycle_label = f"{cycle_index}" if cycle_index > 0 else "?"
    arena_label = "on" if config.arena_enabled else "off"
    print(
        f"[LOOP] cycle={cycle_label} "
        f"selfplay_samples={config.selfplay_target_samples_per_cycle} "
        f"train_steps={config.train_steps_per_cycle} "
        f"train_mode={'probe' if _training_probe_mode(config) else 'train'} "
        f"arena={arena_label} arena_games={config.arena_games} arena_sims={config.arena_sims} "
        f"device={config.device}",
        flush=True,
    )


def _write_cycle_summary(training_output_dir: Path, cycle_summary: dict[str, Any]) -> None:
    training_output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = training_output_dir / "closed_loop_latest.json"
    dated_path = training_output_dir / f"closed_loop_cycle_{int(cycle_summary['cycle']):03d}.json"
    payload = json.dumps(cycle_summary, indent=2, ensure_ascii=False)
    latest_path.write_text(payload, encoding="utf-8")
    dated_path.write_text(payload, encoding="utf-8")


def _available_cpu_ids() -> list[int]:
    if hasattr(os, "sched_getaffinity"):
        return sorted(int(cpu_id) for cpu_id in os.sched_getaffinity(0))
    cpu_count = os.cpu_count() or 1
    return list(range(cpu_count))


def _try_set_process_affinity(cpu_ids: list[int]) -> None:
    if not cpu_ids or not hasattr(os, "sched_setaffinity"):
        return
    try:
        os.sched_setaffinity(0, set(int(cpu_id) for cpu_id in cpu_ids))
    except OSError:
        return


def _split_overlap_cpu_ids(keep_free_cores: int) -> tuple[list[int], list[int]]:
    cpu_ids = _available_cpu_ids()
    if len(cpu_ids) <= 1:
        return cpu_ids, cpu_ids

    keep = min(max(0, keep_free_cores), max(len(cpu_ids) - 2, 0))
    usable_cpu_ids = cpu_ids[:-keep] if keep > 0 else cpu_ids
    if len(usable_cpu_ids) <= 1:
        return cpu_ids, cpu_ids

    midpoint = max(1, len(usable_cpu_ids) // 2)
    selfplay_cpu_ids = usable_cpu_ids[:midpoint]
    train_cpu_ids = usable_cpu_ids[midpoint:]
    if not train_cpu_ids:
        train_cpu_ids = [selfplay_cpu_ids[-1]]
    return selfplay_cpu_ids, train_cpu_ids


def _format_cpu_ids(cpu_ids: list[int]) -> str:
    if not cpu_ids:
        return "all"
    if len(cpu_ids) <= 8:
        return ",".join(str(cpu_id) for cpu_id in cpu_ids)
    head = ",".join(str(cpu_id) for cpu_id in cpu_ids[:4])
    tail = ",".join(str(cpu_id) for cpu_id in cpu_ids[-2:])
    return f"{head},...,{tail} ({len(cpu_ids)} cores)"


def _request_process_interrupt(proc: mp.Process, label: str) -> None:
    if not proc.is_alive():
        return
    try:
        if proc.pid is not None:
            os.kill(proc.pid, signal.SIGINT)
            return
    except OSError:
        pass
    proc.terminate()
    print(f"[LOOP] forced terminate on {label} pid={proc.pid}", flush=True)


def _run_selfplay_phase(
    config: ClosedLoopConfig,
    cycle_index: int,
    training_output_dir: Path,
    *,
    selfplay_num_workers: int | None = None,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    checkpoint_path = _resolve_selfplay_checkpoint(
        training_output_dir,
        source=config.selfplay_checkpoint_source,
    )
    checkpoint_step = _load_checkpoint_step(checkpoint_path)
    output_dir = _next_selfplay_output_dir(Path(config.selfplay_output_root))
    print(
        f"[LOOP] self-play start cycle={cycle_index} checkpoint={checkpoint_path} "
        f"step={checkpoint_step} output_dir={output_dir}",
        flush=True,
    )
    selfplay_config = SelfPlayConfig(
        target_samples=config.selfplay_target_samples_per_cycle,
        target_games=None,
        num_workers=int(selfplay_num_workers or config.selfplay_num_workers),
        max_states_per_batch=config.selfplay_max_states_per_batch,
        num_simulations=config.selfplay_num_simulations,
        eval_batch_size=config.selfplay_eval_batch_size,
        progress_log_games=config.selfplay_progress_log_games,
        progress_log_shards=config.selfplay_progress_log_shards,
        max_plies=config.selfplay_max_plies,
        repeat_limit=config.selfplay_repeat_limit,
        repeat_min_ply=config.selfplay_repeat_min_ply,
        no_capture_limit=config.selfplay_no_capture_limit,
        bootstrap_quality_nocap_draw_threshold=config.selfplay_nocap_stuck_threshold,
        bootstrap_quality_warn_nocap_draw_threshold=config.selfplay_nocap_warn_threshold,
        immediate_draw_candidate_scan=config.selfplay_immediate_draw_candidate_scan,
        immediate_draw_capture_injection_threshold=config.selfplay_immediate_draw_capture_injection_threshold,
        immediate_draw_capture_priority_threshold=config.selfplay_immediate_draw_capture_priority_threshold,
        immediate_draw_nocap_pressure_threshold=config.selfplay_immediate_draw_nocap_pressure_threshold,
        immediate_draw_nocap_pressure_scan=config.selfplay_immediate_draw_nocap_pressure_scan,
        start_position_mode="human_positions",
        human_position_source_dir=str((Path(config.human_data_dir).resolve() / "train")),
        human_position_mix_ratio=config.selfplay_human_position_mix_ratio,
        opening_like_material_min_ratio=config.selfplay_opening_like_material_min_ratio,
        opening_like_min_piece_count=config.selfplay_opening_like_min_piece_count,
        pause_at_local_time=config.pause_at_local_time,
        device=config.device,
        seed=config.seed + cycle_index * 100_000,
        frozen_value_checkpoint=(
            str(config.rl_frozen_value_checkpoint)
            if config.rl_frozen_value_checkpoint
            else None
        ),
    )
    summary = run_selfplay(checkpoint_path=checkpoint_path, output_dir=output_dir, config=selfplay_config)
    manifest = _read_selfplay_manifest(output_dir)
    quality = str(summary.get("quality", manifest.get("quality", "unknown")))
    quality_metrics = summary.get("quality_metrics") or manifest.get("quality_metrics", {})
    start_position_metrics = summary.get("start_position_metrics") or manifest.get("start_position_metrics", {})
    print(
        f"[LOOP] self-play done cycle={cycle_index} games={summary['games_completed']} "
        f"samples={summary['samples_written']} shards={summary['shards_written']} "
        f"quality={quality} "
        f"decisive={float(quality_metrics.get('decisive_rate', 0.0)):.1f}% "
        f"rep_draw={float(quality_metrics.get('rep_draw_rate', 0.0)):.1f}% "
        f"long_check_loss={float(quality_metrics.get('long_check_loss_rate', 0.0)):.1f}% "
        f"nocap_draw={float(quality_metrics.get('nocap_draw_rate', 0.0)):.1f}% "
        f"anti_draw={float(quality_metrics.get('anti_draw_override_rate', 0.0)):.1f}% "
        f"stm_black={float(quality_metrics.get('stm_is_black_rate', 0.0)):.1f}% "
        f"human_start={float(start_position_metrics.get('human_start_rate', 0.0)):.1f}%",
        flush=True,
    )
    if quality == "stuck":
        print(
            "[LOOP] self-play quality=stuck; trainer quality gate will skip this run and keep leaning on human data.",
            flush=True,
        )
    return summary, output_dir, manifest


def _run_training_phase(
    config: ClosedLoopConfig,
    cycle_index: int,
    training_output_dir: Path,
    *,
    train_cpu_reserved_cores: int = 4,
) -> dict[str, Any]:
    latest_checkpoint = _current_latest_checkpoint(training_output_dir)
    best_checkpoint = _current_best_checkpoint(training_output_dir)
    current_step = _load_checkpoint_step(latest_checkpoint) or 0
    previous_best_step = _load_checkpoint_step(best_checkpoint)
    target_step = current_step + config.train_steps_per_cycle
    lr_schedule_max_steps = max(config.train_lr_schedule_max_steps, target_step)
    selfplay_root = Path(config.selfplay_output_root).resolve()
    should_reset_selfplay_ingest_state = bool(
        config.train_reset_selfplay_ingest_state_on_resume and not _selfplay_root_has_completed_runs(selfplay_root)
    )
    resume_path: str | None = None
    if latest_checkpoint.is_file():
        resume_path = str(latest_checkpoint)
    elif best_checkpoint.is_file():
        resume_path = str(best_checkpoint)
    print(
        f"[LOOP] train start cycle={cycle_index} from_step={current_step} target_step={target_step} "
        f"lr_schedule_max_steps={lr_schedule_max_steps} selfplay_root={selfplay_root} "
        f"reset_selfplay_ingest={'on' if should_reset_selfplay_ingest_state else 'off'}",
        flush=True,
    )
    train_config = TrainingConfig(
        human_data_dir=config.human_data_dir,
        # v15-RL: force geo's FULL geometry (both attention biases). The default
        # XiangqiTransformerConfig has both biases OFF, which would build a
        # non-geometry model and fail/cripple loading geo's geometry weights.
        model_config=XiangqiTransformerConfig(
            use_2d_relative_attention_bias=True,
            use_line_of_sight_attention_bias=True,
        ),
        selfplay_dirs=[config.selfplay_output_root],
        output_dir=config.training_output_dir,
        resume_path=resume_path,
        device=config.device,
        shard_cache_size=config.train_shard_cache_size,
        bootstrap_mode=True,
        bootstrap_human_floor=config.train_bootstrap_human_floor,
        # v15-RL: size the replay buffer to ~3 cycles of self-play so it FILLS with
        # self-play (buffer_fill->1 => human mix -> the 5% floor). The default 300k
        # buffer never fills at our self-play volume, which forces a ~73% human mix
        # (bootstrap padding) -- i.e. human-distillation, not self-play. See
        # _compute_mix_ratios: human_ratio = max(floor, 1-(1-floor)*buffer_fill).
        replay_buffer_size=max(1, int(config.selfplay_target_samples_per_cycle) * 3),
        # v15-RL value-anchor fix: pure-z value training COLLAPSED geo's d20-distilled
        # value head (candidate lost 0-24 at the first gate). Down-weight the outcome
        # (z)-based value/wdl heads so z is only a LIGHT AUXILIARY -> the value head
        # stays ~geo's calibrated judgment instead of being washed out by noisy z, while
        # the POLICY keeps learning from self-play visits at full weight. + gentle LR.
        # v16 frozen-evaluator mode: value comes from the frozen reference at PLAY
        # time (chimeras), so the network's own value/wdl heads get ZERO loss and
        # the optimizer is reset so the 1e-4 actually applies (with a short warmup
        # that fits inside one cycle; the default 2000 would never complete).
        learning_rate=1e-4,
        value_loss_weight=0.0 if config.rl_frozen_value_checkpoint else 0.1,
        wdl_loss_weight=0.0 if config.rl_frozen_value_checkpoint else 0.2,
        reset_optimizer_on_resume=bool(config.rl_frozen_value_checkpoint),
        warmup_steps=100 if config.rl_frozen_value_checkpoint else 2000,
        # B-experiment: policy anchor (KL to frozen geo) to stop the self-play policy from
        # forgetting geo's broad d20-distilled knowledge. Off (kl_weight 0) by default.
        anchor_checkpoint=config.rl_anchor_checkpoint,
        anchor_policy_kl_weight=config.rl_anchor_policy_kl_weight,
        anchor_anneal_steps=config.rl_anchor_anneal_steps,
        selfplay_run_quality_gate=True,
        reset_selfplay_ingest_state_on_resume=should_reset_selfplay_ingest_state,
        micro_batch_size=config.train_micro_batch_size,
        grad_accum_steps=config.train_grad_accum_steps,
        max_steps=target_step,
        lr_schedule_max_steps=lr_schedule_max_steps,
        cpu_sampler_workers=config.train_cpu_sampler_workers,
        cpu_reserved_cores=train_cpu_reserved_cores,
        cpu_prefetch_batches=config.train_cpu_prefetch_batches,
        run_detached=False,
        save_on_interrupt=True,
        promote_best_on_human_val=not (config.arena_enabled and config.arena_gates_best_update),
        pause_at_local_time=config.pause_at_local_time,
    )
    summary = run_training(train_config)
    latest_step_after = _load_checkpoint_step(latest_checkpoint)
    best_step_after = _load_checkpoint_step(best_checkpoint)
    latest_snapshot = _snapshot_checkpoint(
        latest_checkpoint,
        training_output_dir,
        family="latest",
        step=latest_step_after,
        keep_count=config.snapshot_keep_latest_count,
    )
    best_snapshot = None
    if best_step_after is not None and best_step_after != previous_best_step:
        best_snapshot = _snapshot_checkpoint(
            best_checkpoint,
            training_output_dir,
            family="best",
            step=best_step_after,
            keep_count=config.snapshot_keep_best_count,
        )
    summary["latest_snapshot_path"] = str(latest_snapshot) if latest_snapshot is not None else None
    summary["best_snapshot_path"] = str(best_snapshot) if best_snapshot is not None else None
    print(
        f"[LOOP] train done cycle={cycle_index} global_step={summary['global_step']} "
        f"best_human_val={summary['best_human_val_metric']:.6f} "
        f"last_human_val={summary['last_human_val_total_loss'] if summary['last_human_val_total_loss'] is not None else 'n/a'} "
        f"selfplay_buffer={summary['selfplay_buffer_samples']}",
        flush=True,
    )
    if latest_snapshot is not None or best_snapshot is not None:
        snapshot_parts: list[str] = []
        if latest_snapshot is not None:
            snapshot_parts.append(f"latest={latest_snapshot}")
        if best_snapshot is not None:
            snapshot_parts.append(f"best={best_snapshot}")
        print(f"[LOOP] checkpoint snapshots {' '.join(snapshot_parts)}", flush=True)
    return summary


def _selfplay_phase_child(
    config: ClosedLoopConfig,
    cycle_index: int,
    training_output_dir: str,
    cpu_ids: list[int],
    selfplay_num_workers: int,
    result_queue: Any,
) -> None:
    try:
        _try_set_process_affinity(cpu_ids)
        summary, output_dir, manifest = _run_selfplay_phase(
            config,
            cycle_index,
            Path(training_output_dir),
            selfplay_num_workers=selfplay_num_workers,
        )
        result_queue.put(
            {
                "name": "selfplay",
                "ok": True,
                "summary": summary,
                "output_dir": str(output_dir),
                "manifest": manifest,
            }
        )
    except Exception as exc:
        result_queue.put({"name": "selfplay", "ok": False, "error": f"{type(exc).__name__}: {exc}"})
        raise


def _training_phase_child(
    config: ClosedLoopConfig,
    cycle_index: int,
    training_output_dir: str,
    cpu_ids: list[int],
    result_queue: Any,
) -> None:
    try:
        _try_set_process_affinity(cpu_ids)
        summary = _run_training_phase(
            config,
            cycle_index,
            Path(training_output_dir),
            train_cpu_reserved_cores=0,
        )
        result_queue.put({"name": "train", "ok": True, "summary": summary})
    except Exception as exc:
        result_queue.put({"name": "train", "ok": False, "error": f"{type(exc).__name__}: {exc}"})
        raise


def _run_overlap_phases(
    config: ClosedLoopConfig,
    cycle_index: int,
    training_output_dir: Path,
) -> tuple[dict[str, Any], Path, dict[str, Any], dict[str, Any]]:
    selfplay_cpu_ids, train_cpu_ids = _split_overlap_cpu_ids(config.overlap_keep_free_cpu_cores)
    effective_selfplay_workers = int(
        min(
            _configured_overlap_selfplay_workers(config),
            max(len(selfplay_cpu_ids), 1),
        )
    )
    print(
        f"[LOOP] overlap start cycle={cycle_index} "
        f"selfplay_workers={effective_selfplay_workers} "
        f"selfplay_cpus={_format_cpu_ids(selfplay_cpu_ids)} "
        f"train_cpus={_format_cpu_ids(train_cpu_ids)}",
        flush=True,
    )
    if _training_probe_mode(config):
        selfplay_summary, selfplay_output_dir, selfplay_manifest = _run_selfplay_phase(
            config,
            cycle_index,
            training_output_dir,
            selfplay_num_workers=effective_selfplay_workers,
        )
        train_summary = _skip_training_phase(
            config,
            cycle_index,
            reason="training_disabled:probe_mode",
            message="probe mode active; training is disabled until draw probes pass",
        )
        return selfplay_summary, selfplay_output_dir, selfplay_manifest, train_summary

    ctx = mp.get_context("spawn")
    result_queue: Any = ctx.Queue()
    processes = {
        "selfplay": ctx.Process(
            target=_selfplay_phase_child,
            name=f"loop-selfplay-{cycle_index}",
            args=(
                config,
                cycle_index,
                str(training_output_dir),
                selfplay_cpu_ids,
                effective_selfplay_workers,
                result_queue,
            ),
        ),
        "train": ctx.Process(
            target=_training_phase_child,
            name=f"loop-train-{cycle_index}",
            args=(
                config,
                cycle_index,
                str(training_output_dir),
                train_cpu_ids,
                result_queue,
            ),
        ),
    }

    for proc in processes.values():
        proc.start()

    results: dict[str, dict[str, Any]] = {}
    try:
        while len(results) < len(processes):
            if _STOP_REQUESTED:
                for name, proc in processes.items():
                    _request_process_interrupt(proc, name)
            try:
                message = result_queue.get(timeout=1.0)
            except queue_module.Empty:
                message = None
            if isinstance(message, dict):
                name = str(message.get("name", "unknown"))
                results[name] = message
                if not bool(message.get("ok", False)):
                    for other_name, proc in processes.items():
                        if other_name != name:
                            _request_process_interrupt(proc, other_name)
                    raise RuntimeError(f"{name} overlap phase failed: {message.get('error', 'unknown error')}")

            for name, proc in processes.items():
                if proc.exitcode not in (None, 0) and name not in results:
                    for other_name, other_proc in processes.items():
                        if other_name != name:
                            _request_process_interrupt(other_proc, other_name)
                    raise RuntimeError(f"{name} overlap phase exited with code {proc.exitcode}")
    finally:
        for proc in processes.values():
            proc.join(timeout=5.0)
        for name, proc in processes.items():
            if proc.is_alive():
                _request_process_interrupt(proc, name)
                proc.join(timeout=5.0)
        try:
            result_queue.close()
            result_queue.join_thread()
        except Exception:
            pass

    selfplay_result = results["selfplay"]
    train_result = results["train"]
    return (
        dict(selfplay_result["summary"]),
        Path(selfplay_result["output_dir"]),
        dict(selfplay_result["manifest"]),
        dict(train_result["summary"]),
    )


def _skip_training_phase(
    config: ClosedLoopConfig,
    cycle_index: int,
    *,
    reason: str,
    message: str,
) -> dict[str, Any]:
    summary = {
        "global_step": _load_checkpoint_step(_current_latest_checkpoint(Path(config.training_output_dir).resolve())) or 0,
        "best_human_val_metric": None,
        "best_val_metric": None,
        "last_human_val_total_loss": None,
        "selfplay_buffer_samples": 0,
        "output_dir": str(Path(config.training_output_dir).resolve()),
        "interrupted": False,
        "skipped": True,
        "reason": reason,
    }
    print(f"[LOOP] train skipped cycle={cycle_index}: {message}", flush=True)
    return summary


def _run_arena_phase(config: ClosedLoopConfig, cycle_index: int, training_output_dir: Path) -> dict[str, Any]:
    candidate_checkpoint, champion_checkpoint = _ensure_candidate_and_champion(training_output_dir)
    if candidate_checkpoint == champion_checkpoint:
        summary = {
            "accepted": False,
            "promoted": False,
            "skipped": True,
            "reason": "candidate and champion checkpoints are identical",
            "candidate_checkpoint": str(candidate_checkpoint),
            "champion_checkpoint": str(champion_checkpoint),
        }
        print(f"[LOOP] arena skipped cycle={cycle_index}: {summary['reason']}", flush=True)
        return summary

    print(
        f"[LOOP] arena start cycle={cycle_index} candidate={candidate_checkpoint.name} "
        f"champion={champion_checkpoint.name}",
        flush=True,
    )
    arena_config = ArenaConfig(
        games=config.arena_games,
        games_per_opening=config.arena_games_per_opening,
        sims=config.arena_sims,
        accept_threshold=config.arena_accept_threshold,
        min_non_draw_games=config.arena_min_non_draw_games,
        log_every_games=config.arena_log_every_games,
        max_plies=config.arena_max_plies,
        repeat_limit=config.arena_repeat_limit,
        repeat_min_ply=config.arena_repeat_min_ply,
        no_capture_limit=config.arena_no_capture_limit,
        device=config.device,
        promote_on_pass=config.arena_promote_on_pass,
        opening_suite_path=config.arena_opening_suite_path,
        shared_value_checkpoint=(
            str(config.rl_frozen_value_checkpoint)
            if config.rl_frozen_value_checkpoint
            else None
        ),
        add_root_noise=config.arena_add_root_noise,
        dirichlet_alpha=config.arena_dirichlet_alpha,
        dirichlet_eps=config.arena_dirichlet_eps,
        temperature_move=config.arena_temperature_move,
    )
    summary = run_arena(
        candidate_checkpoint=candidate_checkpoint,
        champion_checkpoint=champion_checkpoint,
        output_root=config.arena_output_root,
        config=arena_config,
    )
    print(
        f"[LOOP] arena done cycle={cycle_index} accepted={summary['accepted']} promoted={summary['promoted']} "
        f"score={summary['candidate_score_rate'] * 100.0:.1f}% "
        f"red={summary['arena_red_win_rate'] * 100.0:.1f}% "
        f"black={summary['arena_black_win_rate'] * 100.0:.1f}% "
        f"draw={summary['arena_draw_rate'] * 100.0:.1f}% non_draw={summary['non_draw']}",
        flush=True,
    )
    return summary


def _skip_arena_phase(config: ClosedLoopConfig, cycle_index: int) -> dict[str, Any]:
    summary = {
        "accepted": False,
        "promoted": False,
        "skipped": True,
        "reason": "arena_disabled_for_bootstrap",
        "games": int(config.arena_games),
        "sims": int(config.arena_sims),
    }
    print(
        f"[LOOP] arena skipped cycle={cycle_index}: {summary['reason']}",
        flush=True,
    )
    return summary


def run_closed_loop(config: ClosedLoopConfig) -> dict[str, Any]:
    config = _normalize_config(config)
    training_output_dir = Path(config.training_output_dir).resolve()
    selfplay_output_root = Path(config.selfplay_output_root).resolve()
    selfplay_output_root.mkdir(parents=True, exist_ok=True)
    Path(config.arena_output_root).resolve().mkdir(parents=True, exist_ok=True)
    training_output_dir.mkdir(parents=True, exist_ok=True)
    pause_deadline = _next_pause_local_deadline(config.pause_at_local_time)

    legacy_pid = _detect_legacy_detached_trainer(training_output_dir)
    if legacy_pid is not None:
        raise RuntimeError(
            "found an existing detached xiangqi_train.py background process "
            f"(pid={legacy_pid}) for {training_output_dir}. "
            "Please stop it before starting the closed loop, otherwise it will keep competing for the same GPU."
        )

    cycle_summaries: list[dict[str, Any]] = []
    completed_cycles = 0
    start_time = time.time()

    signal_handlers = _install_signal_handlers()
    try:
        cycle_index = 0
        while not _STOP_REQUESTED:
            if config.cycles > 0 and cycle_index >= config.cycles:
                break
            if _pause_deadline_reached(pause_deadline):
                print(
                    "[LOOP] scheduled local pause reached at "
                    f"{pause_deadline.astimezone().strftime('%Y-%m-%d %H:%M') if pause_deadline else config.pause_at_local_time}; "
                    "stopping before the next cycle.",
                    flush=True,
                )
                break

            cycle_index += 1
            completed_cycles = cycle_index
            _print_cycle_header(cycle_index, config)

            if config.execution_mode == "overlap":
                (
                    selfplay_summary,
                    selfplay_output_dir,
                    selfplay_manifest,
                    train_summary,
                ) = _run_overlap_phases(
                    config,
                    cycle_index,
                    training_output_dir,
                )
            else:
                selfplay_summary, selfplay_output_dir, selfplay_manifest = _run_selfplay_phase(
                    config,
                    cycle_index,
                    training_output_dir,
                )
                _release_device_memory(config.device)
                if selfplay_summary.get("fatal_error") is not None:
                    raise RuntimeError(f"self-play cycle {cycle_index} failed: {selfplay_summary['fatal_error']}")
                if bool(selfplay_summary.get("scheduled_pause_requested")):
                    print(
                        "[LOOP] scheduled local pause reached at "
                        f"{pause_deadline.astimezone().strftime('%Y-%m-%d %H:%M') if pause_deadline else config.pause_at_local_time}; "
                        "stopping after self-play.",
                        flush=True,
                    )
                    break
                if _STOP_REQUESTED:
                    break
                if _pause_deadline_reached(pause_deadline):
                    print(
                        "[LOOP] scheduled local pause reached at "
                        f"{pause_deadline.astimezone().strftime('%Y-%m-%d %H:%M') if pause_deadline else config.pause_at_local_time}; "
                        "skipping training.",
                        flush=True,
                    )
                    break

                latest_quality = str(selfplay_manifest.get("quality", selfplay_summary.get("quality", "unknown")))
                if _training_probe_mode(config):
                    train_summary = _skip_training_phase(
                        config,
                        cycle_index,
                        reason="training_disabled:probe_mode",
                        message="probe mode active; training is disabled until draw probes pass",
                    )
                elif _quality_rank(latest_quality) < _quality_rank(config.min_selfplay_quality_for_training):
                    train_summary = _skip_training_phase(
                        config,
                        cycle_index,
                        reason=f"selfplay_quality_below_gate:{latest_quality}",
                        message=(
                            f"latest self-play quality={latest_quality} "
                            f"below required={config.min_selfplay_quality_for_training}"
                        ),
                    )
                else:
                    train_summary = _run_training_phase(config, cycle_index, training_output_dir)
                _release_device_memory(config.device)
                if bool(train_summary.get("interrupted")):
                    break
                if _STOP_REQUESTED:
                    break
                if _pause_deadline_reached(pause_deadline):
                    print(
                        "[LOOP] scheduled local pause reached at "
                        f"{pause_deadline.astimezone().strftime('%Y-%m-%d %H:%M') if pause_deadline else config.pause_at_local_time}; "
                        "stopping before arena.",
                        flush=True,
                    )
                    break

            _release_device_memory(config.device)
            if selfplay_summary.get("fatal_error") is not None:
                raise RuntimeError(f"self-play cycle {cycle_index} failed: {selfplay_summary['fatal_error']}")
            if bool(selfplay_summary.get("scheduled_pause_requested")):
                print(
                    "[LOOP] scheduled local pause reached at "
                    f"{pause_deadline.astimezone().strftime('%Y-%m-%d %H:%M') if pause_deadline else config.pause_at_local_time}; "
                    "stopping after current cycle work.",
                    flush=True,
                )
                break
            if bool(train_summary.get("interrupted")):
                break
            if _STOP_REQUESTED:
                break
            if _pause_deadline_reached(pause_deadline):
                print(
                    "[LOOP] scheduled local pause reached at "
                    f"{pause_deadline.astimezone().strftime('%Y-%m-%d %H:%M') if pause_deadline else config.pause_at_local_time}; "
                    "stopping before arena.",
                    flush=True,
                )
                break

            if config.arena_enabled and (cycle_index % config.arena_every_n_cycles == 0):
                arena_summary = _run_arena_phase(config, cycle_index, training_output_dir)
            else:
                arena_summary = _skip_arena_phase(config, cycle_index)
            _release_device_memory(config.device)

            cycle_summary = {
                "cycle": cycle_index,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "selfplay_output_dir": str(selfplay_output_dir),
                "selfplay_quality": selfplay_manifest.get("quality"),
                "selfplay_summary": selfplay_summary,
                "train_summary": train_summary,
                "arena_summary": arena_summary,
            }
            cycle_summaries.append(cycle_summary)
            _write_cycle_summary(training_output_dir, cycle_summary)

            if _STOP_REQUESTED:
                break
            if config.cycles > 0 and cycle_index >= config.cycles:
                break
            if config.sleep_between_cycles_s > 0.0:
                print(f"[LOOP] sleeping {config.sleep_between_cycles_s:.1f}s before the next cycle", flush=True)
                time.sleep(config.sleep_between_cycles_s)
    finally:
        _release_device_memory(config.device)
        _restore_signal_handlers(signal_handlers)

    return {
        "completed_cycles": completed_cycles,
        "stopped_by_signal": bool(_STOP_REQUESTED),
        "elapsed_seconds": time.time() - start_time,
        "training_output_dir": str(training_output_dir),
        "selfplay_output_root": str(selfplay_output_root),
        "arena_output_root": str(Path(config.arena_output_root).resolve()),
        "last_cycle": cycle_summaries[-1] if cycle_summaries else None,
        "config": asdict(config),
    }


def _parse_args() -> ClosedLoopConfig:
    parser = argparse.ArgumentParser(
        description="Run Xiangqi self-play, training, and arena in one managed closed-loop entrypoint."
    )
    parser.add_argument("--human-data-dir", default=_DEFAULT_CONFIG.human_data_dir)
    parser.add_argument("--training-output-dir", default=_DEFAULT_CONFIG.training_output_dir)
    parser.add_argument("--selfplay-output-root", default=_DEFAULT_CONFIG.selfplay_output_root)
    parser.add_argument("--arena-output-root", default=_DEFAULT_CONFIG.arena_output_root)
    parser.add_argument("--device", default=_DEFAULT_CONFIG.device)
    parser.add_argument("--pause-at-local-time", default=_DEFAULT_CONFIG.pause_at_local_time or "")
    parser.add_argument(
        "--selfplay-checkpoint-source",
        choices=["best", "latest", "auto"],
        default=_DEFAULT_CONFIG.selfplay_checkpoint_source,
    )
    parser.add_argument(
        "--execution-mode",
        choices=["sequential", "overlap"],
        default=_DEFAULT_CONFIG.execution_mode,
    )
    parser.add_argument("--cycles", type=int, default=_DEFAULT_CONFIG.cycles)
    parser.add_argument("--sleep-between-cycles-s", type=float, default=_DEFAULT_CONFIG.sleep_between_cycles_s)
    parser.add_argument("--seed", type=int, default=_DEFAULT_CONFIG.seed)

    parser.add_argument(
        "--selfplay-target-samples-per-cycle",
        type=int,
        default=_DEFAULT_CONFIG.selfplay_target_samples_per_cycle,
    )
    parser.add_argument("--selfplay-num-workers", type=int, default=_DEFAULT_CONFIG.selfplay_num_workers)
    parser.add_argument("--selfplay-num-simulations", type=int, default=_DEFAULT_CONFIG.selfplay_num_simulations)
    parser.add_argument(
        "--selfplay-max-states-per-batch",
        type=int,
        default=_DEFAULT_CONFIG.selfplay_max_states_per_batch,
    )
    parser.add_argument("--selfplay-eval-batch-size", type=int, default=_DEFAULT_CONFIG.selfplay_eval_batch_size)
    parser.add_argument("--selfplay-progress-log-games", type=int, default=_DEFAULT_CONFIG.selfplay_progress_log_games)
    parser.add_argument("--selfplay-progress-log-shards", type=int, default=_DEFAULT_CONFIG.selfplay_progress_log_shards)
    parser.add_argument(
        "--selfplay-human-position-mix-ratio",
        type=float,
        default=_DEFAULT_CONFIG.selfplay_human_position_mix_ratio,
    )
    parser.add_argument(
        "--selfplay-opening-like-material-min-ratio",
        type=float,
        default=_DEFAULT_CONFIG.selfplay_opening_like_material_min_ratio,
    )
    parser.add_argument(
        "--selfplay-opening-like-min-piece-count",
        type=int,
        default=_DEFAULT_CONFIG.selfplay_opening_like_min_piece_count,
    )
    parser.add_argument("--selfplay-max-plies", type=int, default=_DEFAULT_CONFIG.selfplay_max_plies)
    parser.add_argument("--selfplay-repeat-limit", type=int, default=_DEFAULT_CONFIG.selfplay_repeat_limit)
    parser.add_argument("--selfplay-repeat-min-ply", type=int, default=_DEFAULT_CONFIG.selfplay_repeat_min_ply)
    parser.add_argument(
        "--selfplay-no-capture-limit",
        type=int,
        default=_DEFAULT_CONFIG.selfplay_no_capture_limit,
    )
    parser.add_argument(
        "--selfplay-immediate-draw-candidate-scan",
        type=int,
        default=_DEFAULT_CONFIG.selfplay_immediate_draw_candidate_scan,
    )
    parser.add_argument(
        "--selfplay-immediate-draw-nocap-pressure-threshold",
        type=int,
        default=_DEFAULT_CONFIG.selfplay_immediate_draw_nocap_pressure_threshold,
    )
    parser.add_argument(
        "--selfplay-immediate-draw-capture-injection-threshold",
        type=int,
        default=_DEFAULT_CONFIG.selfplay_immediate_draw_capture_injection_threshold,
    )
    parser.add_argument(
        "--selfplay-immediate-draw-capture-priority-threshold",
        type=int,
        default=_DEFAULT_CONFIG.selfplay_immediate_draw_capture_priority_threshold,
    )
    parser.add_argument(
        "--selfplay-immediate-draw-nocap-pressure-scan",
        type=int,
        default=_DEFAULT_CONFIG.selfplay_immediate_draw_nocap_pressure_scan,
    )
    parser.add_argument(
        "--selfplay-nocap-warn-threshold",
        type=float,
        default=_DEFAULT_CONFIG.selfplay_nocap_warn_threshold,
    )
    parser.add_argument(
        "--selfplay-nocap-stuck-threshold",
        type=float,
        default=_DEFAULT_CONFIG.selfplay_nocap_stuck_threshold,
    )
    parser.add_argument(
        "--overlap-selfplay-num-workers",
        type=int,
        default=_DEFAULT_CONFIG.overlap_selfplay_num_workers,
    )
    parser.add_argument(
        "--overlap-keep-free-cpu-cores",
        type=int,
        default=_DEFAULT_CONFIG.overlap_keep_free_cpu_cores,
    )

    parser.add_argument("--train-steps-per-cycle", type=int, default=_DEFAULT_CONFIG.train_steps_per_cycle)
    parser.add_argument(
        "--skip-train",
        action=argparse.BooleanOptionalAction,
        default=_DEFAULT_CONFIG.skip_train,
    )
    parser.add_argument(
        "--train-lr-schedule-max-steps",
        type=int,
        default=_DEFAULT_CONFIG.train_lr_schedule_max_steps,
    )
    parser.add_argument(
        "--train-bootstrap-human-floor",
        type=float,
        default=_DEFAULT_CONFIG.train_bootstrap_human_floor,
    )
    parser.add_argument(
        "--train-reset-selfplay-ingest-state-on-resume",
        action=argparse.BooleanOptionalAction,
        default=_DEFAULT_CONFIG.train_reset_selfplay_ingest_state_on_resume,
    )
    parser.add_argument("--train-micro-batch-size", type=int, default=_DEFAULT_CONFIG.train_micro_batch_size)
    parser.add_argument("--train-grad-accum-steps", type=int, default=_DEFAULT_CONFIG.train_grad_accum_steps)
    parser.add_argument("--train-cpu-sampler-workers", type=int, default=_DEFAULT_CONFIG.train_cpu_sampler_workers)
    parser.add_argument("--train-cpu-prefetch-batches", type=int, default=_DEFAULT_CONFIG.train_cpu_prefetch_batches)
    parser.add_argument("--train-shard-cache-size", type=int, default=_DEFAULT_CONFIG.train_shard_cache_size)
    parser.add_argument(
        "--min-selfplay-quality-for-training",
        choices=["stuck", "warn", "ok"],
        default=_DEFAULT_CONFIG.min_selfplay_quality_for_training,
    )
    parser.add_argument("--snapshot-keep-latest-count", type=int, default=_DEFAULT_CONFIG.snapshot_keep_latest_count)
    parser.add_argument("--snapshot-keep-best-count", type=int, default=_DEFAULT_CONFIG.snapshot_keep_best_count)

    parser.add_argument("--arena-games", type=int, default=_DEFAULT_CONFIG.arena_games)
    parser.add_argument("--arena-games-per-opening", type=int, default=_DEFAULT_CONFIG.arena_games_per_opening)
    parser.add_argument("--arena-sims", type=int, default=_DEFAULT_CONFIG.arena_sims)
    parser.add_argument("--arena-every-n-cycles", type=int, default=_DEFAULT_CONFIG.arena_every_n_cycles,
                        help="Run the promotion arena only every N cycles (cheap-gate cadence). Default 1 = every cycle.")
    parser.add_argument("--rl-anchor-checkpoint", default=_DEFAULT_CONFIG.rl_anchor_checkpoint,
                        help="B-experiment: frozen reference checkpoint (geo) for the policy anchor.")
    parser.add_argument("--rl-anchor-policy-kl-weight", type=float, default=_DEFAULT_CONFIG.rl_anchor_policy_kl_weight,
                        help="B-experiment: weight of KL(anchor||student) policy anchor. 0 = off.")
    parser.add_argument("--rl-anchor-anneal-steps", type=int, default=_DEFAULT_CONFIG.rl_anchor_anneal_steps,
                        help="B-experiment: 0 = constant anchor weight; >0 = linearly anneal to 0 over N steps.")
    parser.add_argument("--rl-frozen-value-checkpoint", default=_DEFAULT_CONFIG.rl_frozen_value_checkpoint,
                        help="v16 frozen-evaluator mode: self-play AND gates run policy/value chimeras with "
                             "value from this frozen ckpt (e.g. geo); trainer becomes policy-only "
                             "(value/wdl weight 0) and resets optimizer on resume (true 1e-4).")
    parser.add_argument("--arena-accept-threshold", type=float, default=_DEFAULT_CONFIG.arena_accept_threshold)
    parser.add_argument("--arena-min-non-draw-games", type=int, default=_DEFAULT_CONFIG.arena_min_non_draw_games)
    parser.add_argument("--arena-log-every-games", type=int, default=_DEFAULT_CONFIG.arena_log_every_games)
    parser.add_argument("--arena-max-plies", type=int, default=_DEFAULT_CONFIG.arena_max_plies)
    parser.add_argument("--arena-repeat-limit", type=int, default=_DEFAULT_CONFIG.arena_repeat_limit)
    parser.add_argument("--arena-repeat-min-ply", type=int, default=_DEFAULT_CONFIG.arena_repeat_min_ply)
    parser.add_argument("--arena-no-capture-limit", type=int, default=_DEFAULT_CONFIG.arena_no_capture_limit)
    parser.add_argument("--arena-temperature-move", type=float, default=_DEFAULT_CONFIG.arena_temperature_move)
    parser.add_argument("--arena-dirichlet-alpha", type=float, default=_DEFAULT_CONFIG.arena_dirichlet_alpha)
    parser.add_argument("--arena-dirichlet-eps", type=float, default=_DEFAULT_CONFIG.arena_dirichlet_eps)
    parser.add_argument("--arena-opening-suite-path", default=_DEFAULT_CONFIG.arena_opening_suite_path)
    parser.add_argument(
        "--enable-arena",
        dest="arena_enabled",
        action="store_true",
        default=_DEFAULT_CONFIG.arena_enabled,
    )
    parser.add_argument("--disable-arena", dest="arena_enabled", action="store_false")
    parser.add_argument("--disable-arena-promote-on-pass", action="store_true")
    parser.add_argument("--disable-arena-gated-best-update", action="store_true")
    args = parser.parse_args()

    return ClosedLoopConfig(
        human_data_dir=args.human_data_dir,
        training_output_dir=args.training_output_dir,
        selfplay_output_root=args.selfplay_output_root,
        arena_output_root=args.arena_output_root,
        device=args.device,
        pause_at_local_time=args.pause_at_local_time,
        selfplay_checkpoint_source=args.selfplay_checkpoint_source,
        execution_mode=args.execution_mode,
        cycles=args.cycles,
        sleep_between_cycles_s=args.sleep_between_cycles_s,
        seed=args.seed,
        selfplay_target_samples_per_cycle=args.selfplay_target_samples_per_cycle,
        selfplay_num_workers=args.selfplay_num_workers,
        selfplay_num_simulations=args.selfplay_num_simulations,
        selfplay_max_states_per_batch=args.selfplay_max_states_per_batch,
        selfplay_eval_batch_size=args.selfplay_eval_batch_size,
        selfplay_progress_log_games=args.selfplay_progress_log_games,
        selfplay_progress_log_shards=args.selfplay_progress_log_shards,
        selfplay_human_position_mix_ratio=args.selfplay_human_position_mix_ratio,
        selfplay_opening_like_material_min_ratio=args.selfplay_opening_like_material_min_ratio,
        selfplay_opening_like_min_piece_count=args.selfplay_opening_like_min_piece_count,
        selfplay_max_plies=args.selfplay_max_plies,
        selfplay_repeat_limit=args.selfplay_repeat_limit,
        selfplay_repeat_min_ply=args.selfplay_repeat_min_ply,
        selfplay_no_capture_limit=args.selfplay_no_capture_limit,
        selfplay_immediate_draw_candidate_scan=args.selfplay_immediate_draw_candidate_scan,
        selfplay_immediate_draw_capture_injection_threshold=args.selfplay_immediate_draw_capture_injection_threshold,
        selfplay_immediate_draw_capture_priority_threshold=args.selfplay_immediate_draw_capture_priority_threshold,
        selfplay_immediate_draw_nocap_pressure_threshold=args.selfplay_immediate_draw_nocap_pressure_threshold,
        selfplay_immediate_draw_nocap_pressure_scan=args.selfplay_immediate_draw_nocap_pressure_scan,
        selfplay_nocap_warn_threshold=args.selfplay_nocap_warn_threshold,
        selfplay_nocap_stuck_threshold=args.selfplay_nocap_stuck_threshold,
        overlap_selfplay_num_workers=args.overlap_selfplay_num_workers,
        overlap_keep_free_cpu_cores=args.overlap_keep_free_cpu_cores,
        train_steps_per_cycle=args.train_steps_per_cycle,
        skip_train=bool(args.skip_train),
        train_lr_schedule_max_steps=args.train_lr_schedule_max_steps,
        train_bootstrap_human_floor=args.train_bootstrap_human_floor,
        train_reset_selfplay_ingest_state_on_resume=bool(args.train_reset_selfplay_ingest_state_on_resume),
        train_micro_batch_size=args.train_micro_batch_size,
        train_grad_accum_steps=args.train_grad_accum_steps,
        train_cpu_sampler_workers=args.train_cpu_sampler_workers,
        train_cpu_prefetch_batches=args.train_cpu_prefetch_batches,
        train_shard_cache_size=args.train_shard_cache_size,
        min_selfplay_quality_for_training=args.min_selfplay_quality_for_training,
        snapshot_keep_latest_count=args.snapshot_keep_latest_count,
        snapshot_keep_best_count=args.snapshot_keep_best_count,
        arena_games=args.arena_games,
        arena_games_per_opening=args.arena_games_per_opening,
        arena_sims=args.arena_sims,
        arena_every_n_cycles=args.arena_every_n_cycles,
        rl_anchor_checkpoint=args.rl_anchor_checkpoint,
        rl_anchor_policy_kl_weight=args.rl_anchor_policy_kl_weight,
        rl_anchor_anneal_steps=args.rl_anchor_anneal_steps,
        rl_frozen_value_checkpoint=args.rl_frozen_value_checkpoint,
        arena_accept_threshold=args.arena_accept_threshold,
        arena_min_non_draw_games=args.arena_min_non_draw_games,
        arena_log_every_games=args.arena_log_every_games,
        arena_max_plies=args.arena_max_plies,
        arena_repeat_limit=args.arena_repeat_limit,
        arena_repeat_min_ply=args.arena_repeat_min_ply,
        arena_no_capture_limit=args.arena_no_capture_limit,
        arena_temperature_move=args.arena_temperature_move,
        arena_dirichlet_alpha=args.arena_dirichlet_alpha,
        arena_dirichlet_eps=args.arena_dirichlet_eps,
        arena_opening_suite_path=args.arena_opening_suite_path,
        arena_enabled=bool(args.arena_enabled),
        arena_promote_on_pass=not bool(args.disable_arena_promote_on_pass),
        arena_gates_best_update=not bool(args.disable_arena_gated_best_update),
    )


def main() -> None:
    config = _parse_args()
    print(
        f"[LOOP] config: cycles={config.cycles or 'infinite'} "
        f"mode={config.execution_mode} "
        f"selfplay_samples={config.selfplay_target_samples_per_cycle} "
        f"train_steps={config.train_steps_per_cycle} "
        f"train_mode={'probe' if _training_probe_mode(config) else 'train'} "
        f"train_lr_schedule_max_steps={config.train_lr_schedule_max_steps} "
        f"min_train_quality={config.min_selfplay_quality_for_training} "
        f"selfplay_ckpt={config.selfplay_checkpoint_source} "
        f"overlap_selfplay_workers={_configured_overlap_selfplay_workers(config)} "
        f"best_update={'arena' if config.arena_enabled and config.arena_gates_best_update else 'human_val'} "
        f"pause_at={config.pause_at_local_time or 'off'} "
        f"arena={'on' if config.arena_enabled else 'off'} "
        f"arena_games={config.arena_games} "
        f"arena_games_per_opening={config.arena_games_per_opening} "
        f"arena_suite={Path(config.arena_opening_suite_path).name if config.arena_opening_suite_path else 'standard'} "
        f"device={config.device}",
        flush=True,
    )
    summary = run_closed_loop(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
