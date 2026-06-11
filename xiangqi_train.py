from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import queue
import random
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections import Counter, OrderedDict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from xiangqi_transformer_model import (
    XiangqiPVTransformer,
    XiangqiTransformerConfig,
    build_model_from_checkpoint_state,
    load_xiangqi_model_state_dict,
)


@dataclass
class TrainingConfig:
    human_data_dir: str | Path = "human_bootstrap_data_elite_wdl"
    selfplay_dirs: list[str | Path] = field(default_factory=lambda: ["selfplay_runs_bootstrap"])
    output_dir: str | Path = "training_runs/run_001"
    resume_path: str | Path | None = None
    reset_optimizer_on_resume: bool = False
    device: str = "cuda:0"

    replay_buffer_size: int = 300_000
    poll_interval_s: float = 2.0
    shard_cache_size: int = 16
    bootstrap_mode: bool = True
    bootstrap_human_floor: float = 0.20
    selfplay_run_quality_gate: bool = True
    selfplay_run_max_rep_draw_rate: float = 60.0
    selfplay_run_min_decisive_rate: float = 25.0
    reset_selfplay_ingest_state_on_resume: bool = False
    # Optional per-selfplay-dir sampling ratios.  When provided, self-play
    # samples are drawn from each configured directory according to these
    # ratios instead of pooled purely by shard size.  This is useful for small
    # tactical/refutation slices that should act as a regularizer rather than
    # dominate a finetune by oversampling.
    selfplay_dir_sampling_ratios: list[float] = field(default_factory=list)

    micro_batch_size: int = 1024
    grad_accum_steps: int = 1
    eval_interval_steps: int = 1000
    save_interval_steps: int = 2000
    # When > 0, also persist a numbered snapshot to `snapshots/latest_step<N>.pt`
    # at the same cadence.  Crucial for never losing a peak checkpoint to a later
    # regression — `latest.pt` is overwritten on every save.  0 = disabled.
    snapshot_interval_steps: int = 0
    log_interval_steps: int = 100
    max_steps: int = 200_000
    lr_schedule_max_steps: int | None = None
    samples_per_unit: int = 256
    cpu_sampler_workers: int = 16       # was 12 — tuned for 7970X 32-core, keeps GPU fed
    cpu_prefetch_batches: int = 16      # was 8 — deeper queue so GPU never starves on slow shards
    cpu_reserved_cores: int = 2         # was 4 — free up cores for selfplay's parallel games
    cpu_sampler_backend: str = "thread"
    run_detached: bool = True
    save_on_interrupt: bool = True
    promote_best_on_human_val: bool = True
    pause_at_local_time: str | None = None

    learning_rate: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    warmup_steps: int = 2000

    wdl_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 0.5
    wdl_value_consistency_weight: float = 0.0
    value_target_scale: float = 0.9
    # When True, samples that have a finite oracle_value (from oracle_value_labeler.py)
    # use it directly as the value-head target instead of scaled_z.  Samples without
    # oracle (NaN) still use scaled_z as fallback.  See _compute_training_losses.
    use_oracle_value: bool = True

    # v11: combine MCTS-visit policy target with Pikafish-multipv-derived oracle policy.
    # alpha=0 -> v10 behavior (MCTS only). alpha=0.5 -> equal blend. alpha=1 -> oracle only.
    # Per-sample empty oracle slots fall back to MCTS automatically (loss term = 0).
    policy_oracle_alpha: float = 0.0
    # v12.5: optional action-value distillation target.  Shards may carry
    # teacher_q_{offsets,idxs,values}, where values are centipawns from the
    # root side-to-move's perspective after trying each candidate action.
    teacher_q_loss_weight: float = 0.0
    teacher_q_temperature_cp: float = 80.0
    teacher_q_pairwise_loss_weight: float = 0.0
    teacher_q_pairwise_margin_logit: float = 0.25
    teacher_q_pairwise_min_gap_cp: float = 80.0
    teacher_q_pairwise_beta: float = 1.0
    teacher_q_pairwise_use_anchor_reference: bool = False
    teacher_q_pairwise_bad_move_only: bool = False
    bad_move_suppression_loss_weight: float = 0.0
    bad_move_suppression_margin_logit: float = 0.75
    bad_move_suppression_min_gap_cp: float = 80.0
    bad_move_suppression_beta: float = 2.0
    anchor_checkpoint: str | Path | None = None
    anchor_policy_kl_weight: float = 0.0
    anchor_policy_top1_ce_weight: float = 0.0
    anchor_value_mse_weight: float = 0.0
    anchor_anneal_steps: int = 0

    use_bfloat16: bool = True
    allow_tf32: bool = True
    cudnn_benchmark: bool = True
    seed: int = 0

    model_config: XiangqiTransformerConfig = field(default_factory=XiangqiTransformerConfig)
    train_only_relative_attention_bias: bool = False
    train_only_policy_head: bool = False
    train_only_value_head: bool = False
    train_only_cnn_local_adapter: bool = False
    train_only_cnn_policy_residual_adapter: bool = False
    adapter_unfreeze_last_n_blocks: int = 0

    def __post_init__(self) -> None:
        if self.replay_buffer_size < 1:
            raise ValueError("replay_buffer_size must be >= 1")
        if self.poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be > 0")
        if self.shard_cache_size < 1:
            raise ValueError("shard_cache_size must be >= 1")
        if not (0.0 <= self.bootstrap_human_floor <= 1.0):
            raise ValueError("bootstrap_human_floor must be within [0, 1]")
        if self.selfplay_run_max_rep_draw_rate < 0.0:
            raise ValueError("selfplay_run_max_rep_draw_rate must be >= 0")
        if self.selfplay_run_min_decisive_rate < 0.0:
            raise ValueError("selfplay_run_min_decisive_rate must be >= 0")
        if any(float(ratio) < 0.0 for ratio in self.selfplay_dir_sampling_ratios):
            raise ValueError("selfplay_dir_sampling_ratios must be non-negative")
        if self.micro_batch_size < 1:
            raise ValueError("micro_batch_size must be >= 1")
        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")
        if self.eval_interval_steps < 1:
            raise ValueError("eval_interval_steps must be >= 1")
        if self.save_interval_steps < 1:
            raise ValueError("save_interval_steps must be >= 1")
        if self.snapshot_interval_steps < 0:
            raise ValueError("snapshot_interval_steps must be >= 0 (0 disables)")
        if self.log_interval_steps < 1:
            raise ValueError("log_interval_steps must be >= 1")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.lr_schedule_max_steps is not None and self.lr_schedule_max_steps < 1:
            raise ValueError("lr_schedule_max_steps must be >= 1 when provided")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.wdl_loss_weight < 0:
            raise ValueError("wdl_loss_weight must be >= 0")
        if self.policy_loss_weight < 0:
            raise ValueError("policy_loss_weight must be >= 0")
        if self.value_loss_weight < 0:
            raise ValueError("value_loss_weight must be >= 0")
        if self.adapter_unfreeze_last_n_blocks < 0:
            raise ValueError("adapter_unfreeze_last_n_blocks must be >= 0")
        if self.wdl_value_consistency_weight < 0:
            raise ValueError("wdl_value_consistency_weight must be >= 0")
        if self.teacher_q_loss_weight < 0:
            raise ValueError("teacher_q_loss_weight must be >= 0")
        if self.teacher_q_temperature_cp <= 0:
            raise ValueError("teacher_q_temperature_cp must be > 0")
        if self.teacher_q_pairwise_loss_weight < 0:
            raise ValueError("teacher_q_pairwise_loss_weight must be >= 0")
        if self.teacher_q_pairwise_margin_logit < 0:
            raise ValueError("teacher_q_pairwise_margin_logit must be >= 0")
        if self.teacher_q_pairwise_min_gap_cp < 0:
            raise ValueError("teacher_q_pairwise_min_gap_cp must be >= 0")
        if self.teacher_q_pairwise_beta <= 0:
            raise ValueError("teacher_q_pairwise_beta must be > 0")
        if (
            self.teacher_q_pairwise_use_anchor_reference
            and self.teacher_q_pairwise_loss_weight > 0.0
            and self.anchor_checkpoint is None
        ):
            raise ValueError(
                "teacher_q_pairwise_use_anchor_reference requires --anchor-checkpoint "
                "when teacher_q_pairwise_loss_weight > 0"
            )
        if self.bad_move_suppression_loss_weight < 0:
            raise ValueError("bad_move_suppression_loss_weight must be >= 0")
        if self.bad_move_suppression_margin_logit < 0:
            raise ValueError("bad_move_suppression_margin_logit must be >= 0")
        if self.bad_move_suppression_min_gap_cp < 0:
            raise ValueError("bad_move_suppression_min_gap_cp must be >= 0")
        if self.bad_move_suppression_beta <= 0:
            raise ValueError("bad_move_suppression_beta must be > 0")
        if self.bad_move_suppression_loss_weight > 0.0 and self.anchor_checkpoint is None:
            raise ValueError(
                "bad_move_suppression_loss_weight > 0 requires --anchor-checkpoint "
                "so the bad move can be suppressed relative to a frozen reference model"
            )
        if self.anchor_policy_kl_weight < 0:
            raise ValueError("anchor_policy_kl_weight must be >= 0")
        if self.anchor_policy_top1_ce_weight < 0:
            raise ValueError("anchor_policy_top1_ce_weight must be >= 0")
        if self.anchor_value_mse_weight < 0:
            raise ValueError("anchor_value_mse_weight must be >= 0")
        if self.anchor_anneal_steps < 0:
            raise ValueError("anchor_anneal_steps must be >= 0")
        if not (0.0 < self.value_target_scale <= 1.0):
            raise ValueError("value_target_scale must be within (0, 1]")
        if self.samples_per_unit < 1:
            raise ValueError("samples_per_unit must be >= 1")
        if self.cpu_sampler_workers < 0:
            raise ValueError("cpu_sampler_workers must be >= 0")
        if self.cpu_prefetch_batches < 1:
            raise ValueError("cpu_prefetch_batches must be >= 1")
        if self.cpu_reserved_cores < 0:
            raise ValueError("cpu_reserved_cores must be >= 0")
        if self.cpu_sampler_backend not in {"auto", "process", "thread", "none"}:
            raise ValueError("cpu_sampler_backend must be one of: auto, process, thread, none")
        _parse_pause_local_time(self.pause_at_local_time)

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


_DEFAULT_LAUNCH_CONFIG = TrainingConfig()


@dataclass(frozen=True)
class _HumanShardSpec:
    path: Path
    sample_count: int


@dataclass(frozen=True)
class _SelfPlaySpan:
    path: str
    start: int
    end: int
    source_group: str = "0"

    @property
    def sample_count(self) -> int:
        return self.end - self.start


class _GracefulStopRequested(Exception):
    pass


class _ShardCache:
    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, path: Path) -> Any:
        key = str(path.resolve())
        with self._lock:
            if key in self._cache:
                value = self._cache.pop(key)
                self._cache[key] = value
                return value

            value = torch.load(path, map_location="cpu", weights_only=False)
            self._cache[key] = value
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)
            return value

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


def _human_sampler_cache_budget_per_worker(config: TrainingConfig, backend: str) -> int:
    if config.shard_cache_size < 1:
        return 1
    if backend == "process":
        workers = max(config.cpu_sampler_workers, 1)
        return max(1, math.ceil(config.shard_cache_size / float(workers)))
    return config.shard_cache_size


def _effective_lr_schedule_max_steps(config: TrainingConfig) -> int:
    if config.lr_schedule_max_steps is None:
        return config.max_steps
    return max(config.lr_schedule_max_steps, config.max_steps)


def _format_ingest_reason_counts(reason_counts: dict[str, int]) -> str:
    if not reason_counts:
        return "-"
    return ",".join(f"{reason}:{count}" for reason, count in sorted(reason_counts.items()))


class _SelfPlayReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.spans: deque[_SelfPlaySpan] = deque()
        self.sample_count = 0

    def add_shard(self, path: Path, shard_sample_count: int, source_group: str = "0") -> int:
        if shard_sample_count <= 0:
            return 0

        self.spans.append(
            _SelfPlaySpan(
                path=str(path.resolve()),
                start=0,
                end=shard_sample_count,
                source_group=str(source_group),
            )
        )
        self.sample_count += shard_sample_count
        self._trim_left()
        return shard_sample_count

    def _trim_left(self) -> None:
        while self.sample_count > self.capacity and self.spans:
            overflow = self.sample_count - self.capacity
            span = self.spans[0]
            if overflow >= span.sample_count:
                self.sample_count -= span.sample_count
                self.spans.popleft()
                continue

            trimmed = _SelfPlaySpan(
                path=span.path,
                start=span.start + overflow,
                end=span.end,
                source_group=span.source_group,
            )
            self.sample_count -= overflow
            self.spans[0] = trimmed

    def restore(self, state: list[dict[str, Any]]) -> None:
        self.spans.clear()
        self.sample_count = 0
        for item in state:
            span = _SelfPlaySpan(
                path=str(item["path"]),
                start=int(item["start"]),
                end=int(item["end"]),
                source_group=str(item.get("source_group", item.get("source", "0"))),
            )
            if span.sample_count <= 0:
                continue
            self.spans.append(span)
            self.sample_count += span.sample_count
        self._trim_left()

    def retain(self, keep_fn: Any) -> tuple[int, int]:
        kept: deque[_SelfPlaySpan] = deque()
        removed_samples = 0
        removed_spans = 0
        for span in self.spans:
            if keep_fn(span):
                kept.append(span)
            else:
                removed_samples += span.sample_count
                removed_spans += 1
        self.spans = kept
        self.sample_count = sum(span.sample_count for span in self.spans)
        self._trim_left()
        return removed_spans, removed_samples

    def to_state(self) -> list[dict[str, Any]]:
        return [
            {
                "path": span.path,
                "start": span.start,
                "end": span.end,
                "source_group": span.source_group,
            }
            for span in self.spans
        ]

    def fill_ratio(self) -> float:
        return min(self.sample_count / float(self.capacity), 1.0)

    def __len__(self) -> int:
        return self.sample_count


class _HumanTrainCatalog:
    def __init__(self, train_specs: list[_HumanShardSpec], val_specs: list[_HumanShardSpec]) -> None:
        self.train_specs = train_specs
        self.val_specs = val_specs
        self.train_weights = [spec.sample_count for spec in train_specs]
        self.val_weights = [spec.sample_count for spec in val_specs]
        self.train_total_samples = int(sum(self.train_weights))
        self.val_total_samples = int(sum(self.val_weights))
        if not self.train_specs:
            raise RuntimeError("human train shard list is empty")
        if not self.val_specs:
            raise RuntimeError("human val shard list is empty")

    @classmethod
    def from_dir(cls, data_dir: Path) -> _HumanTrainCatalog:
        manifest_path = data_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"human data manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        train_specs = _parse_manifest_split(data_dir, manifest, split="train")
        val_specs = _parse_manifest_split(data_dir, manifest, split="val")
        return cls(train_specs=train_specs, val_specs=val_specs)


class _SelfPlayIngestor:
    def __init__(
        self,
        directories: list[Path],
        seen_registry: dict[str, dict[str, Any]],
        quality_gate: bool = False,
        max_rep_draw_rate: float = 100.0,
        min_decisive_rate: float = 0.0,
        bootstrap_soft_gate: bool = False,
        source_groups: list[str] | None = None,
    ) -> None:
        self.directories = directories
        self.seen_registry = seen_registry
        if source_groups is None:
            source_groups = [str(index) for index in range(len(directories))]
        if len(source_groups) != len(directories):
            raise ValueError("source_groups length must match selfplay directories")
        self.source_groups = [str(group) for group in source_groups]
        self.quality_gate = bool(quality_gate)
        self.max_rep_draw_rate = float(max_rep_draw_rate)
        self.min_decisive_rate = float(min_decisive_rate)
        self.bootstrap_soft_gate = bool(bootstrap_soft_gate)
        self._stability: dict[str, tuple[tuple[int, int], int]] = {}
        self._run_gate_cache: dict[str, tuple[tuple[int, int], tuple[bool, str | None, str | None]]] = {}
        self._logged_run_gate_state: dict[str, tuple[str | None, tuple[int, int] | None]] = {}

    def _load_run_manifest(
        self, run_root: Path
    ) -> tuple[dict[str, Any] | None, tuple[int, int] | None, str | None]:
        manifest_path = run_root / "manifest.json"
        if not manifest_path.is_file():
            return None, None, "manifest_missing"

        try:
            stat = manifest_path.stat()
            fingerprint = (int(stat.st_size), int(stat.st_mtime_ns))
        except OSError:
            return None, None, "manifest_unreadable"

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return None, fingerprint, "manifest_unreadable"
        return manifest, fingerprint, None

    def _evaluate_run_root(self, run_root: Path) -> tuple[bool, str | None, str | None]:
        if not self.quality_gate:
            return True, None, None
        run_root = run_root.resolve()
        cache_key = str(run_root)
        manifest, fingerprint, manifest_error = self._load_run_manifest(run_root)
        if manifest_error is not None:
            self._run_gate_cache.pop(cache_key, None)
            return False, manifest_error, None

        assert manifest is not None
        assert fingerprint is not None
        manifest_state = str(manifest.get("manifest_state", "complete")).strip().lower()
        if manifest_state != "complete":
            self._run_gate_cache.pop(cache_key, None)
            return False, "manifest_in_progress", None
        cached = self._run_gate_cache.get(cache_key)
        if cached is not None and cached[0] == fingerprint:
            return cached[1]

        quality = manifest.get("quality")
        quality_metrics = manifest.get("quality_metrics", {})
        rep_draw_rate = float(quality_metrics.get("rep_draw_rate", 0.0))
        decisive_rate = float(quality_metrics.get("decisive_rate", 0.0))
        nocap_draw_rate = float(quality_metrics.get("nocap_draw_rate", 0.0))
        manifest_config = manifest.get("config", {})
        search_defaults = manifest_config.get("search_defaults", {}) if isinstance(manifest_config, dict) else {}

        def _manifest_threshold(key: str) -> float | None:
            raw_value = search_defaults.get(key)
            if raw_value is None:
                return None
            try:
                return float(raw_value)
            except (TypeError, ValueError):
                return None

        warn_nocap_threshold = _manifest_threshold("bootstrap_quality_warn_nocap_draw_threshold")
        stuck_nocap_threshold = _manifest_threshold("bootstrap_quality_nocap_draw_threshold")
        nocap_warn_triggered = warn_nocap_threshold is not None and nocap_draw_rate >= warn_nocap_threshold
        nocap_stuck_triggered = stuck_nocap_threshold is not None and nocap_draw_rate >= stuck_nocap_threshold

        allowed = True
        reason: str | None = None
        soft_allowed_reason: str | None = None
        if rep_draw_rate > self.max_rep_draw_rate:
            allowed = False
            reason = "rep_draw_too_high"
        elif quality == "stuck" and nocap_stuck_triggered:
            if self.bootstrap_soft_gate:
                soft_allowed_reason = "nocap_draw_too_high"
            else:
                allowed = False
                reason = "nocap_draw_too_high"
        elif nocap_warn_triggered:
            if self.bootstrap_soft_gate:
                soft_allowed_reason = "nocap_draw_too_high"
            else:
                allowed = False
                reason = "nocap_draw_too_high"
        elif decisive_rate < self.min_decisive_rate:
            if self.bootstrap_soft_gate:
                soft_allowed_reason = "decisive_rate_too_low"
            else:
                allowed = False
                reason = "decisive_rate_too_low"
        elif quality == "stuck":
            if self.bootstrap_soft_gate:
                soft_allowed_reason = "quality_stuck"
            else:
                allowed = False
                reason = "quality_stuck"

        result = (allowed, reason, soft_allowed_reason)
        self._run_gate_cache[cache_key] = (fingerprint, result)
        return result

    def _train_dir_status(self, train_dir: Path) -> dict[str, Any]:
        run_root = train_dir.parent if train_dir.name == "train" else train_dir
        allowed, reason, soft_allowed_reason = self._evaluate_run_root(run_root)
        cache_key = str(run_root.resolve())
        manifest_error: str | None = None
        fingerprint: tuple[int, int] | None = None
        if not allowed:
            manifest, fingerprint, manifest_error = self._load_run_manifest(run_root)
            log_state = (f"skip:{reason}", fingerprint)
            if self._logged_run_gate_state.get(cache_key) != log_state:
                self._logged_run_gate_state[cache_key] = log_state
                if manifest_error == "manifest_missing":
                    print(
                        f"waiting for self-play run manifest before ingesting: {run_root}",
                        flush=True,
                    )
                    return {
                        "allowed": allowed,
                        "reason": reason,
                        "soft_allowed_reason": None,
                        "manifest_error": manifest_error,
                        "run_root": run_root,
                    }
                if manifest_error == "manifest_unreadable":
                    print(
                        f"waiting for readable self-play manifest before ingesting: {run_root}",
                        flush=True,
                    )
                    return {
                        "allowed": allowed,
                        "reason": reason,
                        "soft_allowed_reason": None,
                        "manifest_error": manifest_error,
                        "run_root": run_root,
                    }
                if reason == "manifest_in_progress":
                    print(
                        f"waiting for self-play run to complete before ingesting: {run_root}",
                        flush=True,
                    )
                    return {
                        "allowed": allowed,
                        "reason": reason,
                        "soft_allowed_reason": None,
                        "manifest_error": manifest_error,
                        "run_root": run_root,
                    }

                manifest = manifest or {}
                quality = manifest.get("quality")
                quality_metrics = manifest.get("quality_metrics", {})
                rep_draw_rate = float(quality_metrics.get("rep_draw_rate", 0.0))
                decisive_rate = float(quality_metrics.get("decisive_rate", 0.0))
                nocap_draw_rate = float(quality_metrics.get("nocap_draw_rate", 0.0))
                print(
                    "skipping self-play run due to quality gate: "
                    f"{run_root} reason={reason} quality={quality} "
                    f"rep_draw={rep_draw_rate:.1f}% decisive={decisive_rate:.1f}% "
                    f"nocap_draw={nocap_draw_rate:.1f}%",
                    flush=True,
                )
            return {
                "allowed": allowed,
                "reason": reason,
                "soft_allowed_reason": None,
                "manifest_error": manifest_error,
                "run_root": run_root,
            }
        if soft_allowed_reason is not None:
            manifest, fingerprint, manifest_error = self._load_run_manifest(run_root)
            log_state = (f"soft:{soft_allowed_reason}", fingerprint)
            if self._logged_run_gate_state.get(cache_key) != log_state:
                self._logged_run_gate_state[cache_key] = log_state
                if manifest_error in {"manifest_missing", "manifest_unreadable"}:
                    print(
                        "allowing self-play run through bootstrap soft gate: "
                        f"{run_root} soft_reason={soft_allowed_reason} quality=pending "
                        "rep_draw=n/a decisive=n/a nocap_draw=n/a",
                        flush=True,
                    )
                else:
                    manifest = manifest or {}
                    quality = manifest.get("quality")
                    quality_metrics = manifest.get("quality_metrics", {})
                    rep_draw_rate = float(quality_metrics.get("rep_draw_rate", 0.0))
                    decisive_rate = float(quality_metrics.get("decisive_rate", 0.0))
                    nocap_draw_rate = float(quality_metrics.get("nocap_draw_rate", 0.0))
                    print(
                        "allowing self-play run through bootstrap soft gate: "
                        f"{run_root} soft_reason={soft_allowed_reason} quality={quality} "
                        f"rep_draw={rep_draw_rate:.1f}% decisive={decisive_rate:.1f}% "
                        f"nocap_draw={nocap_draw_rate:.1f}%",
                        flush=True,
                    )
        else:
            self._logged_run_gate_state.pop(cache_key, None)
        return {
            "allowed": allowed,
            "reason": reason,
            "soft_allowed_reason": soft_allowed_reason,
            "manifest_error": manifest_error,
            "run_root": run_root,
        }

    def prune_replay_buffer(self, replay_buffer: _SelfPlayReplayBuffer) -> tuple[int, int]:
        def _keep(span: _SelfPlaySpan) -> bool:
            shard_path = Path(span.path).resolve()
            run_root = shard_path.parent.parent if shard_path.parent.name == "train" else shard_path.parent
            allowed, _reason, _soft_allowed_reason = self._evaluate_run_root(run_root)
            return allowed

        removed_spans, removed_samples = replay_buffer.retain(_keep)
        if removed_spans > 0:
            print(
                "pruned replay buffer after quality gate update: "
                f"removed_spans={removed_spans} removed_samples={removed_samples} "
                f"remaining_samples={len(replay_buffer)}",
                flush=True,
            )
        return removed_spans, removed_samples

    def poll(self, cache: _ShardCache, replay_buffer: _SelfPlayReplayBuffer) -> dict[str, Any]:
        added_shards = 0
        added_samples = 0
        waiting_manifest_runs = 0
        waiting_unreadable_manifest_runs = 0
        waiting_in_progress_runs = 0
        eligible_runs = 0
        skipped_runs = 0
        soft_allowed_runs = 0
        pruned_spans = 0
        pruned_samples = 0
        skipped_run_reasons: dict[str, int] = {}
        soft_allowed_run_reasons: dict[str, int] = {}
        live_paths: set[str] = set()

        for root_index, root in enumerate(self.directories):
            source_group = self.source_groups[root_index]
            for train_dir in _iter_selfplay_train_dirs(root):
                if not train_dir.exists():
                    continue
                status = self._train_dir_status(train_dir)
                if not bool(status["allowed"]):
                    manifest_error = status.get("manifest_error")
                    if manifest_error == "manifest_missing":
                        waiting_manifest_runs += 1
                    elif manifest_error == "manifest_unreadable":
                        waiting_unreadable_manifest_runs += 1
                    elif status.get("reason") == "manifest_in_progress":
                        waiting_in_progress_runs += 1
                    else:
                        skipped_runs += 1
                        reason = str(status.get("reason") or "unknown")
                        skipped_run_reasons[reason] = skipped_run_reasons.get(reason, 0) + 1
                    continue
                eligible_runs += 1
                soft_allowed_reason = status.get("soft_allowed_reason")
                if soft_allowed_reason:
                    soft_allowed_runs += 1
                    soft_allowed_key = str(soft_allowed_reason)
                    soft_allowed_run_reasons[soft_allowed_key] = soft_allowed_run_reasons.get(soft_allowed_key, 0) + 1
                for shard_path in sorted(train_dir.glob("shard_*.pt")):
                    resolved = shard_path.resolve()
                    key = str(resolved)
                    live_paths.add(key)
                    if key in self.seen_registry:
                        continue

                    stat = resolved.stat()
                    fingerprint = (int(stat.st_size), int(stat.st_mtime_ns))
                    previous = self._stability.get(key)
                    if previous is not None and previous[0] == fingerprint:
                        stable_count = previous[1] + 1
                    else:
                        stable_count = 1
                    self._stability[key] = (fingerprint, stable_count)

                    if stable_count < 2:
                        continue

                    try:
                        shard = cache.get(resolved)
                        sample_count = _get_selfplay_shard_sample_count(shard, resolved)
                    except Exception:
                        continue

                    replay_buffer.add_shard(resolved, sample_count, source_group=source_group)
                    self.seen_registry[key] = {
                        "size": fingerprint[0],
                        "mtime_ns": fingerprint[1],
                        "samples": sample_count,
                        "source_group": source_group,
                    }
                    added_shards += 1
                    added_samples += sample_count

        stale_keys = [key for key in self._stability if key not in live_paths and key not in self.seen_registry]
        for key in stale_keys:
            self._stability.pop(key, None)

        if self.quality_gate:
            pruned_spans, pruned_samples = self.prune_replay_buffer(replay_buffer)

        return {
            "added_shards": added_shards,
            "added_samples": added_samples,
            "waiting_manifest_runs": waiting_manifest_runs,
            "waiting_unreadable_manifest_runs": waiting_unreadable_manifest_runs,
            "waiting_in_progress_runs": waiting_in_progress_runs,
            "eligible_runs": eligible_runs,
            "skipped_runs": skipped_runs,
            "soft_allowed_runs": soft_allowed_runs,
            "pruned_spans": pruned_spans,
            "pruned_samples": pruned_samples,
            "skipped_run_reasons": dict(sorted(skipped_run_reasons.items())),
            "soft_allowed_run_reasons": dict(sorted(soft_allowed_run_reasons.items())),
        }


class _HumanBatchWorkerPool:
    def __init__(self, config: TrainingConfig) -> None:
        self.batch_size = config.micro_batch_size
        self.prefetch_batches = config.cpu_prefetch_batches
        self._ctx = mp.get_context("spawn")
        self._stop_event = self._ctx.Event()
        self._queue: mp.Queue = self._ctx.Queue(maxsize=config.cpu_prefetch_batches)
        self._processes: list[mp.Process] = []

        cpu_groups = _plan_worker_cpu_groups(
            num_workers=config.cpu_sampler_workers,
            reserved_cores=config.cpu_reserved_cores,
        )
        per_worker_cache_size = _human_sampler_cache_budget_per_worker(config, backend="process")
        worker_payload = {
            "human_data_dir": str(config.human_data_dir),
            "shard_cache_size": int(per_worker_cache_size),
            "batch_size": int(config.micro_batch_size),
            "samples_per_unit": int(config.samples_per_unit),
            "seed": int(config.seed),
        }

        for worker_id, cpu_ids in enumerate(cpu_groups):
            process = self._ctx.Process(
                target=_human_batch_prefetch_worker,
                args=(worker_id, worker_payload, cpu_ids, self._queue, self._stop_event),
                name=f"human-batch-worker-{worker_id}",
            )
            process.start()
            self._processes.append(process)

        cpu_preview = ", ".join(
            f"{proc.name}:{cpu_groups[index]}"
            for index, proc in enumerate(self._processes[: min(4, len(self._processes))])
        )
        print(
            "started CPU sampler workers "
            f"count={len(self._processes)} prefetch={self.prefetch_batches} "
            f"samples_per_unit={config.samples_per_unit} "
            f"cache_budget={config.shard_cache_size} per_worker_cache={per_worker_cache_size} "
            f"cpu_groups={cpu_preview}",
            flush=True,
        )
        print(
            "warming CPU sampler prefetch queue; the first resumed step can be noticeably slower",
            flush=True,
        )

    def get_batch(self, timeout_s: float = 300.0) -> dict[str, Tensor]:
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = max(deadline - time.monotonic(), 0.1)
            try:
                item = self._queue.get(timeout=min(1.0, remaining))
            except queue.Empty:
                self._raise_if_worker_failed()
                if time.monotonic() >= deadline:
                    raise TimeoutError("timed out waiting for prefetched human batch")
                continue

            if not isinstance(item, dict):
                raise RuntimeError(f"human sampler worker returned invalid payload type: {type(item)!r}")

            if item.get("kind") == "error":
                raise RuntimeError(
                    f"human sampler worker {item.get('worker_id')} failed: {item.get('error')}"
                )
            if item.get("kind") != "batch":
                raise RuntimeError(f"human sampler worker returned unexpected payload: {item}")
            return item["batch"]

    def _raise_if_worker_failed(self) -> None:
        for process in self._processes:
            if process.exitcode not in (None, 0):
                raise RuntimeError(
                    f"human sampler worker exited unexpectedly: {process.name} exitcode={process.exitcode}"
                )

    def close(self) -> None:
        self._stop_event.set()
        for process in self._processes:
            process.join(timeout=5.0)
        for process in self._processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
        self._queue.close()
        self._queue.join_thread()


class _HumanBatchThreadPool:
    def __init__(self, config: TrainingConfig) -> None:
        self.batch_size = config.micro_batch_size
        self.prefetch_batches = config.cpu_prefetch_batches
        self._stop_event = threading.Event()
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=config.cpu_prefetch_batches)
        self._threads: list[threading.Thread] = []
        shared_shard_cache = _ShardCache(config.shard_cache_size)

        cpu_groups = _plan_worker_cpu_groups(
            num_workers=config.cpu_sampler_workers,
            reserved_cores=config.cpu_reserved_cores,
        )
        worker_payload = {
            "human_data_dir": str(config.human_data_dir),
            "batch_size": int(config.micro_batch_size),
            "samples_per_unit": int(config.samples_per_unit),
            "seed": int(config.seed),
        }

        for worker_id, cpu_ids in enumerate(cpu_groups):
            thread = threading.Thread(
                target=_human_batch_thread_worker,
                args=(worker_id, worker_payload, shared_shard_cache, cpu_ids, self._queue, self._stop_event),
                name=f"human-batch-thread-{worker_id}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

        cpu_preview = ", ".join(
            f"{thread.name}:{cpu_groups[index]}"
            for index, thread in enumerate(self._threads[: min(4, len(self._threads))])
        )
        print(
            "started CPU sampler threads "
            f"count={len(self._threads)} prefetch={self.prefetch_batches} "
            f"samples_per_unit={config.samples_per_unit} "
            f"shared_cache_budget={config.shard_cache_size} cpu_groups={cpu_preview}",
            flush=True,
        )
        print(
            "warming CPU sampler prefetch queue; the first resumed step can be noticeably slower",
            flush=True,
        )

    def get_batch(self, timeout_s: float = 300.0) -> dict[str, Tensor]:
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = max(deadline - time.monotonic(), 0.1)
            try:
                item = self._queue.get(timeout=min(1.0, remaining))
            except queue.Empty:
                if time.monotonic() >= deadline:
                    raise TimeoutError("timed out waiting for prefetched human batch")
                continue

            if not isinstance(item, dict):
                raise RuntimeError(f"human sampler thread returned invalid payload type: {type(item)!r}")
            if item.get("kind") == "error":
                raise RuntimeError(
                    f"human sampler thread {item.get('worker_id')} failed: {item.get('error')}"
                )
            if item.get("kind") != "batch":
                raise RuntimeError(f"human sampler thread returned unexpected payload: {item}")
            return item["batch"]

    def close(self) -> None:
        self._stop_event.set()
        for thread in self._threads:
            thread.join(timeout=2.0)


def _compute_mix_ratios(config: TrainingConfig, replay_buffer: _SelfPlayReplayBuffer) -> tuple[float, float, float]:
    buffer_fill = replay_buffer.fill_ratio()
    human_floor = float(config.bootstrap_human_floor) if config.bootstrap_mode else 0.25
    human_ratio = max(human_floor, 1.0 - (1.0 - human_floor) * buffer_fill)
    selfplay_ratio = 1.0 - human_ratio
    return human_ratio, selfplay_ratio, buffer_fill


def _build_human_batch_pool(config: TrainingConfig) -> _HumanBatchWorkerPool | _HumanBatchThreadPool | None:
    if config.cpu_sampler_workers <= 0 or config.selfplay_dirs:
        return None

    backend = config.cpu_sampler_backend
    if backend == "auto":
        backend = "thread" if os.environ.get("PYCHARM_HOSTED") else "process"

    if backend == "none":
        return None
    if backend == "thread":
        return _HumanBatchThreadPool(config)
    if backend == "process":
        return _HumanBatchWorkerPool(config)
    raise ValueError(f"unsupported cpu sampler backend: {backend}")


def _console_log_path(output_dir: Path) -> Path:
    return output_dir / "train_console.log"


def _pid_file_path(output_dir: Path) -> Path:
    return output_dir / "train_pid.txt"


def _status_file_path(output_dir: Path) -> Path:
    return output_dir / "train_status.json"


def _is_process_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _get_proc_cmdline(pid: int) -> str:
    proc_cmdline = Path("/proc") / str(pid) / "cmdline"
    if not proc_cmdline.is_file():
        return ""
    try:
        return proc_cmdline.read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def _looks_like_training_process(pid: int, output_dir: Path) -> bool:
    if not _is_process_running(pid):
        return False

    cmdline = _get_proc_cmdline(pid)
    if not cmdline:
        return False

    script_name = Path(__file__).name
    output_dir_str = str(output_dir)
    return script_name in cmdline and output_dir_str in cmdline


def _read_pid_file(pid_path: Path) -> int | None:
    if not pid_path.is_file():
        return None
    try:
        raw_text = pid_path.read_text(encoding="utf-8").strip()
        if not raw_text:
            return None
        if raw_text.startswith("{"):
            payload = json.loads(raw_text)
            pid = payload.get("pid")
            return int(pid) if pid is not None else None
        return int(raw_text)
    except Exception:
        return None


def _cleanup_stale_pid_file(output_dir: Path) -> None:
    pid_path = _pid_file_path(output_dir)
    pid = _read_pid_file(pid_path)
    if pid is None:
        return
    if not _looks_like_training_process(pid, output_dir):
        pid_path.unlink(missing_ok=True)


def _write_background_status(output_dir: Path, payload: dict[str, Any]) -> None:
    status_path = _status_file_path(output_dir)
    status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _register_child_pid(output_dir: Path) -> None:
    pid_path = _pid_file_path(output_dir)
    payload = {
        "pid": os.getpid(),
        "output_dir": str(output_dir),
        "script": str(Path(__file__).resolve()),
    }
    pid_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _cleanup() -> None:
        current_pid = _read_pid_file(pid_path)
        if current_pid == os.getpid():
            pid_path.unlink(missing_ok=True)

    atexit.register(_cleanup)


def _tail_log_file(log_path: Path, output_dir: Path, pid: int) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch(exist_ok=True)
    pid_path = _pid_file_path(output_dir)
    status_path = _status_file_path(output_dir)
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(0, os.SEEK_END)
        while True:
            line = handle.readline()
            if line:
                print(line.rstrip("\n"), flush=True)
                continue
            child_finished = (not _is_process_running(pid)) or (not pid_path.exists() and status_path.exists())
            if child_finished:
                remainder = handle.read()
                if remainder:
                    print(remainder, end="", flush=True)
                break
            time.sleep(0.5)


def _spawn_detached_training_child(config: TrainingConfig, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = _console_log_path(output_dir)
    status_path = _status_file_path(output_dir)
    pid_path = _pid_file_path(output_dir)
    status_path.unlink(missing_ok=True)
    pid_path.unlink(missing_ok=True)

    env = os.environ.copy()
    env["XIANGQI_TRAIN_CHILD"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    process = _spawn_detached_process(cmd=cmd, env=env, log_path=log_path)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        pid = _read_pid_file(pid_path)
        if pid is not None:
            return pid
        if process.poll() is not None:
            return process.pid
        time.sleep(0.1)
    return process.pid


def _spawn_detached_process(
    cmd: list[str],
    env: dict[str, str],
    log_path: Path,
) -> subprocess.Popen[Any]:
    wsl_distro = os.environ.get("WSL_DISTRO_NAME")
    wsl_exe = Path("/mnt/c/Windows/System32/wsl.exe")
    script_dir = Path(__file__).resolve().parent

    if wsl_distro and wsl_exe.is_file():
        launch_cmd = _build_wsl_bridge_command(
            distro=wsl_distro,
            script_dir=script_dir,
            cmd=cmd,
            env=env,
            log_path=log_path,
        )
        return subprocess.Popen(
            launch_cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )

    with log_path.open("a", encoding="utf-8") as log_file:
        return subprocess.Popen(
            cmd,
            cwd=str(script_dir),
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
            close_fds=True,
        )


def _build_wsl_bridge_command(
    distro: str,
    script_dir: Path,
    cmd: list[str],
    env: dict[str, str],
    log_path: Path,
) -> list[str]:
    bridge_env = []
    for key in ("XIANGQI_TRAIN_CHILD", "PYTHONUNBUFFERED"):
        value = env.get(key)
        if value is not None:
            bridge_env.append(f"{key}={shlex.quote(value)}")

    quoted_cmd = " ".join(shlex.quote(part) for part in cmd)
    quoted_cwd = shlex.quote(str(script_dir))
    quoted_log = shlex.quote(str(log_path))
    env_prefix = " ".join(bridge_env)
    if env_prefix:
        env_prefix += " "

    shell_command = (
        f"cd {quoted_cwd} && "
        f"{env_prefix}nohup {quoted_cmd} >> {quoted_log} 2>&1 </dev/null"
    )
    return [
        str(Path("/mnt/c/Windows/System32/wsl.exe")),
        "-d",
        distro,
        "bash",
        "-lc",
        shell_command,
    ]


def _run_pycharm_supervisor(config: TrainingConfig) -> int:
    config = _normalize_training_config(config)
    output_dir = Path(config.output_dir)
    _cleanup_stale_pid_file(output_dir)
    pid_path = _pid_file_path(output_dir)
    log_path = _console_log_path(output_dir)

    pid = _read_pid_file(pid_path)
    if pid is not None and _looks_like_training_process(pid, output_dir):
        print(f"attaching to background trainer pid={pid}", flush=True)
    else:
        pid_path.unlink(missing_ok=True)
        pid = _spawn_detached_training_child(config, output_dir)
        print(f"spawned detached trainer pid={pid}", flush=True)
    print(f"streaming logs from {log_path}", flush=True)

    try:
        _tail_log_file(log_path, output_dir, pid)
    except KeyboardInterrupt:
        print(
            "detached trainer is still running in background; re-run this script to attach again",
            flush=True,
        )
        return 0

    status_path = _status_file_path(output_dir)
    if status_path.is_file():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            print(json.dumps(status, indent=2, ensure_ascii=False), flush=True)
        except Exception:
            pass
        return 0

    if not _is_process_running(pid):
        print(
            "background trainer exited without writing train_status.json; "
            "this indicates an unexpected hard stop outside the normal Python shutdown path",
            flush=True,
        )
        return 1

    return 0


def _maybe_handle_background_stop() -> bool:
    if "--stop-background-trainer" not in sys.argv:
        return False

    parser = argparse.ArgumentParser(description="Stop a detached Xiangqi trainer.")
    parser.add_argument("--output-dir", default=_DEFAULT_LAUNCH_CONFIG.output_dir)
    args, _ = parser.parse_known_args()

    output_dir = Path(args.output_dir).resolve()
    pid = _read_pid_file(_pid_file_path(output_dir))
    if pid is None or not _looks_like_training_process(pid, output_dir):
        _cleanup_stale_pid_file(output_dir)
        print(f"no running detached trainer found in {output_dir}", flush=True)
        raise SystemExit(0)

    os.kill(pid, signal.SIGINT)
    print(f"sent SIGINT to detached trainer pid={pid}", flush=True)
    raise SystemExit(0)


def run_training(config: TrainingConfig) -> dict[str, Any]:
    config = _normalize_training_config(config)
    _validate_training_inputs(config)
    _seed_everything(config.seed)

    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if os.environ.get("XIANGQI_TRAIN_CHILD") == "1":
        _register_child_pid(output_dir)
    train_log_path = output_dir / "train_log.jsonl"

    human_catalog = _HumanTrainCatalog.from_dir(Path(config.human_data_dir))
    shard_cache = _ShardCache(config.shard_cache_size)
    replay_buffer = _SelfPlayReplayBuffer(config.replay_buffer_size)
    seen_selfplay_shards: dict[str, dict[str, Any]] = {}
    ingestor = _SelfPlayIngestor(
        [Path(path).resolve() for path in config.selfplay_dirs],
        seen_selfplay_shards,
        quality_gate=config.selfplay_run_quality_gate,
        max_rep_draw_rate=config.selfplay_run_max_rep_draw_rate,
        min_decisive_rate=config.selfplay_run_min_decisive_rate,
        bootstrap_soft_gate=config.bootstrap_mode,
        source_groups=[str(index) for index in range(len(config.selfplay_dirs))],
    )

    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested for training, but CUDA is not available")
    _configure_torch_runtime(config, device)

    model = XiangqiPVTransformer(config.model_config)
    model.to(device)
    _apply_parameter_training_scope(model, config)
    anchor_model: nn.Module | None = None
    needs_anchor_model = (
        config.anchor_checkpoint is not None
        and (
            config.anchor_policy_kl_weight > 0.0
            or config.anchor_value_mse_weight > 0.0
            or config.anchor_policy_top1_ce_weight > 0.0
            or (
                config.teacher_q_pairwise_use_anchor_reference
                and config.teacher_q_pairwise_loss_weight > 0.0
            )
            or config.bad_move_suppression_loss_weight > 0.0
        )
    )
    if needs_anchor_model:
        anchor_state = torch.load(Path(config.anchor_checkpoint), map_location="cpu", weights_only=False)
        anchor_model = build_model_from_checkpoint_state(anchor_state)
        anchor_model.to(device)
        anchor_model.eval()
        for parameter in anchor_model.parameters():
            parameter.requires_grad = False
    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, config)
    signal_handlers = _install_graceful_signal_handlers()
    human_batch_pool = _build_human_batch_pool(config)

    global_step = 0
    best_human_val_metric = float("inf")
    last_human_val_metrics: dict[str, float] | None = None
    cumulative_added_shards = 0
    cumulative_added_samples = 0
    last_ingest_status: dict[str, Any] = {
        "waiting_manifest_runs": 0,
        "waiting_unreadable_manifest_runs": 0,
        "waiting_in_progress_runs": 0,
        "eligible_runs": 0,
        "skipped_runs": 0,
        "soft_allowed_runs": 0,
        "pruned_spans": 0,
        "pruned_samples": 0,
        "skipped_run_reasons": {},
        "soft_allowed_run_reasons": {},
        "added_shards": 0,
        "added_samples": 0,
        "selfplay_buffer_samples": 0,
    }
    if config.resume_path is not None:
        print(f"resuming training from checkpoint: {config.resume_path}", flush=True)
        resume_state = torch.load(Path(config.resume_path), map_location="cpu", weights_only=False)
        load_xiangqi_model_state_dict(model, resume_state["model_state_dict"])
        legacy_wdl_frozen = not _wdl_head_is_nontrivial(model)
        if legacy_wdl_frozen:
            _reinit_wdl_head(model)
            print(
                "resume: legacy wdl_head was frozen; reinitialized and starting fresh optimizer state "
                "(scheduler state preserved)",
                flush=True,
            )
        optimizer = _build_optimizer(model, config)
        if not legacy_wdl_frozen and not config.reset_optimizer_on_resume:
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        scheduler = _build_scheduler(optimizer, config)
        if not config.reset_optimizer_on_resume:
            scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        else:
            print(
                "resume: reset optimizer/scheduler state; using fresh optimizer "
                f"with learning_rate={config.learning_rate:g}",
                flush=True,
            )
        global_step = int(resume_state["global_step"])
        if legacy_wdl_frozen:
            # Loss scale changed (added wdl_loss, scaled value_loss); best metric from the old regime
            # is no longer comparable. Re-establish under the new loss.
            best_human_val_metric = float("inf")
        else:
            best_human_val_metric = float(
                resume_state.get("best_human_val_metric", resume_state.get("best_val_metric", float("inf")))
            )
        if config.reset_selfplay_ingest_state_on_resume:
            replay_buffer.restore([])
            cumulative_added_shards = 0
            cumulative_added_samples = 0
            last_ingest_status["selfplay_buffer_samples"] = 0
        else:
            seen_selfplay_shards.update(resume_state.get("seen_selfplay_shards", {}))
            replay_buffer.restore(resume_state.get("selfplay_buffer_refs", []))
            cumulative_added_shards = int(
                resume_state.get("cumulative_added_shards", len(seen_selfplay_shards))
            )
            cumulative_added_samples = int(
                resume_state.get(
                    "cumulative_added_samples",
                    sum(int(meta.get("samples", 0)) for meta in seen_selfplay_shards.values()),
                )
            )
            restored_ingest_status = resume_state.get("last_ingest_status")
            if isinstance(restored_ingest_status, dict):
                last_ingest_status.update(restored_ingest_status)
            last_ingest_status["selfplay_buffer_samples"] = len(replay_buffer)
            if config.selfplay_run_quality_gate:
                ingestor.prune_replay_buffer(replay_buffer)
                last_ingest_status["selfplay_buffer_samples"] = len(replay_buffer)
        _restore_rng_state(resume_state.get("rng_state"))
        print(f"resume complete at global_step={global_step}", flush=True)
        if config.reset_selfplay_ingest_state_on_resume:
            print("resume weights/state restored, self-play ingest state reset", flush=True)
    print(
        f"training target max_steps={config.max_steps} remaining_steps={max(config.max_steps - global_step, 0)}",
        flush=True,
    )
    print(
        f"training lr_schedule_max_steps={_effective_lr_schedule_max_steps(config)} warmup_steps={config.warmup_steps}",
        flush=True,
    )
    if config.bootstrap_mode:
        print(f"training bootstrap_human_floor={config.bootstrap_human_floor:.2f}", flush=True)
    if config.selfplay_run_quality_gate:
        gate_mode = "bootstrap_soft" if config.bootstrap_mode else "hard"
        print(
            "training selfplay_quality_gate="
            f"on mode={gate_mode} max_rep_draw={config.selfplay_run_max_rep_draw_rate:.1f}% "
            f"min_decisive={config.selfplay_run_min_decisive_rate:.1f}%",
            flush=True,
        )
    if config.selfplay_dir_sampling_ratios:
        ratio_text = ",".join(f"{float(ratio):.4g}" for ratio in config.selfplay_dir_sampling_ratios)
        print(f"training selfplay_dir_sampling_ratios={ratio_text}", flush=True)
    if config.pause_at_local_time:
        print(f"training scheduled_pause_local={config.pause_at_local_time}", flush=True)
    if config.promote_best_on_human_val:
        print("training best_checkpoint_metric=human_val_total_loss", flush=True)
    else:
        print("training best_checkpoint_metric=arena_gated human_val_recorded_only=on", flush=True)
    print(
        f"training wdl_loss_weight={config.wdl_loss_weight:.3f} "
        f"policy_loss_weight={config.policy_loss_weight:.3f} "
        f"value_loss_weight={config.value_loss_weight:.3f} "
        f"wdl_value_consistency_weight={config.wdl_value_consistency_weight:.3f} "
        f"value_target_scale={config.value_target_scale:.3f} "
        f"teacher_q_loss_weight={config.teacher_q_loss_weight:.3f} "
        f"teacher_q_temperature_cp={config.teacher_q_temperature_cp:.1f} "
        f"teacher_q_pairwise_loss_weight={config.teacher_q_pairwise_loss_weight:.3f} "
        f"teacher_q_pairwise_margin_logit={config.teacher_q_pairwise_margin_logit:.3f} "
        f"teacher_q_pairwise_min_gap_cp={config.teacher_q_pairwise_min_gap_cp:.1f} "
        f"teacher_q_pairwise_beta={config.teacher_q_pairwise_beta:.3f} "
        f"teacher_q_pairwise_use_anchor_reference={bool(config.teacher_q_pairwise_use_anchor_reference)} "
        f"teacher_q_pairwise_bad_move_only={bool(config.teacher_q_pairwise_bad_move_only)} "
        f"bad_move_suppression_loss_weight={config.bad_move_suppression_loss_weight:.3f} "
        f"bad_move_suppression_margin_logit={config.bad_move_suppression_margin_logit:.3f} "
        f"bad_move_suppression_min_gap_cp={config.bad_move_suppression_min_gap_cp:.1f} "
        f"bad_move_suppression_beta={config.bad_move_suppression_beta:.3f} "
        f"anchor_policy_kl_weight={config.anchor_policy_kl_weight:.3f} "
        f"anchor_policy_top1_ce_weight={config.anchor_policy_top1_ce_weight:.3f} "
        f"anchor_value_mse_weight={config.anchor_value_mse_weight:.3f} "
        f"anchor_anneal_steps={config.anchor_anneal_steps}",
        flush=True,
    )
    if anchor_model is not None:
        print(f"training anchor_checkpoint={config.anchor_checkpoint}", flush=True)
    print(
        "training model_config="
        f"d_model={config.model_config.d_model} "
        f"layers={config.model_config.num_layers} "
        f"heads={config.model_config.num_heads} "
        f"ffn_dim={config.model_config.ffn_dim} "
        f"policy_head_dim={config.model_config.policy_head_dim} "
        f"use_2d_relative_attention_bias={config.model_config.use_2d_relative_attention_bias} "
        f"use_line_of_sight_attention_bias={config.model_config.use_line_of_sight_attention_bias} "
        f"use_history_memory_attention={config.model_config.use_history_memory_attention} "
        f"use_global_strategic_attention={config.model_config.use_global_strategic_attention} "
        f"use_trunk_global_strategy_tokens={config.model_config.use_trunk_global_strategy_tokens} "
        f"use_value_token_pooling={config.model_config.use_value_token_pooling} "
        f"num_global_strategy_tokens={config.model_config.num_global_strategy_tokens} "
        f"use_cnn_local_tactical_adapter={config.model_config.use_cnn_local_tactical_adapter} "
        f"cnn_local_channels={config.model_config.cnn_local_channels} "
        f"cnn_local_blocks={config.model_config.cnn_local_blocks} "
        f"use_cnn_policy_residual_adapter={config.model_config.use_cnn_policy_residual_adapter} "
        f"cnn_policy_channels={config.model_config.cnn_policy_channels} "
        f"cnn_policy_blocks={config.model_config.cnn_policy_blocks} "
        f"cnn_policy_rank={config.model_config.cnn_policy_rank} "
        f"use_cnn_local_tactical_stem={config.model_config.use_cnn_local_tactical_stem} "
        f"cnn_stem_channels={config.model_config.cnn_stem_channels} "
        f"cnn_stem_blocks={config.model_config.cnn_stem_blocks}",
        flush=True,
    )
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"training trainable_params={trainable_params} total_params={total_params} "
        f"train_only_relative_attention_bias={config.train_only_relative_attention_bias} "
        f"train_only_cnn_local_adapter={config.train_only_cnn_local_adapter} "
        f"train_only_cnn_policy_residual_adapter={config.train_only_cnn_policy_residual_adapter} "
        f"adapter_unfreeze_last_n_blocks={config.adapter_unfreeze_last_n_blocks}",
        flush=True,
    )

    autocast_enabled = bool(config.use_bfloat16 and device.type == "cuda")
    log_file = train_log_path.open("a", encoding="utf-8")
    last_poll_at = 0.0
    last_log_time = time.time()
    last_emitted_ingest_status: dict[str, Any] | None = None
    interrupt_reason: str | None = None
    pause_local_time = _parse_pause_local_time(config.pause_at_local_time)
    pause_deadline = _next_pause_local_deadline(pause_local_time)

    try:
        while global_step < config.max_steps:
            if _pause_deadline_reached(pause_deadline):
                pause_label = pause_deadline.astimezone().strftime("%Y-%m-%d %H:%M") if pause_deadline else "unknown"
                raise _GracefulStopRequested(f"reached scheduled local pause time {pause_label}")
            now = time.monotonic()
            newly_added = {
                "added_shards": 0,
                "added_samples": 0,
                "waiting_manifest_runs": 0,
                "waiting_unreadable_manifest_runs": 0,
                "waiting_in_progress_runs": 0,
                "eligible_runs": 0,
                "skipped_runs": 0,
                "soft_allowed_runs": 0,
                "pruned_spans": 0,
                "pruned_samples": 0,
                "skipped_run_reasons": {},
                "soft_allowed_run_reasons": {},
            }
            if config.selfplay_dirs and (now - last_poll_at) >= config.poll_interval_s:
                newly_added = ingestor.poll(shard_cache, replay_buffer)
                last_poll_at = now
                cumulative_added_shards += int(newly_added["added_shards"])
                cumulative_added_samples += int(newly_added["added_samples"])
                current_ingest_status = {
                    "waiting_manifest_runs": int(newly_added["waiting_manifest_runs"]),
                    "waiting_unreadable_manifest_runs": int(newly_added["waiting_unreadable_manifest_runs"]),
                    "waiting_in_progress_runs": int(newly_added["waiting_in_progress_runs"]),
                    "eligible_runs": int(newly_added["eligible_runs"]),
                    "skipped_runs": int(newly_added["skipped_runs"]),
                    "soft_allowed_runs": int(newly_added["soft_allowed_runs"]),
                    "pruned_spans": int(newly_added["pruned_spans"]),
                    "pruned_samples": int(newly_added["pruned_samples"]),
                    "skipped_run_reasons": dict(newly_added["skipped_run_reasons"]),
                    "soft_allowed_run_reasons": dict(newly_added["soft_allowed_run_reasons"]),
                    "added_shards": int(newly_added["added_shards"]),
                    "added_samples": int(newly_added["added_samples"]),
                    "selfplay_buffer_samples": len(replay_buffer),
                }
                last_ingest_status = current_ingest_status
                if (
                    last_emitted_ingest_status != current_ingest_status
                    or int(newly_added["added_samples"]) > 0
                ):
                    ingest_entry = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "step": global_step,
                        "kind": "ingest",
                        **current_ingest_status,
                        "cumulative_added_shards": cumulative_added_shards,
                        "cumulative_added_samples": cumulative_added_samples,
                    }
                    log_file.write(json.dumps(ingest_entry, ensure_ascii=False) + "\n")
                    log_file.flush()
                    print(
                        "ingest "
                        f"step={global_step} "
                        f"waiting_manifest={current_ingest_status['waiting_manifest_runs']} "
                        f"waiting_unreadable={current_ingest_status['waiting_unreadable_manifest_runs']} "
                        f"waiting_in_progress={current_ingest_status['waiting_in_progress_runs']} "
                        f"eligible_runs={current_ingest_status['eligible_runs']} "
                        f"skipped_runs={current_ingest_status['skipped_runs']} "
                        f"soft_allowed_runs={current_ingest_status['soft_allowed_runs']} "
                        f"pruned_spans={current_ingest_status['pruned_spans']} "
                        f"pruned_samples={current_ingest_status['pruned_samples']} "
                        f"skipped_by_reason={_format_ingest_reason_counts(current_ingest_status['skipped_run_reasons'])} "
                        f"soft_allowed_by_reason={_format_ingest_reason_counts(current_ingest_status['soft_allowed_run_reasons'])} "
                        f"added_shards={current_ingest_status['added_shards']} "
                        f"added_samples={current_ingest_status['added_samples']} "
                        f"selfplay_buffer={current_ingest_status['selfplay_buffer_samples']} "
                        f"cumulative_added_shards={cumulative_added_shards} "
                        f"cumulative_added_samples={cumulative_added_samples}",
                        flush=True,
                    )
                    last_emitted_ingest_status = dict(current_ingest_status)

            human_ratio, selfplay_ratio, buffer_fill = _compute_mix_ratios(config, replay_buffer)

            desired_selfplay = int(round(config.micro_batch_size * selfplay_ratio))
            if len(replay_buffer) == 0:
                desired_selfplay = 0
            desired_selfplay = min(desired_selfplay, config.micro_batch_size)
            human_batch_size = config.micro_batch_size - desired_selfplay

            optimizer.zero_grad(set_to_none=True)
            accum_metrics = {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "wdl_loss": 0.0,
                "wdl_value_consistency_loss": 0.0,
                "teacher_q_loss": 0.0,
                "teacher_q_pairwise_loss": 0.0,
                "bad_move_suppression_loss": 0.0,
                "anchor_policy_kl_loss": 0.0,
                "anchor_policy_top1_ce_loss": 0.0,
                "anchor_value_mse_loss": 0.0,
                "n_teacher_q_samples": 0.0,
                "wdl_entropy": 0.0,
                "total_loss": 0.0,
                "n_oracle_samples": 0.0,  # # of samples whose value loss used oracle_value vs scaled_z
                "n_teacher_q_pairwise_samples": 0.0,
                "n_bad_move_suppression_samples": 0.0,
            }
            accum_source_counts: Counter[str] = Counter()

            for _ in range(config.grad_accum_steps):
                if (
                    human_batch_pool is not None
                    and desired_selfplay == 0
                    and human_batch_size == config.micro_batch_size
                ):
                    batch = human_batch_pool.get_batch()
                else:
                    batch = _sample_training_batch(
                        human_catalog=human_catalog,
                        shard_cache=shard_cache,
                        replay_buffer=replay_buffer,
                        human_batch_size=human_batch_size,
                        selfplay_batch_size=desired_selfplay,
                        samples_per_unit=config.samples_per_unit,
                        selfplay_dir_sampling_ratios=(
                            config.selfplay_dir_sampling_ratios
                            if config.selfplay_dir_sampling_ratios
                            else None
                        ),
                    )
                batch = _move_batch_to_device(batch, device)
                if "source_group_id" in batch:
                    source_ids = batch["source_group_id"].detach().to(device="cpu", dtype=torch.int16).tolist()
                    accum_source_counts.update(str(int(source_id)) for source_id in source_ids)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                    outputs = model(batch["state"])
                    anchor_outputs = None
                    if anchor_model is not None:
                        with torch.no_grad():
                            anchor_outputs = anchor_model(batch["state"])
                    losses = _compute_training_losses(
                        policy_logits=outputs["policy_logits"],
                        value_scalar=outputs["value_scalar"],
                        policy_offsets=batch["policy_offsets"],
                        policy_idxs=batch["policy_idxs"],
                        policy_probs=batch["policy_probs"],
                        z=batch["z"],
                        wdl_logits=outputs.get("wdl_logits"),
                        wdl_target=batch.get("wdl_target"),
                        wdl_loss_weight=config.wdl_loss_weight,
                        policy_loss_weight=config.policy_loss_weight,
                        value_loss_weight=config.value_loss_weight,
                        wdl_value_consistency_weight=config.wdl_value_consistency_weight,
                        value_target_scale=config.value_target_scale,
                        oracle_value=batch.get("oracle_value"),
                        use_oracle_value=bool(getattr(config, "use_oracle_value", True)),
                        oracle_policy_offsets=batch.get("oracle_policy_offsets"),
                        oracle_policy_idxs=batch.get("oracle_policy_idxs"),
                        oracle_policy_probs=batch.get("oracle_policy_probs"),
                        policy_oracle_alpha=float(getattr(config, "policy_oracle_alpha", 0.0)),
                        teacher_q_offsets=batch.get("teacher_q_offsets"),
                        teacher_q_idxs=batch.get("teacher_q_idxs"),
                        teacher_q_values=batch.get("teacher_q_values"),
                        teacher_q_loss_weight=float(getattr(config, "teacher_q_loss_weight", 0.0)),
                        teacher_q_temperature_cp=float(getattr(config, "teacher_q_temperature_cp", 80.0)),
                        teacher_q_pairwise_loss_weight=float(
                            getattr(config, "teacher_q_pairwise_loss_weight", 0.0)
                        ),
                        teacher_q_pairwise_margin_logit=float(
                            getattr(config, "teacher_q_pairwise_margin_logit", 0.25)
                        ),
                        teacher_q_pairwise_min_gap_cp=float(
                            getattr(config, "teacher_q_pairwise_min_gap_cp", 80.0)
                        ),
                        teacher_q_pairwise_beta=float(
                            getattr(config, "teacher_q_pairwise_beta", 1.0)
                        ),
                        teacher_q_ref_policy_logits=(
                            anchor_outputs["policy_logits"]
                            if (
                                anchor_outputs is not None
                                and bool(getattr(config, "teacher_q_pairwise_use_anchor_reference", False))
                            )
                            else None
                        ),
                        teacher_q_pairwise_bad_move_only=bool(
                            getattr(config, "teacher_q_pairwise_bad_move_only", False)
                        ),
                        bad_move_suppression_loss_weight=float(
                            getattr(config, "bad_move_suppression_loss_weight", 0.0)
                        ),
                        bad_move_suppression_margin_logit=float(
                            getattr(config, "bad_move_suppression_margin_logit", 0.75)
                        ),
                        bad_move_suppression_min_gap_cp=float(
                            getattr(config, "bad_move_suppression_min_gap_cp", 80.0)
                        ),
                        bad_move_suppression_beta=float(
                            getattr(config, "bad_move_suppression_beta", 2.0)
                        ),
                        bad_move=batch.get("bad_move"),
                        sample_weight=batch.get("sample_weight"),
                        legal_offsets=batch.get("legal_offsets"),
                        legal_idxs=batch.get("legal_idxs"),
                    )
                    if anchor_outputs is not None:
                        anchor_losses = _compute_anchor_distillation_losses(
                            policy_logits=outputs["policy_logits"],
                            value_scalar=outputs["value_scalar"],
                            anchor_policy_logits=anchor_outputs["policy_logits"],
                            anchor_value_scalar=anchor_outputs["value_scalar"],
                            legal_offsets=batch.get("legal_offsets"),
                            legal_idxs=batch.get("legal_idxs"),
                        )
                        losses["anchor_policy_kl_loss"] = anchor_losses["anchor_policy_kl_loss"]
                        losses["anchor_policy_top1_ce_loss"] = anchor_losses["anchor_policy_top1_ce_loss"]
                        losses["anchor_value_mse_loss"] = anchor_losses["anchor_value_mse_loss"]
                        anchor_scale = _anchor_weight_scale(config, global_step)
                        losses["total_loss"] = (
                            losses["total_loss"]
                            + anchor_scale * float(config.anchor_policy_kl_weight) * losses["anchor_policy_kl_loss"]
                            + anchor_scale
                            * float(config.anchor_policy_top1_ce_weight)
                            * losses["anchor_policy_top1_ce_loss"]
                            + anchor_scale * float(config.anchor_value_mse_weight) * losses["anchor_value_mse_loss"]
                        )
                    else:
                        losses["anchor_policy_kl_loss"] = torch.zeros(
                            (), device=outputs["policy_logits"].device, dtype=outputs["policy_logits"].dtype
                        )
                        losses["anchor_policy_top1_ce_loss"] = torch.zeros(
                            (), device=outputs["policy_logits"].device, dtype=outputs["policy_logits"].dtype
                        )
                        losses["anchor_value_mse_loss"] = torch.zeros(
                            (), device=outputs["policy_logits"].device, dtype=outputs["policy_logits"].dtype
                        )
                    scaled_loss = losses["total_loss"] / float(config.grad_accum_steps)

                scaled_loss.backward()
                for key, value in losses.items():
                    accum_metrics[key] += float(value.detach().item())

            if config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            scheduler.step()
            global_step += 1

            for key in accum_metrics:
                accum_metrics[key] /= float(config.grad_accum_steps)

            if global_step % config.log_interval_steps == 0 or global_step == 1:
                current_time = time.time()
                elapsed = max(current_time - last_log_time, 1e-6)
                last_log_time = current_time
                total_logged_samples = max(1, sum(accum_source_counts.values()))
                source_group_counts = dict(sorted(accum_source_counts.items(), key=lambda item: item[0]))
                source_group_ratios = {
                    key: float(value) / float(total_logged_samples)
                    for key, value in source_group_counts.items()
                }
                entry = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "step": global_step,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "policy_loss": accum_metrics["policy_loss"],
                    "value_loss": accum_metrics["value_loss"],
                    "wdl_loss": accum_metrics["wdl_loss"],
                    "wdl_value_consistency_loss": accum_metrics["wdl_value_consistency_loss"],
                    "teacher_q_loss": accum_metrics["teacher_q_loss"],
                    "teacher_q_pairwise_loss": accum_metrics["teacher_q_pairwise_loss"],
                    "bad_move_suppression_loss": accum_metrics["bad_move_suppression_loss"],
                    "anchor_policy_kl_loss": accum_metrics["anchor_policy_kl_loss"],
                    "anchor_policy_top1_ce_loss": accum_metrics["anchor_policy_top1_ce_loss"],
                    "anchor_value_mse_loss": accum_metrics["anchor_value_mse_loss"],
                    "n_teacher_q_samples": accum_metrics["n_teacher_q_samples"],
                    "n_teacher_q_pairwise_samples": accum_metrics["n_teacher_q_pairwise_samples"],
                    "n_bad_move_suppression_samples": accum_metrics["n_bad_move_suppression_samples"],
                    "wdl_entropy": accum_metrics["wdl_entropy"],
                    "total_loss": accum_metrics["total_loss"],
                    # # of samples whose value loss used the calibrated oracle_value
                    # (rest fell back to scaled_z).  When this is 0, training is using
                    # the legacy noisy-z target only.  Should ramp up to ~micro_batch
                    # once oracle-labeled selfplay shards dominate the batch.
                    "n_oracle_samples": accum_metrics["n_oracle_samples"],
                    "oracle_value_coverage": accum_metrics["n_oracle_samples"] / float(total_logged_samples),
                    "source_group_counts": source_group_counts,
                    "source_group_ratios": source_group_ratios,
                    "human_ratio": human_ratio,
                    "selfplay_ratio": selfplay_ratio,
                    "selfplay_buffer_samples": len(replay_buffer),
                    "selfplay_buffer_fill": buffer_fill,
                    "cache_items": len(shard_cache),
                    "added_shards": newly_added["added_shards"],
                    "added_samples": newly_added["added_samples"],
                    "waiting_manifest_runs": newly_added["waiting_manifest_runs"],
                    "waiting_unreadable_manifest_runs": newly_added["waiting_unreadable_manifest_runs"],
                    "waiting_in_progress_runs": newly_added["waiting_in_progress_runs"],
                    "eligible_runs": newly_added["eligible_runs"],
                    "skipped_runs": newly_added["skipped_runs"],
                    "soft_allowed_runs": newly_added["soft_allowed_runs"],
                    "pruned_spans": newly_added["pruned_spans"],
                    "pruned_samples": newly_added["pruned_samples"],
                    "skipped_run_reasons": dict(newly_added["skipped_run_reasons"]),
                    "soft_allowed_run_reasons": dict(newly_added["soft_allowed_run_reasons"]),
                    "cumulative_added_shards": cumulative_added_shards,
                    "cumulative_added_samples": cumulative_added_samples,
                    "steps_per_sec": config.log_interval_steps / elapsed if global_step > 1 else 0.0,
                }
                log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                log_file.flush()
                print(
                    "train "
                    f"step={global_step} "
                    f"lr={entry['lr']:.6g} "
                    f"policy_loss={entry['policy_loss']:.4f} "
                    f"value_loss={entry['value_loss']:.4f} "
                    f"wdl_loss={entry['wdl_loss']:.4f} "
                    f"wdl_value_consistency={entry['wdl_value_consistency_loss']:.4f} "
                    f"teacher_q_loss={entry['teacher_q_loss']:.4f} "
                    f"teacher_q_pairwise={entry['teacher_q_pairwise_loss']:.4f} "
                    f"bad_move_suppression={entry['bad_move_suppression_loss']:.4f} "
                    f"anchor_kl={entry['anchor_policy_kl_loss']:.4f} "
                    f"anchor_top1_ce={entry['anchor_policy_top1_ce_loss']:.4f} "
                    f"anchor_value={entry['anchor_value_mse_loss']:.4f} "
                    f"n_teacher_q={entry['n_teacher_q_samples']:.0f} "
                    f"n_pairwise={entry['n_teacher_q_pairwise_samples']:.0f} "
                    f"n_bad_suppress={entry['n_bad_move_suppression_samples']:.0f} "
                    f"wdl_entropy={entry['wdl_entropy']:.4f} "
                    f"total_loss={entry['total_loss']:.4f} "
                    f"n_oracle={entry['n_oracle_samples']:.0f} "
                    f"oracle_cov={entry['oracle_value_coverage']:.2f} "
                    f"mix={entry['human_ratio']:.2f}/{entry['selfplay_ratio']:.2f} "
                    f"source_groups={entry['source_group_counts']} "
                    f"selfplay_buffer={entry['selfplay_buffer_samples']} "
                    f"steps_per_sec={entry['steps_per_sec']:.2f}",
                    flush=True,
                )

            should_eval = global_step % config.eval_interval_steps == 0 or global_step == config.max_steps
            should_save = global_step % config.save_interval_steps == 0 or global_step == config.max_steps
            latest_state: dict[str, Any] | None = None

            if should_eval:
                val_metrics = _evaluate_model(
                    model=model,
                    catalog=human_catalog,
                    shard_cache=shard_cache,
                    device=device,
                    batch_size=config.micro_batch_size,
                    autocast_enabled=autocast_enabled,
                    wdl_loss_weight=config.wdl_loss_weight,
                    value_loss_weight=config.value_loss_weight,
                    wdl_value_consistency_weight=config.wdl_value_consistency_weight,
                    value_target_scale=config.value_target_scale,
                )
                eval_entry = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "step": global_step,
                    "kind": "eval",
                    **val_metrics,
                }
                log_file.write(json.dumps(eval_entry, ensure_ascii=False) + "\n")
                log_file.flush()
                print(
                    "eval "
                    f"step={global_step} "
                    f"human_val_policy_loss={val_metrics['human_val_policy_loss']:.4f} "
                    f"human_val_value_loss={val_metrics['human_val_value_loss']:.4f} "
                    f"human_val_wdl_loss={val_metrics['human_val_wdl_loss']:.4f} "
                    f"human_val_wdl_value_consistency={val_metrics['human_val_wdl_value_consistency_loss']:.4f} "
                    f"human_val_total_loss={val_metrics['human_val_total_loss']:.4f} "
                    f"human_val_samples={val_metrics['human_val_samples']}",
                    flush=True,
                )
                last_human_val_metrics = {
                    "human_val_policy_loss": float(val_metrics["human_val_policy_loss"]),
                    "human_val_value_loss": float(val_metrics["human_val_value_loss"]),
                    "human_val_wdl_loss": float(val_metrics["human_val_wdl_loss"]),
                    "human_val_wdl_value_consistency_loss": float(
                        val_metrics["human_val_wdl_value_consistency_loss"]
                    ),
                    "human_val_total_loss": float(val_metrics["human_val_total_loss"]),
                    "human_val_samples": float(val_metrics["human_val_samples"]),
                }

                latest_state = _build_checkpoint_state(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,
                    global_step=global_step,
                    best_human_val_metric=best_human_val_metric,
                    seen_selfplay_shards=seen_selfplay_shards,
                    replay_buffer=replay_buffer,
                    cumulative_added_shards=cumulative_added_shards,
                    cumulative_added_samples=cumulative_added_samples,
                    last_ingest_status=last_ingest_status,
                )
                if (
                    config.promote_best_on_human_val
                    and val_metrics["human_val_total_loss"] < best_human_val_metric
                ):
                    best_human_val_metric = float(val_metrics["human_val_total_loss"])
                    latest_state["best_human_val_metric"] = best_human_val_metric
                    latest_state["best_val_metric"] = best_human_val_metric
                    torch.save(latest_state, output_dir / "best.pt")

            if should_save:
                if latest_state is None:
                    latest_state = _build_checkpoint_state(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        config=config,
                        global_step=global_step,
                        best_human_val_metric=best_human_val_metric,
                        seen_selfplay_shards=seen_selfplay_shards,
                        replay_buffer=replay_buffer,
                        cumulative_added_shards=cumulative_added_shards,
                        cumulative_added_samples=cumulative_added_samples,
                        last_ingest_status=last_ingest_status,
                    )
                torch.save(latest_state, output_dir / "latest.pt")

                # Numbered snapshot: never overwritten, lets us roll back to a
                # specific step (e.g. recover a peak checkpoint after a later
                # training regression).  Only triggers if snapshot_interval_steps>0.
                if (
                    config.snapshot_interval_steps > 0
                    and global_step % config.snapshot_interval_steps == 0
                ):
                    snap_dir = output_dir / "snapshots"
                    snap_dir.mkdir(parents=True, exist_ok=True)
                    snap_path = snap_dir / f"latest_step{global_step}.pt"
                    torch.save(latest_state, snap_path)
                    print(f"  saved numbered snapshot to {snap_path}", flush=True)

    except (KeyboardInterrupt, _GracefulStopRequested) as exc:
        interrupt_reason = str(exc) or exc.__class__.__name__
        interrupt_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": global_step,
            "kind": "interrupt",
            "reason": interrupt_reason,
        }
        log_file.write(json.dumps(interrupt_entry, ensure_ascii=False) + "\n")
        log_file.flush()
        if config.save_on_interrupt and global_step > 0:
            latest_state = _build_checkpoint_state(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                global_step=global_step,
                best_human_val_metric=best_human_val_metric,
                seen_selfplay_shards=seen_selfplay_shards,
                replay_buffer=replay_buffer,
                cumulative_added_shards=cumulative_added_shards,
                cumulative_added_samples=cumulative_added_samples,
                last_ingest_status=last_ingest_status,
            )
            torch.save(latest_state, output_dir / "latest.pt")
            print(
                f"saved interrupt checkpoint to {(output_dir / 'latest.pt')} at global_step={global_step}",
                flush=True,
            )
        print(f"training interrupted: {interrupt_reason}", flush=True)
    finally:
        if interrupt_reason is None and global_step >= config.max_steps:
            complete_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "step": global_step,
                "kind": "complete",
            }
            log_file.write(json.dumps(complete_entry, ensure_ascii=False) + "\n")
            log_file.flush()
            print(f"training completed normally at global_step={global_step}", flush=True)
        if human_batch_pool is not None:
            human_batch_pool.close()
        _restore_signal_handlers(signal_handlers)
        log_file.close()

    return {
        "global_step": global_step,
        "best_human_val_metric": best_human_val_metric,
        "best_val_metric": best_human_val_metric,
        "seen_selfplay_shards": len(seen_selfplay_shards),
        "selfplay_buffer_samples": len(replay_buffer),
        "cumulative_added_shards": cumulative_added_shards,
        "cumulative_added_samples": cumulative_added_samples,
        "last_ingest_status": last_ingest_status,
        "output_dir": str(output_dir),
        "interrupted": interrupt_reason is not None,
        "interrupt_reason": interrupt_reason,
        "last_human_val_policy_loss": None if last_human_val_metrics is None else last_human_val_metrics["human_val_policy_loss"],
        "last_human_val_value_loss": None if last_human_val_metrics is None else last_human_val_metrics["human_val_value_loss"],
        "last_human_val_wdl_loss": None if last_human_val_metrics is None else last_human_val_metrics.get("human_val_wdl_loss"),
        "last_human_val_total_loss": None if last_human_val_metrics is None else last_human_val_metrics["human_val_total_loss"],
        "last_human_val_samples": None if last_human_val_metrics is None else int(last_human_val_metrics["human_val_samples"]),
    }


def _available_cpu_ids() -> list[int]:
    if hasattr(os, "sched_getaffinity"):
        return sorted(int(cpu_id) for cpu_id in os.sched_getaffinity(0))
    cpu_count = os.cpu_count() or 1
    return list(range(cpu_count))


def _plan_worker_cpu_groups(num_workers: int, reserved_cores: int) -> list[list[int]]:
    if num_workers <= 0:
        return []

    cpu_ids = _available_cpu_ids()
    if not cpu_ids:
        return [[] for _ in range(num_workers)]

    reserve = min(max(0, reserved_cores), max(len(cpu_ids) - 1, 0))
    worker_cpu_ids = cpu_ids[reserve:] if reserve < len(cpu_ids) else cpu_ids
    if not worker_cpu_ids:
        worker_cpu_ids = cpu_ids

    chunks: list[list[int]] = []
    base = len(worker_cpu_ids) // num_workers
    extra = len(worker_cpu_ids) % num_workers
    cursor = 0
    for worker_index in range(num_workers):
        width = base + (1 if worker_index < extra else 0)
        if width <= 0:
            width = 1
        chunk = worker_cpu_ids[cursor : cursor + width]
        if not chunk:
            chunk = [worker_cpu_ids[worker_index % len(worker_cpu_ids)]]
        chunks.append(chunk)
        cursor += width
    return chunks


def _try_set_process_affinity(cpu_ids: list[int]) -> None:
    if not cpu_ids or not hasattr(os, "sched_setaffinity"):
        return
    try:
        os.sched_setaffinity(0, set(int(cpu_id) for cpu_id in cpu_ids))
    except Exception:
        return


def _try_set_thread_affinity(cpu_ids: list[int]) -> None:
    if not cpu_ids or not hasattr(os, "sched_setaffinity"):
        return
    try:
        os.sched_setaffinity(threading.get_native_id(), set(int(cpu_id) for cpu_id in cpu_ids))
    except Exception:
        return


def _queue_put_with_stop(
    output_queue: Any,
    item: dict[str, Any],
    stop_event: Any,
    timeout_s: float = 1.0,
) -> bool:
    while not stop_event.is_set():
        try:
            output_queue.put(item, timeout=timeout_s)
            return True
        except queue.Full:
            continue
    return False


def _human_batch_prefetch_worker(
    worker_id: int,
    payload: dict[str, Any],
    cpu_ids: list[int],
    output_queue: mp.Queue,
    stop_event: mp.Event,
) -> None:
    try:
        _try_set_process_affinity(cpu_ids)
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        rng = random.Random(int(payload["seed"]) + worker_id * 1_000_003)
        catalog = _HumanTrainCatalog.from_dir(Path(payload["human_data_dir"]))
        shard_cache = _ShardCache(int(payload["shard_cache_size"]))

        while not stop_event.is_set():
            blobs = _sample_human_samples(
                specs=catalog.train_specs,
                weights=catalog.train_weights,
                shard_cache=shard_cache,
                batch_size=int(payload["batch_size"]),
                rng=rng,
                samples_per_unit=int(payload["samples_per_unit"]),
            )
            batch = _collate_sample_blobs(blobs)
            if not _queue_put_with_stop(
                output_queue=output_queue,
                item={"kind": "batch", "worker_id": worker_id, "batch": batch},
                stop_event=stop_event,
            ):
                break
    except Exception as exc:
        _queue_put_with_stop(
            output_queue=output_queue,
            item={"kind": "error", "worker_id": worker_id, "error": repr(exc)},
            stop_event=stop_event,
        )


def _human_batch_thread_worker(
    worker_id: int,
    payload: dict[str, Any],
    shard_cache: _ShardCache,
    cpu_ids: list[int],
    output_queue: queue.Queue[dict[str, Any]],
    stop_event: threading.Event,
) -> None:
    try:
        _try_set_thread_affinity(cpu_ids)
        rng = random.Random(int(payload["seed"]) + worker_id * 1_000_003)
        catalog = _HumanTrainCatalog.from_dir(Path(payload["human_data_dir"]))

        while not stop_event.is_set():
            blobs = _sample_human_samples(
                specs=catalog.train_specs,
                weights=catalog.train_weights,
                shard_cache=shard_cache,
                batch_size=int(payload["batch_size"]),
                rng=rng,
                samples_per_unit=int(payload["samples_per_unit"]),
            )
            batch = _collate_sample_blobs(blobs)
            if not _queue_put_with_stop(
                output_queue=output_queue,
                item={"kind": "batch", "worker_id": worker_id, "batch": batch},
                stop_event=stop_event,
            ):
                break
    except Exception as exc:
        _queue_put_with_stop(
            output_queue=output_queue,
            item={"kind": "error", "worker_id": worker_id, "error": repr(exc)},
            stop_event=stop_event,
        )


def _sample_training_batch(
    human_catalog: _HumanTrainCatalog,
    shard_cache: _ShardCache,
    replay_buffer: _SelfPlayReplayBuffer,
    human_batch_size: int,
    selfplay_batch_size: int,
    samples_per_unit: int,
    selfplay_dir_sampling_ratios: list[float] | None = None,
) -> dict[str, Tensor]:
    sample_blobs: list[dict[str, Any]] = []
    rng = random

    if human_batch_size > 0:
        sample_blobs.extend(
            _sample_human_samples(
                specs=human_catalog.train_specs,
                weights=human_catalog.train_weights,
                shard_cache=shard_cache,
                batch_size=human_batch_size,
                rng=rng,
                samples_per_unit=samples_per_unit,
            )
        )

    if selfplay_batch_size > 0 and len(replay_buffer) > 0:
        sample_blobs.extend(
            _sample_selfplay_samples(
                replay_buffer=replay_buffer,
                shard_cache=shard_cache,
                batch_size=selfplay_batch_size,
                rng=rng,
                samples_per_unit=samples_per_unit,
                source_ratios=selfplay_dir_sampling_ratios,
            )
        )

    if not sample_blobs:
        raise RuntimeError("training batch sampling produced no samples")

    random.shuffle(sample_blobs)
    return _collate_sample_blobs(sample_blobs)


def _sample_human_samples(
    specs: list[_HumanShardSpec],
    weights: list[int],
    shard_cache: _ShardCache,
    batch_size: int,
    rng: random.Random | Any,
    samples_per_unit: int,
) -> list[dict[str, Any]]:
    selected = _select_weighted_units(weights, batch_size, rng, samples_per_unit)
    result: list[dict[str, Any]] = []
    for spec_index, count in selected:
        spec = specs[spec_index]
        shard = shard_cache.get(spec.path)
        raw_samples = _get_legacy_samples(shard, spec.path)
        local_indices = _sample_indices(count, spec.sample_count, rng)
        for local_index in local_indices:
            blob = _legacy_sample_to_blob(raw_samples[local_index])
            blob["source_group_id"] = -1
            result.append(blob)
    return result


def _sample_selfplay_samples(
    replay_buffer: _SelfPlayReplayBuffer,
    shard_cache: _ShardCache,
    batch_size: int,
    rng: random.Random | Any,
    samples_per_unit: int,
    source_ratios: list[float] | None = None,
) -> list[dict[str, Any]]:
    spans = list(replay_buffer.spans)
    result: list[dict[str, Any]] = []

    def _sample_from_span_subset(subset: list[_SelfPlaySpan], count: int) -> None:
        if count <= 0 or not subset:
            return
        weights = [span.sample_count for span in subset]
        selected = _select_weighted_units(weights, count, rng, samples_per_unit)
        for span_index, span_count in selected:
            span = subset[span_index]
            shard_path = Path(span.path)
            shard = shard_cache.get(shard_path)
            _get_selfplay_shard_sample_count(shard, shard_path)
            local_indices = [rng.randrange(span.start, span.end) for _ in range(span_count)]
            blobs = _extract_tensorized_sample_blobs(shard, local_indices)
            try:
                source_group_id = int(span.source_group)
            except (TypeError, ValueError):
                source_group_id = 0
            for blob in blobs:
                blob["source_group_id"] = source_group_id
            result.extend(blobs)

    if source_ratios:
        grouped_spans: dict[str, list[_SelfPlaySpan]] = {}
        for span in spans:
            grouped_spans.setdefault(str(span.source_group), []).append(span)
        group_counts = _allocate_selfplay_group_counts(
            grouped_spans=grouped_spans,
            source_ratios=source_ratios,
            total_samples=batch_size,
            rng=rng,
        )
        if group_counts:
            for source_group, count in group_counts:
                _sample_from_span_subset(grouped_spans.get(source_group, []), count)
        else:
            _sample_from_span_subset(spans, batch_size)
    else:
        _sample_from_span_subset(spans, batch_size)
    return result


def _allocate_selfplay_group_counts(
    grouped_spans: dict[str, list[_SelfPlaySpan]],
    source_ratios: list[float],
    total_samples: int,
    rng: random.Random | Any,
) -> list[tuple[str, int]]:
    if total_samples < 1:
        return []

    available: list[tuple[str, float]] = []
    for source_index, ratio in enumerate(source_ratios):
        source_group = str(source_index)
        if float(ratio) <= 0.0:
            continue
        group_sample_count = sum(span.sample_count for span in grouped_spans.get(source_group, []))
        if group_sample_count <= 0:
            continue
        available.append((source_group, float(ratio)))

    if not available:
        for source_group, spans in grouped_spans.items():
            if sum(span.sample_count for span in spans) > 0:
                return [(source_group, total_samples)]
        return []

    total_ratio = sum(ratio for _source_group, ratio in available)
    raw_counts = [
        (source_group, total_samples * ratio / total_ratio)
        for source_group, ratio in available
    ]
    counts: dict[str, int] = {source_group: int(math.floor(raw)) for source_group, raw in raw_counts}
    remaining = total_samples - sum(counts.values())
    remainders = [
        (raw - math.floor(raw), rng.random(), source_group)
        for source_group, raw in raw_counts
    ]
    remainders.sort(reverse=True)
    for _fractional, _jitter, source_group in remainders[:remaining]:
        counts[source_group] += 1

    return [(source_group, counts[source_group]) for source_group, _ratio in available if counts[source_group] > 0]


def _select_weighted_units(
    weights: list[int],
    total_samples: int,
    rng: random.Random | Any,
    samples_per_unit: int,
) -> list[tuple[int, int]]:
    if total_samples < 1:
        return []

    available = [index for index, weight in enumerate(weights) if weight > 0]
    if not available:
        return []

    target_units = min(len(available), max(1, math.ceil(total_samples / float(samples_per_unit))))
    selected_units: list[int] = []
    pool = available.copy()
    for _ in range(target_units):
        chosen = _weighted_choice_from_subset(weights, pool, rng)
        selected_units.append(chosen)
        pool.remove(chosen)

    counts = {index: 1 for index in selected_units}
    remaining = total_samples - len(selected_units)
    while remaining > 0:
        chosen = _weighted_choice_from_subset(weights, selected_units, rng)
        counts[chosen] += 1
        remaining -= 1

    return [(index, counts[index]) for index in selected_units]


def _weighted_choice_from_subset(
    weights: list[int],
    indices: list[int],
    rng: random.Random | Any,
) -> int:
    total_weight = float(sum(weights[index] for index in indices))
    if total_weight <= 0:
        return indices[0]

    threshold = rng.random() * total_weight
    cumulative = 0.0
    for index in indices:
        cumulative += float(weights[index])
        if cumulative >= threshold:
            return index
    return indices[-1]


def _sample_indices(count: int, upper_bound: int, rng: random.Random | Any) -> list[int]:
    if count <= upper_bound:
        return rng.sample(range(upper_bound), count)
    return [rng.randrange(upper_bound) for _ in range(count)]


def _legacy_sample_to_blob(sample: dict[str, Any]) -> dict[str, Any]:
    z_value = float(sample["z"])
    return {
        "state": torch.as_tensor(sample["state"], dtype=torch.float32).contiguous(),
        "policy_idxs": torch.as_tensor(sample["idxs"], dtype=torch.int64).contiguous(),
        "policy_probs": torch.as_tensor(sample["probs"], dtype=torch.float32).contiguous(),
        "z": z_value,
        "wdl_target": _wdl_target_from_z(z_value),
    }


def _wdl_target_from_z(z_value: float) -> list[float]:
    # Map scalar outcome in [-1, 1] (side-to-move view) to 3-class W/D/L one-hot.
    # Shard z is the raw game outcome (possibly scaled for human early-game labels);
    # sign determines the class, magnitude is ignored for the classifier.
    if z_value > 1e-6:
        return [1.0, 0.0, 0.0]
    if z_value < -1e-6:
        return [0.0, 0.0, 1.0]
    return [0.0, 1.0, 0.0]


def _extract_tensorized_sample_blobs(shard: dict[str, Any], local_indices: list[int]) -> list[dict[str, Any]]:
    offsets = shard["policy_offsets"].to(torch.int64)
    shard_wdl_target = shard.get("wdl_target")
    # oracle_value (Pikafish d=15+ calibrated value, optional). NaN = no oracle for this position
    # → train falls back to z-based loss for that sample. See tools/oracle_value_labeler.py.
    shard_oracle = shard.get("oracle_value")
    # v11 additions: oracle_policy (CSR triple) and sample_weight, both optional. Empty oracle
    # slots (offsets[i+1] == offsets[i]) and absent sample_weight default to no-op behavior.
    shard_oracle_policy_offsets = shard.get("oracle_policy_offsets")
    shard_oracle_policy_idxs = shard.get("oracle_policy_idxs")
    shard_oracle_policy_probs = shard.get("oracle_policy_probs")
    shard_teacher_q_offsets = shard.get("teacher_q_offsets")
    shard_teacher_q_idxs = shard.get("teacher_q_idxs")
    shard_teacher_q_values = shard.get("teacher_q_values")
    shard_sample_weight = shard.get("sample_weight")
    shard_bad_move = shard.get("bad_move")
    if shard_oracle_policy_offsets is not None:
        shard_oracle_policy_offsets = shard_oracle_policy_offsets.to(torch.int64)
    if shard_teacher_q_offsets is not None:
        shard_teacher_q_offsets = shard_teacher_q_offsets.to(torch.int64)
    # v12: legal moves CSR. Used to mask softmax in policy CE so non-legal logits don't
    # consume model capacity. Per-sample empty slice → trainer treats that sample as no-mask.
    shard_legal_offsets = shard.get("legal_offsets")
    shard_legal_idxs = shard.get("legal_idxs")
    if shard_legal_offsets is not None:
        shard_legal_offsets = shard_legal_offsets.to(torch.int64)
    result: list[dict[str, Any]] = []
    for local_index in local_indices:
        start = int(offsets[local_index].item())
        end = int(offsets[local_index + 1].item())
        z_value = float(shard["z"][local_index].item())
        if shard_wdl_target is not None:
            wdl_target = shard_wdl_target[local_index].to(torch.float32).tolist()
        else:
            wdl_target = _wdl_target_from_z(z_value)
        blob: dict[str, Any] = {
            "state": shard["state"][local_index].to(torch.float32).contiguous(),
            "policy_idxs": shard["policy_idxs"][start:end].to(torch.int64).contiguous(),
            "policy_probs": shard["policy_probs"][start:end].to(torch.float32).contiguous(),
            "z": z_value,
            "wdl_target": wdl_target,
        }
        if shard_oracle is not None:
            blob["oracle_value"] = float(shard_oracle[local_index].item())
        if shard_oracle_policy_offsets is not None and shard_oracle_policy_idxs is not None:
            op_start = int(shard_oracle_policy_offsets[local_index].item())
            op_end = int(shard_oracle_policy_offsets[local_index + 1].item())
            blob["oracle_policy_idxs"] = shard_oracle_policy_idxs[op_start:op_end].to(torch.int64).contiguous()
            blob["oracle_policy_probs"] = shard_oracle_policy_probs[op_start:op_end].to(torch.float32).contiguous()
        if (
            shard_teacher_q_offsets is not None
            and shard_teacher_q_idxs is not None
            and shard_teacher_q_values is not None
        ):
            tq_start = int(shard_teacher_q_offsets[local_index].item())
            tq_end = int(shard_teacher_q_offsets[local_index + 1].item())
            blob["teacher_q_idxs"] = shard_teacher_q_idxs[tq_start:tq_end].to(torch.int64).contiguous()
            blob["teacher_q_values"] = shard_teacher_q_values[tq_start:tq_end].to(torch.float32).contiguous()
        if shard_sample_weight is not None:
            blob["sample_weight"] = float(shard_sample_weight[local_index].item())
        if shard_bad_move is not None:
            blob["bad_move"] = int(shard_bad_move[local_index].item())
        if shard_legal_offsets is not None and shard_legal_idxs is not None:
            lg_start = int(shard_legal_offsets[local_index].item())
            lg_end = int(shard_legal_offsets[local_index + 1].item())
            if lg_end > lg_start:
                blob["legal_idxs"] = shard_legal_idxs[lg_start:lg_end].to(torch.int64).contiguous()
        result.append(blob)
    return result


def _collate_sample_blobs(sample_blobs: list[dict[str, Any]]) -> dict[str, Tensor]:
    states = torch.stack([blob["state"] for blob in sample_blobs], dim=0).to(torch.float32).contiguous()

    policy_offsets = [0]
    idx_chunks: list[Tensor] = []
    prob_chunks: list[Tensor] = []
    z_values = []
    wdl_targets: list[list[float]] = []
    oracle_values: list[float] = []
    has_any_oracle = False
    # v11: oracle_policy CSR + sample_weight. Default to no-op if absent.
    oracle_policy_offsets = [0]
    oracle_policy_idx_chunks: list[Tensor] = []
    oracle_policy_prob_chunks: list[Tensor] = []
    has_any_oracle_policy = False
    # v12.5: optional action-value teacher targets, CSR per sample.
    teacher_q_offsets = [0]
    teacher_q_idx_chunks: list[Tensor] = []
    teacher_q_value_chunks: list[Tensor] = []
    has_any_teacher_q = False
    sample_weights: list[float] = []
    has_any_sample_weight = False
    bad_moves: list[int] = []
    has_any_bad_move = False
    # v12: legal-mask CSR. Per-sample empty slice → unmasked softmax fallback.
    legal_offsets = [0]
    legal_idx_chunks: list[Tensor] = []
    has_any_legal = False
    source_group_ids: list[int] = []
    for blob in sample_blobs:
        idxs = blob["policy_idxs"].to(torch.int64).contiguous()
        probs = blob["policy_probs"].to(torch.float32).contiguous()
        idx_chunks.append(idxs)
        prob_chunks.append(probs)
        policy_offsets.append(policy_offsets[-1] + int(idxs.numel()))
        z_value = float(blob["z"])
        z_values.append(z_value)
        wdl = blob.get("wdl_target")
        if wdl is None:
            wdl_targets.append(_wdl_target_from_z(z_value))
        else:
            wdl_targets.append([float(x) for x in wdl])
        # oracle_value
        ov = blob.get("oracle_value")
        if ov is None:
            oracle_values.append(float("nan"))
        else:
            oracle_values.append(float(ov))
            has_any_oracle = True
        # oracle_policy: per-sample CSR slice (may be empty)
        op_idxs = blob.get("oracle_policy_idxs")
        op_probs = blob.get("oracle_policy_probs")
        if op_idxs is not None and op_probs is not None:
            op_idxs_t = op_idxs.to(torch.int64).contiguous()
            op_probs_t = op_probs.to(torch.float32).contiguous()
            oracle_policy_idx_chunks.append(op_idxs_t)
            oracle_policy_prob_chunks.append(op_probs_t)
            oracle_policy_offsets.append(oracle_policy_offsets[-1] + int(op_idxs_t.numel()))
            has_any_oracle_policy = True
        else:
            oracle_policy_offsets.append(oracle_policy_offsets[-1])
        # teacher_q: per-sample CSR slice (may be empty)
        tq_idxs = blob.get("teacher_q_idxs")
        tq_values = blob.get("teacher_q_values")
        if tq_idxs is not None and tq_values is not None:
            tq_idxs_t = tq_idxs.to(torch.int64).contiguous()
            tq_values_t = tq_values.to(torch.float32).contiguous()
            teacher_q_idx_chunks.append(tq_idxs_t)
            teacher_q_value_chunks.append(tq_values_t)
            teacher_q_offsets.append(teacher_q_offsets[-1] + int(tq_idxs_t.numel()))
            has_any_teacher_q = True
        else:
            teacher_q_offsets.append(teacher_q_offsets[-1])
        # sample_weight: default 1.0 if absent
        sw = blob.get("sample_weight")
        if sw is None:
            sample_weights.append(1.0)
        else:
            sample_weights.append(float(sw))
            has_any_sample_weight = True
        bad_move = blob.get("bad_move")
        if bad_move is None:
            bad_moves.append(-1)
        else:
            bad_moves.append(int(bad_move))
            has_any_bad_move = True
        # legal_idxs: per-sample CSR slice (may be empty → no-mask for that sample)
        leg = blob.get("legal_idxs")
        if leg is not None and leg.numel() > 0:
            leg_t = leg.to(torch.int64).contiguous()
            legal_idx_chunks.append(leg_t)
            legal_offsets.append(legal_offsets[-1] + int(leg_t.numel()))
            has_any_legal = True
        else:
            legal_offsets.append(legal_offsets[-1])
        source_group_ids.append(int(blob.get("source_group_id", -1)))

    all_policy_idxs = torch.cat(idx_chunks, dim=0).contiguous() if idx_chunks else torch.empty(0, dtype=torch.int64)
    all_policy_probs = torch.cat(prob_chunks, dim=0).contiguous() if prob_chunks else torch.empty(0, dtype=torch.float32)

    out: dict[str, Tensor] = {
        "state": states,
        "policy_offsets": torch.tensor(policy_offsets, dtype=torch.int64),
        "policy_idxs": all_policy_idxs,
        "policy_probs": all_policy_probs,
        "z": torch.tensor(z_values, dtype=torch.float32),
        "wdl_target": torch.tensor(wdl_targets, dtype=torch.float32),
        "source_group_id": torch.tensor(source_group_ids, dtype=torch.int16),
    }
    if has_any_oracle:
        out["oracle_value"] = torch.tensor(oracle_values, dtype=torch.float32)
    if has_any_oracle_policy:
        out["oracle_policy_offsets"] = torch.tensor(oracle_policy_offsets, dtype=torch.int64)
        out["oracle_policy_idxs"] = (
            torch.cat(oracle_policy_idx_chunks, dim=0).contiguous()
            if oracle_policy_idx_chunks else torch.empty(0, dtype=torch.int64)
        )
        out["oracle_policy_probs"] = (
            torch.cat(oracle_policy_prob_chunks, dim=0).contiguous()
            if oracle_policy_prob_chunks else torch.empty(0, dtype=torch.float32)
        )
    if has_any_teacher_q:
        out["teacher_q_offsets"] = torch.tensor(teacher_q_offsets, dtype=torch.int64)
        out["teacher_q_idxs"] = (
            torch.cat(teacher_q_idx_chunks, dim=0).contiguous()
            if teacher_q_idx_chunks else torch.empty(0, dtype=torch.int64)
        )
        out["teacher_q_values"] = (
            torch.cat(teacher_q_value_chunks, dim=0).contiguous()
            if teacher_q_value_chunks else torch.empty(0, dtype=torch.float32)
        )
    if has_any_sample_weight:
        out["sample_weight"] = torch.tensor(sample_weights, dtype=torch.float32)
    if has_any_bad_move:
        out["bad_move"] = torch.tensor(bad_moves, dtype=torch.int64)
    if has_any_legal:
        out["legal_offsets"] = torch.tensor(legal_offsets, dtype=torch.int64)
        out["legal_idxs"] = (
            torch.cat(legal_idx_chunks, dim=0).contiguous()
            if legal_idx_chunks else torch.empty(0, dtype=torch.int64)
        )
    return out


def _move_batch_to_device(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    non_blocking = device.type == "cuda"
    out: dict[str, Tensor] = {
        "state": batch["state"].to(device=device, dtype=torch.float32, non_blocking=non_blocking),
        "policy_offsets": batch["policy_offsets"].to(device=device, dtype=torch.int64, non_blocking=non_blocking),
        "policy_idxs": batch["policy_idxs"].to(device=device, dtype=torch.int64, non_blocking=non_blocking),
        "policy_probs": batch["policy_probs"].to(device=device, dtype=torch.float32, non_blocking=non_blocking),
        "z": batch["z"].to(device=device, dtype=torch.float32, non_blocking=non_blocking),
        "wdl_target": batch["wdl_target"].to(device=device, dtype=torch.float32, non_blocking=non_blocking),
    }
    if "oracle_value" in batch:
        out["oracle_value"] = batch["oracle_value"].to(
            device=device, dtype=torch.float32, non_blocking=non_blocking,
        )
    if "oracle_policy_offsets" in batch:
        out["oracle_policy_offsets"] = batch["oracle_policy_offsets"].to(
            device=device, dtype=torch.int64, non_blocking=non_blocking,
        )
        out["oracle_policy_idxs"] = batch["oracle_policy_idxs"].to(
            device=device, dtype=torch.int64, non_blocking=non_blocking,
        )
        out["oracle_policy_probs"] = batch["oracle_policy_probs"].to(
            device=device, dtype=torch.float32, non_blocking=non_blocking,
        )
    if "teacher_q_offsets" in batch:
        out["teacher_q_offsets"] = batch["teacher_q_offsets"].to(
            device=device, dtype=torch.int64, non_blocking=non_blocking,
        )
        out["teacher_q_idxs"] = batch["teacher_q_idxs"].to(
            device=device, dtype=torch.int64, non_blocking=non_blocking,
        )
        out["teacher_q_values"] = batch["teacher_q_values"].to(
            device=device, dtype=torch.float32, non_blocking=non_blocking,
        )
    if "sample_weight" in batch:
        out["sample_weight"] = batch["sample_weight"].to(
            device=device, dtype=torch.float32, non_blocking=non_blocking,
        )
    if "bad_move" in batch:
        out["bad_move"] = batch["bad_move"].to(
            device=device, dtype=torch.int64, non_blocking=non_blocking,
        )
    if "legal_offsets" in batch:
        out["legal_offsets"] = batch["legal_offsets"].to(
            device=device, dtype=torch.int64, non_blocking=non_blocking,
        )
        out["legal_idxs"] = batch["legal_idxs"].to(
            device=device, dtype=torch.int64, non_blocking=non_blocking,
        )
    if "source_group_id" in batch:
        out["source_group_id"] = batch["source_group_id"].to(
            device=device, dtype=torch.int16, non_blocking=non_blocking,
        )
    return out


def _compute_training_losses(
    policy_logits: Tensor,
    value_scalar: Tensor,
    policy_offsets: Tensor,
    policy_idxs: Tensor,
    policy_probs: Tensor,
    z: Tensor,
    wdl_logits: Tensor | None = None,
    wdl_target: Tensor | None = None,
    wdl_loss_weight: float = 0.0,
    policy_loss_weight: float = 1.0,
    value_loss_weight: float = 1.0,
    wdl_value_consistency_weight: float = 0.0,
    value_target_scale: float = 1.0,
    oracle_value: Tensor | None = None,
    use_oracle_value: bool = True,
    oracle_policy_offsets: Tensor | None = None,
    oracle_policy_idxs: Tensor | None = None,
    oracle_policy_probs: Tensor | None = None,
    policy_oracle_alpha: float = 0.0,
    teacher_q_offsets: Tensor | None = None,
    teacher_q_idxs: Tensor | None = None,
    teacher_q_values: Tensor | None = None,
    teacher_q_loss_weight: float = 0.0,
    teacher_q_temperature_cp: float = 80.0,
    teacher_q_pairwise_loss_weight: float = 0.0,
    teacher_q_pairwise_margin_logit: float = 0.25,
    teacher_q_pairwise_min_gap_cp: float = 80.0,
    teacher_q_pairwise_beta: float = 1.0,
    teacher_q_ref_policy_logits: Tensor | None = None,
    teacher_q_pairwise_bad_move_only: bool = False,
    bad_move_suppression_loss_weight: float = 0.0,
    bad_move_suppression_margin_logit: float = 0.75,
    bad_move_suppression_min_gap_cp: float = 80.0,
    bad_move_suppression_beta: float = 2.0,
    bad_move: Tensor | None = None,
    sample_weight: Tensor | None = None,
    legal_offsets: Tensor | None = None,
    legal_idxs: Tensor | None = None,
) -> dict[str, Tensor]:
    """Compute the multi-task training losses.

    Value-target selection (per-sample, in priority order):
      1. ``oracle_value[i]`` if not NaN — calibrated Pikafish-d=15 eval, used as-is.
      2. ``z[i] * value_target_scale`` — noisy game outcome (legacy fallback).

    Policy target combination (v11 oracle-policy distillation):
      ``policy_loss = (1-α) * CE(p, π_MCTS) + α * CE(p, π_oracle)``
    where π_oracle is built from Pikafish multipv at oracle-labeling time.  When
    ``oracle_policy_*`` is None or ``policy_oracle_alpha == 0`` (default v10
    behavior), only the MCTS-visit policy target is used.  Samples whose oracle
    slot is empty (offsets[i+1] == offsets[i]) contribute 0 to the oracle CE
    term — the ``_sparse_policy_cross_entropy`` scatter naturally handles this.

    Sample weights (v11 hard-position mining):
      If ``sample_weight`` is provided, all per-sample losses (policy + value +
      wdl) are weighted-averaged with these weights instead of plain mean.
      Weight=1.0 is the no-op default.

    Action-value distillation (v12.5):
      If ``teacher_q_*`` is present and ``teacher_q_loss_weight > 0``, train the
      policy logits against a per-position softmax over teacher action values.
      The labeler stores centipawns from the root side-to-move's perspective.
    """
    # ---- Policy loss: MCTS visits + optional oracle policy ----
    use_weights = sample_weight is not None
    if use_weights or (policy_oracle_alpha > 0.0 and oracle_policy_offsets is not None):
        policy_loss_mcts_per_sample = _sparse_policy_cross_entropy(
            policy_logits=policy_logits,
            policy_offsets=policy_offsets,
            policy_idxs=policy_idxs,
            policy_probs=policy_probs,
            return_per_sample=True,
            legal_offsets=legal_offsets,
            legal_idxs=legal_idxs,
        )
        if policy_oracle_alpha > 0.0 and oracle_policy_offsets is not None:
            policy_loss_oracle_per_sample = _sparse_policy_cross_entropy(
                policy_logits=policy_logits,
                policy_offsets=oracle_policy_offsets,
                policy_idxs=oracle_policy_idxs,
                policy_probs=oracle_policy_probs,
                return_per_sample=True,
                legal_offsets=legal_offsets,
                legal_idxs=legal_idxs,
            )
            policy_per_sample = (
                (1.0 - float(policy_oracle_alpha)) * policy_loss_mcts_per_sample
                + float(policy_oracle_alpha) * policy_loss_oracle_per_sample
            )
        else:
            policy_per_sample = policy_loss_mcts_per_sample
        if use_weights:
            sw = sample_weight.float()
            policy_loss = (policy_per_sample * sw).sum() / sw.sum().clamp_min(1e-6)
        else:
            policy_loss = policy_per_sample.mean()
    else:
        policy_loss = _sparse_policy_cross_entropy(
            policy_logits=policy_logits,
            policy_offsets=policy_offsets,
            policy_idxs=policy_idxs,
            policy_probs=policy_probs,
            legal_offsets=legal_offsets,
            legal_idxs=legal_idxs,
        )

    # ---- Value loss: oracle_value with z fallback, optionally sample-weighted ----
    scaled_z = z * float(value_target_scale)
    if use_oracle_value and oracle_value is not None:
        oracle_mask = torch.isfinite(oracle_value)
        target = torch.where(oracle_mask, oracle_value, scaled_z)
        n_oracle = int(oracle_mask.sum().item())
    else:
        target = scaled_z
        n_oracle = 0
    value_se = (value_scalar.squeeze(1) - target) ** 2
    if use_weights:
        sw = sample_weight.float()
        value_loss = (value_se * sw).sum() / sw.sum().clamp_min(1e-6)
    else:
        value_loss = value_se.mean()

    # ---- WDL loss: standard CE, optionally sample-weighted ----
    if wdl_logits is not None and wdl_target is not None and wdl_loss_weight > 0.0:
        log_probs = F.log_softmax(wdl_logits.float(), dim=1)
        wdl_per_sample = -(wdl_target.float() * log_probs).sum(dim=1)
        if use_weights:
            sw = sample_weight.float()
            wdl_loss = (wdl_per_sample * sw).sum() / sw.sum().clamp_min(1e-6)
        else:
            wdl_loss = wdl_per_sample.mean()
        probs = log_probs.exp()
        wdl_entropy = -(probs * log_probs).sum(dim=1).mean().detach()
    else:
        wdl_loss = torch.zeros((), device=policy_logits.device, dtype=policy_logits.dtype)
        wdl_entropy = torch.zeros((), device=policy_logits.device, dtype=policy_logits.dtype)

    # ---- Optional scalar/WDL value bridge ----
    # MCTS still consumes value_scalar, but WDL CE usually gives a better-shaped
    # probabilistic target.  A tiny regularizer nudges the scalar head toward
    # E[WDL]=P(win)-P(loss) without letting this term drag the WDL classifier.
    if wdl_logits is not None and wdl_value_consistency_weight > 0.0:
        wdl_probs = torch.softmax(wdl_logits.float(), dim=1)
        wdl_expectation = (wdl_probs[:, 0] - wdl_probs[:, 2]).detach()
        consistency_se = (value_scalar.squeeze(1).float() - wdl_expectation) ** 2
        if use_weights:
            sw = sample_weight.float()
            wdl_value_consistency_loss = (consistency_se * sw).sum() / sw.sum().clamp_min(1e-6)
        else:
            wdl_value_consistency_loss = consistency_se.mean()
    else:
        wdl_value_consistency_loss = torch.zeros(
            (), device=policy_logits.device, dtype=policy_logits.dtype
        )

    # ---- Optional action-value distillation over labeled candidate moves ----
    if (
        teacher_q_loss_weight > 0.0
        and teacher_q_offsets is not None
        and teacher_q_idxs is not None
        and teacher_q_values is not None
        and teacher_q_idxs.numel() > 0
    ):
        teacher_q_per_sample, teacher_q_valid = _teacher_q_cross_entropy(
            policy_logits=policy_logits,
            teacher_q_offsets=teacher_q_offsets,
            teacher_q_idxs=teacher_q_idxs,
            teacher_q_values=teacher_q_values,
            temperature_cp=float(teacher_q_temperature_cp),
            legal_offsets=legal_offsets,
            legal_idxs=legal_idxs,
        )
        if use_weights:
            sw = sample_weight.float() * teacher_q_valid.float()
            teacher_q_loss = (teacher_q_per_sample * sw).sum() / sw.sum().clamp_min(1e-6)
        else:
            teacher_q_loss = (
                (teacher_q_per_sample * teacher_q_valid.float()).sum()
                / teacher_q_valid.float().sum().clamp_min(1e-6)
            )
        n_teacher_q = int(teacher_q_valid.sum().item())
    else:
        teacher_q_loss = torch.zeros((), device=policy_logits.device, dtype=policy_logits.dtype)
        n_teacher_q = 0

    # ---- Optional surgical pairwise action-value ranking ----
    # This nudges only the teacher-best candidate above clearly worse labeled
    # candidates.  It is less style-forcing than the full teacher_q CE because
    # it does not prescribe a complete probability distribution over moves.
    if (
        teacher_q_pairwise_loss_weight > 0.0
        and teacher_q_offsets is not None
        and teacher_q_idxs is not None
        and teacher_q_values is not None
        and teacher_q_idxs.numel() > 0
    ):
        pairwise_per_sample, pairwise_valid = _teacher_q_pairwise_margin_loss(
            policy_logits=policy_logits,
            teacher_q_offsets=teacher_q_offsets,
            teacher_q_idxs=teacher_q_idxs,
            teacher_q_values=teacher_q_values,
            margin_logit=float(teacher_q_pairwise_margin_logit),
            min_gap_cp=float(teacher_q_pairwise_min_gap_cp),
            temperature_cp=float(teacher_q_temperature_cp),
            beta=float(teacher_q_pairwise_beta),
            reference_policy_logits=teacher_q_ref_policy_logits,
            bad_move_idxs=bad_move,
            bad_move_only=bool(teacher_q_pairwise_bad_move_only),
            legal_offsets=legal_offsets,
            legal_idxs=legal_idxs,
        )
        if use_weights:
            sw = sample_weight.float() * pairwise_valid.float()
            teacher_q_pairwise_loss = (pairwise_per_sample * sw).sum() / sw.sum().clamp_min(1e-6)
        else:
            teacher_q_pairwise_loss = (
                (pairwise_per_sample * pairwise_valid.float()).sum()
                / pairwise_valid.float().sum().clamp_min(1e-6)
            )
        n_teacher_q_pairwise = int(pairwise_valid.sum().item())
    else:
        teacher_q_pairwise_loss = torch.zeros(
            (), device=policy_logits.device, dtype=policy_logits.dtype
        )
        n_teacher_q_pairwise = 0

    # ---- Optional bad-move-only local suppression ----
    # Pairwise repair asks the best move to outrank the known bad move, but that
    # can still leave the bad move as top-1 when the whole local distribution is
    # very sharp.  This term directly pushes the known bad move below the frozen
    # V13 reference, without teaching a full teacher distribution.
    if (
        bad_move_suppression_loss_weight > 0.0
        and teacher_q_ref_policy_logits is not None
        and bad_move is not None
        and teacher_q_offsets is not None
        and teacher_q_idxs is not None
        and teacher_q_values is not None
        and teacher_q_idxs.numel() > 0
    ):
        suppress_per_sample, suppress_valid = _teacher_q_bad_move_suppression_loss(
            policy_logits=policy_logits,
            reference_policy_logits=teacher_q_ref_policy_logits,
            teacher_q_offsets=teacher_q_offsets,
            teacher_q_idxs=teacher_q_idxs,
            teacher_q_values=teacher_q_values,
            bad_move_idxs=bad_move,
            margin_logit=float(bad_move_suppression_margin_logit),
            min_gap_cp=float(bad_move_suppression_min_gap_cp),
            beta=float(bad_move_suppression_beta),
            legal_offsets=legal_offsets,
            legal_idxs=legal_idxs,
        )
        if use_weights:
            sw = sample_weight.float() * suppress_valid.float()
            bad_move_suppression_loss = (suppress_per_sample * sw).sum() / sw.sum().clamp_min(1e-6)
        else:
            bad_move_suppression_loss = (
                (suppress_per_sample * suppress_valid.float()).sum()
                / suppress_valid.float().sum().clamp_min(1e-6)
            )
        n_bad_move_suppression = int(suppress_valid.sum().item())
    else:
        bad_move_suppression_loss = torch.zeros(
            (), device=policy_logits.device, dtype=policy_logits.dtype
        )
        n_bad_move_suppression = 0

    total_loss = (
        policy_loss_weight * policy_loss
        + value_loss_weight * value_loss
        + wdl_loss_weight * wdl_loss
        + wdl_value_consistency_weight * wdl_value_consistency_loss
        + teacher_q_loss_weight * teacher_q_loss
        + teacher_q_pairwise_loss_weight * teacher_q_pairwise_loss
        + bad_move_suppression_loss_weight * bad_move_suppression_loss
    )
    return {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "wdl_loss": wdl_loss,
        "wdl_value_consistency_loss": wdl_value_consistency_loss,
        "teacher_q_loss": teacher_q_loss,
        "teacher_q_pairwise_loss": teacher_q_pairwise_loss,
        "bad_move_suppression_loss": bad_move_suppression_loss,
        "n_teacher_q_samples": torch.tensor(float(n_teacher_q), device=policy_logits.device),
        "n_teacher_q_pairwise_samples": torch.tensor(
            float(n_teacher_q_pairwise), device=policy_logits.device
        ),
        "n_bad_move_suppression_samples": torch.tensor(
            float(n_bad_move_suppression), device=policy_logits.device
        ),
        "wdl_entropy": wdl_entropy,
        "total_loss": total_loss,
        "n_oracle_samples": torch.tensor(float(n_oracle), device=policy_logits.device),
    }


def _compute_anchor_distillation_losses(
    *,
    policy_logits: Tensor,
    value_scalar: Tensor,
    anchor_policy_logits: Tensor,
    anchor_value_scalar: Tensor,
    legal_offsets: Tensor | None = None,
    legal_idxs: Tensor | None = None,
) -> dict[str, Tensor]:
    student_log_probs = _policy_log_probs_with_optional_legal_mask(
        policy_logits=policy_logits,
        legal_offsets=legal_offsets,
        legal_idxs=legal_idxs,
    )
    teacher_log_probs = _policy_log_probs_with_optional_legal_mask(
        policy_logits=anchor_policy_logits.detach(),
        legal_offsets=legal_offsets,
        legal_idxs=legal_idxs,
    )
    teacher_probs = teacher_log_probs.exp()
    finite_mask = torch.isfinite(student_log_probs) & torch.isfinite(teacher_log_probs)
    per_action_kl = torch.where(
        finite_mask,
        teacher_probs * (teacher_log_probs - student_log_probs),
        torch.zeros_like(student_log_probs),
    )
    anchor_policy_kl_loss = per_action_kl.sum(dim=1).mean()
    teacher_best = torch.argmax(teacher_log_probs, dim=1)
    batch_ids = torch.arange(int(policy_logits.shape[0]), device=policy_logits.device)
    anchor_policy_top1_ce_loss = -student_log_probs[batch_ids, teacher_best].mean()
    anchor_value_mse_loss = (
        value_scalar.squeeze(1).float() - anchor_value_scalar.detach().squeeze(1).float()
    ).pow(2).mean()
    return {
        "anchor_policy_kl_loss": anchor_policy_kl_loss,
        "anchor_policy_top1_ce_loss": anchor_policy_top1_ce_loss,
        "anchor_value_mse_loss": anchor_value_mse_loss,
    }


def _sparse_policy_cross_entropy(
    policy_logits: Tensor,
    policy_offsets: Tensor,
    policy_idxs: Tensor,
    policy_probs: Tensor,
    return_per_sample: bool = False,
    legal_offsets: Tensor | None = None,
    legal_idxs: Tensor | None = None,
) -> Tensor:
    """Sparse policy CE with optional legal-move masking (v12).

    Without legal mask: log_softmax over all 8100 actions. The model then has to
    burn capacity pushing 8000+ illegal-move logits towards −∞.
    With legal mask: log_softmax only over legal actions. Per-sample empty slice
    (legal_offsets[i+1] == legal_offsets[i]) keeps that sample unmasked — happens
    when older v10/v11 shards (no legal_idxs field) are mixed into the batch.
    """
    batch_size = int(policy_logits.shape[0])
    log_probs = _policy_log_probs_with_optional_legal_mask(
        policy_logits=policy_logits,
        legal_offsets=legal_offsets,
        legal_idxs=legal_idxs,
    )
    counts = policy_offsets[1:] - policy_offsets[:-1]
    batch_ids = torch.repeat_interleave(torch.arange(batch_size, device=policy_logits.device), counts)
    selected_log_probs = log_probs[batch_ids, policy_idxs]
    per_sample = torch.zeros(batch_size, device=policy_logits.device, dtype=torch.float32)
    per_sample.scatter_add_(0, batch_ids, -selected_log_probs * policy_probs.float())
    if return_per_sample:
        return per_sample
    return per_sample.mean()


def _policy_log_probs_with_optional_legal_mask(
    *,
    policy_logits: Tensor,
    legal_offsets: Tensor | None,
    legal_idxs: Tensor | None,
) -> Tensor:
    batch_size = int(policy_logits.shape[0])
    if legal_offsets is not None and legal_idxs is not None and legal_idxs.numel() > 0:
        legal_counts = legal_offsets[1:] - legal_offsets[:-1]
        legal_mask = torch.zeros_like(policy_logits, dtype=torch.bool)
        legal_batch_ids = torch.repeat_interleave(
            torch.arange(batch_size, device=policy_logits.device), legal_counts
        )
        legal_mask[legal_batch_ids, legal_idxs] = True
        no_legal_rows = (legal_counts == 0)
        if bool(no_legal_rows.any()):
            legal_mask[no_legal_rows] = True
        masked_logits = policy_logits.float().masked_fill(~legal_mask, -1e9)
        return torch.log_softmax(masked_logits, dim=1)
    return torch.log_softmax(policy_logits.float(), dim=1)


def _teacher_q_cross_entropy(
    *,
    policy_logits: Tensor,
    teacher_q_offsets: Tensor,
    teacher_q_idxs: Tensor,
    teacher_q_values: Tensor,
    temperature_cp: float,
    legal_offsets: Tensor | None = None,
    legal_idxs: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Per-sample CE against a softmax over teacher action values.

    Returns ``(per_sample_loss, valid_mask)``.  Samples with no teacher-Q slice,
    invalid indices, or indices masked out by the legal mask are skipped instead
    of producing a destructive -1e9 loss.
    """
    batch_size = int(policy_logits.shape[0])
    log_probs = _policy_log_probs_with_optional_legal_mask(
        policy_logits=policy_logits,
        legal_offsets=legal_offsets,
        legal_idxs=legal_idxs,
    )
    per_sample = torch.zeros(batch_size, device=policy_logits.device, dtype=torch.float32)
    valid = torch.zeros(batch_size, device=policy_logits.device, dtype=torch.bool)
    temp = max(1.0, float(temperature_cp))
    for i in range(batch_size):
        start = int(teacher_q_offsets[i].item())
        end = int(teacher_q_offsets[i + 1].item())
        if end <= start:
            continue
        idxs = teacher_q_idxs[start:end]
        values = teacher_q_values[start:end].float()
        if idxs.numel() == 0 or int(idxs.min().item()) < 0 or int(idxs.max().item()) >= policy_logits.shape[1]:
            continue
        selected_log_probs = log_probs[i, idxs]
        # A legal-mask mismatch means the shard is dirty.  Skip the sample so a
        # stale teacher_q label cannot reproduce the v12 infinite-loss failure.
        if bool((selected_log_probs < -1e8).any()):
            continue
        target = torch.softmax((values - values.max()) / temp, dim=0)
        per_sample[i] = -(target * selected_log_probs).sum()
        valid[i] = True
    return per_sample, valid


def _teacher_q_pairwise_margin_loss(
    *,
    policy_logits: Tensor,
    teacher_q_offsets: Tensor,
    teacher_q_idxs: Tensor,
    teacher_q_values: Tensor,
    margin_logit: float,
    min_gap_cp: float,
    temperature_cp: float,
    beta: float = 1.0,
    reference_policy_logits: Tensor | None = None,
    bad_move_idxs: Tensor | None = None,
    bad_move_only: bool = False,
    legal_offsets: Tensor | None = None,
    legal_idxs: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Per-sample pairwise margin loss for teacher-Q candidates.

    For each labeled position, find the teacher-best candidate and compare it
    only against candidates at least ``min_gap_cp`` worse.  Without a reference
    model this is a simple margin-ranking loss.  With ``reference_policy_logits``
    it becomes DPO-style local repair: improve the good-vs-bad log-prob gap
    relative to the frozen reference instead of rewriting the full policy.  If
    ``bad_move_only`` is set, only the known blunder action from the shard's
    ``bad_move`` field is compared against the teacher-best candidate.
    """
    batch_size = int(policy_logits.shape[0])
    log_probs = _policy_log_probs_with_optional_legal_mask(
        policy_logits=policy_logits,
        legal_offsets=legal_offsets,
        legal_idxs=legal_idxs,
    )
    per_sample = torch.zeros(batch_size, device=policy_logits.device, dtype=torch.float32)
    valid = torch.zeros(batch_size, device=policy_logits.device, dtype=torch.bool)
    margin = max(0.0, float(margin_logit))
    min_gap = max(0.0, float(min_gap_cp))
    temp = max(1.0, float(temperature_cp))
    beta = max(1e-6, float(beta))
    ref_log_probs = None
    if reference_policy_logits is not None:
        ref_log_probs = _policy_log_probs_with_optional_legal_mask(
            policy_logits=reference_policy_logits.detach(),
            legal_offsets=legal_offsets,
            legal_idxs=legal_idxs,
        )

    for i in range(batch_size):
        start = int(teacher_q_offsets[i].item())
        end = int(teacher_q_offsets[i + 1].item())
        if end - start < 2:
            continue
        idxs = teacher_q_idxs[start:end]
        values = teacher_q_values[start:end].float().to(policy_logits.device)
        if idxs.numel() < 2 or int(idxs.min().item()) < 0 or int(idxs.max().item()) >= policy_logits.shape[1]:
            continue
        idxs = idxs.to(device=policy_logits.device, dtype=torch.long)
        selected_log_probs = log_probs[i, idxs]
        if bool((selected_log_probs < -1e8).any()):
            continue

        best_pos = int(torch.argmax(values).item())
        best_value = values[best_pos]
        gaps = best_value - values
        if bad_move_only:
            bad_mask = torch.zeros_like(gaps, dtype=torch.bool)
            if bad_move_idxs is None:
                continue
            known_bad = int(bad_move_idxs[i].item())
            if known_bad < 0:
                continue
            matches = (idxs == known_bad).nonzero(as_tuple=False)
            if matches.numel() == 0:
                continue
            bad_pos = int(matches[0].item())
            if bad_pos == best_pos or float(gaps[bad_pos].item()) < min_gap:
                continue
            bad_mask[bad_pos] = True
        else:
            bad_mask = gaps >= min_gap
            bad_mask[best_pos] = False
        if not bool(bad_mask.any()):
            continue

        selected_log_probs = selected_log_probs.float()
        best_logp = selected_log_probs[best_pos]
        bad_logps = selected_log_probs[bad_mask]
        bad_gaps = gaps[bad_mask].clamp_min(0.0)
        weights = torch.softmax(bad_gaps / temp, dim=0).detach()
        if ref_log_probs is not None:
            ref_selected_log_probs = ref_log_probs[i, idxs].float()
            if bool((ref_selected_log_probs < -1e8).any()):
                continue
            ref_best_logp = ref_selected_log_probs[best_pos]
            ref_bad_logps = ref_selected_log_probs[bad_mask]
            new_diff = best_logp - bad_logps
            ref_diff = ref_best_logp - ref_bad_logps
            if margin > 0.0 and min_gap > 0.0:
                margins = margin * (bad_gaps / min_gap).clamp_min(1.0).clamp_max(4.0)
            else:
                margins = torch.zeros_like(bad_gaps) + margin
            losses = F.softplus(-beta * (new_diff - ref_diff - margins))
        else:
            losses = F.softplus(bad_logps - best_logp + margin)
        per_sample[i] = (weights * losses).sum()
        valid[i] = True

    return per_sample, valid


def _teacher_q_bad_move_suppression_loss(
    *,
    policy_logits: Tensor,
    reference_policy_logits: Tensor,
    teacher_q_offsets: Tensor,
    teacher_q_idxs: Tensor,
    teacher_q_values: Tensor,
    bad_move_idxs: Tensor,
    margin_logit: float,
    min_gap_cp: float,
    beta: float = 1.0,
    legal_offsets: Tensor | None = None,
    legal_idxs: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Directly suppress known blunder moves relative to a frozen reference.

    This is intentionally narrower than teacher-Q CE and even narrower than the
    pairwise ranking loss: it only fires when a shard marks a known ``bad_move``
    and teacher-Q says that move is at least ``min_gap_cp`` worse than the best
    labeled candidate.  The loss asks the current model to lower the bad move's
    log-probability below the frozen reference by ``margin_logit``.
    """
    batch_size = int(policy_logits.shape[0])
    # The suppression itself acts on raw logits, not log-probabilities.  Earlier
    # pairwise/log-prob repair can be defeated by softmax competition among many
    # legal moves; this term is a direct "turn down this known bad action" knob.
    logits = policy_logits.float()
    ref_logits = reference_policy_logits.detach().float()
    log_probs = _policy_log_probs_with_optional_legal_mask(
        policy_logits=policy_logits,
        legal_offsets=legal_offsets,
        legal_idxs=legal_idxs,
    )
    ref_log_probs = _policy_log_probs_with_optional_legal_mask(
        policy_logits=reference_policy_logits.detach(),
        legal_offsets=legal_offsets,
        legal_idxs=legal_idxs,
    )
    per_sample = torch.zeros(batch_size, device=policy_logits.device, dtype=torch.float32)
    valid = torch.zeros(batch_size, device=policy_logits.device, dtype=torch.bool)
    margin = max(0.0, float(margin_logit))
    min_gap = max(0.0, float(min_gap_cp))
    beta = max(1e-6, float(beta))

    for i in range(batch_size):
        known_bad = int(bad_move_idxs[i].item())
        if known_bad < 0 or known_bad >= policy_logits.shape[1]:
            continue
        start = int(teacher_q_offsets[i].item())
        end = int(teacher_q_offsets[i + 1].item())
        if end - start < 2:
            continue
        idxs = teacher_q_idxs[start:end].to(device=policy_logits.device, dtype=torch.long)
        values = teacher_q_values[start:end].float().to(policy_logits.device)
        if idxs.numel() < 2 or int(idxs.min().item()) < 0 or int(idxs.max().item()) >= policy_logits.shape[1]:
            continue
        matches = (idxs == known_bad).nonzero(as_tuple=False)
        if matches.numel() == 0:
            continue
        bad_pos = int(matches[0].item())
        best_pos = int(torch.argmax(values).item())
        if bad_pos == best_pos:
            continue
        gap = float((values[best_pos] - values[bad_pos]).item())
        if gap < min_gap:
            continue

        bad_logp = log_probs[i, known_bad].float()
        ref_bad_logp = ref_log_probs[i, known_bad].float()
        if bool(bad_logp < -1e8) or bool(ref_bad_logp < -1e8):
            continue
        bad_logit = logits[i, known_bad]
        ref_bad_logit = ref_logits[i, known_bad]
        scaled_margin = margin
        if margin > 0.0 and min_gap > 0.0:
            scaled_margin = margin * min(max(gap / min_gap, 1.0), 4.0)
        per_sample[i] = F.softplus(beta * (bad_logit - ref_bad_logit + scaled_margin))
        valid[i] = True

    return per_sample, valid


@torch.no_grad()
def _evaluate_model(
    model: nn.Module,
    catalog: _HumanTrainCatalog,
    shard_cache: _ShardCache,
    device: torch.device,
    batch_size: int,
    autocast_enabled: bool,
    wdl_loss_weight: float = 1.0,
    value_loss_weight: float = 1.0,
    wdl_value_consistency_weight: float = 0.0,
    value_target_scale: float = 1.0,
) -> dict[str, float]:
    model.eval()
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_wdl_loss = 0.0
    total_wdl_value_consistency_loss = 0.0
    total_samples = 0

    for spec in catalog.val_specs:
        shard = shard_cache.get(spec.path)
        raw_samples = _get_legacy_samples(shard, spec.path)
        for start in range(0, spec.sample_count, batch_size):
            stop = min(start + batch_size, spec.sample_count)
            samples = [_legacy_sample_to_blob(raw_samples[index]) for index in range(start, stop)]
            batch = _move_batch_to_device(_collate_sample_blobs(samples), device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                outputs = model(batch["state"])
                losses = _compute_training_losses(
                    policy_logits=outputs["policy_logits"],
                    value_scalar=outputs["value_scalar"],
                    policy_offsets=batch["policy_offsets"],
                    policy_idxs=batch["policy_idxs"],
                    policy_probs=batch["policy_probs"],
                    z=batch["z"],
                    wdl_logits=outputs.get("wdl_logits"),
                    wdl_target=batch.get("wdl_target"),
                    wdl_loss_weight=wdl_loss_weight,
                    value_loss_weight=value_loss_weight,
                    wdl_value_consistency_weight=wdl_value_consistency_weight,
                    value_target_scale=value_target_scale,
                    # Eval-time: human-val shards don't carry oracle_value (only training shards do).
                    # If a future eval shard has it, this picks it up automatically.
                    oracle_value=batch.get("oracle_value"),
                    use_oracle_value=True,
                )

            sample_count = int(batch["state"].shape[0])
            total_policy_loss += float(losses["policy_loss"].item()) * sample_count
            total_value_loss += float(losses["value_loss"].item()) * sample_count
            total_wdl_loss += float(losses["wdl_loss"].item()) * sample_count
            total_wdl_value_consistency_loss += (
                float(losses["wdl_value_consistency_loss"].item()) * sample_count
            )
            total_samples += sample_count

    model.train()
    if total_samples == 0:
        raise RuntimeError("validation dataset is empty")

    avg_policy = total_policy_loss / float(total_samples)
    avg_value = total_value_loss / float(total_samples)
    avg_wdl = total_wdl_loss / float(total_samples)
    avg_wdl_value_consistency = total_wdl_value_consistency_loss / float(total_samples)
    # Mirror the training-loop total so best-checkpoint selection and training loss are on the same scale.
    total = (
        avg_policy
        + value_loss_weight * avg_value
        + wdl_loss_weight * avg_wdl
        + wdl_value_consistency_weight * avg_wdl_value_consistency
    )
    return {
        "human_val_policy_loss": avg_policy,
        "human_val_value_loss": avg_value,
        "human_val_wdl_loss": avg_wdl,
        "human_val_wdl_value_consistency_loss": avg_wdl_value_consistency,
        "human_val_total_loss": total,
        "human_val_samples": total_samples,
        "policy_loss": avg_policy,
        "value_loss": avg_value,
        "wdl_loss": avg_wdl,
        "wdl_value_consistency_loss": avg_wdl_value_consistency,
        "total_loss": total,
        "val_samples": total_samples,
    }


def _apply_parameter_training_scope(model: nn.Module, config: TrainingConfig) -> None:
    if config.train_only_cnn_local_adapter or config.train_only_cnn_policy_residual_adapter:
        trainable_prefixes: list[str] = []
        if config.train_only_cnn_local_adapter:
            trainable_prefixes.append("cnn_local_")
        if config.train_only_cnn_policy_residual_adapter:
            trainable_prefixes.append("cnn_policy_")
        matched = 0
        for name, parameter in model.named_parameters():
            trainable = any(name.startswith(prefix) for prefix in trainable_prefixes)
            parameter.requires_grad = trainable
            if trainable:
                matched += int(parameter.numel())
        if matched <= 0:
            raise RuntimeError(
                "CNN adapter train-only scope did not match any parameters. "
                "Use a v14 preset or enable the matching CNN adapter."
            )
        return
    if config.train_only_value_head:
        matched = 0
        for name, parameter in model.named_parameters():
            trainable = (
                name.startswith("value_shared.")
                or name.startswith("wdl_head.")
                or name.startswith("scalar_head.")
                or name == "value_query"
                or name.startswith("value_pool_")
            )
            parameter.requires_grad = trainable
            if trainable:
                matched += int(parameter.numel())
        if matched <= 0:
            raise RuntimeError("--train-only-value-head did not match any value-head parameters")
        return
    if config.train_only_policy_head:
        matched = 0
        for name, parameter in model.named_parameters():
            trainable = (
                name.startswith("from_repr.")
                or name.startswith("to_repr.")
                or name.startswith("from_bias.")
                or name.startswith("to_bias.")
            )
            parameter.requires_grad = trainable
            if trainable:
                matched += int(parameter.numel())
        if matched <= 0:
            raise RuntimeError("--train-only-policy-head did not match any policy-head parameters")
        return
    if not config.train_only_relative_attention_bias:
        return
    matched = 0
    last_n_blocks = int(config.adapter_unfreeze_last_n_blocks)
    first_unfrozen_block = max(0, int(config.model_config.num_layers) - last_n_blocks)
    for name, parameter in model.named_parameters():
        trainable = (
            name.endswith("relative_attention_bias")
            or name.endswith("line_of_sight_attention_bias")
            or name.startswith("history_memory_")
            or name.startswith("global_strategy_")
            or name.startswith("trunk_strategy_")
            or name == "value_query"
            or name.startswith("value_pool_")
            or (
                last_n_blocks > 0
                and (
                    any(
                        name.startswith(f"blocks.{block_idx}.")
                        for block_idx in range(first_unfrozen_block, int(config.model_config.num_layers))
                    )
                    or name.startswith("final_norm.")
                )
            )
        )
        parameter.requires_grad = trainable
        if trainable:
            matched += int(parameter.numel())
    if matched <= 0:
        raise RuntimeError(
            "--train-only-relative-attention-bias requires "
            "--use-2d-relative-attention-bias, --use-line-of-sight-attention-bias, "
            "--use-history-memory-attention, --use-global-strategic-attention, "
            "or --adapter-unfreeze-last-n-blocks > 0"
        )


def _build_optimizer(model: nn.Module, config: TrainingConfig) -> AdamW:
    decay_params: list[Tensor] = []
    no_decay_params: list[Tensor] = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1:
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)

    param_groups = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": config.weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})

    return AdamW(
        param_groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )


def _build_scheduler(optimizer: AdamW, config: TrainingConfig) -> LambdaLR:
    schedule_max_steps = _effective_lr_schedule_max_steps(config)

    def lr_lambda(step: int) -> float:
        if schedule_max_steps <= 1:
            return 1.0
        if config.warmup_steps > 0 and step < config.warmup_steps:
            return max(float(step + 1) / float(config.warmup_steps), 1e-8)
        if schedule_max_steps <= config.warmup_steps:
            return 1.0

        progress = (step - config.warmup_steps) / float(max(schedule_max_steps - config.warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def _reinit_wdl_head(model: XiangqiPVTransformer) -> None:
    with torch.no_grad():
        nn.init.xavier_uniform_(model.wdl_head.weight)
        nn.init.zeros_(model.wdl_head.bias)
    for parameter in model.wdl_head.parameters():
        parameter.requires_grad = True


def _wdl_head_is_nontrivial(model: XiangqiPVTransformer) -> bool:
    # Legacy checkpoints froze wdl_head to all zeros; detect that and reinit.
    with torch.no_grad():
        return bool(model.wdl_head.weight.abs().max().item() > 0.0)


def _build_checkpoint_state(
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    config: TrainingConfig,
    global_step: int,
    best_human_val_metric: float,
    seen_selfplay_shards: dict[str, dict[str, Any]],
    replay_buffer: _SelfPlayReplayBuffer,
    cumulative_added_shards: int,
    cumulative_added_samples: int,
    last_ingest_status: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "global_step": global_step,
        "best_human_val_metric": best_human_val_metric,
        "best_val_metric": best_human_val_metric,
        "seen_selfplay_shards": seen_selfplay_shards,
        "selfplay_buffer_refs": replay_buffer.to_state(),
        "cumulative_added_shards": int(cumulative_added_shards),
        "cumulative_added_samples": int(cumulative_added_samples),
        "last_ingest_status": last_ingest_status,
        "rng_state": _capture_rng_state(),
        "model_config": asdict(config.model_config),
        "training_config": _config_to_jsonable(config),
    }


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_torch_runtime(config: TrainingConfig, device: torch.device) -> None:
    if device.type != "cuda":
        return

    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = bool(config.allow_tf32)
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = bool(config.allow_tf32)
    if hasattr(torch.backends.cudnn, "benchmark"):
        torch.backends.cudnn.benchmark = bool(config.cudnn_benchmark)
    if config.allow_tf32:
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def _install_graceful_signal_handlers() -> dict[int, Any]:
    previous: dict[int, Any] = {}

    def _handler(signum: int, _frame: Any) -> None:
        signal_name = signal.Signals(signum).name if signum in signal.Signals._value2member_map_ else str(signum)
        raise _GracefulStopRequested(f"received {signal_name}")

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


def _parse_manifest_split(data_dir: Path, manifest: dict[str, Any], split: str) -> list[_HumanShardSpec]:
    split_meta = manifest.get(split)
    if not isinstance(split_meta, dict):
        raise KeyError(f"manifest is missing split metadata for '{split}'")

    specs: list[_HumanShardSpec] = []
    for shard_meta in split_meta.get("shards", []):
        shard_path = data_dir / split / str(shard_meta["path"])
        specs.append(_HumanShardSpec(path=shard_path.resolve(), sample_count=int(shard_meta["samples"])))
    return specs


def _resolve_selfplay_train_dir(root: Path) -> Path:
    root = root.resolve()
    train_dir = root / "train"
    if train_dir.is_dir():
        return train_dir
    return root


def _iter_selfplay_train_dirs(root: Path) -> list[Path]:
    root = root.resolve()
    direct_train_dir = root / "train"
    if direct_train_dir.is_dir():
        return [direct_train_dir]
    if root.is_dir():
        nested_train_dirs = sorted(
            child / "train"
            for child in root.iterdir()
            if child.is_dir() and (child / "train").is_dir()
        )
        if nested_train_dirs:
            return nested_train_dirs
    return [root]


def _get_legacy_samples(shard: Any, path: Path) -> list[dict[str, Any]]:
    if not isinstance(shard, dict) or "samples" not in shard:
        raise RuntimeError(f"legacy human shard at {path} is missing 'samples'")
    samples = shard["samples"]
    if not isinstance(samples, list):
        raise RuntimeError(f"legacy human shard at {path} has non-list 'samples'")
    return samples


def _get_selfplay_shard_sample_count(shard: Any, path: Path) -> int:
    required = {"state", "policy_offsets", "policy_idxs", "policy_probs", "z"}
    if not isinstance(shard, dict) or not required.issubset(shard.keys()):
        raise RuntimeError(f"self-play shard at {path} does not match tensorized self-play format")

    state = shard["state"]
    policy_offsets = shard["policy_offsets"]
    z = shard["z"]
    if not isinstance(state, torch.Tensor):
        raise RuntimeError(f"self-play shard at {path} has non-tensor 'state'")
    if state.ndim != 4 or tuple(state.shape[1:]) != (115, 10, 9):
        raise RuntimeError(f"self-play shard at {path} has invalid state shape {tuple(state.shape)}")

    sample_count = int(state.shape[0])
    if tuple(policy_offsets.shape) != (sample_count + 1,):
        raise RuntimeError(f"self-play shard at {path} has invalid policy_offsets shape {tuple(policy_offsets.shape)}")
    if tuple(z.shape) != (sample_count,):
        raise RuntimeError(f"self-play shard at {path} has invalid z shape {tuple(z.shape)}")
    return sample_count


def _normalize_training_config(config: TrainingConfig) -> TrainingConfig:
    config.human_data_dir = str(Path(config.human_data_dir).resolve())
    config.output_dir = str(Path(config.output_dir).resolve())
    config.selfplay_dirs = [str(Path(path).resolve()) for path in config.selfplay_dirs]
    if config.resume_path is not None:
        config.resume_path = str(Path(config.resume_path).resolve())
    else:
        latest_checkpoint = Path(config.output_dir) / "latest.pt"
        if latest_checkpoint.is_file():
            config.resume_path = str(latest_checkpoint.resolve())
    return config


def _validate_training_inputs(config: TrainingConfig) -> None:
    human_data_dir = Path(config.human_data_dir)
    if not human_data_dir.is_dir():
        raise FileNotFoundError(f"human_data_dir not found: {human_data_dir}")

    if config.resume_path is not None and not Path(config.resume_path).is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {config.resume_path}")

    if config.selfplay_dir_sampling_ratios:
        if len(config.selfplay_dir_sampling_ratios) != len(config.selfplay_dirs):
            raise ValueError(
                "--selfplay-dir-sampling-ratios must have one value per --selfplay-dirs entry"
            )
        if sum(float(ratio) for ratio in config.selfplay_dir_sampling_ratios) <= 0.0:
            raise ValueError("--selfplay-dir-sampling-ratios must contain at least one positive value")


def _config_to_jsonable(config: TrainingConfig) -> dict[str, Any]:
    result = asdict(config)
    result["human_data_dir"] = str(config.human_data_dir)
    result["selfplay_dirs"] = [str(path) for path in config.selfplay_dirs]
    result["output_dir"] = str(config.output_dir)
    result["resume_path"] = None if config.resume_path is None else str(config.resume_path)
    result["model_config"] = asdict(config.model_config)
    return result


def _model_config_from_args(args: argparse.Namespace) -> XiangqiTransformerConfig:
    preset = str(getattr(args, "model_preset", "legacy"))
    if preset == "legacy":
        return XiangqiTransformerConfig(
            use_2d_relative_attention_bias=bool(args.use_2d_relative_attention_bias),
            use_line_of_sight_attention_bias=bool(args.use_line_of_sight_attention_bias),
            use_history_memory_attention=bool(args.use_history_memory_attention),
            use_global_strategic_attention=bool(args.use_global_strategic_attention),
            use_trunk_global_strategy_tokens=bool(args.use_trunk_global_strategy_tokens),
            use_value_token_pooling=bool(args.use_value_token_pooling),
            num_global_strategy_tokens=int(args.num_global_strategy_tokens),
            policy_head_dim=int(args.policy_head_dim),
            use_cnn_local_tactical_adapter=bool(args.use_cnn_local_tactical_adapter),
            cnn_local_channels=int(args.cnn_local_channels),
            cnn_local_blocks=int(args.cnn_local_blocks),
            use_cnn_policy_residual_adapter=bool(args.use_cnn_policy_residual_adapter),
            cnn_policy_channels=int(args.cnn_policy_channels),
            cnn_policy_blocks=int(args.cnn_policy_blocks),
            cnn_policy_rank=int(args.cnn_policy_rank),
            use_cnn_local_tactical_stem=bool(args.use_cnn_local_tactical_stem),
            cnn_stem_channels=int(args.cnn_stem_channels),
            cnn_stem_blocks=int(args.cnn_stem_blocks),
        )

    common = {
        "d_model": 896,
        "num_layers": 20,
        "num_heads": 14,
        "ffn_dim": 3584,
        "policy_head_dim": 384,
        "use_2d_relative_attention_bias": True,
        "use_line_of_sight_attention_bias": False,
        "use_history_memory_attention": False,
        "use_global_strategic_attention": False,
        "use_value_token_pooling": True,
        "num_global_strategy_tokens": 8,
    }
    if preset == "v13_200m_dense":
        return XiangqiTransformerConfig(
            **common,
            use_trunk_global_strategy_tokens=False,
        )
    if preset == "v13_200m_dense_nopool":
        return XiangqiTransformerConfig(
            **{
                **common,
                "use_value_token_pooling": False,
            },
            use_trunk_global_strategy_tokens=False,
        )
    if preset == "v13_200m_strategy":
        return XiangqiTransformerConfig(
            **common,
            use_trunk_global_strategy_tokens=True,
        )
    if preset == "v14a_200m_cnn_adapter":
        return XiangqiTransformerConfig(
            **{
                **common,
                "use_value_token_pooling": False,
            },
            use_trunk_global_strategy_tokens=False,
            use_cnn_local_tactical_adapter=True,
            cnn_local_channels=int(args.cnn_local_channels),
            cnn_local_blocks=int(args.cnn_local_blocks),
        )
    if preset == "v14b_200m_cnn_policy_residual":
        return XiangqiTransformerConfig(
            **{
                **common,
                "use_value_token_pooling": False,
            },
            use_trunk_global_strategy_tokens=False,
            use_cnn_policy_residual_adapter=True,
            cnn_policy_channels=int(args.cnn_policy_channels),
            cnn_policy_blocks=int(args.cnn_policy_blocks),
            cnn_policy_rank=int(args.cnn_policy_rank),
        )
    if preset == "v14r_200m_hybrid":
        return XiangqiTransformerConfig(
            **{
                **common,
                "use_value_token_pooling": False,
            },
            use_trunk_global_strategy_tokens=False,
            use_cnn_local_tactical_stem=True,
            cnn_stem_channels=int(args.cnn_stem_channels),
            cnn_stem_blocks=int(args.cnn_stem_blocks),
        )
    if preset == "v15_207m_geo":
        # 207M FULL-geometry: BOTH attention biases ON (unlike the v13/v14 200M
        # presets, which omit line-of-sight). Pure transformer — no CNN stem,
        # no global-strategy tokens, no value pooling (geo-style). 206,998,556 params.
        return XiangqiTransformerConfig(
            d_model=896,
            num_layers=21,
            num_heads=14,
            ffn_dim=3648,
            policy_head_dim=384,
            use_2d_relative_attention_bias=True,
            use_line_of_sight_attention_bias=True,
            use_history_memory_attention=False,
            use_global_strategic_attention=False,
            use_trunk_global_strategy_tokens=False,
            use_value_token_pooling=False,
            num_global_strategy_tokens=6,
        )
    raise ValueError(f"unknown model preset: {preset}")


def _anchor_weight_scale(config: TrainingConfig, global_step: int) -> float:
    if int(config.anchor_anneal_steps) <= 0:
        return 1.0
    progress = min(max(float(global_step) / float(config.anchor_anneal_steps), 0.0), 1.0)
    return max(0.0, 1.0 - progress)


def _parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train the Xiangqi transformer with human + self-play data.")
    parser.add_argument("--human-data-dir", default=_DEFAULT_LAUNCH_CONFIG.human_data_dir)
    parser.add_argument("--selfplay-dirs", nargs="*", default=list(_DEFAULT_LAUNCH_CONFIG.selfplay_dirs))
    parser.add_argument("--output-dir", default=_DEFAULT_LAUNCH_CONFIG.output_dir)
    parser.add_argument("--resume-path", default=_DEFAULT_LAUNCH_CONFIG.resume_path)
    parser.add_argument("--reset-optimizer-on-resume", action="store_true",
                        help="Load model weights/global_step from --resume-path but start a fresh "
                             "optimizer and scheduler. Useful for short finetunes from a checkpoint "
                             "whose learning rate has already decayed near zero.")
    parser.add_argument("--device", default=_DEFAULT_LAUNCH_CONFIG.device)
    parser.add_argument("--replay-buffer-size", type=int, default=_DEFAULT_LAUNCH_CONFIG.replay_buffer_size)
    parser.add_argument("--poll-interval-s", type=float, default=_DEFAULT_LAUNCH_CONFIG.poll_interval_s)
    parser.add_argument("--shard-cache-size", type=int, default=_DEFAULT_LAUNCH_CONFIG.shard_cache_size)
    parser.add_argument("--bootstrap-human-floor", type=float, default=_DEFAULT_LAUNCH_CONFIG.bootstrap_human_floor)
    parser.add_argument("--disable-bootstrap-mode", action="store_true")
    parser.add_argument("--disable-selfplay-run-quality-gate", action="store_true")
    parser.add_argument("--reset-selfplay-ingest-state-on-resume", action="store_true")
    parser.add_argument(
        "--selfplay-dir-sampling-ratios",
        nargs="*",
        type=float,
        default=list(_DEFAULT_LAUNCH_CONFIG.selfplay_dir_sampling_ratios),
        help="Optional ratios for sampling from each --selfplay-dirs entry. "
             "When omitted, all self-play shards are pooled by sample count. "
             "When provided, length must match --selfplay-dirs.",
    )
    parser.add_argument(
        "--selfplay-run-max-rep-draw-rate",
        type=float,
        default=_DEFAULT_LAUNCH_CONFIG.selfplay_run_max_rep_draw_rate,
    )
    parser.add_argument(
        "--selfplay-run-min-decisive-rate",
        type=float,
        default=_DEFAULT_LAUNCH_CONFIG.selfplay_run_min_decisive_rate,
    )
    parser.add_argument("--micro-batch-size", type=int, default=_DEFAULT_LAUNCH_CONFIG.micro_batch_size)
    parser.add_argument("--grad-accum-steps", type=int, default=_DEFAULT_LAUNCH_CONFIG.grad_accum_steps)
    parser.add_argument("--eval-interval-steps", type=int, default=_DEFAULT_LAUNCH_CONFIG.eval_interval_steps)
    parser.add_argument("--save-interval-steps", type=int, default=_DEFAULT_LAUNCH_CONFIG.save_interval_steps)
    parser.add_argument("--snapshot-interval-steps", type=int,
                        default=_DEFAULT_LAUNCH_CONFIG.snapshot_interval_steps,
                        help="If >0, also write 'snapshots/latest_step<N>.pt' every N steps "
                             "(in addition to latest.pt). Lets you roll back to peak checkpoints "
                             "after later regressions. 0 = disabled (default).")
    parser.add_argument("--log-interval-steps", type=int, default=_DEFAULT_LAUNCH_CONFIG.log_interval_steps)
    parser.add_argument("--max-steps", type=int, default=_DEFAULT_LAUNCH_CONFIG.max_steps)
    parser.add_argument("--lr-schedule-max-steps", type=int, default=_DEFAULT_LAUNCH_CONFIG.lr_schedule_max_steps)
    parser.add_argument("--samples-per-unit", type=int, default=_DEFAULT_LAUNCH_CONFIG.samples_per_unit)
    parser.add_argument("--cpu-sampler-workers", type=int, default=_DEFAULT_LAUNCH_CONFIG.cpu_sampler_workers)
    parser.add_argument("--cpu-prefetch-batches", type=int, default=_DEFAULT_LAUNCH_CONFIG.cpu_prefetch_batches)
    parser.add_argument("--cpu-reserved-cores", type=int, default=_DEFAULT_LAUNCH_CONFIG.cpu_reserved_cores)
    parser.add_argument(
        "--cpu-sampler-backend",
        choices=["auto", "process", "thread", "none"],
        default=_DEFAULT_LAUNCH_CONFIG.cpu_sampler_backend,
    )
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--disable-save-on-interrupt", action="store_true")
    parser.add_argument("--disable-promote-best-on-human-val", action="store_true")
    parser.add_argument("--pause-at-local-time", default=_DEFAULT_LAUNCH_CONFIG.pause_at_local_time or "")
    parser.add_argument("--learning-rate", type=float, default=_DEFAULT_LAUNCH_CONFIG.learning_rate)
    parser.add_argument("--beta1", type=float, default=_DEFAULT_LAUNCH_CONFIG.beta1)
    parser.add_argument("--beta2", type=float, default=_DEFAULT_LAUNCH_CONFIG.beta2)
    parser.add_argument("--weight-decay", type=float, default=_DEFAULT_LAUNCH_CONFIG.weight_decay)
    parser.add_argument("--grad-clip-norm", type=float, default=_DEFAULT_LAUNCH_CONFIG.grad_clip_norm)
    parser.add_argument("--warmup-steps", type=int, default=_DEFAULT_LAUNCH_CONFIG.warmup_steps)
    parser.add_argument("--wdl-loss-weight", type=float, default=_DEFAULT_LAUNCH_CONFIG.wdl_loss_weight)
    parser.add_argument(
        "--policy-loss-weight",
        type=float,
        default=_DEFAULT_LAUNCH_CONFIG.policy_loss_weight,
        help="Weight for the sparse policy CE term. Default 1.0. Set to 0 for "
             "teacher_q/anchor-only micro-finetunes that should not relearn the "
             "base MCTS policy targets.",
    )
    parser.add_argument("--value-loss-weight", type=float, default=_DEFAULT_LAUNCH_CONFIG.value_loss_weight)
    parser.add_argument(
        "--wdl-value-consistency-weight",
        type=float,
        default=_DEFAULT_LAUNCH_CONFIG.wdl_value_consistency_weight,
        help="Optional scalar/WDL bridge. When >0, adds MSE(value_scalar, "
             "stopgrad(P(win)-P(loss))) so the search scalar tracks WDL calibration.",
    )
    parser.add_argument("--value-target-scale", type=float, default=_DEFAULT_LAUNCH_CONFIG.value_target_scale)
    parser.add_argument(
        "--use-oracle-value", action=argparse.BooleanOptionalAction,
        default=_DEFAULT_LAUNCH_CONFIG.use_oracle_value,
        help="When shards contain oracle_value (Pikafish-d=15+ calibrated value), "
             "use it as the value-head target instead of noisy scaled_z. "
             "Pass --no-use-oracle-value for an ablation that ignores oracle even "
             "when present. Samples without oracle_value always fall back to scaled_z.",
    )
    parser.add_argument(
        "--policy-oracle-alpha", type=float,
        default=_DEFAULT_LAUNCH_CONFIG.policy_oracle_alpha,
        help="v11: blend Pikafish-multipv oracle_policy with MCTS-visit policy target. "
             "α=0 (default, v10 behavior) → MCTS only. α=0.5 → equal mix. α=1 → oracle only. "
             "Samples whose shard lacks oracle_policy_* fields fall back to MCTS-only "
             "automatically (no error, the oracle CE term contributes 0 for those samples).",
    )
    parser.add_argument(
        "--teacher-q-loss-weight", type=float,
        default=_DEFAULT_LAUNCH_CONFIG.teacher_q_loss_weight,
        help="v12.5: optional action-value distillation loss. When >0 and shards contain "
             "teacher_q_{offsets,idxs,values}, adds this weighted listwise CE term to total_loss. "
             "Default 0.0 keeps legacy behavior.",
    )
    parser.add_argument(
        "--teacher-q-temperature-cp", type=float,
        default=_DEFAULT_LAUNCH_CONFIG.teacher_q_temperature_cp,
        help="Softmax temperature, in centipawns, for teacher_q_values. Lower is sharper. "
             "Default 80.",
    )
    parser.add_argument(
        "--teacher-q-pairwise-loss-weight", type=float,
        default=_DEFAULT_LAUNCH_CONFIG.teacher_q_pairwise_loss_weight,
        help="Weight for surgical pairwise teacher-Q ranking loss. Unlike full teacher_q CE, "
             "this only pushes the teacher-best candidate above clearly worse candidates.",
    )
    parser.add_argument(
        "--teacher-q-pairwise-margin-logit", type=float,
        default=_DEFAULT_LAUNCH_CONFIG.teacher_q_pairwise_margin_logit,
        help="Required logit margin for teacher-best over worse teacher-Q candidates. Default 0.25.",
    )
    parser.add_argument(
        "--teacher-q-pairwise-min-gap-cp", type=float,
        default=_DEFAULT_LAUNCH_CONFIG.teacher_q_pairwise_min_gap_cp,
        help="Only compare teacher-Q candidates at least this many centipawns below the best. Default 80.",
    )
    parser.add_argument(
        "--teacher-q-pairwise-beta", type=float,
        default=_DEFAULT_LAUNCH_CONFIG.teacher_q_pairwise_beta,
        help="DPO-style inverse temperature when --teacher-q-pairwise-use-anchor-reference is enabled. Default 1.0.",
    )
    parser.add_argument(
        "--teacher-q-pairwise-use-anchor-reference",
        action=argparse.BooleanOptionalAction,
        default=_DEFAULT_LAUNCH_CONFIG.teacher_q_pairwise_use_anchor_reference,
        help="Use --anchor-checkpoint as the frozen reference for DPO-style teacher-Q pairwise repair. "
             "This trains relative improvement over the V13 reference instead of absolute imitation.",
    )
    parser.add_argument(
        "--teacher-q-pairwise-bad-move-only",
        action=argparse.BooleanOptionalAction,
        default=_DEFAULT_LAUNCH_CONFIG.teacher_q_pairwise_bad_move_only,
        help="When shards contain bad_move, compare teacher-Q best only against that known blunder. "
             "This is the narrowest local repair mode.",
    )
    parser.add_argument(
        "--bad-move-suppression-loss-weight",
        type=float,
        default=_DEFAULT_LAUNCH_CONFIG.bad_move_suppression_loss_weight,
        help="Weight for direct bad_move suppression relative to --anchor-checkpoint. "
             "Only fires when teacher_q marks the bad move as clearly worse than the best candidate.",
    )
    parser.add_argument(
        "--bad-move-suppression-margin-logit",
        type=float,
        default=_DEFAULT_LAUNCH_CONFIG.bad_move_suppression_margin_logit,
        help="How far below the frozen reference log-prob the known bad move should be pushed. Default 0.75.",
    )
    parser.add_argument(
        "--bad-move-suppression-min-gap-cp",
        type=float,
        default=_DEFAULT_LAUNCH_CONFIG.bad_move_suppression_min_gap_cp,
        help="Only suppress bad_move when teacher-Q says it is at least this many centipawns worse. Default 80.",
    )
    parser.add_argument(
        "--bad-move-suppression-beta",
        type=float,
        default=_DEFAULT_LAUNCH_CONFIG.bad_move_suppression_beta,
        help="Softplus sharpness for direct bad_move suppression. Default 2.0.",
    )
    parser.add_argument(
        "--anchor-checkpoint",
        type=Path,
        default=None,
        help="Optional frozen teacher checkpoint for behavior anchoring. When paired with "
             "anchor loss weights, the current model is regularized toward this checkpoint "
             "on legal policy distribution and/or scalar value.",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=None,
        help="Alias for --anchor-checkpoint, intended for short v12->v13 bootstrap distillation.",
    )
    parser.add_argument(
        "--anchor-policy-kl-weight",
        type=float,
        default=_DEFAULT_LAUNCH_CONFIG.anchor_policy_kl_weight,
        help="Weight for KL(anchor_policy || current_policy) over legal moves. Default 0.",
    )
    parser.add_argument(
        "--anchor-policy-top1-ce-weight",
        type=float,
        default=_DEFAULT_LAUNCH_CONFIG.anchor_policy_top1_ce_weight,
        help="Weight for hard CE toward the anchor model's legal top-1 move. Default 0.",
    )
    parser.add_argument(
        "--anchor-value-mse-weight",
        type=float,
        default=_DEFAULT_LAUNCH_CONFIG.anchor_value_mse_weight,
        help="Weight for MSE(current value_scalar, anchor value_scalar). Default 0.",
    )
    parser.add_argument(
        "--anchor-anneal-steps",
        type=int,
        default=_DEFAULT_LAUNCH_CONFIG.anchor_anneal_steps,
        help="Linearly decay anchor/teacher distillation weights to zero over this many "
             "global steps. 0 keeps weights constant.",
    )
    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument("--disable-cudnn-benchmark", action="store_true")
    parser.add_argument("--seed", type=int, default=_DEFAULT_LAUNCH_CONFIG.seed)
    parser.add_argument("--disable-bfloat16", action="store_true")
    parser.add_argument(
        "--model-preset",
        choices=[
            "legacy",
            "v13_200m_dense",
            "v13_200m_dense_nopool",
            "v13_200m_strategy",
            "v14a_200m_cnn_adapter",
            "v14b_200m_cnn_policy_residual",
            "v14r_200m_hybrid",
            "v15_207m_geo",
        ],
        default="legacy",
        help="Named model size/architecture preset. v13 presets fix the 200M "
             "architecture and ignore legacy architecture toggles except training losses.",
    )
    parser.add_argument(
        "--policy-head-dim",
        type=int,
        default=_DEFAULT_LAUNCH_CONFIG.model_config.policy_head_dim,
        help="Low-rank policy head dimension for legacy/custom models. v13 presets use 384.",
    )
    parser.add_argument(
        "--use-2d-relative-attention-bias",
        action="store_true",
        help="Enable zero-init 2D relative attention bias in every transformer block. "
             "Safe for warm-starting old checkpoints because missing relative-bias "
             "parameters are initialized to zero.",
    )
    parser.add_argument(
        "--use-line-of-sight-attention-bias",
        action="store_true",
        help="Enable zero-init dynamic attention bias for Xiangqi rank/file relations "
             "(clear line, one screen, multiple blockers).",
    )
    parser.add_argument(
        "--use-history-memory-attention",
        action="store_true",
        help="Enable a zero-init cross-attention adapter that lets current board tokens "
             "query the previous seven history frames as memory tokens.",
    )
    parser.add_argument(
        "--use-global-strategic-attention",
        action="store_true",
        help="Enable a zero-init global strategic-token adapter. The strategic tokens "
             "pool the current board globally and broadcast a high-level context back "
             "to board tokens.",
    )
    parser.add_argument(
        "--use-trunk-global-strategy-tokens",
        action="store_true",
        help="Insert global strategy tokens into the main Transformer sequence. "
             "Used by the v13 strategy-token arm.",
    )
    parser.add_argument(
        "--use-value-token-pooling",
        action="store_true",
        help="Use a learned attention pooling query over material/strategy/board tokens "
             "for WDL/scalar value heads instead of material-token-only value input.",
    )
    parser.add_argument(
        "--num-global-strategy-tokens",
        type=int,
        default=_DEFAULT_LAUNCH_CONFIG.model_config.num_global_strategy_tokens,
        help="Number of learned global strategic query tokens when "
             "--use-global-strategic-attention is enabled. Default 6.",
    )
    parser.add_argument(
        "--use-cnn-local-tactical-adapter",
        action="store_true",
        help="Enable the V14A zero-init local CNN tactical adapter. It reads only "
             "the current 14 piece planes and adds local features back into board tokens.",
    )
    parser.add_argument(
        "--cnn-local-channels",
        type=int,
        default=_DEFAULT_LAUNCH_CONFIG.model_config.cnn_local_channels,
        help="Hidden channel count for the local CNN tactical adapter. Default 128.",
    )
    parser.add_argument(
        "--cnn-local-blocks",
        type=int,
        default=_DEFAULT_LAUNCH_CONFIG.model_config.cnn_local_blocks,
        help="Number of 3x3 residual blocks in the local CNN tactical adapter. Default 4.",
    )
    parser.add_argument(
        "--use-cnn-policy-residual-adapter",
        action="store_true",
        help="Enable the V14B zero-init CNN policy residual adapter. It reads the "
             "current 14 piece planes and adds a low-rank local tactical delta to "
             "policy logits directly.",
    )
    parser.add_argument(
        "--cnn-policy-channels",
        type=int,
        default=_DEFAULT_LAUNCH_CONFIG.model_config.cnn_policy_channels,
        help="Hidden channel count for the CNN policy residual adapter. Default 128.",
    )
    parser.add_argument(
        "--cnn-policy-blocks",
        type=int,
        default=_DEFAULT_LAUNCH_CONFIG.model_config.cnn_policy_blocks,
        help="Number of 3x3 residual blocks in the CNN policy residual adapter. Default 4.",
    )
    parser.add_argument(
        "--cnn-policy-rank",
        type=int,
        default=_DEFAULT_LAUNCH_CONFIG.model_config.cnn_policy_rank,
        help="Low-rank from/to dimension for the CNN policy residual logits. Default 64.",
    )
    parser.add_argument(
        "--use-cnn-local-tactical-stem",
        action="store_true",
        help="Enable the V14R trunk-native local CNN stem. Unlike V14A, this is "
             "not zero-init and is intended for scratch/hybrid training rather "
             "than late adapter warm-starting.",
    )
    parser.add_argument(
        "--cnn-stem-channels",
        type=int,
        default=_DEFAULT_LAUNCH_CONFIG.model_config.cnn_stem_channels,
        help="Hidden channel count for the V14R local CNN stem. Default 128.",
    )
    parser.add_argument(
        "--cnn-stem-blocks",
        type=int,
        default=_DEFAULT_LAUNCH_CONFIG.model_config.cnn_stem_blocks,
        help="Number of 3x3 residual blocks in the V14R local CNN stem. Default 4.",
    )
    parser.add_argument(
        "--train-only-relative-attention-bias",
        action="store_true",
        help="Freeze the base model and train only relative_attention_bias parameters. "
             "Intended for the first v12.8 adapter warmup.",
    )
    parser.add_argument(
        "--train-only-attention-biases",
        action="store_true",
        help="Alias for --train-only-relative-attention-bias; also includes "
             "line_of_sight_attention_bias parameters when enabled.",
    )
    parser.add_argument(
        "--train-only-transformer-adapters",
        action="store_true",
        help="Freeze the base model and train only optional Transformer adapter parameters "
             "(relative/line-of-sight attention biases, history-memory attention, "
             "and global strategic attention).",
    )
    parser.add_argument(
        "--train-only-cnn-local-adapter",
        action="store_true",
        help="Freeze the base model and train only the V14A cnn_local_* tactical adapter.",
    )
    parser.add_argument(
        "--train-only-cnn-policy-residual-adapter",
        action="store_true",
        help="Freeze the base model and train only the V14B cnn_policy_* tactical "
             "policy residual adapter.",
    )
    parser.add_argument(
        "--adapter-unfreeze-last-n-blocks",
        type=int,
        default=_DEFAULT_LAUNCH_CONFIG.adapter_unfreeze_last_n_blocks,
        help="When using an adapter-only training scope, also unfreeze the last N "
             "Transformer blocks plus final_norm. This keeps the early trunk frozen "
             "while letting a new adapter integrate into high-level representations.",
    )
    parser.add_argument(
        "--train-only-policy-head",
        action="store_true",
        help="Freeze the trunk/value heads and train only from/to policy-head projections. "
             "Intended for tiny verified-blunder repair probes that should not rewrite "
             "the main representation.",
    )
    parser.add_argument(
        "--train-only-value-head",
        action="store_true",
        help="Freeze the trunk/policy head and train only value_shared, WDL, and scalar "
             "value heads. Intended for post-blunder value-calibration probes.",
    )
    args = parser.parse_args()

    if args.anchor_checkpoint is not None and args.teacher_checkpoint is not None:
        if Path(args.anchor_checkpoint).resolve() != Path(args.teacher_checkpoint).resolve():
            raise ValueError("--anchor-checkpoint and --teacher-checkpoint point to different files")
    if args.anchor_checkpoint is None and args.teacher_checkpoint is not None:
        args.anchor_checkpoint = args.teacher_checkpoint

    return TrainingConfig(
        human_data_dir=args.human_data_dir,
        selfplay_dirs=args.selfplay_dirs,
        output_dir=args.output_dir,
        resume_path=args.resume_path,
        reset_optimizer_on_resume=bool(args.reset_optimizer_on_resume),
        device=args.device,
        replay_buffer_size=args.replay_buffer_size,
        poll_interval_s=args.poll_interval_s,
        shard_cache_size=args.shard_cache_size,
        bootstrap_mode=not args.disable_bootstrap_mode,
        bootstrap_human_floor=args.bootstrap_human_floor,
        selfplay_run_quality_gate=not args.disable_selfplay_run_quality_gate,
        selfplay_run_max_rep_draw_rate=args.selfplay_run_max_rep_draw_rate,
        selfplay_run_min_decisive_rate=args.selfplay_run_min_decisive_rate,
        reset_selfplay_ingest_state_on_resume=bool(args.reset_selfplay_ingest_state_on_resume),
        selfplay_dir_sampling_ratios=list(args.selfplay_dir_sampling_ratios),
        micro_batch_size=args.micro_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        eval_interval_steps=args.eval_interval_steps,
        save_interval_steps=args.save_interval_steps,
        snapshot_interval_steps=args.snapshot_interval_steps,
        log_interval_steps=args.log_interval_steps,
        max_steps=args.max_steps,
        lr_schedule_max_steps=args.lr_schedule_max_steps,
        samples_per_unit=args.samples_per_unit,
        cpu_sampler_workers=args.cpu_sampler_workers,
        cpu_prefetch_batches=args.cpu_prefetch_batches,
        cpu_reserved_cores=args.cpu_reserved_cores,
        cpu_sampler_backend=args.cpu_sampler_backend,
        run_detached=not args.foreground,
        save_on_interrupt=not args.disable_save_on_interrupt,
        promote_best_on_human_val=not args.disable_promote_best_on_human_val,
        pause_at_local_time=args.pause_at_local_time,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        warmup_steps=args.warmup_steps,
        wdl_loss_weight=args.wdl_loss_weight,
        policy_loss_weight=args.policy_loss_weight,
        value_loss_weight=args.value_loss_weight,
        wdl_value_consistency_weight=args.wdl_value_consistency_weight,
        value_target_scale=args.value_target_scale,
        use_oracle_value=bool(args.use_oracle_value),
        policy_oracle_alpha=float(args.policy_oracle_alpha),
        teacher_q_loss_weight=float(args.teacher_q_loss_weight),
        teacher_q_temperature_cp=float(args.teacher_q_temperature_cp),
        teacher_q_pairwise_loss_weight=float(args.teacher_q_pairwise_loss_weight),
        teacher_q_pairwise_margin_logit=float(args.teacher_q_pairwise_margin_logit),
        teacher_q_pairwise_min_gap_cp=float(args.teacher_q_pairwise_min_gap_cp),
        teacher_q_pairwise_beta=float(args.teacher_q_pairwise_beta),
        teacher_q_pairwise_use_anchor_reference=bool(args.teacher_q_pairwise_use_anchor_reference),
        teacher_q_pairwise_bad_move_only=bool(args.teacher_q_pairwise_bad_move_only),
        bad_move_suppression_loss_weight=float(args.bad_move_suppression_loss_weight),
        bad_move_suppression_margin_logit=float(args.bad_move_suppression_margin_logit),
        bad_move_suppression_min_gap_cp=float(args.bad_move_suppression_min_gap_cp),
        bad_move_suppression_beta=float(args.bad_move_suppression_beta),
        anchor_checkpoint=args.anchor_checkpoint,
        anchor_policy_kl_weight=float(args.anchor_policy_kl_weight),
        anchor_policy_top1_ce_weight=float(args.anchor_policy_top1_ce_weight),
        anchor_value_mse_weight=float(args.anchor_value_mse_weight),
        anchor_anneal_steps=int(args.anchor_anneal_steps),
        use_bfloat16=not args.disable_bfloat16,
        allow_tf32=not args.disable_tf32,
        cudnn_benchmark=not args.disable_cudnn_benchmark,
        seed=args.seed,
        model_config=_model_config_from_args(args),
        train_only_relative_attention_bias=bool(
            args.train_only_relative_attention_bias or args.train_only_attention_biases
            or args.train_only_transformer_adapters
            or int(args.adapter_unfreeze_last_n_blocks) > 0
        ),
        train_only_policy_head=bool(args.train_only_policy_head),
        train_only_value_head=bool(args.train_only_value_head),
        train_only_cnn_local_adapter=bool(args.train_only_cnn_local_adapter),
        train_only_cnn_policy_residual_adapter=bool(
            args.train_only_cnn_policy_residual_adapter
        ),
        adapter_unfreeze_last_n_blocks=int(args.adapter_unfreeze_last_n_blocks),
    )


def main() -> None:
    _maybe_handle_background_stop()
    config = _parse_args()
    if config.run_detached and os.environ.get("XIANGQI_TRAIN_CHILD") != "1":
        raise SystemExit(_run_pycharm_supervisor(config))
    summary = run_training(config)
    _write_background_status(Path(summary["output_dir"]), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
