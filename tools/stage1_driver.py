"""Stage 1 driver: alternates between shard generation and training.

Cycle (once per iteration):
  1. Generate distillation shards (30% of target samples, from random rollouts + Pikafish labels)
  2. Generate vs-Pikafish shards (70%, with Tier-1 noise)
  3. Train on the accumulated shards via xiangqi_train.py

The shards are written under `selfplay_runs/stage1/` in cycle-specific subdirs so
that xiangqi_train's ingest logic picks them up.  We run training as a subprocess
that consumes all eligible shards and advances the model by a fixed step budget.

Watchdog note: this script also re-executes the training subprocess if it ever
exits abnormally; no separate marathon_watchdog is needed for Stage 1.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


@dataclass(frozen=True)
class VspikaProfile:
    name: str
    games: int
    opp_depth: int
    noise_ratio: float
    our_sims: int


@dataclass(frozen=True)
class SanityLadderProfile:
    name: str
    games: int
    opp_depth: int
    our_sims: int


@dataclass(frozen=True)
class SelfplayProfile:
    """One self-play config block: our model plays against a frozen snapshot opponent.

    Format on the CLI: ``NAME:GAMES:OPP_SIMS:OUR_SIMS``

    The opponent checkpoint is a TOP-LEVEL flag (--selfplay-opp-checkpoint) rather than
    per-profile, since we typically want the same frozen snapshot across all self-play
    games in the curriculum.  If you need multiple opponents you can run multiple driver
    invocations or extend this dataclass.
    """
    name: str
    games: int
    opp_sims: int
    our_sims: int


def _safe_profile_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(name).strip())
    return safe.strip("_") or "profile"


def _parse_vspika_profile(raw: str) -> VspikaProfile:
    parts = str(raw).split(":")
    if len(parts) != 5:
        raise ValueError("expected NAME:GAMES:DEPTH:NOISE:SIMS")
    name = _safe_profile_name(parts[0])
    games = int(parts[1])
    opp_depth = int(parts[2])
    noise_ratio = float(parts[3])
    our_sims = int(parts[4])
    if games < 0:
        raise ValueError("GAMES must be >= 0")
    if opp_depth < 1:
        raise ValueError("DEPTH must be >= 1")
    if not (0.0 <= noise_ratio <= 1.0):
        raise ValueError("NOISE must be in [0, 1]")
    if our_sims < 1:
        raise ValueError("SIMS must be >= 1")
    return VspikaProfile(
        name=name,
        games=games,
        opp_depth=opp_depth,
        noise_ratio=noise_ratio,
        our_sims=our_sims,
    )


def _parse_selfplay_profile(raw: str) -> SelfplayProfile:
    parts = str(raw).split(":")
    if len(parts) != 4:
        raise ValueError("expected NAME:GAMES:OPP_SIMS:OUR_SIMS")
    name = _safe_profile_name(parts[0])
    games = int(parts[1])
    opp_sims = int(parts[2])
    our_sims = int(parts[3])
    if games < 0:
        raise ValueError("GAMES must be >= 0")
    if opp_sims < 1:
        raise ValueError("OPP_SIMS must be >= 1")
    if our_sims < 1:
        raise ValueError("OUR_SIMS must be >= 1")
    return SelfplayProfile(
        name=name,
        games=games,
        opp_sims=opp_sims,
        our_sims=our_sims,
    )


def _parse_sanity_ladder_profile(raw: str) -> SanityLadderProfile:
    parts = str(raw).split(":")
    if len(parts) != 4:
        raise ValueError("expected NAME:GAMES:DEPTH:SIMS")
    name = _safe_profile_name(parts[0])
    games = int(parts[1])
    opp_depth = int(parts[2])
    our_sims = int(parts[3])
    if games < 1:
        raise ValueError("GAMES must be >= 1")
    if opp_depth < 1:
        raise ValueError("DEPTH must be >= 1")
    if our_sims < 1:
        raise ValueError("SIMS must be >= 1")
    return SanityLadderProfile(
        name=name,
        games=games,
        opp_depth=opp_depth,
        our_sims=our_sims,
    )


def _profile_payload(profile: VspikaProfile | SanityLadderProfile | SelfplayProfile) -> dict:
    return dict(profile.__dict__)


def _sh(cmd: list[str], env: dict | None = None, log_path: Path | None = None,
        timeout_s: float | None = None) -> int:
    """Run a subprocess, streaming stdout+stderr. Returns the exit code.

    If ``timeout_s`` is provided and exceeded, subprocess.run raises TimeoutExpired
    which we translate to exit code 124 (same convention as coreutils `timeout`).
    """
    try:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a") as f:
                f.write(f"\n{'='*70}\n[{datetime.now().isoformat()}] CMD: {' '.join(cmd)}\n{'='*70}\n")
                f.flush()
                r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env,
                                   timeout=timeout_s)
        else:
            r = subprocess.run(cmd, env=env, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT ({timeout_s}s) on: {' '.join(cmd)}", flush=True)
        return 124
    return int(r.returncode)


def _spawn(cmd: list[str], env: dict | None = None, log_path: Path | None = None):
    """Launch a subprocess non-blocking.

    Returns (Popen, log_file_handle_or_None).  The caller MUST NOT close the log
    file until the Popen has exited — the subprocess writes to it asynchronously.
    Use _wait_parallel() which handles cleanup.
    """
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        f = log_path.open("a")
        f.write(f"\n{'='*70}\n[{datetime.now().isoformat()}] CMD: {' '.join(cmd)}\n{'='*70}\n")
        f.flush()
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        return proc, f
    proc = subprocess.Popen(cmd, env=env)
    return proc, None


def _wait_parallel(
    entries: dict,
    poll_interval: float = 1.0,
    heartbeat_interval: float = 60.0,
    kill_on_first_failure: bool = True,
) -> dict:
    """Wait for all spawned subprocesses in `entries` (dict[name] = (Popen, file_handle)).

    If `kill_on_first_failure`, the first non-zero rc triggers SIGTERM/SIGKILL on siblings
    so we don't leave orphan GPU processes.  Emits a heartbeat line every `heartbeat_interval`
    seconds so the outer marathon watchdog doesn't stall-detect.

    Returns dict[name] -> int return code.
    """
    results: dict[str, int] = {}
    pending = dict(entries)
    kill_triggered = False
    last_beat = time.monotonic()
    while pending:
        time.sleep(poll_interval)
        now = time.monotonic()
        if now - last_beat > heartbeat_interval:
            names = ",".join(sorted(pending.keys()))
            print(f"  [parallel] heartbeat: still running [{names}]", flush=True)
            last_beat = now
        for name in list(pending.keys()):
            proc, _ = pending[name]
            rc = proc.poll()
            if rc is None:
                continue
            results[name] = int(rc)
            del pending[name]
            if rc != 0 and kill_on_first_failure and not kill_triggered:
                kill_triggered = True
                for other_name, (other_proc, _) in list(pending.items()):
                    print(
                        f"  [parallel] killing {other_name} (sibling {name} exited rc={rc})",
                        flush=True,
                    )
                    try:
                        other_proc.terminate()
                        other_proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        other_proc.kill()
                        try:
                            other_proc.wait(timeout=5)
                        except Exception:
                            pass
                    results[other_name] = int(other_proc.returncode if other_proc.returncode is not None else -1)
                pending.clear()
                break
    # Close log files held open for the subprocesses
    for _, pair in entries.items():
        _, f = pair
        if f is not None:
            try:
                f.close()
            except Exception:
                pass
    return results


def _run_sanity_probe(
    *,
    cycle: int,
    py: str,
    repo: Path,
    training_dir: Path,
    stage_log_dir: Path,
    device: str,
    games: int,
    opp_depth: int,
    our_sims: int,
    opp_noise_ratio: float = 0.0,
    timeout_s: float = 1800.0,
    seed_base: int = 20260423,
) -> dict:
    """Run a small external arena against Pikafish and return the parsed result.

    Purpose: catch pessimism-collapse early.  After the overnight Stage-2 pilot
    disaster (44 cycles, 0-1712-5 WLD because we cranked opp_depth too fast),
    we now gate each training cycle with a ~20-game arena.  If winrate drops
    below `sanity_probe_min_winrate`, the driver halts so we don't waste
    another 11 hours of compute.

    Returns dict with keys:
      {'cycle', 'wins', 'losses', 'draws', 'total', 'winrate', 'score_rate',
       'elo_estimate', 'duration_s', 'error' (if failed)}
    """
    probe_root = training_dir / "stage2_sanity_probes"
    probe_root.mkdir(parents=True, exist_ok=True)
    probe_dir = probe_root / f"cycle_{cycle:03d}"
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe_log = stage_log_dir / f"sanity_probe_cycle_{cycle:03d}.log"

    ckpt = training_dir / "latest.pt"
    if not ckpt.is_file():
        return {"cycle": cycle, "error": f"no checkpoint at {ckpt}"}

    cmd = [
        py,
        str(repo / "tools" / "external_arena.py"),
        "--checkpoint", str(ckpt),
        "--output-dir", str(probe_dir),
        "--games", str(int(games)),
        "--opp-depth", str(int(opp_depth)),
        "--our-sims", str(int(our_sims)),
        "--opp-noise-ratio", str(float(opp_noise_ratio)),
        "--device", device,
        "--seed", str(int(seed_base) + cycle * 13 + 7),
    ]
    rc = _sh(cmd, log_path=probe_log, timeout_s=timeout_s)
    if rc != 0:
        return {"cycle": cycle, "error": f"probe subprocess rc={rc}, see {probe_log}"}

    json_files = sorted(probe_dir.glob("external_arena_*.json"))
    if not json_files:
        return {"cycle": cycle, "error": "no JSON output from external_arena"}
    try:
        data = json.loads(json_files[-1].read_text(encoding="utf-8"))
    except Exception as exc:
        return {"cycle": cycle, "error": f"JSON read failed: {type(exc).__name__}: {exc}"}

    wins = int(data.get("our_wins", 0))
    losses = int(data.get("opp_wins", 0))
    draws = int(data.get("draws", 0))
    total = wins + losses + draws
    winrate = (wins / total) if total > 0 else 0.0
    result = {
        "cycle": cycle,
        "timestamp": datetime.now().isoformat(),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "total": total,
        "winrate": winrate,
        "score_rate": float(data.get("score_rate", 0.0)),
        "elo_estimate": data.get("elo_estimate"),
        "duration_s": float(data.get("duration_s", 0.0)),
        "opp_depth": opp_depth,
        "our_sims": our_sims,
        "checkpoint_step": _get_current_step(ckpt),
    }
    # Append to the run-level probe log so we can chart Elo over time.
    probe_jsonl = stage_log_dir / "sanity_probe.jsonl"
    with probe_jsonl.open("a") as f:
        f.write(json.dumps(result) + "\n")
    return result


def _run_sanity_ladder(
    *,
    cycle: int,
    py: str,
    repo: Path,
    training_dir: Path,
    stage_log_dir: Path,
    device: str,
    profiles: list[SanityLadderProfile],
    timeout_s: float = 1800.0,
    seed_base: int = 20260423,
) -> list[dict]:
    """Run one or more no-noise arena probes and append them to sanity_ladder.jsonl."""
    ladder_root = training_dir / "stage2_sanity_ladder"
    cycle_root = ladder_root / f"cycle_{cycle:03d}"
    cycle_root.mkdir(parents=True, exist_ok=True)
    ckpt = training_dir / "latest.pt"
    results: list[dict] = []
    ladder_jsonl = stage_log_dir / "sanity_ladder.jsonl"

    if not ckpt.is_file():
        result = {"cycle": cycle, "error": f"no checkpoint at {ckpt}"}
        with ladder_jsonl.open("a") as f:
            f.write(json.dumps(result) + "\n")
        return [result]

    for idx, profile in enumerate(profiles):
        probe_dir = cycle_root / profile.name
        probe_dir.mkdir(parents=True, exist_ok=True)
        probe_log = stage_log_dir / f"sanity_ladder_cycle_{cycle:03d}_{profile.name}.log"
        cmd = [
            py,
            str(repo / "tools" / "external_arena.py"),
            "--checkpoint", str(ckpt),
            "--output-dir", str(probe_dir),
            "--games", str(int(profile.games)),
            "--opp-depth", str(int(profile.opp_depth)),
            "--our-sims", str(int(profile.our_sims)),
            "--device", device,
            "--seed", str(int(seed_base) + cycle * 1009 + idx * 37),
        ]
        rc = _sh(cmd, log_path=probe_log, timeout_s=timeout_s)
        if rc != 0:
            result = {
                "cycle": cycle,
                "timestamp": datetime.now().isoformat(),
                "profile": profile.name,
                "error": f"ladder subprocess rc={rc}, see {probe_log}",
            }
            results.append(result)
            with ladder_jsonl.open("a") as f:
                f.write(json.dumps(result) + "\n")
            continue

        json_files = sorted(probe_dir.glob("external_arena_*.json"))
        if not json_files:
            result = {
                "cycle": cycle,
                "timestamp": datetime.now().isoformat(),
                "profile": profile.name,
                "error": "no JSON output from external_arena",
            }
            results.append(result)
            with ladder_jsonl.open("a") as f:
                f.write(json.dumps(result) + "\n")
            continue

        try:
            data = json.loads(json_files[-1].read_text(encoding="utf-8"))
        except Exception as exc:
            result = {
                "cycle": cycle,
                "timestamp": datetime.now().isoformat(),
                "profile": profile.name,
                "error": f"JSON read failed: {type(exc).__name__}: {exc}",
            }
            results.append(result)
            with ladder_jsonl.open("a") as f:
                f.write(json.dumps(result) + "\n")
            continue

        wins = int(data.get("our_wins", 0))
        losses = int(data.get("opp_wins", 0))
        draws = int(data.get("draws", 0))
        total = wins + losses + draws
        result = {
            "cycle": cycle,
            "timestamp": datetime.now().isoformat(),
            "profile": profile.name,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "total": total,
            "winrate": (wins / total) if total > 0 else 0.0,
            "score_rate": float(data.get("score_rate", 0.0)),
            "elo_estimate": data.get("elo_estimate"),
            "duration_s": float(data.get("duration_s", 0.0)),
            "avg_plies": float(data.get("avg_plies", 0.0)),
            "termination_counts": data.get("termination_counts", {}),
            "opp_depth": profile.opp_depth,
            "our_sims": profile.our_sims,
            "checkpoint_step": _get_current_step(ckpt),
        }
        results.append(result)
        with ladder_jsonl.open("a") as f:
            f.write(json.dumps(result) + "\n")
    return results


def _read_last_human_val_total_loss(training_dir: Path) -> float | None:
    status_path = training_dir / "train_status.json"
    if not status_path.is_file():
        return None
    try:
        data = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    value = data.get("last_human_val_total_loss")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _verify_devices_available(devs: list[str]) -> None:
    """Fail fast if any requested `cuda:N` isn't actually present in this process.

    We can only check torch-visible devices; the subprocess might see a different set
    (e.g. if CUDA_VISIBLE_DEVICES is set differently) so this is best-effort.
    """
    cuda_devs = [d for d in devs if str(d).startswith("cuda:")]
    if not cuda_devs:
        return
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return  # torch not here; let the subprocess fail instead
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA not available but cuda:* devices requested: {cuda_devs}")
    count = torch.cuda.device_count()
    for d in cuda_devs:
        try:
            idx = int(str(d).split(":", 1)[1])
        except (IndexError, ValueError):
            continue
        if idx >= count:
            raise RuntimeError(
                f"device {d} requested but only {count} CUDA GPU(s) visible to torch"
            )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--venv-python", default="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/python")
    p.add_argument("--repo", default="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer")
    p.add_argument("--training-output-dir", default="/home/laure/alphaxiang/training_runs/run_002_pikafish_curriculum")
    p.add_argument("--selfplay-root", default="/home/laure/alphaxiang/selfplay_runs")
    p.add_argument("--human-data-dir", default="/home/laure/alphaxiang/human_bootstrap_data_elite_wdl")

    # per-cycle targets
    p.add_argument("--cycles", type=int, default=0, help="0 = run forever until killed")
    p.add_argument("--samples-per-cycle", type=int, default=12000)
    p.add_argument("--distill-fraction", type=float, default=0.30)
    p.add_argument("--train-steps-per-cycle", type=int, default=1500)
    p.add_argument("--train-lr-schedule-max-steps", type=int, default=200000)
    p.add_argument("--train-snapshot-interval-steps", type=int, default=2000,
                   help="Pass-through to xiangqi_train --snapshot-interval-steps. "
                        "If >0, training writes numbered snapshots every N steps under "
                        "<training_dir>/snapshots/. Default 2000 — never lose a peak again.")
    p.add_argument("--reset-buffer-on-first-cycle", action="store_true",
                   help="On the FIRST cycle of this run, pass --reset-selfplay-ingest-state-on-resume "
                        "to xiangqi_train so it discards the resume checkpoint's stale shard buffer "
                        "(useful when starting from a transplanted checkpoint whose original "
                        "selfplay_runs no longer exist on disk). Subsequent cycles run normally.")

    # distillation
    p.add_argument("--distill-depth", type=int, default=6)
    p.add_argument("--distill-workers", type=int, default=12)
    p.add_argument("--distill-threads-per-worker", type=int, default=1)
    p.add_argument("--distill-hash-mb", type=int, default=16)
    p.add_argument("--distill-random-opening-plies", type=int, default=20)

    # vs-pikafish
    p.add_argument("--vspika-opp-depth", type=int, default=3)
    p.add_argument("--vspika-noise-ratio", type=float, default=0.15)
    p.add_argument("--vspika-our-sims", type=int, default=256)
    p.add_argument("--vspika-games-per-batch", type=int, default=40,
                   help="approx number of games (model plays both sides of half each)")
    p.add_argument("--vspika-parallel-games", type=int, default=8,
                   help="How many vs-Pikafish games run concurrently in worker threads. "
                        "Each thread gets own Pikafish subprocess + evaluator. "
                        "Set to 1 for legacy serial. Default 8 tuned for TR 7970X + 5080.")
    p.add_argument("--vspika-profile", action="append", default=None,
                   help="Repeatable Stage 2 profile NAME:GAMES:DEPTH:NOISE:SIMS. "
                        "GAMES=0 keeps the profile in the manifest but skips generation. "
                        "When omitted, the legacy --vspika-* knobs create one profile.")

    # Self-play configuration: training shards generated by playing OUR latest model
    # against a frozen snapshot.  These shards are written to selfplay_root/<cycle_tag>_selfplay_*
    # and consumed by training the same way vspika shards are.
    #
    # Background: v6/v7 plateaued because the curriculum was 100% Pikafish, biasing the
    # policy toward beating that one engine.  Adding self-play games injects a different
    # training distribution (positions where opponent has the same style/blunders as us)
    # which has historically been the missing ingredient (cf. v9 plan).
    p.add_argument("--selfplay-profile", action="append", default=None,
                   help="Repeatable self-play profile NAME:GAMES:OPP_SIMS:OUR_SIMS. "
                        "If omitted, no self-play games are generated. "
                        "OPP_SIMS=400 (half of OUR_SIMS=800) is a reasonable starting point.")
    p.add_argument("--selfplay-opp-checkpoint", default=None,
                   help="Frozen snapshot used as the self-play opponent. Required when "
                        "any --selfplay-profile is specified. Typically a previous PEAK "
                        "checkpoint (e.g. /path/to/PEAK_step232500_v7.pt). "
                        "The opponent is held fixed for the entire run — it does NOT update "
                        "with our latest checkpoint, by design.")
    p.add_argument("--selfplay-parallel-games", type=int, default=8,
                   help="How many self-play games run concurrently in worker threads of "
                        "each pikafish_selfplay subprocess. Each thread shares the snapshot "
                        "ModelOpponent (single GPU instance) and the cross-game batcher. "
                        "Default 8.")

    p.add_argument("--device", default="cuda:0",
                   help="Fallback device. Used when --train-device / --selfplay-device "
                        "aren't set. Kept for backward compat with the single-GPU driver.")
    p.add_argument("--train-device", default=None,
                   help="Device for the training phase (e.g. cuda:0). Defaults to --device.")
    p.add_argument("--selfplay-device", default=None,
                   help="Device for the vs-Pikafish self-play phase (e.g. cuda:1). "
                        "Defaults to --device. If this differs from --train-device, vspika "
                        "and train run IN PARALLEL inside each cycle.")
    # Oracle value labeling (Pikafish-d=N) — post-process step after each cycle that
    # tags every position in the cycle's shards with a calibrated value target,
    # replacing the noisy z={-1,0,+1} for those positions during training.
    # Directly addresses Lemma 4 (OOD over-search trap from value miscalibration).
    p.add_argument("--oracle-label", action=argparse.BooleanOptionalAction, default=True,
                   help="Run tools/oracle_value_labeler.py on each cycle's shards before "
                        "the next cycle starts.  Adds 'oracle_value' field (Pikafish-d=N "
                        "calibrated value) which xiangqi_train uses as a much cleaner "
                        "value-head target than the noisy game-outcome z.  Default ON.")
    p.add_argument("--oracle-depth", type=int, default=12,
                   help="Pikafish depth for the oracle value labeler.  d=12 is a balance "
                        "between strength (much stronger than the d=6 used in distillation) "
                        "and wall time (~3 min/cycle vs d=15's ~10 min/cycle).  Bump to 15 "
                        "or 20 if compute allows.  Default 12.")
    p.add_argument("--oracle-workers", type=int, default=8,
                   help="Number of parallel Pikafish processes for labeling. Default 8.")
    p.add_argument("--oracle-hash-mb", type=int, default=64,
                   help="Pikafish hash table per worker (MB).  Larger -> better at high depth.")
    p.add_argument("--oracle-max-wait-per-shard-s", type=float, default=1800.0,
                   help="Per-shard timeout for tools/oracle_value_labeler.py. "
                        "High-depth d20 labels can need longer than the default on "
                        "tactical shards.")
    # v11: oracle policy distillation.  Adds Pikafish multipv-derived policy targets to
    # each shard, which the trainer blends with MCTS visits via --policy-oracle-alpha.
    p.add_argument("--policy-oracle-label", action=argparse.BooleanOptionalAction, default=False,
                   help="v11: run tools/oracle_policy_labeler.py after value labeling. "
                        "Adds Pikafish-multipv-derived policy targets. Default OFF (v10 behavior).")
    p.add_argument("--policy-oracle-depth", type=int, default=8,
                   help="Pikafish depth for policy oracle multipv. Lower than --oracle-depth "
                        "because multipv adds K-fold per-move work. Default 8.")
    p.add_argument("--policy-oracle-multipv", type=int, default=5,
                   help="Top-K moves per position. Default 5.")
    p.add_argument("--policy-oracle-temperature-cp", type=float, default=200.0,
                   help="Softmax temperature on cp evals. Default 200.")
    p.add_argument("--policy-oracle-adaptive-temperature",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="v12.5: pass --adaptive-temperature to oracle_policy_labeler.py.")
    p.add_argument("--policy-oracle-min-temperature-cp", type=float, default=35.0)
    p.add_argument("--policy-oracle-max-temperature-cp", type=float, default=200.0)
    p.add_argument("--policy-oracle-soft-gap-cp", type=float, default=30.0)
    p.add_argument("--policy-oracle-hard-gap-cp", type=float, default=180.0)
    p.add_argument("--policy-oracle-legal-smoothing", type=float, default=0.02,
                   help="v12.5: probability mass spread over legal moves in oracle policy labels.")
    p.add_argument("--policy-oracle-alpha", type=float, default=0.0,
                   help="Trainer-side blend weight: 0=MCTS only, 0.5=equal mix, 1=oracle only. "
                        "Forwarded to xiangqi_train.py --policy-oracle-alpha. "
                        "Set together with --policy-oracle-label. Default 0.0 (v10 behavior).")
    p.add_argument("--teacher-q-loss-weight", type=float, default=0.0,
                   help="v12.5: forwarded to xiangqi_train.py. Use >0 only with shards "
                        "labeled by action_value_labeler.py.")
    p.add_argument("--teacher-q-temperature-cp", type=float, default=80.0,
                   help="v12.5: softmax temperature for teacher_q_values in training.")
    # v11: hard-position mining (active-learning over value disagreement)
    p.add_argument("--hard-mining", action=argparse.BooleanOptionalAction, default=False,
                   help="v11: after labeling, run tools/hard_position_mining.py to flag "
                        "positions where the latest model disagrees most with oracle_value, "
                        "and assign sample_weight=heavy to them for next cycle. Default OFF.")
    p.add_argument("--hard-mining-top-percent", type=float, default=10.0,
                   help="Per-shard top-X%% by |oracle_v - pred_v| flagged hard. Default 10.")
    p.add_argument("--hard-mining-heavy-weight", type=float, default=3.0,
                   help="Sample weight for hard positions (others get 1.0). Default 3.0.")
    p.add_argument("--hard-mining-policy-regret-weight", type=float, default=0.0,
                   help="v12: when > 0, combine policy regret with value disagreement in mining. "
                        "Forwarded to hard_position_mining.py --policy-regret-weight. "
                        "Default 0 (v11 behavior).")
    p.add_argument("--action-value-label", action=argparse.BooleanOptionalAction, default=False,
                   help="v12.5: after hard mining, run tools/action_value_labeler.py to add "
                        "teacher_q_* action-value targets. Default OFF.")
    p.add_argument("--action-value-depth", type=int, default=12,
                   help="Pikafish depth for teacher_q child-position evals. Default 12.")
    p.add_argument("--action-value-oracle-top-k", type=int, default=6)
    p.add_argument("--action-value-mcts-top-k", type=int, default=3)
    p.add_argument("--action-value-max-candidates", type=int, default=8)
    p.add_argument("--action-value-only-hard", action=argparse.BooleanOptionalAction, default=True,
                   help="Only label rows whose sample_weight >= --action-value-min-sample-weight.")
    p.add_argument("--action-value-min-sample-weight", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=20260418)
    p.add_argument("--stop-at-local-time", default=None,
                   help="HH:MM 24-hour local time. Driver checks at the top of each "
                        "cycle; if current local time is >= this, it exits cleanly (no "
                        "new subprocesses spawned). Does NOT kill in-flight subprocesses. "
                        "Example: '07:30' to stop around 7:30 AM.")

    # Sanity probe: small arena every N cycles to catch pessimism-collapse early.
    # Added after the overnight Stage-2 pilot disaster (44 cycles @ 0% winrate) so
    # that if the curriculum is too aggressive, we halt after ~5 cycles, not 44.
    p.add_argument("--sanity-probe-every", type=int, default=5,
                   help="Run a small arena against Pikafish every N cycles. "
                        "0 = disabled. Default 5.")
    p.add_argument("--sanity-probe-games", type=int, default=20,
                   help="Games per sanity probe (default 20). Binomial CI at 20 games "
                        "is wide but enough to catch 0 percent winrate disasters reliably.")
    p.add_argument("--sanity-probe-opp-depth", type=int, default=8,
                   help="Pikafish depth for probe (matches Stage-1 Elo reference).")
    p.add_argument("--sanity-probe-our-sims", type=int, default=800,
                   help="MCTS sims for probe (matches Stage-1 Elo reference).")
    p.add_argument("--sanity-probe-opp-noise-ratio", type=float, default=0.0,
                   help="Probability per move of replacing Pikafish's choice with a random "
                        "legal move. Use 0.15 to match Stage-1 baseline opponent (d=1+n0.15).")
    p.add_argument("--sanity-probe-min-winrate", type=float, default=0.08,
                   help="If probe winrate < this, halt the driver. Default 0.08 (8 percent). "
                        "Set to 0 to never halt (still log the probe).")
    p.add_argument("--sanity-probe-device", default=None,
                   help="Device for sanity probe arena. Default: same as --train-device. "
                        "Probe runs BETWEEN cycles so device doesn't conflict with training.")
    p.add_argument("--sanity-probe-timeout-s", type=float, default=1800.0,
                   help="Kill the probe subprocess if it runs longer than this (default 1800s=30min). "
                        "Prevents hangs.")
    p.add_argument("--sanity-ladder-profile", action="append", default=None,
                   help="Repeatable no-noise ladder profile NAME:GAMES:DEPTH:SIMS. "
                        "When present, replaces the single sanity probe at probe time.")

    p.add_argument("--train-bootstrap-human-floor", type=float, default=0.10,
                   help="Passed to xiangqi_train.py --bootstrap-human-floor. "
                        "Stage 2 v2.2 uses 0.20 to keep a stronger human anchor.")
    p.add_argument("--train-learning-rate", type=float, default=3e-4,
                   help="Passed to xiangqi_train.py --learning-rate.")
    p.add_argument("--train-replay-buffer-size", type=int, default=0,
                   help="If >0, passed to xiangqi_train.py --replay-buffer-size. Set small "
                        "(~= a few cycles of samples) so buffer_fill->1.0 quickly and the "
                        "selfplay/oracle pool dominates the batch instead of human bootstrap data.")
    p.add_argument("--train-reset-selfplay-ingest-state-on-resume", action="store_true",
                   help="Pass --reset-selfplay-ingest-state-on-resume to xiangqi_train.py. "
                        "Useful for control runs that should not restore the checkpoint replay buffer.")
    p.add_argument("--halt-human-val-threshold", type=float, default=0.0,
                   help="If >0, halt after N consecutive cycles with last_human_val_total_loss above this value.")
    p.add_argument("--halt-human-val-patience", type=int, default=2,
                   help="Consecutive bad human-val cycles before halting when --halt-human-val-threshold is set.")

    args = p.parse_args()
    try:
        if args.vspika_profile:
            args.vspika_profiles = [_parse_vspika_profile(raw) for raw in args.vspika_profile]
        else:
            args.vspika_profiles = [
                VspikaProfile(
                    name="vspika",
                    games=int(args.vspika_games_per_batch),
                    opp_depth=int(args.vspika_opp_depth),
                    noise_ratio=float(args.vspika_noise_ratio),
                    our_sims=int(args.vspika_our_sims),
                )
            ]
        args.sanity_ladder_profiles = [
            _parse_sanity_ladder_profile(raw) for raw in (args.sanity_ladder_profile or [])
        ]
        args.selfplay_profiles = [
            _parse_selfplay_profile(raw) for raw in (args.selfplay_profile or [])
        ]
    except ValueError as exc:
        p.error(str(exc))
    if len({profile.name for profile in args.vspika_profiles}) != len(args.vspika_profiles):
        p.error("--vspika-profile names must be unique")
    if len({profile.name for profile in args.sanity_ladder_profiles}) != len(args.sanity_ladder_profiles):
        p.error("--sanity-ladder-profile names must be unique")
    if len({profile.name for profile in args.selfplay_profiles}) != len(args.selfplay_profiles):
        p.error("--selfplay-profile names must be unique")
    if args.selfplay_profiles and not args.selfplay_opp_checkpoint:
        p.error("--selfplay-profile requires --selfplay-opp-checkpoint")
    if args.selfplay_opp_checkpoint:
        opp_ckpt = Path(args.selfplay_opp_checkpoint)
        if not opp_ckpt.is_file():
            p.error(f"--selfplay-opp-checkpoint not found: {opp_ckpt}")
    if args.halt_human_val_patience < 1:
        p.error("--halt-human-val-patience must be >= 1")
    if not (0.0 <= args.train_bootstrap_human_floor <= 1.0):
        p.error("--train-bootstrap-human-floor must be in [0, 1]")
    return args


def _resolve_stop_datetime(stop_spec: str | None, launch_time: datetime) -> datetime | None:
    """Resolve --stop-at-local-time 'HH:MM' to the NEXT occurrence >= launch_time.

    If HH:MM today has already passed by the time we launched, stop is tomorrow's
    HH:MM.  Otherwise stop is today's HH:MM.  This prevents the driver from
    exiting immediately when launched after the nominal stop time.
    """
    if not stop_spec:
        return None
    try:
        h, m = [int(x) for x in stop_spec.split(":", 1)]
    except Exception:
        print(f"  WARN: bad --stop-at-local-time value {stop_spec!r}, ignoring", flush=True)
        return None
    stop_today = launch_time.replace(hour=h, minute=m, second=0, microsecond=0)
    if stop_today <= launch_time:
        stop_today += timedelta(days=1)
    return stop_today


def main() -> int:
    args = _parse_args()
    repo = Path(args.repo)
    py = args.venv_python
    training_dir = Path(args.training_output_dir)
    selfplay_root = Path(args.selfplay_root)
    selfplay_root.mkdir(parents=True, exist_ok=True)

    stage_log_dir = training_dir / "stage1_logs"
    stage_log_dir.mkdir(parents=True, exist_ok=True)
    driver_log = stage_log_dir / "stage1_driver.log"

    distill_target = int(args.samples_per_cycle * args.distill_fraction)
    vspika_target = int(args.samples_per_cycle - distill_target)
    vspika_profiles: list[VspikaProfile] = list(args.vspika_profiles)
    active_vspika_profiles = [p for p in vspika_profiles if p.games > 0]
    explicit_vspika_profiles = bool(args.vspika_profile)
    sanity_ladder_profiles: list[SanityLadderProfile] = list(args.sanity_ladder_profiles)
    selfplay_profiles: list[SelfplayProfile] = list(args.selfplay_profiles)
    active_selfplay_profiles = [p for p in selfplay_profiles if p.games > 0]

    # Resolve dual-GPU devices. If train and selfplay land on different CUDA indices we
    # run vspika+selfplay and train IN PARALLEL inside each cycle.  Distill stays serial —
    # it's CPU-only (Pikafish NNUE) and finishes quickly relative to the train+vspika pair.
    train_device = args.train_device or args.device
    selfplay_device = args.selfplay_device or args.device
    has_phase2_jobs = bool(active_vspika_profiles) or bool(active_selfplay_profiles)
    run_parallel = train_device != selfplay_device and has_phase2_jobs
    _verify_devices_available([train_device, selfplay_device])

    launch_time = datetime.now()
    stop_at = _resolve_stop_datetime(args.stop_at_local_time, launch_time)

    print(f"Stage 1 driver starting", flush=True)
    print(f"  training_dir : {training_dir}", flush=True)
    print(f"  selfplay_root: {selfplay_root}", flush=True)
    vspika_label = f"~{vspika_target} vs-pikafish samples" if active_vspika_profiles else "0 vs-pikafish samples"
    print(f"  per-cycle    : {distill_target} distill + {vspika_label}", flush=True)
    print(
        f"  distill      : depth={args.distill_depth} workers={args.distill_workers} "
        f"threads/worker={args.distill_threads_per_worker} hash={args.distill_hash_mb}MB",
        flush=True,
    )
    print(f"  train_steps  : {args.train_steps_per_cycle}", flush=True)
    print(
        "  vs-pikafish  : "
        + ", ".join(
            f"{p.name}(games={p.games}, depth={p.opp_depth}, noise={p.noise_ratio}, sims={p.our_sims})"
            + (" [skip]" if p.games <= 0 else "")
            for p in vspika_profiles
        ),
        flush=True,
    )
    if selfplay_profiles:
        print(
            "  self-play    : opp="
            + str(args.selfplay_opp_checkpoint) + "  "
            + ", ".join(
                f"{p.name}(games={p.games}, opp_sims={p.opp_sims}, our_sims={p.our_sims})"
                + (" [skip]" if p.games <= 0 else "")
                for p in selfplay_profiles
            ),
            flush=True,
        )
    print(f"  human_floor  : {args.train_bootstrap_human_floor}", flush=True)
    if sanity_ladder_profiles:
        print(
            "  sanity ladder: "
            + ", ".join(
                f"{p.name}(games={p.games}, depth={p.opp_depth}, sims={p.our_sims})"
                for p in sanity_ladder_profiles
            ),
            flush=True,
        )
    print(f"  devices      : train={train_device}  selfplay={selfplay_device}  "
          f"mode={'PARALLEL' if run_parallel else 'serial'}", flush=True)
    if stop_at is not None:
        print(f"  stop_at      : {stop_at.isoformat(timespec='minutes')} (local time)", flush=True)

    cycle = 0
    bad_human_val_streak = 0
    while True:
        cycle += 1
        if args.cycles and cycle > args.cycles:
            print(f"reached requested cycle limit ({args.cycles}); exiting", flush=True)
            return 0
        if stop_at is not None and datetime.now() >= stop_at:
            print(
                f"reached stop-at-local-time {stop_at.isoformat(timespec='minutes')}; "
                f"exiting before cycle {cycle}",
                flush=True,
            )
            return 0

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cycle_tag = f"stage1_c{cycle:03d}_{stamp}"
        print(f"\n{'='*72}\nCYCLE {cycle} ({cycle_tag})\n{'='*72}", flush=True)

        # 1. Distillation shards
        # IMPORTANT: each phase must live at the TOP level of selfplay_root as its own
        # "run" (with manifest.json + train/) so xiangqi_train's _SelfPlayIngestor picks
        # it up.  Putting distill/vspika as sub-dirs of a single run_dir makes the
        # ingestor see NO manifest and silently skip the shards (the bug that cost us
        # the first overnight session).
        dt_d = 0.0
        if distill_target > 0:
            distill_dir = selfplay_root / f"{cycle_tag}_distill"
            distill_log = stage_log_dir / f"{cycle_tag}_distill.log"
            t0 = time.monotonic()
            rc = _sh(
                [
                    py,
                    str(repo / "tools" / "distillation_generator.py"),
                    "--output-dir", str(distill_dir / "train"),
                    "--num-positions", str(distill_target),
                    "--depth", str(args.distill_depth),
                    "--workers", str(args.distill_workers),
                    "--threads-per-worker", str(args.distill_threads_per_worker),
                    "--hash-mb", str(args.distill_hash_mb),
                    "--shard-size", "2048",
                    "--seed", str(args.seed + cycle * 101),
                    "--random-opening-plies", str(args.distill_random_opening_plies),
                ],
                log_path=distill_log,
            )
            dt_d = time.monotonic() - t0
            print(f"  distill:    rc={rc}  dt={dt_d:.1f}s", flush=True)
            if rc != 0:
                print(f"  distill FAILED; see {distill_log}", flush=True)
                # continue trying; it may be a transient Pikafish issue
                time.sleep(5)
                continue
        else:
            print(f"  distill:    SKIPPED (distill_target=0)", flush=True)

        # 2. vspika + 3. train — run in parallel if we have two distinct devices,
        # otherwise fall back to the serial path.
        #
        # Parallel semantics: training ingests shards via _SelfPlayIngestor which polls
        # `selfplay_root` for new TOP-LEVEL runs that have a manifest.json.  vspika only
        # writes its manifest when all games finish.  So training during this cycle reads:
        #   - distill shards just written this cycle (manifest done)
        #   - all prior cycles' distill + vspika shards
        # The CURRENT cycle's vspika shards only become visible to the NEXT cycle's train.
        # That's fine — we still get 1.3-1.5x speedup by overlapping the two heaviest phases.
        current_step = _get_current_step(training_dir / "latest.pt")
        target_step = current_step + args.train_steps_per_cycle
        train_log = stage_log_dir / f"{cycle_tag}_train.log"

        vspika_jobs = []
        for profile_idx, profile in enumerate(active_vspika_profiles):
            suffix = "vspika" if (not explicit_vspika_profiles and len(vspika_profiles) == 1) else f"vspika_{profile.name}"
            vspika_dir = selfplay_root / f"{cycle_tag}_{suffix}"
            vspika_log = stage_log_dir / f"{cycle_tag}_{suffix}.log"
            vspika_cmd = [
                py,
                str(repo / "tools" / "pikafish_selfplay.py"),
                "--checkpoint", str(training_dir / "latest.pt"),
                "--output-dir", str(vspika_dir / "train"),
                "--num-games", str(profile.games),
                "--parallel-games", str(args.vspika_parallel_games),
                "--opp-depth", str(profile.opp_depth),
                "--noise-ratio", str(profile.noise_ratio),
                "--our-sims", str(profile.our_sims),
                "--shard-size-samples", "2048",
                "--seed", str(args.seed + cycle * 103 + 17 + profile_idx * 1009),
                "--device", selfplay_device,
            ]
            vspika_jobs.append(
                {
                    "name": profile.name,
                    "profile": profile,
                    "dir": vspika_dir,
                    "log": vspika_log,
                    "cmd": vspika_cmd,
                }
            )

        # Self-play jobs: same pikafish_selfplay.py binary, but --opp-type=model.
        # Output dirs use the "selfplay_<name>" prefix so training's _SelfPlayIngestor
        # picks them up alongside vspika shards.  Profile name is included for unique dirs.
        selfplay_jobs = []
        for profile_idx, profile in enumerate(active_selfplay_profiles):
            suffix = f"selfplay_{profile.name}"
            sp_dir = selfplay_root / f"{cycle_tag}_{suffix}"
            sp_log = stage_log_dir / f"{cycle_tag}_{suffix}.log"
            sp_cmd = [
                py,
                str(repo / "tools" / "pikafish_selfplay.py"),
                "--checkpoint", str(training_dir / "latest.pt"),
                "--output-dir", str(sp_dir / "train"),
                "--num-games", str(profile.games),
                "--parallel-games", str(args.selfplay_parallel_games),
                "--opp-type", "model",
                "--opp-model-checkpoint", str(args.selfplay_opp_checkpoint),
                "--opp-model-sims", str(profile.opp_sims),
                "--our-sims", str(profile.our_sims),
                "--shard-size-samples", "2048",
                "--seed", str(args.seed + cycle * 103 + 17 + (profile_idx + 100) * 1009),
                "--device", selfplay_device,
            ]
            selfplay_jobs.append(
                {
                    "name": profile.name,
                    "profile": profile,
                    "dir": sp_dir,
                    "log": sp_log,
                    "cmd": sp_cmd,
                }
            )

        train_cmd = [
            py, "-u",
            str(repo / "xiangqi_train.py"),
            "--foreground",
            "--human-data-dir", args.human_data_dir,
            "--selfplay-dirs", str(selfplay_root),
            "--output-dir", str(training_dir),
            "--resume-path", str(training_dir / "latest.pt"),
            "--device", train_device,
            "--max-steps", str(target_step),
            "--lr-schedule-max-steps", str(args.train_lr_schedule_max_steps),
            "--learning-rate", str(args.train_learning_rate),
            "--log-interval-steps", "100",
            "--eval-interval-steps", "500",
            "--save-interval-steps", "2000",
            "--snapshot-interval-steps", str(int(args.train_snapshot_interval_steps)),
            "--disable-selfplay-run-quality-gate",  # trust our generator-labelled shards
            # By default, keep the replay buffer from the checkpoint so stage runs grow
            # monotonically. Control runs can opt into a clean selfplay buffer.
            "--bootstrap-human-floor", str(args.train_bootstrap_human_floor),
            "--wdl-loss-weight", "1.0",
            "--value-loss-weight", "0.5",
            "--value-target-scale", "0.9",
        ]
        if int(getattr(args, "train_replay_buffer_size", 0)) > 0:
            train_cmd += ["--replay-buffer-size", str(int(args.train_replay_buffer_size))]
        if float(args.policy_oracle_alpha) > 0.0:
            train_cmd += ["--policy-oracle-alpha", str(float(args.policy_oracle_alpha))]
        if float(args.teacher_q_loss_weight) > 0.0:
            train_cmd += [
                "--teacher-q-loss-weight", str(float(args.teacher_q_loss_weight)),
                "--teacher-q-temperature-cp", str(float(args.teacher_q_temperature_cp)),
            ]
        if (args.train_reset_selfplay_ingest_state_on_resume
                or (args.reset_buffer_on_first_cycle and cycle == 1)):
            train_cmd.append("--reset-selfplay-ingest-state-on-resume")

        # dt_v / dt_t are per-phase wall times in serial mode; in parallel mode we
        # only know the combined wall time (`dt_phase23`), so we record that separately.
        dt_v = 0.0
        dt_t = 0.0
        dt_phase23 = 0.0
        vspika_results: dict[str, dict] = {}
        selfplay_results: dict[str, dict] = {}

        if run_parallel:
            print(
                f"  launching {len(vspika_jobs)} vspika + {len(selfplay_jobs)} self-play "
                f"profile(s) (device={selfplay_device}) and "
                f"train (device={train_device}, step {current_step}->{target_step}) IN PARALLEL",
                flush=True,
            )
            t0 = time.monotonic()
            procs = {"train": _spawn(train_cmd, log_path=train_log)}
            # Subprocess-level stagger: spawn each vspika/selfplay subprocess with a
            # short delay so the K Pikafish processes inside subprocess A finish
            # initialising before subprocess B starts spawning its own.  Without
            # this, 4 subprocesses each spawning 8 Pikafish concurrently overwhelms
            # WSL's file/memory subsystem and Pikafish dies mid-game with BrokenPipe
            # (observed reproducibly in v10 attempts 1-4).  10s gap × 4 subprocesses
            # = 30s extra startup per cycle, negligible vs 17 min cycle wall.
            stagger_s = 10.0
            for i, job in enumerate(vspika_jobs):
                if i > 0:
                    time.sleep(stagger_s)
                procs[f"vspika:{job['name']}"] = _spawn(job["cmd"], log_path=job["log"])
            for i, job in enumerate(selfplay_jobs):
                time.sleep(stagger_s)
                procs[f"selfplay:{job['name']}"] = _spawn(job["cmd"], log_path=job["log"])
            results = _wait_parallel(procs)
            dt_phase23 = time.monotonic() - t0
            rc_t = results.get("train", -1)
            rc_vspika = {job["name"]: results.get(f"vspika:{job['name']}", -1) for job in vspika_jobs}
            rc_selfplay = {job["name"]: results.get(f"selfplay:{job['name']}", -1) for job in selfplay_jobs}
            print(
                f"  parallel:   rc_vspika={rc_vspika}  rc_selfplay={rc_selfplay}  "
                f"rc_train={rc_t}  dt_total={dt_phase23:.1f}s  "
                f"(step {current_step} -> {target_step})",
                flush=True,
            )
            for job in vspika_jobs:
                vspika_results[job["name"]] = {
                    "rc": int(rc_vspika[job["name"]]),
                    "seconds": 0.0,
                    "log": str(job["log"]),
                    "profile": _profile_payload(job["profile"]),
                }
            for job in selfplay_jobs:
                selfplay_results[job["name"]] = {
                    "rc": int(rc_selfplay[job["name"]]),
                    "seconds": 0.0,
                    "log": str(job["log"]),
                    "profile": _profile_payload(job["profile"]),
                }
            failed_vspika = [name for name, rc in rc_vspika.items() if rc != 0]
            failed_selfplay = [name for name, rc in rc_selfplay.items() if rc != 0]
            if failed_vspika:
                print(f"  vspika FAILED for {failed_vspika}; see stage logs", flush=True)
                time.sleep(5)
                continue
            if failed_selfplay:
                print(f"  self-play FAILED for {failed_selfplay}; see stage logs", flush=True)
                time.sleep(5)
                continue
            if rc_t != 0:
                print(f"  train FAILED; see {train_log}", flush=True)
                time.sleep(10)
                continue
        else:
            # Serial path (single-GPU or explicitly-equal devices).
            phase2_failed = False
            for job in vspika_jobs:
                t0 = time.monotonic()
                rc_v = _sh(job["cmd"], log_path=job["log"])
                dt_one = time.monotonic() - t0
                dt_v += dt_one
                vspika_results[job["name"]] = {
                    "rc": int(rc_v),
                    "seconds": dt_one,
                    "log": str(job["log"]),
                    "profile": _profile_payload(job["profile"]),
                }
                print(f"  vs-pikafish[{job['name']}]: rc={rc_v}  dt={dt_one:.1f}s", flush=True)
                if rc_v != 0:
                    print(f"  vspika FAILED; see {job['log']}", flush=True)
                    phase2_failed = True
                    break
            if not phase2_failed:
                for job in selfplay_jobs:
                    t0 = time.monotonic()
                    rc_v = _sh(job["cmd"], log_path=job["log"])
                    dt_one = time.monotonic() - t0
                    dt_v += dt_one
                    selfplay_results[job["name"]] = {
                        "rc": int(rc_v),
                        "seconds": dt_one,
                        "log": str(job["log"]),
                        "profile": _profile_payload(job["profile"]),
                    }
                    print(f"  self-play[{job['name']}]: rc={rc_v}  dt={dt_one:.1f}s", flush=True)
                    if rc_v != 0:
                        print(f"  self-play FAILED; see {job['log']}", flush=True)
                        phase2_failed = True
                        break
            if phase2_failed:
                time.sleep(5)
                continue

            t0 = time.monotonic()
            rc_t = _sh(train_cmd, log_path=train_log)
            dt_t = time.monotonic() - t0
            print(f"  train:      rc={rc_t}  dt={dt_t:.1f}s  (step {current_step} -> {target_step})", flush=True)
            if rc_t != 0:
                print(f"  train FAILED; see {train_log}", flush=True)
                time.sleep(10)
                continue
            dt_phase23 = dt_v + dt_t

        # 3.5. Oracle value labeling — post-cycle, before next cycle starts.
        # This is the Phase 1 distillation upgrade (paper Lemma 4 fix).  We label
        # every shard generated this cycle so the NEXT cycle's training reads
        # them with a calibrated Pikafish-d=N value target instead of noisy z.
        # The labeler is idempotent (skip-already-labeled), so re-running is safe.
        # Distill shards labeled here will only become "oracle-trained" in the
        # NEXT cycle's training; the CURRENT cycle's training already consumed
        # them with z-loss.  Acceptable trade-off: ~30% of any single cycle's
        # samples train with z, the other 70% with oracle.
        dt_label = 0.0
        oracle_label_results: dict[str, dict] = {}
        if args.oracle_label:
            oracle_label_failed = False
            label_dirs: list[tuple[str, Path, Path]] = []
            if distill_target > 0:
                label_dirs.append(("distill", distill_dir / "train",
                                   stage_log_dir / f"{cycle_tag}_label_distill.log"))
            for job in vspika_jobs:
                label_dirs.append((f"vspika_{job['name']}", job["dir"] / "train",
                                   stage_log_dir / f"{cycle_tag}_label_{job['name']}.log"))
            for job in selfplay_jobs:
                label_dirs.append((f"selfplay_{job['name']}", job["dir"] / "train",
                                   stage_log_dir / f"{cycle_tag}_label_sp_{job['name']}.log"))
            print(f"  oracle-label: starting {len(label_dirs)} dir(s) at d={args.oracle_depth} "
                  f"with {args.oracle_workers} workers...", flush=True)
            t0 = time.monotonic()
            for tag, shard_dir, log_path in label_dirs:
                if not shard_dir.is_dir():
                    print(f"  oracle-label[{tag}]: SKIP (no dir at {shard_dir})", flush=True)
                    oracle_label_results[tag] = {"rc": -1, "skipped": "no_dir"}
                    continue
                label_cmd = [
                    py,
                    str(repo / "tools" / "oracle_value_labeler.py"),
                    "--input-shard-dir", str(shard_dir),
                    "--depth", str(int(args.oracle_depth)),
                    "--workers", str(int(args.oracle_workers)),
                    "--hash-mb", str(int(args.oracle_hash_mb)),
                    "--max-wait-per-shard-s", str(float(args.oracle_max_wait_per_shard_s)),
                    "--skip-already-labeled",
                ]
                t_one = time.monotonic()
                rc_l = _sh(label_cmd, log_path=log_path)
                dt_one = time.monotonic() - t_one
                oracle_label_results[tag] = {
                    "rc": int(rc_l),
                    "seconds": dt_one,
                    "log": str(log_path),
                }
                print(f"  oracle-label[{tag}]: rc={rc_l} dt={dt_one:.0f}s", flush=True)
                if rc_l != 0:
                    print(f"  oracle-label[{tag}] FAILED; see {log_path}", flush=True)
                    oracle_label_failed = True
            dt_label = time.monotonic() - t0
            print(f"  oracle-label: done in {dt_label:.0f}s total", flush=True)
            if oracle_label_failed:
                raise SystemExit("oracle-label failed; halting stage1_driver")

        # 3b. v11: policy oracle labeling (after value oracle, mirror logic)
        dt_policy_label = 0.0
        policy_label_results: dict[str, dict] = {}
        if args.policy_oracle_label:
            policy_label_failed = False
            policy_label_dirs: list[tuple[str, Path, Path]] = []
            if distill_target > 0:
                policy_label_dirs.append(("distill", distill_dir / "train",
                                          stage_log_dir / f"{cycle_tag}_polabel_distill.log"))
            for job in vspika_jobs:
                policy_label_dirs.append((f"vspika_{job['name']}", job["dir"] / "train",
                                          stage_log_dir / f"{cycle_tag}_polabel_{job['name']}.log"))
            for job in selfplay_jobs:
                policy_label_dirs.append((f"selfplay_{job['name']}", job["dir"] / "train",
                                          stage_log_dir / f"{cycle_tag}_polabel_sp_{job['name']}.log"))
            print(f"  policy-oracle-label: starting {len(policy_label_dirs)} dir(s) at "
                  f"d={args.policy_oracle_depth} multipv={args.policy_oracle_multipv} "
                  f"with {args.oracle_workers} workers...", flush=True)
            t0 = time.monotonic()
            for tag, shard_dir, log_path in policy_label_dirs:
                if not shard_dir.is_dir():
                    print(f"  policy-oracle-label[{tag}]: SKIP (no dir at {shard_dir})", flush=True)
                    policy_label_results[tag] = {"rc": -1, "skipped": "no_dir"}
                    continue
                p_label_cmd = [
                    py,
                    str(repo / "tools" / "oracle_policy_labeler.py"),
                    "--input-shard-dir", str(shard_dir),
                    "--depth", str(int(args.policy_oracle_depth)),
                    "--multipv", str(int(args.policy_oracle_multipv)),
                    "--temperature-cp", str(float(args.policy_oracle_temperature_cp)),
                    "--min-temperature-cp", str(float(args.policy_oracle_min_temperature_cp)),
                    "--max-temperature-cp", str(float(args.policy_oracle_max_temperature_cp)),
                    "--soft-gap-cp", str(float(args.policy_oracle_soft_gap_cp)),
                    "--hard-gap-cp", str(float(args.policy_oracle_hard_gap_cp)),
                    "--legal-smoothing", str(float(args.policy_oracle_legal_smoothing)),
                    "--workers", str(int(args.oracle_workers)),
                    "--hash-mb", str(int(args.oracle_hash_mb)),
                    "--skip-already-labeled",
                ]
                if args.policy_oracle_adaptive_temperature:
                    p_label_cmd.append("--adaptive-temperature")
                t_one = time.monotonic()
                rc_l = _sh(p_label_cmd, log_path=log_path)
                dt_one = time.monotonic() - t_one
                policy_label_results[tag] = {
                    "rc": int(rc_l),
                    "seconds": dt_one,
                    "log": str(log_path),
                }
                print(f"  policy-oracle-label[{tag}]: rc={rc_l} dt={dt_one:.0f}s", flush=True)
                if rc_l != 0:
                    print(f"  policy-oracle-label[{tag}] FAILED; see {log_path}", flush=True)
                    policy_label_failed = True
            dt_policy_label = time.monotonic() - t0
            print(f"  policy-oracle-label: done in {dt_policy_label:.0f}s total", flush=True)
            if policy_label_failed:
                raise SystemExit("policy-oracle-label failed; halting stage1_driver")

        # 3c. v11: hard-position mining (uses latest.pt model after train phase)
        dt_hard_mine = 0.0
        hard_mine_results: dict[str, dict] = {}
        if args.hard_mining:
            ckpt_for_mining = training_dir / "latest.pt"
            if not ckpt_for_mining.is_file():
                print(f"  hard-mining: SKIP (no checkpoint at {ckpt_for_mining})", flush=True)
            else:
                hard_mining_failed = False
                hm_dirs: list[tuple[str, Path, Path]] = []
                if distill_target > 0:
                    hm_dirs.append(("distill", distill_dir / "train",
                                    stage_log_dir / f"{cycle_tag}_mine_distill.log"))
                for job in vspika_jobs:
                    hm_dirs.append((f"vspika_{job['name']}", job["dir"] / "train",
                                    stage_log_dir / f"{cycle_tag}_mine_{job['name']}.log"))
                for job in selfplay_jobs:
                    hm_dirs.append((f"selfplay_{job['name']}", job["dir"] / "train",
                                    stage_log_dir / f"{cycle_tag}_mine_sp_{job['name']}.log"))
                print(f"  hard-mining: starting {len(hm_dirs)} dir(s), "
                      f"top-{args.hard_mining_top_percent}%, weight={args.hard_mining_heavy_weight}",
                      flush=True)
                t0 = time.monotonic()
                for tag, shard_dir, log_path in hm_dirs:
                    if not shard_dir.is_dir():
                        hard_mine_results[tag] = {"rc": -1, "skipped": "no_dir"}
                        continue
                    hm_cmd = [
                        py,
                        str(repo / "tools" / "hard_position_mining.py"),
                        "--checkpoint", str(ckpt_for_mining),
                        "--input-shard-dir", str(shard_dir),
                        "--top-percent", str(float(args.hard_mining_top_percent)),
                        "--heavy-weight", str(float(args.hard_mining_heavy_weight)),
                        "--policy-regret-weight", str(float(args.hard_mining_policy_regret_weight)),
                        "--device", train_device,
                    ]
                    t_one = time.monotonic()
                    rc_h = _sh(hm_cmd, log_path=log_path)
                    dt_one = time.monotonic() - t_one
                    hard_mine_results[tag] = {"rc": int(rc_h), "seconds": dt_one,
                                              "log": str(log_path)}
                    print(f"  hard-mining[{tag}]: rc={rc_h} dt={dt_one:.0f}s", flush=True)
                    if rc_h != 0:
                        print(f"  hard-mining[{tag}] FAILED; see {log_path}", flush=True)
                        hard_mining_failed = True
                dt_hard_mine = time.monotonic() - t0
                print(f"  hard-mining: done in {dt_hard_mine:.0f}s total", flush=True)
                if hard_mining_failed:
                    raise SystemExit("hard-mining failed; halting stage1_driver")

        # 3d. v12.5: action-value labeling for hard tactical rows.
        dt_action_value = 0.0
        action_value_results: dict[str, dict] = {}
        if args.action_value_label:
            action_value_failed = False
            av_dirs: list[tuple[str, Path, Path]] = []
            if distill_target > 0:
                av_dirs.append(("distill", distill_dir / "train",
                                stage_log_dir / f"{cycle_tag}_teacherq_distill.log"))
            for job in vspika_jobs:
                av_dirs.append((f"vspika_{job['name']}", job["dir"] / "train",
                                stage_log_dir / f"{cycle_tag}_teacherq_{job['name']}.log"))
            for job in selfplay_jobs:
                av_dirs.append((f"selfplay_{job['name']}", job["dir"] / "train",
                                stage_log_dir / f"{cycle_tag}_teacherq_sp_{job['name']}.log"))
            print(f"  action-value-label: starting {len(av_dirs)} dir(s) at "
                  f"d={args.action_value_depth}, max_candidates={args.action_value_max_candidates}",
                  flush=True)
            t0 = time.monotonic()
            for tag, shard_dir, log_path in av_dirs:
                if not shard_dir.is_dir():
                    action_value_results[tag] = {"rc": -1, "skipped": "no_dir"}
                    continue
                av_cmd = [
                    py,
                    str(repo / "tools" / "action_value_labeler.py"),
                    "--input-shard-dir", str(shard_dir),
                    "--depth", str(int(args.action_value_depth)),
                    "--workers", str(int(args.oracle_workers)),
                    "--hash-mb", str(int(args.oracle_hash_mb)),
                    "--oracle-top-k", str(int(args.action_value_oracle_top_k)),
                    "--mcts-top-k", str(int(args.action_value_mcts_top_k)),
                    "--max-candidates", str(int(args.action_value_max_candidates)),
                    "--min-sample-weight", str(float(args.action_value_min_sample_weight)),
                    "--skip-already-labeled",
                ]
                if args.action_value_only_hard:
                    av_cmd.append("--only-hard")
                else:
                    av_cmd.append("--no-only-hard")
                t_one = time.monotonic()
                rc_av = _sh(av_cmd, log_path=log_path)
                dt_one = time.monotonic() - t_one
                action_value_results[tag] = {"rc": int(rc_av), "seconds": dt_one,
                                             "log": str(log_path)}
                print(f"  action-value-label[{tag}]: rc={rc_av} dt={dt_one:.0f}s", flush=True)
                if rc_av != 0:
                    print(f"  action-value-label[{tag}] FAILED; see {log_path}", flush=True)
                    action_value_failed = True
            dt_action_value = time.monotonic() - t0
            print(f"  action-value-label: done in {dt_action_value:.0f}s total", flush=True)
            if action_value_failed:
                raise SystemExit("action-value-label failed; halting stage1_driver")

        # 4. Summary
        cycle_wall = dt_d + dt_phase23 + dt_label + dt_policy_label + dt_hard_mine + dt_action_value
        summary = {
            "cycle": cycle,
            "tag": cycle_tag,
            "timestamp": datetime.now().isoformat(),
            "mode": "parallel" if run_parallel else "serial",
            "distill_seconds": dt_d,
            "vspika_seconds": dt_v,  # 0.0 in parallel mode — see phase23_seconds
            "train_seconds": dt_t,   # 0.0 in parallel mode — see phase23_seconds
            "phase23_seconds": dt_phase23,
            "cycle_wall_seconds": cycle_wall,
            "from_step": current_step,
            "to_step": target_step,
            "distill_depth": int(args.distill_depth),
            "distill_workers": int(args.distill_workers),
            "distill_threads_per_worker": int(args.distill_threads_per_worker),
            "distill_hash_mb": int(args.distill_hash_mb),
            "train_bootstrap_human_floor": args.train_bootstrap_human_floor,
            "vspika_profiles": [_profile_payload(profile) for profile in vspika_profiles],
            "vspika_results": vspika_results,
            "selfplay_profiles": [_profile_payload(profile) for profile in selfplay_profiles],
            "selfplay_results": selfplay_results,
            "selfplay_opp_checkpoint": str(args.selfplay_opp_checkpoint) if args.selfplay_opp_checkpoint else None,
            "oracle_label_seconds": dt_label,
            "oracle_label_results": oracle_label_results,
            "oracle_depth": int(args.oracle_depth) if args.oracle_label else None,
            "policy_oracle_label_seconds": dt_policy_label,
            "policy_oracle_label_results": policy_label_results,
            "policy_oracle_depth": int(args.policy_oracle_depth) if args.policy_oracle_label else None,
            "policy_oracle_multipv": int(args.policy_oracle_multipv) if args.policy_oracle_label else None,
            "policy_oracle_alpha": float(args.policy_oracle_alpha),
            "hard_mining_seconds": dt_hard_mine,
            "hard_mining_results": hard_mine_results,
            "hard_mining_top_percent": float(args.hard_mining_top_percent) if args.hard_mining else None,
            "hard_mining_heavy_weight": float(args.hard_mining_heavy_weight) if args.hard_mining else None,
            "action_value_label_seconds": dt_action_value,
            "action_value_label_results": action_value_results,
            "action_value_depth": int(args.action_value_depth) if args.action_value_label else None,
            "teacher_q_loss_weight": float(args.teacher_q_loss_weight),
        }
        with driver_log.open("a") as f:
            f.write(json.dumps(summary) + "\n")
        print(f"  cycle {cycle} complete in {cycle_wall:.0f}s", flush=True)

        if args.halt_human_val_threshold > 0:
            last_human_val = _read_last_human_val_total_loss(training_dir)
            if last_human_val is None:
                print("  human-val guard: no last_human_val_total_loss yet", flush=True)
            elif last_human_val > args.halt_human_val_threshold:
                bad_human_val_streak += 1
                print(
                    f"  human-val guard: {last_human_val:.4f} > "
                    f"{args.halt_human_val_threshold:.4f} "
                    f"({bad_human_val_streak}/{args.halt_human_val_patience})",
                    flush=True,
                )
                if bad_human_val_streak >= args.halt_human_val_patience:
                    print(
                        f"\n  HUMAN-VAL TRIGGERED HALT at cycle {cycle}: "
                        f"last_human_val_total_loss={last_human_val:.4f}",
                        flush=True,
                    )
                    return 3
            else:
                if bad_human_val_streak:
                    print("  human-val guard: recovered; streak reset", flush=True)
                bad_human_val_streak = 0

        # 5. Sanity probe (every N cycles) — halt on pessimism collapse.
        if args.sanity_probe_every > 0 and cycle % args.sanity_probe_every == 0:
            probe_device = args.sanity_probe_device or train_device
            if sanity_ladder_profiles:
                print(
                    f"\n  === SANITY LADDER after cycle {cycle} "
                    f"({len(sanity_ladder_profiles)} profile(s), device={probe_device}) ===",
                    flush=True,
                )
                ladder = _run_sanity_ladder(
                    cycle=cycle,
                    py=py,
                    repo=repo,
                    training_dir=training_dir,
                    stage_log_dir=stage_log_dir,
                    device=probe_device,
                    profiles=sanity_ladder_profiles,
                    timeout_s=args.sanity_probe_timeout_s,
                    seed_base=args.seed,
                )
                for probe in ladder:
                    if "error" in probe:
                        print(
                            f"  ladder WARN[{probe.get('profile', '?')}]: {probe['error']}",
                            flush=True,
                        )
                        continue
                    print(
                        f"  ladder[{probe['profile']}]: "
                        f"{probe['wins']}W-{probe['losses']}L-{probe['draws']}D "
                        f"score={probe['score_rate']*100:.1f}% "
                        f"avg_plies={probe['avg_plies']:.1f}",
                        flush=True,
                    )
            else:
                print(
                    f"\n  === SANITY PROBE after cycle {cycle} "
                    f"({args.sanity_probe_games} games vs Pikafish d={args.sanity_probe_opp_depth}, "
                    f"our sims={args.sanity_probe_our_sims}, device={probe_device}) ===",
                    flush=True,
                )
                probe = _run_sanity_probe(
                    cycle=cycle,
                    py=py,
                    repo=repo,
                    training_dir=training_dir,
                    stage_log_dir=stage_log_dir,
                    device=probe_device,
                    games=args.sanity_probe_games,
                    opp_depth=args.sanity_probe_opp_depth,
                    our_sims=args.sanity_probe_our_sims,
                    opp_noise_ratio=args.sanity_probe_opp_noise_ratio,
                    timeout_s=args.sanity_probe_timeout_s,
                    seed_base=args.seed,
                )
                if "error" in probe:
                    # Probe itself failed; log but don't halt the driver (could be
                    # transient: GPU hiccup, Pikafish startup issue, etc).
                    print(f"  probe WARN: {probe['error']} -- continuing without halt",
                          flush=True)
                else:
                    wr = probe["winrate"]
                    elo = probe.get("elo_estimate")
                    elo_str = f"Elo~{elo:.0f}" if isinstance(elo, (int, float)) else f"Elo=?"
                    print(
                        f"  probe RESULT: {probe['wins']}W-{probe['losses']}L-{probe['draws']}D "
                        f"= {wr*100:.1f}% winrate  {elo_str}  "
                        f"({probe['duration_s']:.0f}s)",
                        flush=True,
                    )
                    if wr < args.sanity_probe_min_winrate:
                        print(
                            f"\n  SANITY PROBE TRIGGERED HALT at cycle {cycle}:\n"
                            f"    winrate {wr*100:.1f}% < threshold "
                            f"{args.sanity_probe_min_winrate*100:.1f}%\n"
                            f"    checkpoint step: {probe['checkpoint_step']}\n"
                            f"    the training curriculum may be too aggressive for the "
                            f"current model strength.\n"
                            f"    review stage2_sanity_probes/cycle_{cycle:03d}/ and "
                            f"stage1_logs/sanity_probe.jsonl",
                            flush=True,
                        )
                        return 2  # distinctive exit code for "halted by sanity probe"


def _get_current_step(checkpoint_path: Path) -> int:
    if not checkpoint_path.is_file():
        return 0
    # Cheap: just torch.load with weights_only=False and read global_step; defer the import
    import torch
    try:
        s = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return int(s.get("global_step", 0))
    except Exception:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
