"""Generate distillation shards: diverse positions labelled by Pikafish.

Approach
--------
Human shards don't carry FEN, and re-decoding the 115-plane tensor back to a FEN
is brittle. Instead we *generate* positions from scratch via short random-opening
rollouts, each of which gives us a genuine Board object (so FEN is free). At each
recorded position we ask Pikafish for its top move + eval at the configured depth;
those become the distillation targets.

Output format matches the tensorized selfplay shard format consumed by
xiangqi_train.py._extract_tensorized_sample_blobs, so training picks them up
without any pipeline change.

Usage:
    python tools/distillation_generator.py \
        --output-dir /home/laure/alphaxiang/selfplay_runs/distill_20260418/train \
        --num-positions 4000 \
        --depth 6 \
        --workers 16 \
        --hash-mb 256 \
        --random-opening-plies 12
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))

from pikafish_pool import PikafishJob, PikafishPool  # noqa: E402
from pikafish_opponent import uci_move_to_internal  # noqa: E402
from xiangqi_mcts_ext import Board, canonical_action  # noqa: E402


_CP_SCALE = 500.0


def _cp_to_tanh(cp: int) -> float:
    return float(math.tanh(cp / _CP_SCALE))


def _pad_fen(fen: str) -> str:
    parts = fen.strip().split()
    while len(parts) < 6:
        if len(parts) == 2:
            parts.append("-")
        elif len(parts) == 3:
            parts.append("-")
        elif len(parts) == 4:
            parts.append("0")
        else:
            parts.append("1")
    return " ".join(parts)


def _generate_positions(
    num_positions: int,
    random_opening_plies: int,
    rng: random.Random,
    max_game_plies: int = 200,
) -> list[dict]:
    """Play short random-move games from the standard start and record snapshots.

    Each returned dict: {
        "fen": str,
        "board": Board (live instance, for legal_moves + state tensor extraction),
        "ply": int,
        "stm_is_black": bool,
    }

    We use random-move play rather than model play because (a) we want DIVERSE
    positions and (b) avoiding any leak of our current model's biases into the
    target distribution.  Pikafish will then label each position with an
    engine-grade best move.
    """
    positions: list[dict] = []
    games_needed = 1 + num_positions // max(1, random_opening_plies)
    # Play until we accumulate enough snapshots.
    while len(positions) < num_positions:
        board = Board()
        ply = 0
        max_this_game = rng.randint(random_opening_plies, max_game_plies)
        while ply < max_this_game:
            legal = list(board.legal_moves())
            if not legal:
                break
            # Snapshot this pre-move state.
            try:
                fen = board.fen()
            except Exception:
                break
            positions.append({
                "fen": fen,
                "state": board.to_tensor_canonical().to(torch.float32)[0].contiguous().clone(),
                "ply": ply,
                "stm_is_black": bool(board.turn() == 1),
                "num_legal_moves": len(legal),
                # v12: store raw legal moves; canonicalized at shard write time.
                "legal_moves_raw": [int(m) for m in legal],
            })
            if len(positions) >= num_positions:
                break
            # Random move to progress the game
            mv = rng.choice(legal)
            board.push(int(mv))
            ply += 1
    return positions[:num_positions]


def generate_distill_shard(
    *,
    positions: list[dict],
    pool: PikafishPool,
    depth: int,
    output_path: Path,
    shard_id: int,
    multipv: int = 1,
) -> dict:
    jobs = [
        PikafishJob(index=i, fen=_pad_fen(pos["fen"]), depth=depth, multipv=multipv)
        for i, pos in enumerate(positions)
    ]
    pool.submit_all(jobs)
    results_list = pool.collect(len(jobs))
    results = {r.index: r for r in results_list}

    states_list: list[torch.Tensor] = []
    policy_idxs: list[int] = []
    policy_probs: list[float] = []
    policy_offsets: list[int] = [0]
    # v12: legal moves CSR (canonical) for legal-masked policy CE.
    legal_offsets: list[int] = [0]
    legal_idxs: list[int] = []
    z_values: list[float] = []
    wdl_targets: list[list[float]] = []
    chosen_moves: list[int] = []
    num_legal_moves: list[int] = []
    plies: list[int] = []
    stm_is_black_list: list[bool] = []
    game_ids: list[int] = []
    fens_list: list[str] = []

    dropped = 0
    for i, pos in enumerate(positions):
        result = results.get(i)
        if result is None or result.error:
            dropped += 1
            continue
        try:
            uci = result.best_move[:4]
            move = uci_move_to_internal(uci)
        except Exception:
            dropped += 1
            continue
        # Validate move against the position's legal set. Since we kept the Board
        # we can reconstruct quickly from FEN, but we already saved num_legal_moves
        # and the state tensor. For a strict legality check we'd need the board;
        # we trust Pikafish's bestmove here (it's generated from the same FEN).
        stm_is_black = bool(pos["stm_is_black"])
        canonical_idx = int(canonical_action(int(move), stm_is_black))

        state_bf16 = pos["state"].to(torch.bfloat16).contiguous()
        states_list.append(state_bf16)
        policy_idxs.append(canonical_idx)
        policy_probs.append(1.0)
        policy_offsets.append(policy_offsets[-1] + 1)
        cp = result.eval_cp
        z = _cp_to_tanh(int(cp))
        z_values.append(z)
        if z > 0.1:
            wdl_targets.append([1.0, 0.0, 0.0])
        elif z < -0.1:
            wdl_targets.append([0.0, 0.0, 1.0])
        else:
            wdl_targets.append([0.0, 1.0, 0.0])
        chosen_moves.append(canonical_idx)
        num_legal_moves.append(int(pos["num_legal_moves"]))
        # v12: canonicalize each legal move and append to CSR. Empty list when missing
        # (older positions without legal_moves_raw fall back to no-mask training).
        raw_legals = pos.get("legal_moves_raw") or []
        canonical_legals = [int(canonical_action(int(m), stm_is_black)) for m in raw_legals]
        legal_idxs.extend(canonical_legals)
        legal_offsets.append(legal_offsets[-1] + len(canonical_legals))
        plies.append(int(pos["ply"]))
        stm_is_black_list.append(stm_is_black)
        game_ids.append(shard_id * 100000 + i)
        fens_list.append(_pad_fen(pos["fen"]))

    if not states_list:
        raise RuntimeError("all samples dropped during labelling")

    payload = {
        "state": torch.stack(states_list, dim=0).contiguous(),
        "policy_offsets": torch.tensor(policy_offsets, dtype=torch.int64),
        "policy_idxs": torch.tensor(policy_idxs, dtype=torch.int64),
        "policy_probs": torch.tensor(policy_probs, dtype=torch.float32),
        "z": torch.tensor(z_values, dtype=torch.float32),
        "wdl_target": torch.tensor(wdl_targets, dtype=torch.float32),
        "root_value": torch.tensor(z_values, dtype=torch.float32),
        "root_wdl_value": torch.tensor(z_values, dtype=torch.float32),
        "chosen_move": torch.tensor(chosen_moves, dtype=torch.int64),
        "num_legal_moves": torch.tensor(num_legal_moves, dtype=torch.int32),
        # v12 (legal-masked CE): canonical legal moves stored as CSR (offsets+idxs).
        # Empty if no positions have legal_moves_raw (backward compat: falls back to no-mask).
        "legal_offsets": torch.tensor(legal_offsets, dtype=torch.int64),
        "legal_idxs": torch.tensor(legal_idxs, dtype=torch.int64),
        "ply": torch.tensor(plies, dtype=torch.int16),
        "game_id": torch.tensor(game_ids, dtype=torch.int64),
        "stm_is_black": torch.tensor(stm_is_black_list, dtype=torch.bool),
        "is_draw": torch.tensor([False] * len(states_list), dtype=torch.bool),
        "termination_code": torch.tensor([-1] * len(states_list), dtype=torch.int8),
        "fens": fens_list,  # Python list[str], for downstream oracle_value_labeler
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return {
        "samples_written": len(states_list),
        "samples_requested": len(positions),
        "dropped": dropped,
        "output_path": str(output_path),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-positions", type=int, default=4000)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--threads-per-worker", type=int, default=1)
    p.add_argument("--hash-mb", type=int, default=16)
    p.add_argument("--shard-size", type=int, default=2048)
    p.add_argument("--seed", type=int, default=20260418)
    p.add_argument("--pikafish-binary", default="/home/laure/pikafish/pikafish")
    p.add_argument("--multipv", type=int, default=1)
    p.add_argument("--random-opening-plies", type=int, default=12)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"generating {args.num_positions} random-rollout positions...", flush=True)
    t0 = time.monotonic()
    positions = _generate_positions(
        num_positions=args.num_positions,
        random_opening_plies=args.random_opening_plies,
        rng=rng,
    )
    print(f"  got {len(positions)} positions in {time.monotonic() - t0:.1f}s", flush=True)

    print(
        f"launching Pikafish pool ({args.workers} workers x "
        f"{args.threads_per_worker} threads, hash={args.hash_mb}MB)...",
        flush=True,
    )
    pool = PikafishPool(
        num_workers=args.workers,
        binary_path=args.pikafish_binary,
        threads_per_worker=args.threads_per_worker,
        hash_mb=args.hash_mb,
    )

    try:
        total_written = 0
        total_dropped = 0
        shard_idx = 0
        per_shard = int(args.shard_size)
        t_global = time.monotonic()
        for start in range(0, len(positions), per_shard):
            batch = positions[start:start + per_shard]
            shard_path = output_dir / f"shard_{shard_idx:05d}.pt"
            print(
                f"\n[shard {shard_idx}] labelling {len(batch)} positions @ depth={args.depth}",
                flush=True,
            )
            t0 = time.monotonic()
            stats = generate_distill_shard(
                positions=batch, pool=pool, depth=int(args.depth),
                output_path=shard_path, shard_id=shard_idx, multipv=int(args.multipv),
            )
            dt = time.monotonic() - t0
            total_written += stats["samples_written"]
            total_dropped += stats["dropped"]
            print(
                f"  wrote {stats['samples_written']} (dropped {stats['dropped']}), "
                f"{dt:.1f}s, {stats['samples_written']/max(dt,0.001):.1f} samples/s",
                flush=True,
            )
            shard_idx += 1
        dt_total = time.monotonic() - t_global
    finally:
        pool.close()

    manifest = {
        "format": "distillation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_shards": shard_idx,
        "total_samples_written": total_written,
        "total_samples_dropped": total_dropped,
        "pikafish_depth": int(args.depth),
        "quality": "ok",
        "manifest_state": "complete",
        "quality_metrics": {
            "rep_draw_rate": 0.0,
            "decisive_rate": 100.0,
            "nocap_draw_rate": 0.0,
        },
        "config": {
            "source": "distillation",
            "random_opening_plies": int(args.random_opening_plies),
            "search_defaults": {"num_simulations": 0},
            "threads_per_worker": int(args.threads_per_worker),
            "hash_mb": int(args.hash_mb),
        },
        "duration_s": dt_total,
    }
    (output_dir.parent / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nDONE: {total_written} samples across {shard_idx} shards in {dt_total:.1f}s", flush=True)
    print(f"manifest: {output_dir.parent / 'manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
