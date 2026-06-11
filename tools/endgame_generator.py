"""Generate ENDGAME-rich distillation shards to fix the diagnosed value-head endgame gap.

Why: the random-rollout distillation_generator almost never reaches endgames (median ~29
pieces, only ~2% <=16 pieces) because random play has no drive to trade down. Real games
simplify into endgames. So here we play STRONG (Pikafish d6) self-play from random openings
to game end, and KEEP only low-piece (endgame) positions. Output shards match the
distillation_generator format (reuses generate_distill_shard) so oracle_value_labeler.py /
oracle_policy_labeler.py (d20) and xiangqi_train.py ingest them unchanged.

Usage:
    python tools/endgame_generator.py --output-dir <dir>/train --num-positions 4000 \
        --depth 6 --workers 24 --keep-max-pieces 18
"""
from __future__ import annotations
import argparse, json, random, sys, time
from datetime import datetime, timezone
from pathlib import Path
import torch

_REPO = Path(__file__).resolve().parent.parent
for p in (str(_REPO), str(_REPO / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from pikafish_pool import PikafishJob, PikafishPool          # noqa: E402
from pikafish_opponent import uci_move_to_internal           # noqa: E402
from xiangqi_mcts_ext import Board                            # noqa: E402
from distillation_generator import generate_distill_shard, _pad_fen  # noqa: E402


def _count_pieces(fen: str) -> int:
    return sum(1 for c in fen.split()[0] if c.isalpha())


def _rec_prob(pcs: int, base: float) -> float:
    # Depth-weighted recording: oversample DEEP endgames so the distribution matches the
    # real failure region (d3 endgames: median ~13). Shallow 16-18-piece positions (every
    # game passes through them) get undersampled; deep <=12-piece positions get oversampled.
    if pcs <= 12:
        return min(1.0, base * 2.4)
    if pcs <= 15:
        return base
    return base * 0.28


def _snapshot(board: Board, ply: int) -> dict:
    legal = list(board.legal_moves())
    return {
        "fen": board.fen(),
        "state": board.to_tensor_canonical().to(torch.float32)[0].contiguous().clone(),
        "ply": int(ply),
        "stm_is_black": bool(board.turn() == 1),
        "num_legal_moves": len(legal),
        "legal_moves_raw": [int(m) for m in legal],
    }


def _fresh_game(rng: random.Random, open_min: int, open_max: int):
    """Standard start + a few random opening plies (diversity). Returns (board, ply) or (None, 0)."""
    b = Board()
    r = rng.randint(open_min, open_max)
    for _ in range(r):
        legal = list(b.legal_moves())
        if not legal:
            return None, 0
        b.push(int(rng.choice(legal)))
    return b, r


def generate_endgame_positions(
    *, num_positions: int, pool: PikafishPool, depth: int, rng: random.Random,
    keep_max_pieces: int, games_parallel: int, open_min: int, open_max: int,
    max_plies: int, record_prob: float, per_game_cap: int,
) -> list[dict]:
    boards: list = []
    plies: list[int] = []
    egcount: list[int] = []
    for _ in range(games_parallel):
        b, r = _fresh_game(rng, open_min, open_max)
        boards.append(b); plies.append(r); egcount.append(0)

    out: list[dict] = []
    rounds = 0
    # total-round safety cap (independent of per-game max_plies, which restarts a single game).
    # BUG FIX: previously used `rounds < max_plies` which capped TOTAL generation at ~300 rounds
    # (=> only ~1469 positions regardless of --num-positions). This must scale with the target.
    safety_rounds = max(20000, num_positions * 30)
    while len(out) < num_positions and rounds < safety_rounds:
        rounds += 1
        active = [gi for gi, b in enumerate(boards) if b is not None]
        if not active:
            break
        jobs = []
        for gi in active:
            b = boards[gi]
            try:
                fen = b.fen()
            except Exception:
                boards[gi] = None; continue
            # record this position if it's an endgame (depth-weighted subsample, capped per game)
            pcs = _count_pieces(fen)
            if (pcs <= keep_max_pieces and egcount[gi] < per_game_cap
                    and rng.random() < _rec_prob(pcs, record_prob)):
                out.append(_snapshot(b, plies[gi])); egcount[gi] += 1
                if len(out) >= num_positions:
                    break
            jobs.append(PikafishJob(index=gi, fen=_pad_fen(fen), depth=depth))
        if len(out) >= num_positions or not jobs:
            break
        pool.submit_all(jobs)
        res = {r.index: r for r in pool.collect(len(jobs))}
        for gi in active:
            if boards[gi] is None:
                continue
            r = res.get(gi)
            if r is None or r.error:
                boards[gi], plies[gi], egcount[gi] = _fresh_game(rng, open_min, open_max) + (0,); continue
            try:
                mv = uci_move_to_internal(r.best_move[:4])
            except Exception:
                boards[gi], plies[gi], egcount[gi] = _fresh_game(rng, open_min, open_max) + (0,); continue
            b = boards[gi]
            if not b.is_legal(mv):
                boards[gi], plies[gi], egcount[gi] = _fresh_game(rng, open_min, open_max) + (0,); continue
            b.push_legal(mv); plies[gi] += 1
            if not list(b.legal_moves()) or plies[gi] >= max_plies:
                # game over (mate/stalemate) or too long -> restart this slot
                nb, nr = _fresh_game(rng, open_min, open_max)
                boards[gi], plies[gi], egcount[gi] = nb, nr, 0
    return out[:num_positions]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-positions", type=int, default=4000)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--threads-per-worker", type=int, default=1)
    p.add_argument("--hash-mb", type=int, default=32)
    p.add_argument("--shard-size", type=int, default=2048)
    p.add_argument("--seed", type=int, default=70260530)
    p.add_argument("--keep-max-pieces", type=int, default=18)
    p.add_argument("--open-min", type=int, default=10)
    p.add_argument("--open-max", type=int, default=24)
    p.add_argument("--max-plies", type=int, default=300)
    p.add_argument("--games-parallel", type=int, default=0, help="0 => 2x workers")
    p.add_argument("--record-prob", type=float, default=0.45)
    p.add_argument("--per-game-cap", type=int, default=40)
    args = p.parse_args()

    rng = random.Random(args.seed)
    gpar = args.games_parallel if args.games_parallel > 0 else max(8, 2 * args.workers)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"endgame-gen: target={args.num_positions} keep<= {args.keep_max_pieces} pieces, "
          f"d{args.depth} self-play, {gpar} parallel games, {args.workers} workers", flush=True)

    pool = PikafishPool(num_workers=args.workers, threads_per_worker=args.threads_per_worker,
                        hash_mb=args.hash_mb)
    t0 = time.monotonic()
    try:
        positions = generate_endgame_positions(
            num_positions=args.num_positions, pool=pool, depth=args.depth, rng=rng,
            keep_max_pieces=args.keep_max_pieces, games_parallel=gpar,
            open_min=args.open_min, open_max=args.open_max, max_plies=args.max_plies,
            record_prob=args.record_prob, per_game_cap=args.per_game_cap,
        )
        pcs = sorted(_count_pieces(p["fen"]) for p in positions)
        n = len(pcs)
        if n:
            print(f"generated {n} endgame positions in {time.monotonic()-t0:.0f}s | "
                  f"pieces min={pcs[0]} median={pcs[n//2]} max={pcs[-1]} "
                  f"<=16={sum(1 for x in pcs if x<=16)/n:.0%} <=10={sum(1 for x in pcs if x<=10)/n:.0%}", flush=True)
        # write shards (reuse distill format; this also labels with depth-d best move/eval)
        total = 0; shard_idx = 0
        for i in range(0, len(positions), args.shard_size):
            chunk = positions[i:i + args.shard_size]
            stats = generate_distill_shard(
                positions=chunk, pool=pool, depth=args.depth,
                output_path=out_dir / f"shard_{shard_idx:05d}.pt", shard_id=shard_idx, multipv=1)
            total += stats["samples_written"]; shard_idx += 1
            print(f"  shard {shard_idx}: wrote {stats['samples_written']}", flush=True)
    finally:
        pool.close()

    manifest = {
        "kind": "endgame_distill", "depth": args.depth, "keep_max_pieces": args.keep_max_pieces,
        "total_samples_written": total, "num_shards": shard_idx,
        "created": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir.parent / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"DONE: {total} endgame samples in {shard_idx} shards -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
