"""vs-Pikafish selfplay: our model plays against Pikafish (with optional random noise).

This generates training shards where OUR moves have MCTS-visit-based policy targets
and the game outcome provides the z value.  Pikafish's moves are just opponent
responses (not training samples).

The output shards are drop-in compatible with xiangqi_train's selfplay shard format,
so the existing training pipeline consumes them unchanged.

Curriculum
----------
The "noise_ratio" parameter lets us weaken Pikafish by replacing its chosen move
with a random legal move with probability p.  This is how we build Tier 1 (p=0.15)
at the start of the curriculum.  As our model strengthens, run with p=0.0 and raise
--opp-depth / --opp-nodes.

Usage:
    python tools/pikafish_selfplay.py \
        --checkpoint /home/laure/alphaxiang/training_runs/run_002_pikafish_curriculum/latest.pt \
        --output-dir /home/laure/alphaxiang/selfplay_runs/vspika_tier1_20260418/train \
        --num-games 30 \
        --opp-depth 5 \
        --noise-ratio 0.15 \
        --our-sims 256
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))

from cross_game_batcher import CrossGameBatcher  # noqa: E402
from model_opponent import ModelOpponent  # noqa: E402
from pikafish_opponent import (  # noqa: E402
    PikafishOpponent,
    internal_move_to_uci,
    uci_move_to_internal,
)
from xiangqi_mcts_ext import Board, canonical_action, make_gpu_evaluator, mcts_search  # noqa: E402
from xiangqi_transformer_model import build_model_from_checkpoint_state  # noqa: E402


TERMINAL_ONGOING = -1


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


def _load_model(checkpoint: Path, device: torch.device, use_bf16: bool):
    """Load model once.  Returns (model, global_step).  Caller builds per-thread
    evaluators via _make_thread_evaluator() so concurrent games don't share any
    evaluator-internal state (batch queues etc.)."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model_from_checkpoint_state(state)
    model.to(device)
    model.eval()
    return model, int(state.get("global_step", 0))


def _make_thread_evaluator(model, device: torch.device, use_bf16: bool):
    """Create a fresh evaluator wrapping the shared model weights.  Each
    worker thread owns one of these so their MCTS searches don't contend on
    the same C++ evaluator's internal batch buffer."""
    return make_gpu_evaluator(model, device=str(device), use_bfloat16=use_bf16)


def play_one_game(
    *,
    evaluator,
    pf: PikafishOpponent,
    our_is_red: bool,
    our_sims: int,
    our_c_puct: float,
    our_temp_move: float,
    our_add_root_noise: bool,
    our_dirichlet_alpha: float,
    our_dirichlet_eps: float,
    opp_depth: int,
    opp_nodes: int,
    noise_ratio: float,
    seed_base: int,
    max_plies: int,
    repeat_limit: int,
    repeat_min_ply: int,
    no_capture_limit: int,
    rng: random.Random,
    opp_model: ModelOpponent | None = None,
) -> dict:
    """Play one game and return a dict of training samples + metadata.

    If ``opp_model`` is provided, opponent moves are computed by running MCTS
    on the snapshot model (self-play mode).  Otherwise the legacy Pikafish
    subprocess (``pf``) is used.
    """
    board = Board()
    moves_uci: list[str] = []

    our_samples: list[dict] = []
    ply = 0
    game_termination: int | None = None
    start_fen = _pad_fen(board.fen())

    if pf is not None:
        pf.new_game()

    while True:
        term = int(board.terminal_code(
            max_plies, repeat_limit, repeat_min_ply, no_capture_limit,
        ))
        if term != TERMINAL_ONGOING:
            game_termination = term
            break

        red_to_move = (int(board.turn()) == 0)
        our_turn = (red_to_move == our_is_red)

        if our_turn:
            # Our move: MCTS search, record as training sample.
            state_cpu = board.to_tensor_canonical().to(torch.float32).contiguous()
            best_move, policy_idxs, policy_probs, root_value = mcts_search(
                board=board,
                net=evaluator,
                num_simulations=int(our_sims),
                c_puct=float(our_c_puct),
                q_weight=1.0,
                q_clip=1.0,
                add_root_noise=bool(our_add_root_noise),
                dirichlet_alpha=float(our_dirichlet_alpha),
                dirichlet_eps=float(our_dirichlet_eps),
                temperature_move=float(our_temp_move),
                temperature_target=1.0,
                eval_batch_size=16,
                seed=int(seed_base + ply * 31),
                canonical_input=True,
                canonical_policy=True,
                max_plies=max_plies,
                repeat_limit=repeat_limit,
                repeat_min_ply=repeat_min_ply,
                no_capture_limit=no_capture_limit,
            )
            if int(best_move) < 0:
                # No legal move → game end handled next iter
                break

            stm_is_black = bool(board.turn() == 1)
            chosen_canonical = int(canonical_action(int(best_move), stm_is_black))
            # Capture pre-move FEN so a post-process tool (oracle_value_labeler) can
            # query a strong Pikafish (d=15+) to provide a calibrated value target,
            # replacing the noisy game-outcome z. See PROJECT_FINAL_RESULTS Lemma 4
            # (OOD over-search trap caused by miscalibrated value head).
            sample_fen = _pad_fen(board.fen())
            # v12: capture full legal-move set (canonicalized) for legal-masked policy CE.
            # MCTS only visits a subset of legal moves at low sims; the model still emits
            # logits over all 8100 actions, so masking softmax to legal-only is a real fix.
            our_legal_raw = list(board.legal_moves())
            our_legal_canonical = [int(canonical_action(int(m), stm_is_black)) for m in our_legal_raw]
            our_samples.append({
                "state": state_cpu[0].to(torch.bfloat16).contiguous().clone(),
                "policy_idxs": policy_idxs.to(torch.int64).contiguous().clone(),
                "policy_probs": policy_probs.to(torch.float32).contiguous().clone(),
                "root_value": float(root_value),
                "chosen_move": chosen_canonical,
                "num_legal_moves": int(policy_idxs.numel()),
                "ply": ply,
                "stm_is_black": stm_is_black,
                "fen": sample_fen,
                "legal_idxs_canonical": torch.tensor(our_legal_canonical, dtype=torch.int64).contiguous(),
            })
            uci = internal_move_to_uci(int(best_move))
            moves_uci.append(uci)
            board.push(int(best_move))
        else:
            # Opponent move.  Three paths:
            #   1. Random move (with probability `noise_ratio`) — same in both modes
            #   2. ModelOpponent search — self-play snapshot picks the move
            #   3. Pikafish UCI subprocess — the legacy path
            legal = list(board.legal_moves())
            if not legal:
                break
            if rng.random() < noise_ratio:
                mv = int(rng.choice(legal))
                uci = internal_move_to_uci(mv)
                moves_uci.append(uci)
                board.push(mv)
            elif opp_model is not None:
                # Self-play: snapshot model runs its own MCTS on the live board.
                # Different seed offset (+777_777) so opp's MCTS noise is independent of ours.
                try:
                    mv = opp_model.search(board, seed=int(seed_base + ply * 31 + 777_777))
                except Exception:
                    mv = int(rng.choice(legal))
                if int(mv) < 0 or int(mv) not in [int(x) for x in legal]:
                    mv = int(rng.choice(legal))
                uci = internal_move_to_uci(int(mv))
                moves_uci.append(uci)
                board.push(int(mv))
            else:
                assert pf is not None, "Pikafish subprocess required when opp_model not given"
                pf.set_position(start_fen, moves=moves_uci)
                if opp_depth > 0:
                    best_uci, _ = pf.go_depth(int(opp_depth), max_wait_s=600.0)
                elif opp_nodes > 0:
                    best_uci, _ = pf.go_nodes(int(opp_nodes), max_wait_s=60.0)
                else:
                    best_uci, _ = pf.go_depth(1)
                uci = best_uci[:4]
                try:
                    mv = uci_move_to_internal(uci)
                except Exception:
                    # Pikafish returned something we can't parse; fall back to a legal move.
                    mv = int(rng.choice(legal))
                    uci = internal_move_to_uci(mv)
                moves_uci.append(uci)
                board.push(mv)
        ply += 1

    if game_termination is None:
        game_termination = int(board.terminal_code(
            max_plies, repeat_limit, repeat_min_ply, no_capture_limit,
        ))
    final_red_result = int(board.terminal_result_red_view(game_termination))

    return {
        "samples": our_samples,
        "our_is_red": our_is_red,
        "final_red_result": final_red_result,
        "termination_code": int(game_termination),
        "plies": ply,
    }


def _game_outcome_for_our_side(our_is_red: bool, final_red_result: int) -> str:
    if final_red_result == 0:
        return "draw"
    if (final_red_result > 0) == our_is_red:
        return "our_win"
    return "opp_win"


def _z_for_sample(stm_is_black: bool, final_red_result: int) -> float:
    """z is from side-to-move perspective.  If red wins and sample was red's move: +1."""
    if final_red_result == 0:
        return 0.0
    red_wins = (final_red_result > 0)
    sample_is_red = not stm_is_black
    return 1.0 if (red_wins == sample_is_red) else -1.0


def _flush_shard(samples: list[dict], game_results: list[dict], output_dir: Path,
                 shard_id: int) -> dict:
    """Write one shard containing the given per-move samples, with z values resolved
    from the parent game's outcome (looked up by game_id)."""
    states_list: list[torch.Tensor] = []
    policy_idxs_chunks: list[torch.Tensor] = []
    policy_probs_chunks: list[torch.Tensor] = []
    policy_offsets: list[int] = [0]
    # v12: legal moves CSR for legal-masked policy CE.
    legal_idxs_chunks: list[torch.Tensor] = []
    legal_offsets: list[int] = [0]
    z_values: list[float] = []
    wdl_targets: list[list[float]] = []
    root_values: list[float] = []
    root_wdl_values: list[float] = []
    chosen_moves: list[int] = []
    num_legal_moves: list[int] = []
    plies: list[int] = []
    stm_is_black_list: list[bool] = []
    game_ids: list[int] = []
    is_draw_list: list[bool] = []
    termination_codes: list[int] = []
    fens_list: list[str] = []

    for samp in samples:
        gid = int(samp["game_id"])
        gr = game_results[gid]
        z = _z_for_sample(bool(samp["stm_is_black"]), int(gr["final_red_result"]))
        states_list.append(samp["state"])
        idxs = samp["policy_idxs"].to(torch.int64).contiguous()
        probs = samp["policy_probs"].to(torch.float32).contiguous()
        policy_idxs_chunks.append(idxs)
        policy_probs_chunks.append(probs)
        policy_offsets.append(policy_offsets[-1] + int(idxs.numel()))
        z_values.append(z)
        if z > 0.1:
            wdl_targets.append([1.0, 0.0, 0.0])
        elif z < -0.1:
            wdl_targets.append([0.0, 0.0, 1.0])
        else:
            wdl_targets.append([0.0, 1.0, 0.0])
        root_values.append(float(samp["root_value"]))
        root_wdl_values.append(float(samp["root_value"]))
        chosen_moves.append(int(samp["chosen_move"]))
        num_legal_moves.append(int(samp["num_legal_moves"]))
        # v12: legal moves CSR. Old samples without the field get an empty slice
        # (legal_offsets[i+1] == legal_offsets[i]) — trainer treats that as no-mask.
        leg = samp.get("legal_idxs_canonical")
        if leg is not None and leg.numel() > 0:
            legal_idxs_chunks.append(leg.to(torch.int64).contiguous())
            legal_offsets.append(legal_offsets[-1] + int(leg.numel()))
        else:
            legal_offsets.append(legal_offsets[-1])
        plies.append(int(samp["ply"]))
        stm_is_black_list.append(bool(samp["stm_is_black"]))
        game_ids.append(gid)
        is_draw_list.append(int(gr["final_red_result"]) == 0)
        termination_codes.append(int(gr["termination_code"]))
        # FEN may be missing in old samples produced before the schema bump; that's OK,
        # oracle_value_labeler will skip those positions and fall back to z-based loss.
        fens_list.append(str(samp.get("fen", "")))

    payload = {
        "state": torch.stack(states_list, dim=0).contiguous(),
        "policy_offsets": torch.tensor(policy_offsets, dtype=torch.int64),
        "policy_idxs": torch.cat(policy_idxs_chunks, dim=0).contiguous(),
        "policy_probs": torch.cat(policy_probs_chunks, dim=0).contiguous(),
        "z": torch.tensor(z_values, dtype=torch.float32),
        "wdl_target": torch.tensor(wdl_targets, dtype=torch.float32),
        "root_value": torch.tensor(root_values, dtype=torch.float32),
        "root_wdl_value": torch.tensor(root_wdl_values, dtype=torch.float32),
        "chosen_move": torch.tensor(chosen_moves, dtype=torch.int64),
        "num_legal_moves": torch.tensor(num_legal_moves, dtype=torch.int32),
        # v12 (legal-masked CE): canonical legal moves stored as CSR.
        "legal_offsets": torch.tensor(legal_offsets, dtype=torch.int64),
        "legal_idxs": (
            torch.cat(legal_idxs_chunks, dim=0).contiguous()
            if legal_idxs_chunks else torch.empty(0, dtype=torch.int64)
        ),
        "ply": torch.tensor(plies, dtype=torch.int16),
        "game_id": torch.tensor(game_ids, dtype=torch.int64),
        "stm_is_black": torch.tensor(stm_is_black_list, dtype=torch.bool),
        "is_draw": torch.tensor(is_draw_list, dtype=torch.bool),
        "termination_code": torch.tensor(termination_codes, dtype=torch.int8),
        "fens": fens_list,  # Python list[str], saved alongside tensors in .pt
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = output_dir / f"shard_{shard_id:05d}.pt"
    torch.save(payload, shard_path)
    return {"path": str(shard_path), "samples": len(states_list)}


def _run_single_game(
    *,
    gi: int,
    args: argparse.Namespace,
    model,
    device: torch.device,
    use_bf16: bool,
    shared_evaluator=None,
    shared_opp_model: ModelOpponent | None = None,
) -> dict:
    """Run one game in isolation — own RNG, own (optional) Pikafish subprocess.

    Runs inside a worker thread from the ThreadPoolExecutor in main().  The
    shared `model` is read-only (we call .eval() before launching workers) so
    multiple evaluators wrapping the same model don't race on weights.

    Modes:
      * ``shared_evaluator`` provided  → cross-game-batched OUR-side evaluator.
      * ``shared_opp_model`` provided  → self-play (snapshot model is opponent).
        No Pikafish subprocess is spawned in this mode.
      * Otherwise                      → legacy Pikafish opponent path (one
        Pikafish subprocess per worker thread).
    """
    if shared_evaluator is not None:
        evaluator = shared_evaluator
    else:
        evaluator = _make_thread_evaluator(model, device, use_bf16)

    pf: PikafishOpponent | None = None
    if shared_opp_model is None:
        pf = PikafishOpponent(binary_path=args.pikafish_binary, threads=1, hash_mb=16)

    per_game_rng = random.Random(int(args.seed) + gi * 10_007 + 7)
    our_is_red = (gi % 2 == 0)
    gi_seed = int(args.seed) + gi * 10_007
    try:
        game = play_one_game(
            evaluator=evaluator,
            pf=pf,
            our_is_red=our_is_red,
            our_sims=args.our_sims,
            our_c_puct=args.our_c_puct,
            our_temp_move=args.our_temperature_move,
            our_add_root_noise=args.our_add_root_noise,
            our_dirichlet_alpha=args.our_dirichlet_alpha,
            our_dirichlet_eps=args.our_dirichlet_eps,
            opp_depth=args.opp_depth,
            opp_nodes=args.opp_nodes,
            noise_ratio=args.noise_ratio,
            seed_base=gi_seed,
            max_plies=args.max_plies,
            repeat_limit=args.repeat_limit,
            repeat_min_ply=args.repeat_min_ply,
            no_capture_limit=args.no_capture_limit,
            rng=per_game_rng,
            opp_model=shared_opp_model,
        )
    finally:
        if pf is not None:
            try:
                pf.close()
            except Exception:
                pass
    return {"gi": gi, "our_is_red": our_is_red, "game": game}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-games", type=int, default=30)
    p.add_argument("--parallel-games", type=int, default=8,
                   help="Number of games to play concurrently in worker threads. "
                        "Each thread has its own Pikafish subprocess, evaluator and RNG. "
                        "Set to 1 to force serial execution (legacy behaviour).")
    p.add_argument("--cross-game-batching", action=argparse.BooleanOptionalAction, default=True,
                   help="Aggregate MCTS leaf evaluations across all parallel games "
                        "into single GPU forward passes (default ON). When parallel-games "
                        ">1 this gives substantially higher GPU utilization. "
                        "Pass --no-cross-game-batching to use the legacy per-thread "
                        "evaluator path.")
    p.add_argument("--cross-game-batch-cap", type=int, default=256,
                   help="Maximum batch size the cross-game batcher will assemble before "
                        "forcing a GPU forward pass. Default 256.")
    p.add_argument("--cross-game-coalesce-ms", type=float, default=2.0,
                   help="Time the batcher waits to accumulate more leaf requests before "
                        "running a partial batch. Higher = bigger batches but more latency "
                        "per MCTS step. Default 2.0 ms.")
    p.add_argument("--shard-size-samples", type=int, default=2048)
    p.add_argument("--seed", type=int, default=20260418)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--disable-bf16", action="store_true")

    p.add_argument("--pikafish-binary", default="/home/laure/pikafish/pikafish")
    p.add_argument("--opp-depth", type=int, default=0)
    p.add_argument("--opp-nodes", type=int, default=0)
    p.add_argument("--noise-ratio", type=float, default=0.15,
                   help="Probability to replace Pikafish's (or model opponent's) move with a random legal move")
    p.add_argument("--opp-type", default="pikafish", choices=["pikafish", "model"],
                   help="Opponent type. 'pikafish' (default, NNUE+alphabeta) launches one Pikafish "
                        "subprocess per worker thread.  'model' uses a frozen snapshot of our own "
                        "model (path: --opp-model-checkpoint) running its own MCTS.  No Pikafish "
                        "subprocess is started in 'model' mode.")
    p.add_argument("--opp-model-checkpoint", default=None,
                   help="Required when --opp-type=model. Path to the snapshot .pt to use as opponent.")
    p.add_argument("--opp-model-sims", type=int, default=400,
                   help="MCTS sims for the snapshot opponent (default 400 — half of typical "
                        "our_sims=800, gives the opponent meaningful play but lets us win sometimes).")

    p.add_argument("--our-sims", type=int, default=256)
    p.add_argument("--our-c-puct", type=float, default=1.25)
    p.add_argument("--our-temperature-move", type=float, default=0.7,
                   help=">0 for training diversity; >=1 for early plies ideal")
    p.add_argument("--our-add-root-noise", action="store_true", default=True)
    p.add_argument("--our-dirichlet-alpha", type=float, default=0.3)
    p.add_argument("--our-dirichlet-eps", type=float, default=0.25)

    p.add_argument("--max-plies", type=int, default=240)
    p.add_argument("--repeat-limit", type=int, default=6)
    p.add_argument("--repeat-min-ply", type=int, default=30)
    p.add_argument("--no-capture-limit", type=int, default=60)

    args = p.parse_args()
    # In Pikafish-opponent mode, exactly one of --opp-depth / --opp-nodes is required.
    # In ModelOpponent mode neither is consulted (snapshot uses --opp-model-sims).
    if args.opp_type == "pikafish":
        if bool(args.opp_depth) == bool(args.opp_nodes):
            p.error("--opp-type=pikafish requires exactly one of --opp-depth or --opp-nodes")
    else:
        if not args.opp_model_checkpoint:
            p.error("--opp-type=model requires --opp-model-checkpoint")
    if int(args.parallel_games) < 1:
        p.error("--parallel-games must be >= 1")
    return args


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model, model_step = _load_model(Path(args.checkpoint), device, use_bf16=not args.disable_bf16)
    use_bf16 = not args.disable_bf16
    parallel_games = min(int(args.parallel_games), int(args.num_games))
    print(f"loaded model from step {model_step}", flush=True)

    # If cross-game batching is enabled AND we have multiple parallel games,
    # build one shared batcher.  All worker threads call into it; the batcher's
    # internal worker thread coalesces leaves across all games into one big GPU
    # forward pass per call.  This makes the GPU actually busy instead of doing
    # 16 small per-thread forwards in sequence.
    shared_evaluator = None
    if bool(args.cross_game_batching) and parallel_games > 1:
        shared_evaluator = CrossGameBatcher(
            model=model,
            device=device,
            use_bfloat16=use_bf16,
            max_batch_size=int(args.cross_game_batch_cap),
            coalesce_timeout_ms=float(args.cross_game_coalesce_ms),
        )
        print(f"parallelism: {parallel_games} concurrent game threads "
              f"+ shared cross-game batcher (cap={args.cross_game_batch_cap}, "
              f"coalesce={args.cross_game_coalesce_ms:.1f}ms)", flush=True)
    else:
        print(f"parallelism: {parallel_games} concurrent game threads "
              f"(each with own per-thread evaluator)", flush=True)

    # Self-play opponent setup: one shared ModelOpponent across all worker threads.
    # We share rather than per-thread because PyTorch model.forward is thread-safe
    # in eval mode and per-thread copies would consume 16× VRAM on selfplay GPU.
    shared_opp_model: ModelOpponent | None = None
    if args.opp_type == "model":
        if not args.opp_model_checkpoint:
            raise SystemExit("--opp-type=model requires --opp-model-checkpoint")
        print(f"opponent: ModelOpponent (snapshot from {args.opp_model_checkpoint}, "
              f"sims={args.opp_model_sims})", flush=True)
        shared_opp_model = ModelOpponent(
            checkpoint_path=args.opp_model_checkpoint,
            device=device,
            sims=int(args.opp_model_sims),
            use_bfloat16=use_bf16,
            max_plies=int(args.max_plies),
            repeat_limit=int(args.repeat_limit),
            repeat_min_ply=int(args.repeat_min_ply),
            no_capture_limit=int(args.no_capture_limit),
        )
        print(f"  opponent step: {shared_opp_model.checkpoint_step}", flush=True)
    else:
        print(f"opponent: Pikafish "
              f"(depth={args.opp_depth} nodes={args.opp_nodes} noise={args.noise_ratio})",
              flush=True)

    # Shared state across worker threads, protected by output_lock.
    output_lock = threading.Lock()
    game_results: list[dict | None] = [None] * int(args.num_games)
    buffered_samples: list[dict] = []
    total_samples = 0
    shard_idx = 0
    total_plies = 0
    win_count = loss_count = draw_count = 0
    completed = 0
    t_start = time.monotonic()

    with ThreadPoolExecutor(max_workers=parallel_games, thread_name_prefix="vspika") as ex:
        futures = {
            ex.submit(_run_single_game,
                      gi=gi, args=args, model=model, device=device,
                      use_bf16=use_bf16, shared_evaluator=shared_evaluator,
                      shared_opp_model=shared_opp_model): gi
            for gi in range(int(args.num_games))
        }
        for fut in as_completed(futures):
            try:
                result = fut.result()
            except Exception as exc:
                gi = futures[fut]
                # Keep going — a single crashed game shouldn't kill the batch.
                import traceback
                tb = traceback.format_exc()
                with output_lock:
                    print(f"  [game {gi+1}] FAILED: {type(exc).__name__}: {exc}", flush=True)
                    print(tb, flush=True)
                continue

            gi = result["gi"]
            game = result["game"]
            our_is_red = result["our_is_red"]
            outcome = _game_outcome_for_our_side(our_is_red, game["final_red_result"])
            with output_lock:
                if outcome == "our_win":
                    win_count += 1
                elif outcome == "opp_win":
                    loss_count += 1
                else:
                    draw_count += 1
                total_plies += game["plies"]
                game_results[gi] = game
                for samp in game["samples"]:
                    samp["game_id"] = gi
                    buffered_samples.append(samp)
                total_samples += len(game["samples"])
                completed += 1
                dt = time.monotonic() - t_start
                print(
                    f"game {completed}/{args.num_games} (gi={gi}) "
                    f"our={'red' if our_is_red else 'black'} "
                    f"outcome={outcome} plies={game['plies']} "
                    f"running={win_count}-{loss_count}-{draw_count} "
                    f"samples={total_samples} elapsed={dt:.0f}s",
                    flush=True,
                )
                # Flush shard(s) if we've buffered enough.  Safe under lock —
                # the samples we flush are all from games whose game_results[gi]
                # is already populated (same lock-acquire above).
                while len(buffered_samples) >= args.shard_size_samples:
                    chunk = buffered_samples[:args.shard_size_samples]
                    buffered_samples = buffered_samples[args.shard_size_samples:]
                    info = _flush_shard(chunk, game_results, output_dir, shard_idx)
                    print(f"  -> flushed shard {shard_idx}: {info['samples']} samples", flush=True)
                    shard_idx += 1

    # Executor has shut down; no worker threads remain.  Final flush is single-threaded.
    if buffered_samples:
        info = _flush_shard(buffered_samples, game_results, output_dir, shard_idx)
        print(f"  -> final shard {shard_idx}: {info['samples']} samples", flush=True)
        shard_idx += 1

    dt_total = time.monotonic() - t_start
    final_samples = sum(len(gr["samples"]) for gr in game_results if gr is not None)
    manifest = {
        "format": "vs_pikafish_selfplay_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_step": model_step,
        "num_games": args.num_games,
        "wins": win_count,
        "losses": loss_count,
        "draws": draw_count,
        "total_samples": final_samples,
        "total_plies": total_plies,
        "shards": shard_idx,
        "opp_type": str(args.opp_type),
        "opp_depth": int(args.opp_depth) if args.opp_depth else None,
        "opp_nodes": int(args.opp_nodes) if args.opp_nodes else None,
        "noise_ratio": float(args.noise_ratio),
        "our_sims": int(args.our_sims),
        "duration_s": dt_total,
        "quality": "ok",
        "manifest_state": "complete",
        "quality_metrics": {
            "rep_draw_rate": (draw_count / max(args.num_games, 1)) * 100.0,
            "decisive_rate": ((win_count + loss_count) / max(args.num_games, 1)) * 100.0,
            "nocap_draw_rate": 0.0,
        },
        "config": {
            "source": "self_play" if args.opp_type == "model" else "vs_pikafish",
            "search_defaults": {"num_simulations": int(args.our_sims)},
            "opp_model_checkpoint": str(args.opp_model_checkpoint) if args.opp_type == "model" else None,
            "opp_model_sims": int(args.opp_model_sims) if args.opp_type == "model" else None,
        },
    }
    (output_dir.parent / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Shut down the cross-game batcher cleanly + dump its telemetry.
    if shared_evaluator is not None:
        stats = shared_evaluator.stats()
        shared_evaluator.close()
        print(
            f"  cross-game batcher: {stats['batches_run']} GPU batches "
            f"served {stats['calls_received']} thread calls "
            f"(coalesce ratio={stats['coalesce_ratio']:.1f}x, "
            f"avg_leaves/batch={stats['avg_leaves_per_batch']:.1f}, "
            f"max_batch={stats['max_observed_batch']})",
            flush=True,
        )

    # Release the snapshot opponent so its GPU memory can be reclaimed before
    # the next stage1_driver cycle launches another subprocess on this GPU.
    if shared_opp_model is not None:
        try:
            shared_opp_model.close()
        except Exception:
            pass

    print("", flush=True)
    print(
        f"DONE: {win_count}-{loss_count}-{draw_count} in {args.num_games} games, "
        f"{final_samples} samples across {shard_idx} shards in {dt_total:.0f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
