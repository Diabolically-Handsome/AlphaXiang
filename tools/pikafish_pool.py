"""Parallel Pikafish worker pool for batch position labelling.

Each worker owns one Pikafish subprocess and can answer one query at a time.
The pool exposes a simple blocking `query_many(items, ...)` API that distributes
work across processes using a Python multiprocessing queue.

Unlike distillation-time usage, this pool is **not** shared between selfplay and
labelling; selfplay games spin up their own Pikafish instances directly.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


@dataclass
class PikafishJob:
    """One unit of work: evaluate a position and return (best_move_uci, eval_cp)."""

    index: int
    fen: str  # full UCI-style FEN (placement + stm + castling + ep + hm + fm)
    # Only one of depth/nodes/movetime_ms should be > 0:
    depth: int = 0
    nodes: int = 0
    movetime_ms: int = 0

    # MultiPV: number of top moves to return. 1 = only best.
    multipv: int = 1


@dataclass
class PikafishResult:
    index: int
    best_move: str
    eval_cp: int  # centipawns, from side-to-move's perspective
    mate_in: int | None = None
    multipv_moves: list[tuple[str, int]] | None = None  # list of (move, cp) for top-k
    error: str | None = None


class PikafishPoolTimeout(TimeoutError):
    """Raised when a pool collection times out after producing partial results."""

    def __init__(self, message: str, partial_results: list[PikafishResult]):
        super().__init__(message)
        self.partial_results = partial_results


def _parse_info_score(tokens: list[str]) -> tuple[int, int | None]:
    """Parse 'score cp N' or 'score mate N' out of an info line token stream."""
    cp_value = 0
    mate_in: int | None = None
    for i, tok in enumerate(tokens):
        if tok == "score" and i + 2 < len(tokens):
            kind = tokens[i + 1]
            try:
                value = int(tokens[i + 2])
            except ValueError:
                continue
            if kind == "cp":
                cp_value = value
            elif kind == "mate":
                mate_in = value
                # Map mate in N to a large cp value (positive = we mate; negative = we get mated)
                cp_value = 20000 if value > 0 else -20000
            return cp_value, mate_in
    return cp_value, mate_in


def _parse_info_multipv(tokens: list[str]) -> tuple[int | None, str | None, int, int | None]:
    """Return (multipv_idx, pv_move, score_cp, mate_in) from one 'info' line tokens."""
    multipv_idx: int | None = None
    pv_move: str | None = None
    score_cp = 0
    mate_in: int | None = None
    for i, tok in enumerate(tokens):
        if tok == "multipv" and i + 1 < len(tokens):
            try:
                multipv_idx = int(tokens[i + 1])
            except ValueError:
                pass
        elif tok == "score" and i + 2 < len(tokens):
            kind = tokens[i + 1]
            try:
                value = int(tokens[i + 2])
            except ValueError:
                continue
            if kind == "cp":
                score_cp = value
            elif kind == "mate":
                mate_in = value
                score_cp = 20000 if value > 0 else -20000
        elif tok == "pv" and i + 1 < len(tokens):
            pv_move = tokens[i + 1]
    return multipv_idx, pv_move, score_cp, mate_in


def _worker_run(worker_id: int, binary_path: str, threads: int, hash_mb: int,
                in_queue: "mp.Queue[PikafishJob | None]",
                out_queue: "mp.Queue[PikafishResult]") -> None:
    """Long-lived worker: pull PikafishJob, run it, push PikafishResult."""
    import subprocess

    # Stagger NNUE-load by worker_id so we don't have N processes simultaneously
    # mmap'ing the same big network weights file.  Without this, on WSL we have
    # observed silent zombification of 12 simultaneously-spawned workers, which
    # leaves the parent's queue.collect() blocked indefinitely waiting for results
    # that will never arrive.  0.4s × worker_id keeps total spinup overhead tiny
    # (4.4s for 12 workers) while completely eliminating the race.
    if worker_id > 0:
        time.sleep(0.4 * float(worker_id))
    proc = subprocess.Popen(
        [binary_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(Path(binary_path).parent),
        text=True,
        bufsize=1,
    )

    def send(cmd: str) -> None:
        assert proc.stdin is not None
        proc.stdin.write(cmd + "\n")
        proc.stdin.flush()

    def readline() -> str:
        assert proc.stdout is not None
        return (proc.stdout.readline() or "").rstrip("\n")

    # Handshake
    send("uci")
    while True:
        line = readline()
        if "uciok" in line:
            break
    send(f"setoption name Threads value {threads}")
    send(f"setoption name Hash value {hash_mb}")
    send("isready")
    while True:
        line = readline()
        if "readyok" in line:
            break

    try:
        while True:
            job = in_queue.get()
            if job is None:
                break
            # MultiPV setting
            send(f"setoption name MultiPV value {int(max(1, job.multipv))}")
            send("ucinewgame")
            send(f"position fen {job.fen}")
            if job.depth > 0:
                send(f"go depth {job.depth}")
            elif job.nodes > 0:
                send(f"go nodes {job.nodes}")
            elif job.movetime_ms > 0:
                send(f"go movetime {job.movetime_ms}")
            else:
                send("go depth 1")

            # Collect info lines until bestmove; track last-seen score + multipv set
            multipv_snapshot: dict[int, tuple[str, int, int | None]] = {}
            best_move = "0000"
            while True:
                line = readline()
                if not line:
                    continue
                if line.startswith("info "):
                    toks = line.split()
                    mpv_idx, pv_move, score_cp, mate_in = _parse_info_multipv(toks)
                    if mpv_idx is not None and pv_move is not None:
                        multipv_snapshot[mpv_idx] = (pv_move, score_cp, mate_in)
                elif line.startswith("bestmove"):
                    parts = line.split()
                    if len(parts) >= 2:
                        best_move = parts[1]
                    break

            # Compile result
            if 1 in multipv_snapshot:
                pv_move, score_cp, mate_in = multipv_snapshot[1]
            else:
                pv_move = best_move
                score_cp = 0
                mate_in = None

            multipv_moves = None
            if job.multipv > 1 and multipv_snapshot:
                multipv_moves = [
                    (multipv_snapshot[i][0], multipv_snapshot[i][1])
                    for i in sorted(multipv_snapshot.keys())
                ]

            out_queue.put(PikafishResult(
                index=job.index,
                best_move=best_move,
                eval_cp=score_cp,
                mate_in=mate_in,
                multipv_moves=multipv_moves,
            ))
    finally:
        try:
            send("quit")
            proc.wait(timeout=3)
        except Exception:
            proc.kill()


class PikafishPool:
    def __init__(
        self,
        num_workers: int = 8,
        binary_path: str | Path = "/home/laure/pikafish/pikafish",
        threads_per_worker: int = 1,
        hash_mb: int = 16,
    ) -> None:
        self.num_workers = int(num_workers)
        self.binary_path = str(Path(binary_path).resolve())
        self.threads_per_worker = int(threads_per_worker)
        self.hash_mb = int(hash_mb)
        ctx = mp.get_context("spawn")
        self.in_queue: mp.Queue[PikafishJob | None] = ctx.Queue()
        self.out_queue: mp.Queue[PikafishResult] = ctx.Queue()
        self._procs: list[mp.Process] = []
        for wid in range(self.num_workers):
            p = ctx.Process(
                target=_worker_run,
                args=(wid, self.binary_path, self.threads_per_worker, self.hash_mb,
                      self.in_queue, self.out_queue),
                daemon=True,
            )
            p.start()
            self._procs.append(p)

    def submit_all(self, jobs: Sequence[PikafishJob]) -> None:
        for j in jobs:
            self.in_queue.put(j)

    def collect(self, total: int, timeout_s: float | None = None,
                progress_cb: Callable[[int, int], None] | None = None) -> list[PikafishResult]:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        out: list[PikafishResult] = []
        while len(out) < total:
            remaining = None if deadline is None else max(1.0, deadline - time.monotonic())
            try:
                r = self.out_queue.get(timeout=remaining)
            except queue.Empty:
                raise PikafishPoolTimeout(
                    f"pikafish pool: got {len(out)}/{total} within {timeout_s}s",
                    out,
                )
            out.append(r)
            if progress_cb is not None and len(out) % max(1, total // 20) == 0:
                progress_cb(len(out), total)
        return out

    def close(self) -> None:
        for _ in self._procs:
            self.in_queue.put(None)
        for p in self._procs:
            p.join(timeout=5)
        for p in self._procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)

    def __enter__(self) -> "PikafishPool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _self_test() -> None:
    START_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
    jobs = [PikafishJob(index=i, fen=START_FEN, depth=5) for i in range(4)]
    t0 = time.monotonic()
    with PikafishPool(num_workers=2) as pool:
        pool.submit_all(jobs)
        results = pool.collect(len(jobs), timeout_s=30.0)
    dt = time.monotonic() - t0
    for r in sorted(results, key=lambda x: x.index):
        print(f"idx={r.index} best={r.best_move} cp={r.eval_cp} mate_in={r.mate_in}")
    print(f"4 depth-5 queries via 2-worker pool: {dt:.2f}s")


if __name__ == "__main__":
    _self_test()
