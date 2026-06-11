"""Pikafish UCI opponent wrapper.

Pikafish uses the UCI protocol (Universal Chess Interface), not UCCI, despite being a
Chinese Chess engine (it's a Stockfish fork).  This wrapper handles the subprocess,
FEN-based position updates, and UCI<->internal move conversion.

Square index conventions
------------------------
Our internal board (xqcpp_ext_hist8_115.cpp):
    square = y * 9 + x
    y = 0 is the TOP row (Black side)
    y = 9 is the BOTTOM row (Red side)
    x = 0 is the leftmost file

UCI Xiangqi convention (as used by Pikafish):
    file: 'a' (leftmost) to 'i' (rightmost)
    rank: 0 (Red's bottom / palace) to 9 (Black's top)

Move encoding:
    Our move integer = from_sq * 90 + to_sq
    UCI move string  = f"{file_from}{rank_from}{file_to}{rank_to}" (e.g. "b2e2")
"""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

# Module-level startup throttle.  When many threads (e.g. 16 worker threads in
# pikafish_selfplay's ThreadPoolExecutor) all instantiate PikafishOpponent at
# the same instant, they each fork+exec a Pikafish subprocess and that process
# tries to mmap the same NNUE network weights file simultaneously.  On WSL we
# have observed silent zombification when 16+ Pikafish processes mmap NNUE in
# the same kernel tick — the children exit with no diagnostic and the parent
# blocks on a pipe read forever.  Serialising the *handshake* portion of __init__
# (under a lock with a tiny grace sleep) eliminates the race entirely with a
# negligible (<10s for 16 threads) total startup overhead.
_PIKAFISH_STARTUP_LOCK = threading.Lock()
_PIKAFISH_STARTUP_DELAY_S = 0.4


def uci_sq_to_internal(uci_sq: str) -> int:
    """Convert a 2-char UCI square like "b2" to our internal 0..89 index."""
    if len(uci_sq) != 2:
        raise ValueError(f"invalid UCI square: {uci_sq!r}")
    file_ch, rank_ch = uci_sq[0], uci_sq[1]
    x = ord(file_ch.lower()) - ord("a")
    if not (0 <= x <= 8):
        raise ValueError(f"UCI file out of range: {uci_sq!r}")
    rank = int(rank_ch)
    if not (0 <= rank <= 9):
        raise ValueError(f"UCI rank out of range: {uci_sq!r}")
    y = 9 - rank
    return y * 9 + x


def internal_sq_to_uci(sq: int) -> str:
    if not (0 <= sq < 90):
        raise ValueError(f"internal square out of range: {sq}")
    y, x = divmod(sq, 9)
    rank = 9 - y
    file_ch = chr(ord("a") + x)
    return f"{file_ch}{rank}"


def uci_move_to_internal(uci_move: str) -> int:
    """Convert UCI move string (e.g. "b2e2") to our internal move integer."""
    core = uci_move.strip().lower()
    if len(core) != 4:
        # Some engines may append promotion suffix; Xiangqi has no promotions, so strip if present.
        core = core[:4]
    from_sq = uci_sq_to_internal(core[:2])
    to_sq = uci_sq_to_internal(core[2:4])
    return from_sq * 90 + to_sq


def internal_move_to_uci(move: int) -> str:
    from_sq = move // 90
    to_sq = move % 90
    return internal_sq_to_uci(from_sq) + internal_sq_to_uci(to_sq)


def _parse_info_score(tokens: list[str]) -> tuple[int, int | None]:
    """Parse a UCI ``score cp`` / ``score mate`` token sequence.

    Pikafish reports scores from the side-to-move perspective.  Mate scores are
    mapped to a large centipawn sentinel so callers can compare candidates with
    ordinary cp margins.
    """

    cp_value = 0
    mate_in: int | None = None
    for i, tok in enumerate(tokens):
        if tok != "score" or i + 2 >= len(tokens):
            continue
        kind = tokens[i + 1]
        try:
            value = int(tokens[i + 2])
        except ValueError:
            continue
        if kind == "cp":
            cp_value = value
        elif kind == "mate":
            mate_in = value
            cp_value = 20000 if value > 0 else -20000
        return cp_value, mate_in
    return cp_value, mate_in


class PikafishOpponent:
    """Subprocess wrapper around Pikafish in UCI mode."""

    def __init__(
        self,
        binary_path: str | Path = "/home/laure/pikafish/pikafish",
        threads: int = 1,
        hash_mb: int = 64,
        startup_timeout_s: float = 10.0,
        extra_setoption_lines: list[str] | tuple[str, ...] = (),
        extra_argv: list[str] | tuple[str, ...] = (),
        handshake: str = "uci",
        send_threads_and_hash: bool = True,
        required_option_names: list[str] | tuple[str, ...] = (),
    ) -> None:
        """Generic UCI/UCCI subprocess wrapper.  Extension points:

        - ``extra_setoption_lines`` — additional ``setoption ...`` lines sent
          right after the standard Threads/Hash options.  Used by Fairy-Stockfish
          to engage xiangqi+UCCI mode.
        - ``extra_argv`` — additional command-line args (after the binary path)
          when launching the subprocess.  Used by ElephantArt which needs
          ``-m ucci -w <weights>`` to start in UCCI mode.
        - ``handshake`` — initial protocol handshake command (default ``"uci"``,
          can be ``"ucci"`` for engines that only speak UCCI).  We then wait for
          either ``uciok`` or ``ucciok`` accordingly.
        - ``send_threads_and_hash`` — Pikafish/Fairy-SF accept the standard
          ``setoption name Threads/Hash`` lines.  ElephantArt does not — its
          options are passed via CLI flags (``-t``, ``-p``).  Set False to skip.
        - ``required_option_names`` — fail during startup if the engine's UCI/UCCI
          option list does not expose every requested option.  This prevents
          silent no-op ``setoption`` calls from producing misleading benchmarks.
        """
        self.binary_path = Path(binary_path)
        if not self.binary_path.is_file():
            raise FileNotFoundError(f"engine binary not found: {binary_path}")
        self.threads = int(threads)
        self.hash_mb = int(hash_mb)
        self.extra_setoption_lines = list(extra_setoption_lines)
        self.extra_argv = list(extra_argv)
        self.handshake = handshake
        self.send_threads_and_hash = bool(send_threads_and_hash)
        self.required_option_names = tuple(str(x) for x in required_option_names)
        self._proc: subprocess.Popen[str] | None = None
        self._start(startup_timeout_s)

    def _start(self, timeout_s: float) -> None:
        argv = [str(self.binary_path), *self.extra_argv]
        # Serialise the spawn + NNUE-load + handshake under a global lock so that
        # N concurrent PikafishOpponent() constructions don't all mmap NNUE at the
        # same instant — see _PIKAFISH_STARTUP_LOCK comment near top of file.
        # We hold the lock for the full handshake (until readyok) because that's
        # the window during which Pikafish loads NNUE from disk; once it's idle
        # waiting for `position`/`go`, subsequent operations are safe to interleave.
        with _PIKAFISH_STARTUP_LOCK:
            self._proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(self.binary_path.parent),
                text=True,
                bufsize=1,
            )
            # Protocol handshake (uci or ucci).  Reply token is "uciok" / "ucciok".
            self._send(self.handshake)
            ok_token = "ucciok" if self.handshake == "ucci" else "uciok"
            handshake_lines = self._expect(ok_token, timeout_s)
            if self.required_option_names:
                option_names = self._parse_option_names(handshake_lines)
                missing = [name for name in self.required_option_names if name not in option_names]
                if missing:
                    raise RuntimeError(
                        f"engine {self.binary_path} does not expose required UCI options "
                        f"{missing}; available={sorted(option_names)}"
                    )
            if self.send_threads_and_hash:
                self._send(f"setoption name Threads value {self.threads}")
                self._send(f"setoption name Hash value {self.hash_mb}")
            for opt in self.extra_setoption_lines:
                self._send(opt)
            self._send("isready")
            self._expect("readyok", timeout_s)
            # Tiny grace pause before releasing the lock — gives Pikafish a moment
            # to settle internal state (search tree allocation etc.) before the
            # next PikafishOpponent in line starts mmap'ing NNUE.
            time.sleep(_PIKAFISH_STARTUP_DELAY_S)

    def _send(self, line: str) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("pikafish process not started")
        self._proc.stdin.write(line + "\n")
        self._proc.stdin.flush()

    def _readline(self, timeout_s: float) -> str:
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("pikafish process not started")
        deadline = time.monotonic() + timeout_s
        # Subprocess readline is blocking; we don't have non-blocking reads here, but the
        # caller passes a reasonable deadline.  If the engine never answers, we'll hang
        # and rely on wall-clock supervision by the harness.
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("pikafish closed stdout unexpectedly")
        return line.rstrip("\n")

    @staticmethod
    def _parse_option_names(lines: list[str]) -> set[str]:
        names: set[str] = set()
        for line in lines:
            if not line.startswith("option name "):
                continue
            rest = line[len("option name "):]
            name = rest.split(" type ", 1)[0].strip()
            if name:
                names.add(name)
        return names

    def _expect(self, needle: str, timeout_s: float) -> list[str]:
        deadline = time.monotonic() + timeout_s
        lines: list[str] = []
        while time.monotonic() < deadline:
            line = self._readline(max(0.1, deadline - time.monotonic()))
            lines.append(line)
            if needle in line:
                return lines
        raise TimeoutError(f"pikafish: timed out waiting for '{needle}'")

    def new_game(self) -> None:
        self._send("ucinewgame")
        self._send("isready")
        self._expect("readyok", 5.0)

    def set_position(self, fen: str, moves: list[str] | None = None) -> None:
        """Set the position via a FEN, optionally followed by a sequence of UCI moves."""
        cmd = f"position fen {fen}"
        if moves:
            cmd += " moves " + " ".join(moves)
        self._send(cmd)

    def go_depth(self, depth: int, max_wait_s: float = 120.0) -> tuple[str, str | None]:
        """Search to a given depth.  Returns (bestmove_uci, ponder_uci_or_None)."""
        self._send(f"go depth {int(depth)}")
        return self._await_bestmove(max_wait_s)

    def go_depth_eval(self, depth: int, max_wait_s: float = 120.0) -> tuple[str, str | None, int, int | None]:
        """Search to a given depth and return bestmove plus the final root score."""
        self._send(f"go depth {int(depth)}")
        return self._await_bestmove_with_score(max_wait_s)

    def go_movetime(self, movetime_ms: int, max_wait_s: float | None = None) -> tuple[str, str | None]:
        """Search with a fixed wall-clock budget."""
        self._send(f"go movetime {int(movetime_ms)}")
        deadline = (movetime_ms / 1000.0) + 10.0 if max_wait_s is None else max_wait_s
        return self._await_bestmove(deadline)

    def go_movetime_eval(self, movetime_ms: int, max_wait_s: float | None = None) -> tuple[str, str | None, int, int | None]:
        """Search with a fixed wall-clock budget and return the final root score."""
        self._send(f"go movetime {int(movetime_ms)}")
        deadline = (movetime_ms / 1000.0) + 10.0 if max_wait_s is None else max_wait_s
        return self._await_bestmove_with_score(deadline)

    def go_nodes(self, nodes: int, max_wait_s: float = 60.0) -> tuple[str, str | None]:
        """Search bounded by max nodes (weakest practical knob Pikafish offers)."""
        self._send(f"go nodes {int(nodes)}")
        return self._await_bestmove(max_wait_s)

    def go_nodes_eval(self, nodes: int, max_wait_s: float = 60.0) -> tuple[str, str | None, int, int | None]:
        """Search bounded by max nodes and return the final root score."""
        self._send(f"go nodes {int(nodes)}")
        return self._await_bestmove_with_score(max_wait_s)

    def _await_bestmove(self, timeout_s: float) -> tuple[str, str | None]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            line = self._readline(max(0.1, deadline - time.monotonic()))
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) < 2:
                    raise RuntimeError(f"malformed bestmove line: {line!r}")
                best = parts[1]
                ponder = None
                if len(parts) >= 4 and parts[2] == "ponder":
                    ponder = parts[3]
                return best, ponder
        raise TimeoutError("pikafish: timed out waiting for bestmove")

    def _await_bestmove_with_score(self, timeout_s: float) -> tuple[str, str | None, int, int | None]:
        deadline = time.monotonic() + timeout_s
        score_cp = 0
        mate_in: int | None = None
        while time.monotonic() < deadline:
            line = self._readline(max(0.1, deadline - time.monotonic()))
            if line.startswith("info "):
                cp, mate = _parse_info_score(line.split())
                score_cp = int(cp)
                mate_in = mate
                continue
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) < 2:
                    raise RuntimeError(f"malformed bestmove line: {line!r}")
                best = parts[1]
                ponder = None
                if len(parts) >= 4 and parts[2] == "ponder":
                    ponder = parts[3]
                return best, ponder, int(score_cp), mate_in
        raise TimeoutError("pikafish: timed out waiting for bestmove")

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            self._send("quit")
            self._proc.wait(timeout=3.0)
        except Exception:
            self._proc.kill()
        finally:
            self._proc = None

    def __enter__(self) -> "PikafishOpponent":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _self_test() -> None:
    """Quick sanity check: ask Pikafish for the best opening move at depth 5."""
    # Move conversion round-trip
    assert uci_sq_to_internal("a0") == 9 * 9  # red corner = bottom-left in our frame = y=9,x=0 → 81
    assert internal_sq_to_uci(81) == "a0"
    # Red cannon opening move "b2e2": b=file1, rank2; e=file4, rank2
    m = uci_move_to_internal("b2e2")
    assert internal_move_to_uci(m) == "b2e2", f"round-trip failed: got {internal_move_to_uci(m)!r}"
    print("move conversion ok")

    # Engine handshake
    START_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
    with PikafishOpponent() as p:
        p.new_game()
        p.set_position(START_FEN)
        best, ponder = p.go_depth(5)
        print(f"depth=5 from startpos: bestmove={best} ponder={ponder}")
        # Sanity: best move should be a legal opening like b2e2, h2e2, b0c2, etc.
        assert len(best) == 4


if __name__ == "__main__":
    _self_test()
