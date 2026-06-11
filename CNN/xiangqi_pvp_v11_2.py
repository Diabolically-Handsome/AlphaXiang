# -*- coding: utf-8 -*-
"""
Xiangqi PVP / Human-vs-AI GUI (V11.2)
====================================

This UI uses the C++ board and move generator (`xqcpp.Board`) end-to-end.
Search runs through `xqcpp.mcts_search` in a worker thread to keep the GUI responsive.

Supported features:
- AI strength levels (`1` / `2` / `3`)
- Two-sided clock + byoyomi (30:00 + 3x60s)
- Side swap (`S`) in human-vs-AI mode
- Undo (`U` / `Backspace`)
- New game (`N`)
- Reload weights (`R`) without resetting board
- Debug overlay (`D`) with root value and top moves
- Optional exploration temperature toggle (`E`)

Example:
  python xiangqi_pvp_v11_2.py --engine Chessv11_cpp_hist8_115_mps_fp16 --weights v11_runs/best.pth --human red
"""

import os
import time
import argparse
import threading
import queue
import importlib
import glob
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
import pygame


# ----------------------------
# 1) xqcpp extension loading
# ----------------------------

def ensure_xqcpp_loaded():
    """Build and load the local xqcpp extension from source."""
    from torch.utils.cpp_extension import load

    this_dir = os.path.dirname(os.path.abspath(__file__))
    src = None
    for fname in ("xqcpp_ext_hist8_115.cpp", "xqcpp_ext.cpp"):
        cand = os.path.join(this_dir, fname)
        if os.path.exists(cand):
            src = cand
            break
    if src is None:
        raise FileNotFoundError(
            f"C++ source not found: {os.path.join(this_dir, 'xqcpp_ext_hist8_115.cpp')} or xqcpp_ext.cpp"
        )

    return load(
        name="xqcpp",
        sources=[src],
        extra_cflags=["-O3", "-std=c++17"],
        with_cuda=False,
        verbose=True,
    )


# ----------------------------
# 2) Coordinates / move format
# ----------------------------

FILE_CHARS = "abcdefghi"


def square_to_iccs(square: int) -> str:
    y, x = divmod(int(square), 9)
    return f"{FILE_CHARS[x]}{9 - y}"


def iccs_to_square(s: str) -> int:
    s = s.strip().lower()
    if len(s) < 2:
        raise ValueError(s)
    f = FILE_CHARS.index(s[0])
    r = int(s[1:])
    y = 9 - r
    return y * 9 + f


def action_to_from_to(action_id: int) -> Tuple[int, int]:
    a = int(action_id)
    return a // 90, a % 90


def from_to_to_action(fr: int, to: int) -> int:
    return int(fr) * 90 + int(to)


def action_to_iccs(action_id: int) -> str:
    fr, to = action_to_from_to(action_id)
    return square_to_iccs(fr) + square_to_iccs(to)


# ----------------------------
# 3) Piece labels (xqcpp piece code -> display text)
# ----------------------------

# xqcpp: RED piece codes are +1..+7, BLACK are -1..-7
# 1 K/king, 2 A/advisor, 3 B/elephant, 4 N/horse, 5 R/rook, 6 C/cannon, 7 P/pawn

# Chinese piece labels for board rendering (UI text remains English).
CN_RED = {1: "帅", 2: "仕", 3: "相", 4: "马", 5: "车", 6: "炮", 7: "兵"}
CN_BLACK = {1: "将", 2: "士", 3: "象", 4: "马", 5: "车", 6: "炮", 7: "卒"}

ASCII_RED = {1: "K", 2: "A", 3: "B", 4: "N", 5: "R", 6: "C", 7: "P"}
ASCII_BLACK = {1: "k", 2: "a", 3: "b", 4: "n", 5: "r", 6: "c", 7: "p"}


def piece_label(piece_code: int, prefer_cn: bool = True) -> str:
    p = int(piece_code)
    if p == 0:
        return ""
    side = 1 if p > 0 else -1
    t = abs(p)
    if prefer_cn:
        return CN_RED[t] if side > 0 else CN_BLACK[t]
    return ASCII_RED[t] if side > 0 else ASCII_BLACK[t]


def scalar_int(value: Any) -> int:
    """Robustly coerce extension / numpy / torch scalars to a Python int."""
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("empty tensor scalar")
        return int(value.detach().reshape(-1)[0].item())
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("empty ndarray scalar")
        return int(value.reshape(-1)[0].item())
    if isinstance(value, (bytes, bytearray)):
        if len(value) != 1:
            raise ValueError(f"unsupported byte scalar length={len(value)}")
        return int.from_bytes(value, byteorder="little", signed=True)
    return int(value)


def as_numpy_1d(value: Any, dtype) -> np.ndarray:
    """Convert list / tensor / ndarray outputs to a flat numpy array."""
    if torch.is_tensor(value):
        arr = value.detach().cpu().reshape(-1).numpy()
    elif isinstance(value, np.ndarray):
        arr = value.reshape(-1)
    else:
        arr = np.asarray(value)
        arr = arr.reshape(-1) if arr.ndim > 0 else arr.reshape(1)
    return arr.astype(dtype, copy=False)


# ----------------------------
# 4) Load network (import from training module)
# ----------------------------


def load_engine_module(engine: str):
    # 1) If a .py file path is provided, load it dynamically with spec_from_file_location.
    if engine.endswith(".py"):
        spec = importlib.util.spec_from_file_location("xiangqi_engine", engine)
        if spec is None or spec.loader is None:
            raise ImportError(engine)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        return mod

    # 2) Otherwise import as a module name (e.g. Chessv11_cpp_hist8_115_mps_fp16).
    return importlib.import_module(engine)


class EvalNetWrapper:
    """Feed CPU states to C++ MCTS while running model forward on the selected device.

    A lock is used to avoid races between weight reload and forward calls.
    """

    def __init__(self, net: torch.nn.Module, device: torch.device, use_fp16: bool, lock: threading.Lock):
        self.device = device
        self.use_fp16 = bool(use_fp16) and (device.type in ("cuda", "mps"))
        self.net = net.to(device)
        if self.use_fp16:
            self.net = self.net.half()
        self.net.eval()
        self.lock = lock

    def __call__(self, x_cpu: torch.Tensor):
        with self.lock:
            x = x_cpu.to(self.device)
            x = x.half() if self.use_fp16 else x.float()
            with torch.inference_mode():
                out = self.net(x)
            # C++ extension expects CPU tensors.
            if self.device.type == "mps":
                try:
                    torch.mps.synchronize()  # type: ignore
                except Exception:
                    pass
            return tuple(t.float().cpu().contiguous() for t in out)


def pick_net_class(engine_mod):
    for name in ("XiangqiNetV11", "XiangqiNetV10", "XiangqiNet"):
        if hasattr(engine_mod, name):
            return getattr(engine_mod, name)
    raise AttributeError("XiangqiNetV11/XiangqiNetV10/XiangqiNet not found in engine module")


def extract_state_dict(weights_obj: Any) -> Dict[str, torch.Tensor]:
    """Accept either pure state_dict or checkpoint dict with a `model` entry."""
    if isinstance(weights_obj, dict) and isinstance(weights_obj.get("model"), dict):
        return weights_obj["model"]
    if isinstance(weights_obj, dict):
        return weights_obj
    raise TypeError("Unsupported weights format")


def normalize_state_dict_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip common wrappers from key names: module./_orig_mod./model."""
    if not sd:
        return sd
    keys = list(sd.keys())
    for prefix in ("module.", "_orig_mod.", "model."):
        if all(k.startswith(prefix) for k in keys):
            return {k[len(prefix):]: v for k, v in sd.items()}
    return sd


def count_nonfinite_state(sd: Dict[str, torch.Tensor]) -> int:
    bad = 0
    for v in sd.values():
        if torch.is_tensor(v):
            bad += int((~torch.isfinite(v)).sum().item())
    return int(bad)


def resolve_weights_path(requested: str, cfg: Dict[str, Any]) -> str:
    """Resolve a usable weights path. `auto` prefers latest valid weights over stale best."""
    requested = (requested or "").strip()
    if requested and requested.lower() != "auto":
        return requested

    this_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = str(cfg.get("SAVE_DIR", "v11_runs"))
    if not os.path.isabs(save_dir):
        save_dir = os.path.join(this_dir, save_dir)

    latest_name = str(cfg.get("LATEST_NAME", "latest.pth"))
    best_name = str(cfg.get("BEST_NAME", "best.pth"))
    latest_path = os.path.join(save_dir, latest_name)
    best_path = os.path.join(save_dir, best_name)

    candidates: List[str] = []
    if os.path.isfile(latest_path):
        candidates.append(latest_path)
    if os.path.isfile(best_path):
        candidates.append(best_path)

    snapshots = sorted(glob.glob(os.path.join(save_dir, "model_step*.pth")), reverse=True)
    for path in snapshots[:10]:
        if path not in candidates:
            candidates.append(path)

    for path in candidates:
        try:
            obj = torch.load(path, map_location="cpu")
            sd = normalize_state_dict_keys(extract_state_dict(obj))
            if count_nonfinite_state(sd) == 0:
                return path
        except Exception:
            continue

    return latest_path if candidates == [] else candidates[0]


# ----------------------------
# 5) UI state + clock
# ----------------------------


@dataclass
class UIState:
    selected_sq: Optional[int] = None
    legal_dests: Optional[List[int]] = None
    last_move: Optional[int] = None
    game_over: bool = False
    result_text: str = ""
    debug: bool = False
    status_text: str = ""


@dataclass
class PlayerClock:
    main_seconds: float = 30 * 60.0
    byoyomi_seconds: float = 60.0
    periods_total: int = 3
    periods_remaining: int = 3

    def reset(self):
        self.main_seconds = 30 * 60.0
        self.byoyomi_seconds = 60.0
        self.periods_remaining = self.periods_total

    def in_byoyomi(self) -> bool:
        return self.main_seconds <= 0.0

    def on_turn_start(self):
        # In byoyomi, reset the per-move clock to 60 seconds when turn starts.
        if self.in_byoyomi():
            self.byoyomi_seconds = 60.0

    def tick(self, dt: float) -> bool:
        """Return True if the player flags on time."""
        if dt <= 0:
            return False

        if self.main_seconds > 0.0:
            self.main_seconds -= dt
            if self.main_seconds <= 0.0:
                self.main_seconds = 0.0
                self.byoyomi_seconds = 60.0
            return False

        # Byoyomi phase
        self.byoyomi_seconds -= dt
        if self.byoyomi_seconds > 0.0:
            return False

        # Current byoyomi period expired
        if self.periods_remaining <= 1:
            return True

        # Consume one period and continue the same move.
        self.periods_remaining -= 1
        self.byoyomi_seconds = 60.0
        return False

    def fmt(self) -> str:
        def mmss(sec: float) -> str:
            sec_i = max(0, int(sec + 0.5))
            m = sec_i // 60
            s = sec_i % 60
            return f"{m:02d}:{s:02d}"

        if self.main_seconds > 0:
            return mmss(self.main_seconds)
        return f"{mmss(self.byoyomi_seconds)}  Byo x{self.periods_remaining}"


def dynamic_sims(
    level_sims: int,
    clk: PlayerClock,
    enable_time_manager: bool,
    hard_min: int,
    hard_max: int,
) -> int:
    sims = int(level_sims)
    if not enable_time_manager:
        return max(hard_min, min(sims, hard_max))

    if not clk.in_byoyomi():
        t = float(clk.main_seconds)
        if t > 8 * 60:
            sims = sims
        elif t > 3 * 60:
            sims = int(sims * 0.75)
        elif t > 60:
            sims = int(sims * 0.55)
        else:
            sims = min(sims, 240)
    else:
        bt = float(clk.byoyomi_seconds)
        if bt > 40:
            sims = min(sims, 240)
        elif bt > 20:
            sims = min(sims, 160)
        else:
            sims = min(sims, hard_min)

    sims = max(hard_min, sims)
    sims = min(hard_max, sims)
    return sims


# ----------------------------
# 6) Move legality helpers (xqcpp)
# ----------------------------


def side_to_move(board) -> int:
    """0 = red to move, 1 = black to move."""
    return int(board.turn())


def legal_dest_squares(board, from_sq: int) -> List[int]:
    from_sq = int(from_sq)
    dests: List[int] = []
    for a in board.legal_moves():
        fr, to = action_to_from_to(int(a))
        if fr == from_sq:
            dests.append(to)
    return dests


def try_push_move(board, from_sq: int, to_sq: int) -> Optional[int]:
    """Push (from,to) if legal and return action_id, else return None."""
    target = from_to_to_action(from_sq, to_sq)
    for a in board.legal_moves():
        if int(a) == target:
            board.push(int(a))
            return int(a)
    return None


def outcome_text_from_red_view(res: int) -> str:
    if res > 0:
        return "Red wins"
    if res < 0:
        return "Black wins"
    return "Draw"


# ----------------------------
# 7) AI worker (thread)
# ----------------------------


class AIWorker(threading.Thread):
    def __init__(
        self,
        xqcpp,
        board_fen_queue: "queue.Queue[Tuple[int,str]]",
        result_queue: "queue.Queue[Tuple[int,int,dict]]",
        eval_net,
        sims_ref: Dict[str, int],
        temp_ref: Dict[str, float],
        c_puct: float,
        eval_batch: int,
        seed0: int,
    ):
        super().__init__(daemon=True)
        self.xqcpp = xqcpp
        self.in_q = board_fen_queue
        self.out_q = result_queue
        self.eval_net = eval_net
        self.sims_ref = sims_ref
        self.temp_ref = temp_ref
        self.c_puct = float(c_puct)
        self.eval_batch = int(eval_batch)
        self.seed0 = int(seed0)

    def run(self):
        while True:
            req_id, fen, cancel_event = self.in_q.get()
            try:
                t0 = time.time()
                board = self.xqcpp.Board()
                board.set_fen(str(fen))

                sims = int(self.sims_ref.get("sims", 800))
                temperature_move = float(self.temp_ref.get("t_move", 0.0))

                seed = (self.seed0 + int(time.time() * 1000) + req_id * 10007) & 0x7fffffff
                best_mv, idxs, probs, root_v, debug_stats = self.xqcpp.mcts_search(
                    board,
                    self.eval_net,
                    sims,
                    self.c_puct,
                    False,  # add_root_noise
                    0.3,
                    0.0,
                    float(temperature_move),  # temperature_move
                    1.0,  # temperature_target
                    self.eval_batch,
                    int(seed),
                    True,
                    cancel_event,
                    64,
                )

                if isinstance(debug_stats, dict) and bool(debug_stats.get("cancelled", False)):
                    continue

                # Build top-K list
                topk = []
                try:
                    idxs_np = as_numpy_1d(idxs, np.int64)
                    probs_np = as_numpy_1d(probs, np.float32)
                    if idxs_np.size > 0:
                        order = np.argsort(-probs_np)
                        for j in order[:8]:
                            topk.append((action_to_iccs(int(idxs_np[j])), float(probs_np[j])))
                except Exception:
                    pass

                info = {
                    "root_v": float(root_v),
                    "topk": topk,
                    "sims": sims,
                    "t_move": temperature_move,
                    "elapsed_ms": float((time.time() - t0) * 1000.0),
                }
                if isinstance(debug_stats, dict):
                    info.update(debug_stats)
                self.out_q.put((req_id, int(best_mv), info))
            except Exception as e:
                self.out_q.put((req_id, -1, {"error": repr(e)}))


# ----------------------------
# 8) GUI rendering + board flip
# ----------------------------


def init_fonts(prefer_cn: bool):
    pygame.font.init()

    candidates = [
        # macOS
        "PingFang SC",
        "Heiti SC",
        "Songti SC",
        # Windows
        "Microsoft YaHei",
        "SimHei",
        # Linux
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        # Generic fallback
        "Arial Unicode MS",
    ]

    font_path = None
    if prefer_cn:
        for name in candidates:
            try:
                path = pygame.font.match_font(name)
            except Exception:
                path = None
            if path:
                font_path = path
                break

    if prefer_cn and font_path is None:
        try:
            for fam in pygame.font.get_fonts():
                if any(k in fam.lower() for k in ("pingfang", "yahei", "simhei", "heiti", "noto", "wqy")):
                    path = pygame.font.match_font(fam)
                    if path:
                        font_path = path
                        break
        except Exception:
            pass

    if prefer_cn and font_path is None and os.name == "nt":
        win_dir = os.environ.get("WINDIR", r"C:\Windows")
        direct_candidates = [
            os.path.join(win_dir, "Fonts", "msyh.ttc"),
            os.path.join(win_dir, "Fonts", "msyhbd.ttc"),
            os.path.join(win_dir, "Fonts", "simhei.ttf"),
            os.path.join(win_dir, "Fonts", "simsun.ttc"),
            os.path.join(win_dir, "Fonts", "simkai.ttf"),
        ]
        for path in direct_candidates:
            if os.path.isfile(path):
                font_path = path
                break

    try:
        if font_path:
            f_piece = pygame.font.Font(font_path, 36)
            f_piece.set_bold(True)
            f_ui = pygame.font.Font(font_path, 24)
        else:
            f_piece = pygame.font.SysFont(None, 36, bold=True)
            f_ui = pygame.font.SysFont(None, 24)
    except Exception:
        f_piece = pygame.font.SysFont(None, 36, bold=True)
        f_ui = pygame.font.SysFont(None, 24)

    return f_piece, f_ui


def draw_board(screen, rect, color_line=(0, 0, 0)):
    x0, y0, w, h = rect
    cell_w = w / 8.0
    cell_h = h / 9.0

    for r in range(10):
        y = y0 + r * cell_h
        pygame.draw.line(screen, color_line, (x0, y), (x0 + w, y), 1)

    for c in range(9):
        x = x0 + c * cell_w
        if c == 0 or c == 8:
            pygame.draw.line(screen, color_line, (x, y0), (x, y0 + h), 1)
        else:
            pygame.draw.line(screen, color_line, (x, y0), (x, y0 + 4 * cell_h), 1)
            pygame.draw.line(screen, color_line, (x, y0 + 5 * cell_h), (x, y0 + h), 1)

    def line_sq(sx, sy, tx, ty):
        pygame.draw.line(screen, color_line, (sx, sy), (tx, ty), 1)

    # black palace (top)
    line_sq(x0 + 3 * cell_w, y0 + 0 * cell_h, x0 + 5 * cell_w, y0 + 2 * cell_h)
    line_sq(x0 + 5 * cell_w, y0 + 0 * cell_h, x0 + 3 * cell_w, y0 + 2 * cell_h)
    # red palace (bottom)
    line_sq(x0 + 3 * cell_w, y0 + 7 * cell_h, x0 + 5 * cell_w, y0 + 9 * cell_h)
    line_sq(x0 + 5 * cell_w, y0 + 7 * cell_h, x0 + 3 * cell_w, y0 + 9 * cell_h)


def board_sq_to_screen(board_rect, sq: int, flip: bool) -> Tuple[float, float]:
    x0, y0, w, h = board_rect
    cell_w = w / 8.0
    cell_h = h / 9.0

    y, x = divmod(int(sq), 9)
    if flip:
        y = 9 - y
        x = 8 - x

    sx = x0 + x * cell_w
    sy = y0 + y * cell_h
    return sx, sy


def mouse_to_square(board_rect, mx: int, my: int, flip: bool) -> Optional[int]:
    x0, y0, w, h = board_rect
    if mx < x0 - 10 or mx > x0 + w + 10 or my < y0 - 10 or my > y0 + h + 10:
        return None
    cell_w = w / 8.0
    cell_h = h / 9.0
    x = int(round((mx - x0) / cell_w))
    y = int(round((my - y0) / cell_h))
    x = max(0, min(8, x))
    y = max(0, min(9, y))

    if flip:
        y = 9 - y
        x = 8 - x

    return y * 9 + x


def draw_pieces(screen, board_rect, board, font_piece, prefer_cn: bool, ui: UIState, flip: bool):
    x0, y0, w, h = board_rect
    cell_w = w / 8.0
    cell_h = h / 9.0

    # Highlight last move
    if ui.last_move is not None:
        fr, to = action_to_from_to(int(ui.last_move))
        for sq in (fr, to):
            sx, sy = board_sq_to_screen(board_rect, sq, flip)
            rr = pygame.Rect(sx - 0.45 * cell_w, sy - 0.45 * cell_h, 0.9 * cell_w, 0.9 * cell_h)
            pygame.draw.rect(screen, (230, 230, 160), rr, 0)

    # Highlight selected square and legal targets
    if ui.selected_sq is not None:
        sx, sy = board_sq_to_screen(board_rect, ui.selected_sq, flip)
        rr = pygame.Rect(sx - 0.45 * cell_w, sy - 0.45 * cell_h, 0.9 * cell_w, 0.9 * cell_h)
        pygame.draw.rect(screen, (180, 230, 180), rr, 0)

    if ui.legal_dests:
        for dsq in ui.legal_dests:
            sx, sy = board_sq_to_screen(board_rect, dsq, flip)
            pygame.draw.circle(screen, (120, 180, 120), (int(sx), int(sy)), int(min(cell_w, cell_h) * 0.10), 0)

    # Draw pieces
    for sq in range(90):
        p = scalar_int(board.piece_at(int(sq)))
        if p == 0:
            continue

        sx, sy = board_sq_to_screen(board_rect, sq, flip)
        radius = int(min(cell_w, cell_h) * 0.35)
        pygame.draw.circle(screen, (250, 250, 245), (int(sx), int(sy)), radius, 0)
        pygame.draw.circle(screen, (40, 40, 40), (int(sx), int(sy)), radius, 2)

        label = piece_label(p, prefer_cn)
        color = (200, 30, 30) if p > 0 else (20, 20, 20)
        surf = font_piece.render(label, True, color)
        if prefer_cn and surf.get_bounding_rect().width == 0:
            surf = font_piece.render(piece_label(p, False), True, color)
        rect = surf.get_rect(center=(int(sx), int(sy)))
        screen.blit(surf, rect)


def draw_ui_text(screen, font_ui, x: int, y: int, lines: List[str]):
    yy = y
    for t in lines:
        surf = font_ui.render(t, True, (20, 20, 20))
        screen.blit(surf, (x, yy))
        yy += 30


# ----------------------------
# 9) Main
# ----------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=str, default="Chessv11_cpp_hist8_115_mps_fp16")
    parser.add_argument("--weights", type=str, default="auto", help="Path to weights, or `auto` to prefer the latest valid snapshot")
    parser.add_argument("--human", type=str, default="red", choices=["red", "black", "both"], help="red/black=human-vs-ai, both=local two-player")
    parser.add_argument("--ai_device", type=str, default="cpu", help="cpu/mps/cuda")
    parser.add_argument("--ai_fp16", action="store_true")

    parser.add_argument("--c_puct", type=float, default=None)
    parser.add_argument("--sim_level", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--time_manager", action="store_true", help="Adapt sims based on remaining clock time")
    parser.add_argument(
        "--opening_cpu_cap",
        type=int,
        default=1600,
        help="Cap sims on CPU for the first 2 plies to avoid very long opening searches; 0 disables it",
    )
    args = parser.parse_args()

    # xqcpp
    xqcpp = ensure_xqcpp_loaded()

    # engine module
    if args.engine == "none" or args.engine == "":
        engine_mod = None
        NetCls = None
        cfg = {}
    else:
        engine_mod = load_engine_module(args.engine)
        NetCls = pick_net_class(engine_mod)
        cfg = {}
        try:
            if hasattr(engine_mod, "CFG") and isinstance(getattr(engine_mod, "CFG"), dict):
                cfg = dict(getattr(engine_mod, "CFG"))
        except Exception:
            cfg = {}

    if args.c_puct is None:
        args.c_puct = float(cfg.get("C_PUCT", 1.5))

    # ---- Network ----
    net_lock = threading.Lock()
    if NetCls is not None:
        net = NetCls(cfg).cpu()
    else:
        raise RuntimeError("Engine/net is required for human-vs-AI mode.")

    resolved_weights = resolve_weights_path(args.weights, cfg)

    def load_weights_into_net(path: str) -> bool:
        if not os.path.isfile(path):
            return False
        obj = torch.load(path, map_location="cpu")
        sd = extract_state_dict(obj)
        sd = normalize_state_dict_keys(sd)

        bad = count_nonfinite_state(sd)
        if bad > 0:
            print(f"[WARN] non-finite weights detected: {path} bad={bad}")
            return False

        try:
            with net_lock:
                net.load_state_dict(sd, strict=True)
                net.eval()
            return True
        except RuntimeError as e:
            print(f"[WARN] strict load failed for {path}: {e}")
            return False

    if load_weights_into_net(resolved_weights):
        print(f"[LOAD] weights: {resolved_weights}")
    else:
        raise RuntimeError(f"Failed to load valid weights: {resolved_weights}")

    dev = torch.device(args.ai_device)
    eval_net = EvalNetWrapper(net, dev, bool(args.ai_fp16), lock=net_lock)

    # ---- Mode ----
    mode = args.human  # 'red'/'black'/'both'

    def human_side_code() -> Optional[int]:
        if mode == "red":
            return 0
        if mode == "black":
            return 1
        return None

    def ai_side_code() -> Optional[int]:
        if mode == "red":
            return 1
        if mode == "black":
            return 0
        return None

    enable_ai = mode in ("red", "black")

    # ---- Shared sims/temperature values read by AI worker ----
    SIM_LEVELS = {1: 3200, 2: 6400, 3: 12800}
    sim_level = int(args.sim_level)
    sims_ref: Dict[str, int] = {"sims": int(SIM_LEVELS[sim_level])}

    # Exploration toggle on key E
    AI_TEMPERATURE = 0.0
    AI_EXPLORE_TEMPERATURE = 0.15
    temp_ref: Dict[str, float] = {"t_move": float(AI_TEMPERATURE)}

    HARD_MAX_SIMS = 12800
    HARD_MIN_SIMS = 3200
    WINDOW_W = 1240
    WINDOW_H = 840
    BOARD_RECT = (60, 60, 620, 690)  # x, y, w, h
    UI_X = 730
    UI_Y = 60

    # pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Xiangqi V11.2 (xqcpp)")

    prefer_cn = True
    f_piece, f_ui = init_fonts(prefer_cn)

    board_rect = BOARD_RECT

    # Keep the human side at the bottom in human-vs-AI mode.
    view_flip = (mode == "black")

    board = xqcpp.Board()
    move_hist: List[int] = []
    ui = UIState()

    clocks: Dict[str, PlayerClock] = {"red": PlayerClock(), "black": PlayerClock()}
    for c in clocks.values():
        c.reset()

    def current_ai_sims() -> int:
        aicode = ai_side_code()
        if aicode is None:
            return int(sims_ref.get("sims", SIM_LEVELS[sim_level]))
        ak = "red" if aicode == 0 else "black"
        sims = dynamic_sims(
            SIM_LEVELS[sim_level],
            clocks[ak],
            enable_time_manager=bool(args.time_manager),
            hard_min=HARD_MIN_SIMS,
            hard_max=HARD_MAX_SIMS,
        )
        if args.ai_device == "cpu":
            cap = int(args.opening_cpu_cap)
            if cap > 0 and len(move_hist) < 2:
                sims = min(sims, cap)
        return int(sims)

    # AI thread queues
    req_q: "queue.Queue[Tuple[int,str,threading.Event]]" = queue.Queue()
    res_q: "queue.Queue[Tuple[int,int,dict]]" = queue.Queue()
    req_id = 0
    pending_req: Optional[Tuple[int, str]] = None
    pending_cancel: Optional[threading.Event] = None
    last_ai_info: Dict = {}

    ai_worker = AIWorker(
        xqcpp,
        req_q,
        res_q,
        eval_net,
        sims_ref=sims_ref,
        temp_ref=temp_ref,
        c_puct=float(args.c_puct),
        eval_batch=int(cfg.get("EVAL_BATCH_SIZE", 32)),
        seed0=777,
    )
    if enable_ai:
        ai_worker.start()

    def cancel_pending_ai():
        nonlocal pending_req, pending_cancel, req_id, last_ai_info
        if pending_cancel is not None:
            pending_cancel.set()
            pending_cancel = None
        pending_req = None
        last_ai_info = {}
        # Invalidate old requests by incrementing req_id and clearing queues.
        req_id += 1
        while True:
            try:
                _rid, _fen, ev = req_q.get_nowait()
                ev.set()
            except queue.Empty:
                break
        while True:
            try:
                res_q.get_nowait()
            except queue.Empty:
                break

    def request_ai_move():
        nonlocal req_id, pending_req, pending_cancel
        if not enable_ai:
            return
        if ui.game_over:
            return
        if side_to_move(board) != ai_side_code():
            return
        sims_ref["sims"] = current_ai_sims()
        req_id += 1
        fen = board.fen()
        pending_req = (req_id, fen)
        pending_cancel = threading.Event()
        req_q.put((req_id, fen, pending_cancel))

    def apply_ai_result(req_id_done: int, best_mv: int, info: dict):
        nonlocal pending_req, pending_cancel, last_ai_info
        if pending_req is None:
            return
        if req_id_done != pending_req[0]:
            return
        if board.fen() != pending_req[1]:
            return
        pending_req = None
        pending_cancel = None
        last_ai_info = info or {}
        if "error" in last_ai_info:
            ui.status_text = f"AI error: {last_ai_info['error']}"
            print(f"[AI] request failed: {last_ai_info['error']}", flush=True)
            return
        if best_mv < 0:
            ui.game_over = True
            res = -1 if side_to_move(board) == 0 else 1
            ui.result_text = outcome_text_from_red_view(res)
            return
        board.push(int(best_mv))
        move_hist.append(int(best_mv))
        ui.last_move = int(best_mv)
        ui.selected_sq = None
        ui.legal_dests = None

    def check_game_over_after_move():
        if len(board.legal_moves()) == 0:
            ui.game_over = True
            res = -1 if side_to_move(board) == 0 else 1
            ui.result_text = outcome_text_from_red_view(res)

    def reset_game(new_board: bool = True):
        nonlocal board, ui, last_ai_info, view_flip
        cancel_pending_ai()
        if new_board:
            board = xqcpp.Board()
            move_hist.clear()
        ui = UIState(debug=ui.debug)
        ui.last_move = move_hist[-1] if move_hist else None
        ui.game_over = False
        ui.result_text = ""
        ui.status_text = ""
        last_ai_info = {}
        for c in clocks.values():
            c.reset()
        # In human-vs-AI mode, keep human side at the bottom.
        view_flip = (mode == "black")
        # turn start
        clocks["red"].on_turn_start()
        clocks["black"].on_turn_start()
        request_ai_move()

    # Initialize turn timers
    last_turn = side_to_move(board)
    clocks["red"].on_turn_start()
    clocks["black"].on_turn_start()

    # Start: request AI move if AI moves first
    request_ai_move()

    clock = pygame.time.Clock()
    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        # poll ai result
        try:
            while True:
                rid, mv, info = res_q.get_nowait()
                apply_ai_result(rid, mv, info)
                check_game_over_after_move()
                # Turn changed: reset per-move byoyomi timer.
                cur_turn = side_to_move(board)
                if cur_turn != last_turn:
                    last_turn = cur_turn
                    if cur_turn == 0:
                        clocks["red"].on_turn_start()
                    else:
                        clocks["black"].on_turn_start()
                request_ai_move()
        except queue.Empty:
            pass

        # Tick active side clock while game is running.
        if not ui.game_over:
            cur = side_to_move(board)
            side_key = "red" if cur == 0 else "black"
            if clocks[side_key].tick(dt):
                # Flag loss on time
                ui.game_over = True
                ui.result_text = "Black wins (Red flagged)" if side_key == "red" else "Red wins (Black flagged)"
                cancel_pending_ai()

        # Time manager for AI side in human-vs-AI mode
        if enable_ai and not ui.game_over:
            aicode = ai_side_code()
            if aicode is not None:
                sims_ref["sims"] = current_ai_sims()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_d:
                    ui.debug = not ui.debug

                elif event.key == pygame.K_n:
                    reset_game(new_board=True)

                elif event.key == pygame.K_r:
                    # Reload weights (without resetting board)
                    cancel_pending_ai()
                    resolved_weights = resolve_weights_path(args.weights, cfg)
                    ok = load_weights_into_net(resolved_weights)
                    ui.status_text = (
                        f"Weights reloaded: {os.path.basename(resolved_weights)}"
                        if ok else "Weight file missing or failed to load"
                    )
                    request_ai_move()

                elif event.key in (pygame.K_u, pygame.K_BACKSPACE):
                    steps = 2 if enable_ai else 1
                    for _ in range(steps):
                        if move_hist:
                            try:
                                board.pop()
                            except Exception:
                                pass
                            move_hist.pop()
                    ui.selected_sq = None
                    ui.legal_dests = None
                    ui.last_move = move_hist[-1] if move_hist else None
                    ui.game_over = False
                    ui.result_text = ""
                    ui.status_text = f"Undo applied ({steps} ply)"
                    cancel_pending_ai()
                    # Reset byoyomi at turn start
                    cur_turn = side_to_move(board)
                    last_turn = cur_turn
                    clocks["red"].on_turn_start()
                    clocks["black"].on_turn_start()
                    request_ai_move()

                elif event.key == pygame.K_s:
                    # Swap human side in human-vs-AI mode and flip view
                    if mode in ("red", "black"):
                        mode = "black" if mode == "red" else "red"
                        enable_ai = True
                        view_flip = (mode == "black")
                        ui.status_text = "Side swapped"
                        cancel_pending_ai()
                        request_ai_move()

                elif event.key == pygame.K_e:
                    # Exploration temperature toggle
                    if abs(temp_ref["t_move"]) < 1e-6:
                        temp_ref["t_move"] = float(AI_EXPLORE_TEMPERATURE)
                        ui.status_text = f"Explore mode ON (t={AI_EXPLORE_TEMPERATURE})"
                    else:
                        temp_ref["t_move"] = float(AI_TEMPERATURE)
                        ui.status_text = "Explore mode OFF"
                    cancel_pending_ai()
                    request_ai_move()

                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    sim_level = int(event.unicode)
                    sims_ref["sims"] = int(SIM_LEVELS[sim_level])
                    ui.status_text = f"AI level: {sim_level} (sims={SIM_LEVELS[sim_level]})"
                    cancel_pending_ai()
                    request_ai_move()

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if ui.game_over:
                    continue

                hs = human_side_code()
                if hs is not None and side_to_move(board) != hs:
                    continue

                sq = mouse_to_square(board_rect, *event.pos, flip=view_flip)
                if sq is None:
                    continue

                piece = scalar_int(board.piece_at(sq))
                stm = side_to_move(board)

                # If nothing selected, only own-side pieces can be selected.
                if ui.selected_sq is None:
                    if piece == 0:
                        continue
                    if (stm == 0 and piece < 0) or (stm == 1 and piece > 0):
                        continue
                    ui.selected_sq = sq
                    ui.legal_dests = legal_dest_squares(board, sq)
                    continue

                # If selected and clicked own piece, reselection is allowed.
                if piece != 0:
                    if (stm == 0 and piece > 0) or (stm == 1 and piece < 0):
                        ui.selected_sq = sq
                        ui.legal_dests = legal_dest_squares(board, sq)
                        continue

                mv = try_push_move(board, ui.selected_sq, sq)
                if mv is not None:
                    move_hist.append(int(mv))
                    ui.last_move = int(mv)
                    ui.selected_sq = None
                    ui.legal_dests = None
                    ui.status_text = ""
                    last_ai_info = {}

                    check_game_over_after_move()

                    # Turn changed: reset byoyomi at turn start.
                    cur_turn = side_to_move(board)
                    if cur_turn != last_turn:
                        last_turn = cur_turn
                        if cur_turn == 0:
                            clocks["red"].on_turn_start()
                        else:
                            clocks["black"].on_turn_start()

                    cancel_pending_ai()
                    request_ai_move()
                else:
                    ui.selected_sq = None
                    ui.legal_dests = None

        # draw
        screen.fill((250, 248, 239))
        draw_board(screen, board_rect)
        draw_pieces(screen, board_rect, board, f_piece, prefer_cn, ui, flip=view_flip)

        stm = side_to_move(board)
        turn_text = "Red to move" if stm == 0 else "Black to move"

        mode_txt = "Two-player" if mode == "both" else "Human-vs-AI"
        human_txt = "Red" if mode == "red" else ("Black" if mode == "black" else "Both")

        base_sims = SIM_LEVELS[sim_level]
        show_sims = sims_ref.get("sims", base_sims)

        lines = [
            f"Turn: {turn_text}",
            f"Mode: {mode_txt}  Human: {human_txt}",
            f"Weights: {resolved_weights}",
            f"Device: {args.ai_device}",
            f"AI sims: {show_sims} (level {sim_level}: {base_sims})",
            f"T:{temp_ref['t_move']:.2f}",
            f"Red: {clocks['red'].fmt()}",
            f"Black: {clocks['black'].fmt()}",
            "U:Undo  N:New Game  R:Reload Weights",
            "S:Swap Side  1/2/3:Strength  E:Explore",
            "D:Debug  ESC:Quit",
        ]

        if ui.status_text:
            lines.append(f"Hint: {ui.status_text}")

        if ui.game_over:
            lines.append(f"Result: {ui.result_text}")

        if enable_ai and pending_req is not None:
            lines.append("AlphaXiang is thinking...")

        if ui.debug and enable_ai:
            if "error" in last_ai_info:
                lines.append(f"AI error: {last_ai_info['error']}")
            else:
                if last_ai_info:
                    lines.append("Last AI search:")
                rv = last_ai_info.get("root_v", None)
                if rv is not None:
                    lines.append(f"root_v: {rv:+.3f}")
                used_sims = last_ai_info.get("sims", None)
                if used_sims is not None:
                    lines.append(f"used_sims: {int(used_sims)}")
                root_children = last_ai_info.get("root_children", None)
                if root_children is not None:
                    lines.append(f"root_children: {int(root_children)}")
                expanded_nodes = last_ai_info.get("expanded_nodes", None)
                if expanded_nodes is not None:
                    lines.append(f"expanded_nodes: {int(expanded_nodes)}")
                nodes_total = last_ai_info.get("nodes_total", None)
                if nodes_total is not None:
                    lines.append(f"nodes_total: {int(nodes_total)}")
                nn_eval_batches = last_ai_info.get("nn_eval_batches", None)
                if nn_eval_batches is not None:
                    lines.append(f"nn_eval_batches: {int(nn_eval_batches)}")
                nn_eval_states = last_ai_info.get("nn_eval_states", None)
                if nn_eval_states is not None:
                    lines.append(f"nn_eval_states: {int(nn_eval_states)}")
                root_child_visits = last_ai_info.get("root_child_visits", None)
                if root_child_visits is not None:
                    lines.append(f"root_child_visits: {int(root_child_visits)}")
                elapsed_ms = last_ai_info.get("elapsed_ms", None)
                if elapsed_ms is not None:
                    lines.append(f"search_ms: {float(elapsed_ms):.1f}")
                topk = last_ai_info.get("topk", [])
                if topk:
                    lines.append("TopK:")
                    for mv_str, p in topk:
                        lines.append(f"  {mv_str}  {p:.3f}")

        draw_ui_text(screen, f_ui, UI_X, UI_Y, lines)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
