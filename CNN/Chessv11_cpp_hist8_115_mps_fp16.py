# -*- coding: utf-8 -*-
"""
Xiangqi AlphaZero-style trainer (V11, C++ core)
===============================================

Current architecture:
- C++ extension (`xqcpp`) for board rules + MCTS search
- PyTorch ResNet trunk with policy/value/material heads
- Sparse policy targets (`idxs + probs`) instead of dense 8100 labels
- Multiprocess self-play with periodic weight reload
- Auto-resume checkpointing (`latest_ckpt.pt`) + latest weights (`best.pth`)
- Arena gating (`current` vs `best`) before best-model promotion
- LR scheduler + runtime LR recovery logic
"""

import os
import re
import time
import math
import random
import datetime  # Power-price pause helper
import queue
import threading
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
from contextlib import nullcontext

# ----------------------------
# 0) C++ extension
# ----------------------------

def load_xqcpp():
    """Build/load the xqcpp extension from local C++ source."""
    from torch.utils.cpp_extension import load
    this_dir = os.path.dirname(os.path.abspath(__file__))
    # Prefer the V11 source file; fall back to legacy name if present.
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

    # O3 + c++17
    # On Windows, the extension uses MSVC toolchain flags under the hood.
    mod = load(
        name="xqcpp",
        sources=[src],
        extra_cflags=["-O3", "-std=c++17", "-march=native"],
        with_cuda=False,
        verbose=True,
    )
    return mod


# ----------------------------
# 1) CPU thread controls
# ----------------------------

def set_cpu_threads(n: int):
    """
    Set process-local CPU thread counts.

    Typical values:
    - self-play worker: 1-2
    - trainer process: 4-8
    """
    n = max(1, int(n))
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    torch.set_num_threads(n)
    # Keep inter-op threads low to reduce contention.
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def make_grad_scaler(enabled: bool):
    """
    Build a GradScaler across old and new PyTorch AMP APIs.
    """
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


# ----------------------------
# 2) Configuration
# ----------------------------

CFG = {
    # Process settings
    "NUM_WORKERS": 10,
    "CPU_THREADS_SELFPLAY": 1,
    "CPU_THREADS_TRAIN": 4,
    "USE_TORCH_COMPILE": True,

    # Draw and termination settings
    "MAX_STEPS": 360,
    "NO_CAPTURE_LIMIT": 80,
    "REPEAT_LIMIT": 4,
    "REPEAT_MIN_PLY": 20,       # Only apply repetition rule after this ply.

    # MCTS
    "MCTS_SIMS": 200,
    "C_PUCT": 1.5,
    "DIRICHLET_ALPHA": 0.3,
    "DIRICHLET_EPS": 0.25,
    "ROOT_NOISE_END_PLY": 60,
    "EVAL_BATCH_SIZE": 32,     # NN eval batch size inside C++ MCTS.
    "TEMP_TARGET": 1.0,        # Target temperature used for training policy.
    # Move temperature schedule by ply.
    # (ply<=PLY1 -> T1), (ply<=PLY2 -> T2), else -> T3
    "TEMP_MOVE_SCHEDULE": [(16, 1.0), (60, 0.7), (10**9, 0.2)],

    # Resign / claim win MCTS root value forward
    "ENABLE_RESIGN": True,
    "RESIGN_START": 140,     # Enable resign check when ply >= this value.
    "RESIGN_V": -0.98,       # Resign if root value (side-to-move view) <= this.
    "RESIGN_CONSEC": 3,

    "ENABLE_CLAIM_WIN": False,
    "CLAIM_START": 140,
    "CLAIM_V": 0.98,
    "CLAIM_CONSEC": 3,

    # Draw sample handling:
    # - "weighted": keep draws with reduced sample weight + shaped value target
    # - "discard": drop draw samples from training
    "TRAIN_DRAW_MODE": "weighted",
    "DRAW_SAMPLE_W": 0.25,
    "DRAW_VALUE_SCALE": 0.15,

    # Replay buffer / optimization
    "BUFFER_CAPACITY": 270000,
    "BATCH_SIZE": 1024,
    "TRAIN_STEPS_PER_GAME": 2,
    "LR": 2e-4,
    "WEIGHT_DECAY": 1e-4,

    # Max number of sparse policy moves kept per sample.
    "MAX_POLICY_MOVES": 128,

    # Input layout: history frames and channel count from C++ Board.to_tensor().
    "INPUT_HISTORY_FRAMES": 8,
    "INPUT_CHANNELS": 115,

    # Device / AMP settings
    "TRAIN_DEVICE": "cuda",      # "cpu" / "mps"
    "USE_AMP": True,            # mixed precision autocast
    "AMP_DTYPE": "float16",     # "float16" / "bfloat16"
    "ALLOW_MPS_TRAIN_AMP": False,  # MPS FP16 training can be unstable on long runs.
    "SELFPLAY_DEVICE": "cuda",   # Prefer CPU for many workers; MPS/CUDA for fewer workers.
    "SELFPLAY_FP16": True,      # FP16 only applies on MPS/CUDA.

    # Checkpoint / weight files
    "SAVE_LATEST": True,
    "AUTO_RESUME_LATEST": True,  # Resume from checkpoint/weights at startup.
    "LATEST_NAME": "best.pth",
    # V11: full checkpoint (model+optimizer+scheduler+step)
    "CHECKPOINT_NAME": "latest_ckpt.pt",
    # V11: Arena best model
    "BEST_NAME": "best.pth",
    "WORKER_RELOAD_EVERY_GAMES": 20,

    # Model size
    "MODEL_CHANNELS": 128,
    "MODEL_RES_BLOCKS": 20,

    # Save behavior
    "SAVE_DIR": "v11_runs",
    "SAVE_EVERY_TRAIN_STEPS": 8000,

    # --- V11: LR scheduler (loss driven) ---
    "LR_SCHED_ENABLE": False,
    "LR_SCHED_EVERY": 200,
    "LR_PLATEAU_FACTOR": 0.67,
    "LR_PLATEAU_PATIENCE": 16,
    "LR_PLATEAU_THRESHOLD": 5e-4,
    "LR_MIN": 2e-5,
    "LR_COOLDOWN": 0,
    # If resumed LR is too low, lift it once and rebuild scheduler.
    "LR_RECOVER_IF_BELOW": 2.5e-5,
    "LR_RECOVER_TO": 8e-5,
    # Runtime LR recovery when loss EMA stalls at LR floor.
    "LR_RUNTIME_RECOVER_ENABLE": True,
    "LR_RUNTIME_RECOVER_CHECK_EVERY": 200,      # Check interval in train steps.
    "LR_RUNTIME_RECOVER_PATIENCE_STEPS": 5000,  # Must stay at floor this long.
    "LR_RUNTIME_RECOVER_COOLDOWN_STEPS": 12000, # Cooldown between recoveries.
    "LR_RUNTIME_RECOVER_STALL_EPS": 2e-3,       # Loss-EMA improvement threshold.
    "LR_RUNTIME_RECOVER_TO": 8e-5,
    "LR_RUNTIME_RECOVER_MAX": 1.2e-4,
    "LR_RUNTIME_RECOVER_FLOOR_MARGIN": 1.02,    # Treat LR as "at floor" under this margin.
    # Non-finite loss/output recovery
    "NONFINITE_RECOVER_STREAK": 8,
    "NONFINITE_RECOVER_LR_SCALE": 0.5,

    # --- V11: Arena (best model gating) ---
    "ARENA_ENABLE": True,
    "ARENA_EVERY_TRAIN_STEPS": 10000,
    "ARENA_GAMES": 100,
    "ARENA_SIMS": 800,
    "ARENA_EVAL_BATCH_SIZE": 64,
    "ARENA_WINRATE_THRESHOLD": 0.55,
    "ARENA_SEED": 2026011530,
    # Arena anti-draw settings (only for model-vs-model gating)
    "ARENA_MAX_STEPS": 480,
    "ARENA_NO_CAPTURE_LIMIT": 120,
    "ARENA_REPEAT_LIMIT": 6,
    "ARENA_REPEAT_MIN_PLY": 30,
    "ARENA_TEMP_MOVE": 0.0,
    "ARENA_TEMP_UNTIL_PLY": 0,
    "ARENA_ADD_ROOT_NOISE": False,
    "ARENA_ROOT_NOISE_END_PLY": 0,
    "ARENA_DIRICHLET_ALPHA": 0.30,
    "ARENA_DIRICHLET_EPS": 0.10,

    # selfplay uses best model (AlphaZero style)
    "SELFPLAY_USE_BEST": True,
}

# ----------------------------
# 2.5) Device / AMP / Eval wrapper
# ----------------------------

def pick_device(name: str) -> torch.device:
    name = (name or "cpu").lower()
    if name == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def resolve_amp_dtype(cfg: dict) -> torch.dtype:
    name = str(cfg.get("AMP_DTYPE", "float16")).strip().lower()
    if name in ("float16", "fp16", "half"):
        return torch.float16
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16
    raise ValueError(f"unsupported AMP_DTYPE={cfg.get('AMP_DTYPE')!r}")

def amp_context(device: torch.device, enabled: bool, amp_dtype: torch.dtype):
    """Return autocast context for CUDA/MPS AMP, else a no-op context."""
    if not enabled:
        return nullcontext()
    try:
        if device.type in ("cuda", "mps"):
            return torch.autocast(device_type=device.type, dtype=amp_dtype)
    except Exception:
        pass
    return nullcontext()

class NonFiniteEvalError(RuntimeError):
    pass

class EvalNetWrapper:
    """
    Adapter for C++ MCTS:
    - input: CPU float32 states from C++
    - forward: on target device (CPU/MPS/CUDA)
    - output: CPU float32 tensors for C++ (`p_logits`, `wdl_logits`, `m_pred`)
    """
    def __init__(self, net: nn.Module, device: torch.device, use_amp: bool, amp_dtype: torch.dtype):
        self.device = device
        self.use_amp = bool(use_amp) and (device.type in ("cuda", "mps"))
        self.amp_dtype = amp_dtype
        self.net = net.to(device)
        self.net.eval()

    def __call__(self, x_cpu: torch.Tensor):
        # x_cpu: CPU float32 tensor, [B,C,10,9]
        x = x_cpu.to(self.device, dtype=torch.float32)
        with torch.inference_mode():
            with amp_context(self.device, self.use_amp, self.amp_dtype):
                out = self.net(x)
        if any(not torch.isfinite(t).all() for t in out):
            raise NonFiniteEvalError(f"non-finite eval output on device={self.device}")
        # C++ side expects CPU tensors
        return tuple(t.float().cpu() for t in out)

# ----------------------------
# 3) Move encoding / symmetry utilities
# ----------------------------

FILE_CHARS = "abcdefghi"

def square_to_iccs(square: int) -> str:
    y, x = divmod(square, 9)
    return f"{FILE_CHARS[x]}{9 - y}"

def action_to_iccs(action_id: int) -> str:
    from_sq = action_id // 90
    to_sq = action_id % 90
    return square_to_iccs(from_sq) + square_to_iccs(to_sq)

def create_flip_map() -> np.ndarray:
    """
    Build left-right mirror map for action ids.

    action_id = from_sq * 90 + to_sq
    mirror: x -> 8 - x
    """
    flip = np.zeros(8100, dtype=np.int64)
    for a in range(8100):
        f = a // 90
        t = a % 90
        fy, fx = divmod(f, 9)
        ty, tx = divmod(t, 9)
        ff = fy * 9 + (8 - fx)
        tt = ty * 9 + (8 - tx)
        flip[a] = ff * 90 + tt
    return flip

FLIP_MAP = create_flip_map()

# Use numpy flip + contiguous copy so torch.from_numpy() stays safe.
def flip_state_tensor(state: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.flip(state, axis=2))

def flip_sparse_policy(idxs: np.ndarray, probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    fidx = FLIP_MAP[idxs]
    return fidx.astype(np.int64), probs.copy()

# ----------------------------
# 4) Network
# ----------------------------

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        r = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + r)

class XiangqiNetV10(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        c = int(cfg["MODEL_CHANNELS"])
        n = int(cfg["MODEL_RES_BLOCKS"])
        self.stem = nn.Sequential(
            nn.Conv2d(int(cfg.get("INPUT_CHANNELS", 115)), c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(c) for _ in range(n)])

        # policy head: 8100
        self.p_conv = nn.Conv2d(c, 2, 1, bias=False)
        self.p_bn = nn.BatchNorm2d(2)
        self.p_fc = nn.Linear(2 * 10 * 9, 8100)  # Compact policy head for lower parameter count.

        # value head (WDL)
        self.v_conv = nn.Conv2d(c, 32, 1, bias=False)
        self.v_bn = nn.BatchNorm2d(32)
        self.v_fc1 = nn.Linear(32 * 10 * 9, 128)
        self.v_fc2 = nn.Linear(128, 3)

        # material head (optional aux)
        self.m_conv = nn.Conv2d(c, 16, 1, bias=False)
        self.m_bn = nn.BatchNorm2d(16)
        self.m_fc1 = nn.Linear(16 * 10 * 9, 64)
        self.m_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.blocks(self.stem(x))

        # policy
        p = F.relu(self.p_bn(self.p_conv(x)))
        p = p.flatten(1)
        p_logits = self.p_fc(p)

        # wdl
        v = F.relu(self.v_bn(self.v_conv(x)))
        v = v.flatten(1)
        v = F.relu(self.v_fc1(v))
        wdl_logits = self.v_fc2(v)

        # material
        m = F.relu(self.m_bn(self.m_conv(x)))
        m = m.flatten(1)
        m = F.relu(self.m_fc1(m))
        m_pred = torch.tanh(self.m_fc2(m))  # [-1,1]

        return p_logits, wdl_logits, m_pred

def wdl_logits_to_value(wdl_logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(wdl_logits, dim=1)
    return probs[:, 0] - probs[:, 2]  # win - loss

def scalar_value_to_wdl_target(z: torch.Tensor) -> torch.Tensor:
    # z: +1.0 win, -1.0 loss, 0.0 draw
    p_win = F.relu(z)
    p_loss = F.relu(-z)
    p_draw = torch.clamp(1.0 - p_win - p_loss, 0.0, 1.0)
    tgt = torch.stack([p_win, p_draw, p_loss], dim=1)
    return tgt

# ----------------------------
# 5) Replay buffer
# ----------------------------

@dataclass
class Sample:
    state: np.ndarray  # Stored as numpy instead of torch.Tensor
    idxs: np.ndarray
    probs: np.ndarray
    z: float
    w: float                      # sample weight

class ReplayBuffer:
    def __init__(self, capacity: int, max_policy_moves: int):
        self.capacity = int(capacity)
        self.max_policy_moves = int(max_policy_moves)
        self.data: List[Sample] = []
        self.pos = 0
        self.lock = threading.Lock()

    def __len__(self):
        return len(self.data)

    def push(self, s: Sample):
        with self.lock:
            if len(self.data) < self.capacity:
                self.data.append(s)
            else:
                self.data[self.pos] = s
                self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        with self.lock:
            bs = min(int(batch_size), len(self.data))
            batch = random.sample(self.data, bs)

        states_u8 = torch.from_numpy(np.stack([x.state for x in batch], axis=0))  # [B,115,10,9] uint8

        # pad sparse policy
        maxm = self.max_policy_moves
        idx_mat = np.full((bs, maxm), -1, dtype=np.int64)
        prob_mat = np.zeros((bs, maxm), dtype=np.float32)
        for i, x in enumerate(batch):
            k = min(len(x.idxs), maxm)
            if k > 0:
                raw_idx = np.asarray(x.idxs[:k], dtype=np.int64)
                raw_prob = np.asarray(x.probs[:k], dtype=np.float32)

                # Drop invalid actions/probabilities to avoid NaN/Inf propagation.
                raw_prob = np.nan_to_num(raw_prob, nan=0.0, posinf=0.0, neginf=0.0)
                valid = (raw_idx >= 0) & (raw_idx < 8100) & (raw_prob > 0.0)
                raw_idx = raw_idx[valid]
                raw_prob = raw_prob[valid]
                k2 = min(len(raw_idx), maxm)
                if k2 <= 0:
                    continue

                idx_mat[i, :k2] = raw_idx[:k2]
                prob_mat[i, :k2] = raw_prob[:k2]

                # Renormalize per-row for safety.
                ssum = float(prob_mat[i, :k2].sum())
                if ssum > 1e-12:
                    prob_mat[i, :k2] /= ssum

        z = torch.tensor([x.z for x in batch], dtype=torch.float32)
        w = torch.tensor([x.w for x in batch], dtype=torch.float32)

        return states_u8, torch.from_numpy(idx_mat), torch.from_numpy(prob_mat), z, w

# ----------------------------
# 6) Draw shaping / value target helpers
# ----------------------------

def outcome_red_to_value_stm(z_red: float, stm_is_black: bool) -> float:
    # Convert red-view outcome to side-to-move value.
    return -z_red if stm_is_black else z_red

def draw_value_shaping(material_red: float, scale: float) -> float:
    # Shape draw target by material advantage into [-scale, +scale].
    return float(math.tanh(material_red / 8.0) * scale)

# ----------------------------
# 7) Temperature schedule
# ----------------------------

def get_temp_from_schedule(ply: int, schedule: List[Tuple[int, float]]) -> float:
    for up_to, t in schedule:
        if ply <= up_to:
            return float(t)
    return float(schedule[-1][1])

# ----------------------------
# 8) Self-play worker (C++ MCTS)
# ----------------------------

def selfplay_worker(rank: int, cfg: dict, model_state_dict: dict, out_q: mp.Queue, stop_flag: mp.Event):
    # CUDA perf toggles for self-play workers.
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    set_cpu_threads(cfg["CPU_THREADS_SELFPLAY"])

    # Load C++ extension inside each worker process.
    xqcpp = load_xqcpp()

    net = XiangqiNetV10(cfg).cpu()
    net.load_state_dict(model_state_dict, strict=True)
    net.eval()

    # Run NN forward on selected self-play device.
    sp_dev = pick_device(cfg.get("SELFPLAY_DEVICE", "cpu"))
    amp_dtype = resolve_amp_dtype(cfg)
    eval_net = EvalNetWrapper(net, sp_dev, bool(cfg.get("SELFPLAY_FP16", True)), amp_dtype)

    # Workers periodically reload updated weights from disk.
    weights_path = cfg.get("_SELFPLAY_PATH") or cfg.get("_LATEST_PATH")
    reload_every = int(cfg.get("WORKER_RELOAD_EVERY_GAMES", 1))
    last_mtime: Optional[float] = None
    games_local = 0

    # Worker-local RNG for seeds/noise.
    rng = np.random.default_rng(seed=((rank + 1) * 1234567) ^ (os.getpid() << 16) ^ int(time.time() * 1e6))


    def _maybe_reload():
        nonlocal last_mtime
        if (not weights_path) or (not os.path.isfile(weights_path)):
            return
        try:
            mtime = os.path.getmtime(weights_path)
        except OSError:
            return
        if (last_mtime is None) or (mtime > last_mtime + 1e-9):
            obj = torch.load(weights_path, map_location="cpu")
            sd = extract_model_state_dict(obj)
            if sd is None:
                return
            bad, first = nonfinite_count_in_state_dict(sd)
            if bad > 0:
                print(
                    f"[SELFPLAY-{rank}] skip non-finite reload: {weights_path} "
                    f"bad={bad} first_tensor={first}",
                    flush=True,
                )
                return
            net.load_state_dict(sd, strict=True)
            last_mtime = mtime

    # Main self-play loop.
    while not stop_flag.is_set():
        games_local += 1
        # Unique seed per game.
        game_seed = int(rng.integers(0, 2**31 - 1))


        t_game0 = time.time()
        if reload_every > 0 and (games_local % reload_every == 0):
            _maybe_reload()

        board = xqcpp.Board()
        pos_count: Dict[int, int] = {}
        no_cap = 0
        ply = 0

        tag = "UNKNOWN"
        # resign / claim counters
        bad_cnt = 0
        good_cnt = 0

        game_mem = []  # list of (state[115,10,9] uint8, idxs, probs, stm_is_black)
        moves_iccs = []
        discard_game = False

        while True:
            if ply >= cfg["MAX_STEPS"]:
                # draw by max steps
                outcome = 0.0
                reason = "draw_maxsteps"
                tag = "DRAW_LIMIT"
                break

            # repetition
            key = int(board.key())
            pos_count[key] = pos_count.get(key, 0) + 1
            rep_min_ply = int(cfg.get("REPEAT_MIN_PLY", 20))
            if ply >= rep_min_ply and pos_count[key] >= int(cfg.get("REPEAT_LIMIT", 4)):
                outcome = 0.0
                reason = "draw_repeat"
                tag = "DRAW_REPEAT"
                break

            if no_cap >= cfg["NO_CAPTURE_LIMIT"]:
                outcome = 0.0
                reason = "draw_nocap"
                tag = "DRAW_NOPROGRESS"
                break

            temp_move = get_temp_from_schedule(ply, cfg["TEMP_MOVE_SCHEDULE"])
            temp_target = float(cfg["TEMP_TARGET"])
            add_noise = (ply <= int(cfg["ROOT_NOISE_END_PLY"]))

            try:
                with torch.inference_mode():
                    best_mv, idxs_t, probs_t, root_v = xqcpp.mcts_search(
                        board,
                        eval_net,
                        int(cfg["MCTS_SIMS"]),
                        float(cfg["C_PUCT"]),
                        bool(add_noise),
                        float(cfg["DIRICHLET_ALPHA"]),
                        float(cfg["DIRICHLET_EPS"]),
                        float(temp_move),
                        float(temp_target),
                        int(cfg["EVAL_BATCH_SIZE"]),
                        int((game_seed + ply * 10007) & 0x7fffffff),
                    )
            except Exception as e:
                outcome = 0.0
                reason = "eval_error"
                tag = f"EVAL_ERROR_{type(e).__name__}"
                discard_game = True
                print(f"[SELFPLAY-{rank}] skipped game due to eval error: {e}", flush=True)
                break

            if best_mv < 0:
                # no legal moves -> loss for side to move
                # result from red view: if side-to-move is red => red loses (-1), else red wins (+1)
                outcome = -1.0 if board.turn() == 0 else 1.0
                reason = "terminal_nomove"
                tag = "RED_WIN" if outcome > 0 else "BLACK_WIN"
                break

            # store sample
            st = board.to_tensor().squeeze(0).contiguous()  # [115,10,9]
            st_u8 = torch.clamp((st * 255.0).round(), 0, 255).to(torch.uint8).numpy()
            idxs = idxs_t.cpu().numpy().astype(np.int64, copy=False)
            probs = probs_t.cpu().numpy().astype(np.float32, copy=False)
            stm_is_black = (board.turn() == 1)
            game_mem.append((st_u8, idxs, probs, stm_is_black))

            # resign/claim with root_v (stm)
            if cfg["ENABLE_RESIGN"] and ply >= cfg["RESIGN_START"]:
                if float(root_v) <= float(cfg["RESIGN_V"]):
                    bad_cnt += 1
                else:
                    bad_cnt = 0
                if bad_cnt >= int(cfg["RESIGN_CONSEC"]):
                    # side to move resigns => lose
                    outcome = -1.0 if board.turn() == 0 else 1.0
                    reason = "resign"
                    stm = int(board.turn())
                    mat_red = float(board.material_score())
                    mat_stm = mat_red if stm == 0 else -mat_red
                    tag = f"RESIGN_{'RED' if stm==0 else 'BLACK'}_v={float(root_v):.2f}_mat={mat_stm:.1f}"
                    break

            if cfg["ENABLE_CLAIM_WIN"] and ply >= cfg["CLAIM_START"]:
                if float(root_v) >= float(cfg["CLAIM_V"]):
                    good_cnt += 1
                else:
                    good_cnt = 0
                if good_cnt >= int(cfg["CLAIM_CONSEC"]):
                    # side to move claims win => win
                    outcome = 1.0 if board.turn() == 0 else -1.0
                    reason = "claimwin"
                    stm = int(board.turn())
                    mat_red = float(board.material_score())
                    mat_stm = mat_red if stm == 0 else -mat_red
                    tag = f"CLAIM_{'RED' if stm==0 else 'BLACK'}_v={float(root_v):.2f}_mat={mat_stm:.1f}"
                    break

            # play move
            is_cap = bool(board.is_capture(int(best_mv)))
            if is_cap:
                no_cap = 0
            else:
                no_cap += 1

            moves_iccs.append(action_to_iccs(int(best_mv)))
            board.push(int(best_mv))
            ply += 1
        duration = time.time() - t_game0


        # Decide keep & shaping
        # outcome is from red view in {-1,0,+1}
        draw_mode = cfg["TRAIN_DRAW_MODE"]
        samples = []
        if discard_game:
            samples = []
        elif outcome != 0.0:
            # decisive: keep all
            w_game = 1.0
            for (st, idxs, probs, stm_is_black) in game_mem:
                z_stm = outcome_red_to_value_stm(outcome, stm_is_black)
                samples.append(Sample(state=st, idxs=idxs, probs=probs, z=z_stm, w=w_game))
        else:
            if draw_mode == "discard":
                samples = []
            else:
                # weighted draw
                w_game = float(cfg["DRAW_SAMPLE_W"])
                # material shaping based on red advantage
                mat = float(board.material_score())
                z_red = draw_value_shaping(mat, float(cfg["DRAW_VALUE_SCALE"]))
                for (st, idxs, probs, stm_is_black) in game_mem:
                    z_stm = outcome_red_to_value_stm(z_red, stm_is_black)
                    samples.append(Sample(state=st, idxs=idxs, probs=probs, z=z_stm, w=w_game))

        # put to queue
        if samples:
            out_q.put(("game", samples, outcome, reason, ply, moves_iccs, tag, float(duration)))
        else:
            out_q.put(("skip", outcome, reason, ply, tag, float(duration)))

# ----------------------------
# 9) Trainer
# ----------------------------

# ----------------------------
# 9) Trainer (V11)
# ----------------------------

def atomic_torch_save(obj, path: str):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def infer_last_step(save_dir: str) -> int:
    """Infer the latest step number from model/best snapshot filenames."""
    best = 0
    if not os.path.isdir(save_dir):
        return 0
    pat = re.compile(r"^(?:model_step|best_step)(\d+)\.pth$")
    for fn in os.listdir(save_dir):
        m = pat.match(fn)
        if not m:
            continue
        try:
            best = max(best, int(m.group(1)))
        except Exception:
            pass
    return int(best)


_STEP_FILE_PAT = re.compile(r"^(model_step|best_step)(\d+)\.pth$")


def parse_step_from_name(path_or_name: str) -> Optional[int]:
    m = _STEP_FILE_PAT.match(os.path.basename(str(path_or_name)))
    if not m:
        return None
    try:
        return int(m.group(2))
    except Exception:
        return None


def extract_model_state_dict(obj: Any) -> Optional[Dict[str, torch.Tensor]]:
    if isinstance(obj, dict) and isinstance(obj.get("model"), dict):
        return obj["model"]
    if isinstance(obj, dict) and any(torch.is_tensor(v) for v in obj.values()):
        return obj  # weights-only state_dict
    return None


def nonfinite_count_in_state_dict(sd: Dict[str, torch.Tensor]) -> Tuple[int, Optional[str]]:
    bad = 0
    first = None
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        nbad = int((~torch.isfinite(v)).sum().item())
        if nbad > 0:
            bad += nbad
            if first is None:
                first = k
    return bad, first


def nonfinite_count_in_optimizer_state(opt_state_dict: dict) -> int:
    bad = 0
    if not isinstance(opt_state_dict, dict):
        return 0
    state = opt_state_dict.get("state", {})
    if not isinstance(state, dict):
        return 0
    for slot in state.values():
        if not isinstance(slot, dict):
            continue
        for v in slot.values():
            if torch.is_tensor(v):
                bad += int((~torch.isfinite(v)).sum().item())
    return int(bad)


def load_latest_finite_weights(
    save_dir: str,
    latest_path: str,
    best_path: str,
    max_step: Optional[int] = None,
) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[str], Optional[int]]:
    """
    Find the newest finite weights snapshot.
    Search order:
    1) model_step*.pth / best_step*.pth (descending step, optionally <= max_step)
    2) best.pth
    3) best.pth
    """
    candidates: List[Tuple[int, int, str]] = []
    if os.path.isdir(save_dir):
        for fn in os.listdir(save_dir):
            m = _STEP_FILE_PAT.match(fn)
            if not m:
                continue
            kind = m.group(1)
            step = int(m.group(2))
            if (max_step is not None) and (step > int(max_step)):
                continue
            # model_step preferred over best_step at same step
            pri = 0 if kind == "model_step" else 1
            candidates.append((step, pri, os.path.join(save_dir, fn)))
    candidates.sort(key=lambda x: (-x[0], x[1]))

    ordered_paths = [p for _, _, p in candidates]
    ordered_paths.extend([latest_path, best_path])

    seen = set()
    for p in ordered_paths:
        if (not p) or (p in seen) or (not os.path.isfile(p)):
            continue
        seen.add(p)
        try:
            obj = torch.load(p, map_location="cpu")
        except Exception as e:
            print(f"[RESUME] skip unreadable snapshot: {p} err={e}", flush=True)
            continue
        sd = extract_model_state_dict(obj)
        if sd is None:
            continue
        bad, first = nonfinite_count_in_state_dict(sd)
        if bad == 0:
            return sd, p, parse_step_from_name(p)
        print(f"[RESUME] skip non-finite snapshot: {p} bad={bad} first_tensor={first}", flush=True)

    return None, None, None


def move_optimizer_state_to_device(opt: torch.optim.Optimizer, device: torch.device):
    """Move optimizer state tensors to the target device after `load_state_dict`."""
    for st in opt.state.values():
        for k, v in list(st.items()):
            if torch.is_tensor(v):
                st[k] = v.to(device)


def build_lr_scheduler(opt: torch.optim.Optimizer, cfg: dict):
    if not bool(cfg.get("LR_SCHED_ENABLE", True)):
        return None
    try:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=float(cfg.get("LR_PLATEAU_FACTOR", 0.5)),
            patience=int(cfg.get("LR_PLATEAU_PATIENCE", 6)),
            threshold=float(cfg.get("LR_PLATEAU_THRESHOLD", 1e-3)),
            cooldown=int(cfg.get("LR_COOLDOWN", 0)),
            min_lr=float(cfg.get("LR_MIN", 0.0)),
            verbose=True,
        )
    except TypeError:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=float(cfg.get("LR_PLATEAU_FACTOR", 0.5)),
            patience=int(cfg.get("LR_PLATEAU_PATIENCE", 6)),
            threshold=float(cfg.get("LR_PLATEAU_THRESHOLD", 1e-3)),
            cooldown=int(cfg.get("LR_COOLDOWN", 0)),
            min_lr=float(cfg.get("LR_MIN", 0.0)),
        )


def maybe_recover_learning_rate(
    opt: torch.optim.Optimizer,
    scheduler,
    cfg: dict,
):
    recover_if = float(cfg.get("LR_RECOVER_IF_BELOW", 0.0))
    recover_to = float(cfg.get("LR_RECOVER_TO", 0.0))
    if recover_if <= 0.0 or recover_to <= 0.0:
        return scheduler
    cur_lr = min(float(pg.get("lr", 0.0)) for pg in opt.param_groups)
    if cur_lr >= recover_if:
        return scheduler
    for pg in opt.param_groups:
        pg["lr"] = recover_to
    # Rebuild scheduler state after manually overriding LR.
    scheduler = build_lr_scheduler(opt, cfg)
    print(
        f"[LR] recover from {cur_lr:.6g} -> {recover_to:.6g} and reset scheduler",
        flush=True,
    )
    return scheduler


def maybe_runtime_recover_learning_rate(
    opt: torch.optim.Optimizer,
    scheduler,
    cfg: dict,
    global_step: int,
    loss_ema: Optional[float],
    state: dict,
):
    if not bool(cfg.get("LR_RUNTIME_RECOVER_ENABLE", True)):
        return scheduler
    if loss_ema is None:
        return scheduler

    check_every = int(cfg.get("LR_RUNTIME_RECOVER_CHECK_EVERY", cfg.get("LR_SCHED_EVERY", 200)))
    if check_every <= 0 or (global_step % check_every) != 0:
        return scheduler

    cur_lr = min(float(pg.get("lr", 0.0)) for pg in opt.param_groups)
    lr_min = float(cfg.get("LR_MIN", 0.0))
    floor_margin = float(cfg.get("LR_RUNTIME_RECOVER_FLOOR_MARGIN", 1.02))
    at_floor = (lr_min > 0.0) and (cur_lr <= lr_min * floor_margin)

    if not at_floor:
        state["floor_since"] = None
        state["floor_best"] = None
        return scheduler

    if state.get("floor_since") is None:
        state["floor_since"] = int(global_step)
        state["floor_best"] = float(loss_ema)
        return scheduler

    state["floor_best"] = min(float(state.get("floor_best", loss_ema)), float(loss_ema))
    floor_since = int(state.get("floor_since", global_step))
    stay_steps = int(global_step - floor_since)
    if stay_steps < int(cfg.get("LR_RUNTIME_RECOVER_PATIENCE_STEPS", 5000)):
        return scheduler

    last_recover = int(state.get("last_recover_step", -10**9))
    if int(global_step - last_recover) < int(cfg.get("LR_RUNTIME_RECOVER_COOLDOWN_STEPS", 12000)):
        return scheduler

    # Detect stall near LR floor using loss EMA.
    stall_eps = float(cfg.get("LR_RUNTIME_RECOVER_STALL_EPS", 2e-3))
    floor_best = float(state.get("floor_best", loss_ema))
    stalled = (float(loss_ema) >= floor_best - stall_eps)
    if not stalled:
        return scheduler

    target = float(cfg.get("LR_RUNTIME_RECOVER_TO", cfg.get("LR_RECOVER_TO", 0.0)))
    max_lr = float(cfg.get("LR_RUNTIME_RECOVER_MAX", cfg.get("LR", target)))
    if target <= 0.0:
        target = max(cur_lr * 2.0, lr_min * 2.0)
    target = min(max_lr, max(target, cur_lr))
    if target <= cur_lr:
        return scheduler

    for pg in opt.param_groups:
        pg["lr"] = target
    scheduler = build_lr_scheduler(opt, cfg)

    state["last_recover_step"] = int(global_step)
    state["floor_since"] = int(global_step)
    state["floor_best"] = float(loss_ema)
    state["recover_count"] = int(state.get("recover_count", 0)) + 1

    print(
        f"[LR] runtime recover at step={global_step}: {cur_lr:.6g} -> {target:.6g} "
        f"(count={state['recover_count']})",
        flush=True,
    )
    return scheduler


def arena_play_one(xqcpp, cfg: dict, eval_red, eval_black, seed: int) -> int:
    """Play one arena game. Return +1 (red win), -1 (black win), 0 (draw)."""
    board = xqcpp.Board()
    pos_count = {}
    no_cap = 0
    ply = 0

    while True:
        if ply >= int(cfg.get("ARENA_MAX_STEPS", cfg.get("MAX_STEPS", 360))):
            return 0

        key = int(board.key())
        pos_count[key] = pos_count.get(key, 0) + 1
        rep_min_ply = int(cfg.get("ARENA_REPEAT_MIN_PLY", cfg.get("REPEAT_MIN_PLY", 20)))
        rep_limit = int(cfg.get("ARENA_REPEAT_LIMIT", cfg.get("REPEAT_LIMIT", 4)))
        if ply >= rep_min_ply and pos_count[key] >= rep_limit:
            return 0

        if no_cap >= int(cfg.get("ARENA_NO_CAPTURE_LIMIT", cfg.get("NO_CAPTURE_LIMIT", 80))):
            return 0

        # choose eval net by side-to-move
        eval_net = eval_red if board.turn() == 0 else eval_black
        temp_move = 0.0
        if ply < int(cfg.get("ARENA_TEMP_UNTIL_PLY", 0)):
            temp_move = float(cfg.get("ARENA_TEMP_MOVE", 0.0))
        add_root_noise = bool(cfg.get("ARENA_ADD_ROOT_NOISE", False)) and (
            ply < int(cfg.get("ARENA_ROOT_NOISE_END_PLY", 0))
        )

        side_to_move = int(board.turn())
        try:
            best_mv, _idxs, _probs, _root_v = xqcpp.mcts_search(
                board,
                eval_net,
                int(cfg["ARENA_SIMS"]),
                float(cfg["C_PUCT"]),
                bool(add_root_noise),
                float(cfg.get("ARENA_DIRICHLET_ALPHA", cfg["DIRICHLET_ALPHA"])),
                float(cfg.get("ARENA_DIRICHLET_EPS", cfg["DIRICHLET_EPS"])),
                float(temp_move),
                1.0,    # temperature_target (unused)
                int(cfg.get("ARENA_EVAL_BATCH_SIZE", cfg.get("EVAL_BATCH_SIZE", 64))),
                int((seed + ply * 10007) & 0x7fffffff),
            )
        except Exception as e:
            print(f"[ARENA] eval error at ply={ply}: {e}", flush=True)
            return -1 if side_to_move == 0 else 1

        if best_mv < 0:
            # side-to-move has no legal moves -> loses
            return -1 if board.turn() == 0 else 1

        is_cap = bool(board.is_capture(int(best_mv)))
        no_cap = 0 if is_cap else (no_cap + 1)
        board.push(int(best_mv))
        ply += 1


def arena_evaluate(xqcpp, cfg: dict, net_cls, cur_sd: dict, best_sd: dict, device: torch.device):
    """Evaluate current model vs best model and report arena statistics."""
    games = int(cfg.get("ARENA_GAMES", 20))
    seed0 = int(cfg.get("ARENA_SEED", 20260115))

    # build nets
    net_cur = net_cls(cfg).to(device)
    net_best = net_cls(cfg).to(device)
    net_cur.load_state_dict(cur_sd, strict=True)
    net_best.load_state_dict(best_sd, strict=True)
    net_cur.eval()
    net_best.eval()

    # Always use wrapper so arena forward runs in inference_mode on every device.
    amp_dtype = resolve_amp_dtype(cfg)
    cur_eval = EvalNetWrapper(net_cur, device, False, amp_dtype)
    best_eval = EvalNetWrapper(net_best, device, False, amp_dtype)

    cur_win = 0
    best_win = 0
    draw = 0

    for g in range(games):
        # Alternate colors every game.
        cur_is_red = (g % 2 == 0)
        eval_red = cur_eval if cur_is_red else best_eval
        eval_black = best_eval if cur_is_red else cur_eval
        res = arena_play_one(xqcpp, cfg, eval_red, eval_black, seed0 + g * 9973)
        if res == 0:
            draw += 1
        else:
            winner_is_red = (res > 0)
            if winner_is_red == cur_is_red:
                cur_win += 1
            else:
                best_win += 1

    non_draw = cur_win + best_win
    winrate = (cur_win + 0.5 * draw) / games

    return {
        "games": games,
        "cur_win": cur_win,
        "best_win": best_win,
        "draw": draw,
        "non_draw": non_draw,
        "winrate": float(winrate),
    }


def trainer_main():
    # CUDA perf toggles for the trainer.
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    os.makedirs(CFG["SAVE_DIR"], exist_ok=True)

    save_dir = str(CFG["SAVE_DIR"])
    latest_path = os.path.join(save_dir, str(CFG.get("LATEST_NAME", "best.pth")))
    ckpt_path = os.path.join(save_dir, str(CFG.get("CHECKPOINT_NAME", "latest_ckpt.pt")))
    best_path = os.path.join(save_dir, str(CFG.get("BEST_NAME", "best.pth")))

    # Paths used by self-play workers for weight reload.
    CFG["_LATEST_PATH"] = latest_path if bool(CFG.get("SAVE_LATEST", True)) else None
    CFG["_BEST_PATH"] = best_path
    if bool(CFG.get("SELFPLAY_USE_BEST", True)):
        CFG["_SELFPLAY_PATH"] = best_path
    else:
        CFG["_SELFPLAY_PATH"] = latest_path

    set_cpu_threads(CFG["CPU_THREADS_TRAIN"])

    seed = 12345
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Seed reproducibility.
    xqcpp = load_xqcpp()

    train_device = pick_device(CFG.get("TRAIN_DEVICE", "cpu"))
    amp_dtype = resolve_amp_dtype(CFG)
    amp_enabled = bool(CFG.get("USE_AMP", True)) and (train_device.type in ("cuda", "mps"))
    if amp_enabled and train_device.type == "mps" and (not bool(CFG.get("ALLOW_MPS_TRAIN_AMP", False))):
        amp_enabled = False
        print("[AMP] disabled on MPS for training stability (set ALLOW_MPS_TRAIN_AMP=True to force).", flush=True)
    scaler_enabled = bool(amp_enabled and train_device.type == "cuda" and amp_dtype == torch.float16)
    scaler = make_grad_scaler(scaler_enabled)

    net = XiangqiNetV10(CFG).to(train_device)
    train_net = net
    use_torch_compile = bool(CFG.get("USE_TORCH_COMPILE", True))
    if train_device.type == "cuda" and use_torch_compile and hasattr(torch, "compile"):
        try:
            print("[CUDA] using torch.compile for the training forward graph...", flush=True)
            train_net = torch.compile(net)
        except Exception as exc:
            print(f"[CUDA] torch.compile unavailable; falling back to eager mode: {exc}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=float(CFG["LR"]), weight_decay=float(CFG["WEIGHT_DECAY"]))
    scheduler = build_lr_scheduler(opt, CFG)

    global_step = 0
    game_count = 0
    lr_runtime_state = {
        "floor_since": None,
        "floor_best": None,
        "last_recover_step": -10**9,
        "recover_count": 0,
    }

    auto_resume = bool(CFG.get("AUTO_RESUME_LATEST", True))

    # 1) Resume from full checkpoint first.
    if auto_resume and os.path.isfile(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            ckpt_model = ckpt.get("model")
            if not isinstance(ckpt_model, dict):
                raise RuntimeError("checkpoint has no model state_dict")

            bad_model, first_model = nonfinite_count_in_state_dict(ckpt_model)
            if bad_model > 0:
                print(
                    f"[RESUME] skip ckpt due to non-finite model: {ckpt_path} "
                    f"bad={bad_model} first_tensor={first_model}",
                    flush=True,
                )
            else:
                net.load_state_dict(ckpt_model, strict=True)

                bad_opt = nonfinite_count_in_optimizer_state(ckpt.get("optimizer", {}))
                if bad_opt == 0 and ckpt.get("optimizer") is not None:
                    opt.load_state_dict(ckpt["optimizer"])
                    move_optimizer_state_to_device(opt, train_device)
                    if scheduler is not None and ckpt.get("scheduler") is not None:
                        scheduler.load_state_dict(ckpt["scheduler"])
                    if scaler is not None and ckpt.get("scaler") is not None:
                        scaler.load_state_dict(ckpt["scaler"])
                else:
                    print(
                        f"[RESUME] ckpt optimizer has non-finite state (bad={bad_opt}); "
                        f"using fresh optimizer/scheduler.",
                        flush=True,
                    )

                global_step = int(ckpt.get("global_step", 0))
                game_count = int(ckpt.get("game_count", 0))
                if isinstance(ckpt.get("lr_runtime_state"), dict):
                    lr_runtime_state.update(ckpt["lr_runtime_state"])
                print(f"[RESUME] loaded ckpt: {ckpt_path} step={global_step} games={game_count}", flush=True)
        except Exception as e:
            print(f"[RESUME] failed to load ckpt ({ckpt_path}). err={e}", flush=True)

    # 2) Fallback: resume from latest weights only.
    if global_step == 0 and auto_resume and os.path.isfile(latest_path):
        try:
            obj_latest = torch.load(latest_path, map_location="cpu")
            sd = extract_model_state_dict(obj_latest)
            if sd is None:
                raise RuntimeError("latest file does not contain a valid state_dict")
            bad_latest, first_latest = nonfinite_count_in_state_dict(sd)
            if bad_latest > 0:
                print(
                    f"[RESUME] skip latest due to non-finite model: {latest_path} "
                    f"bad={bad_latest} first_tensor={first_latest}",
                    flush=True,
                )
            else:
                net.load_state_dict(sd, strict=True)
                _sd_tmp, _path_tmp, finite_step = load_latest_finite_weights(
                    save_dir=save_dir,
                    latest_path=latest_path,
                    best_path=best_path,
                )
                if finite_step is not None:
                    global_step = int(finite_step)
                else:
                    global_step = infer_last_step(save_dir)
                print(f"[RESUME] loaded latest: {latest_path} inferred_step={global_step}", flush=True)
        except Exception as e:
            print(f"[RESUME] failed to load latest ({latest_path}). err={e}", flush=True)

    # 3) Last-resort fallback: scan snapshots and pick the newest finite weights.
    if global_step == 0:
        sd_fallback, path_fallback, step_fallback = load_latest_finite_weights(
            save_dir=save_dir,
            latest_path=latest_path,
            best_path=best_path,
        )
        if sd_fallback is not None and path_fallback is not None:
            net.load_state_dict(sd_fallback, strict=True)
            if step_fallback is not None:
                global_step = int(step_fallback)
            print(
                f"[RESUME] loaded fallback finite weights: {path_fallback} "
                f"step={global_step}",
                flush=True,
            )

    scheduler = maybe_recover_learning_rate(opt, scheduler, CFG)

    train_dtype = amp_dtype if amp_enabled else torch.float32
    print(f"[TRAIN] device={train_device} amp={amp_enabled} dtype={train_dtype} step={global_step}", flush=True)

    # ensure best exists
    if os.path.isfile(best_path):
        try:
            best_obj = torch.load(best_path, map_location="cpu")
            best_sd = extract_model_state_dict(best_obj)
            if best_sd is None:
                raise RuntimeError("best file does not contain a valid state_dict")
            bad_best, first_best = nonfinite_count_in_state_dict(best_sd)
            if bad_best > 0:
                print(
                    f"[INIT] best is non-finite; rewriting from current net "
                    f"(bad={bad_best} first_tensor={first_best})",
                    flush=True,
                )
                best_sd = {k: v.detach().cpu() for k, v in net.state_dict().items()}
                atomic_torch_save(best_sd, best_path)
        except Exception:
            best_sd = {k: v.detach().cpu() for k, v in net.state_dict().items()}
            atomic_torch_save(best_sd, best_path)
    else:
        best_sd = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        atomic_torch_save(best_sd, best_path)
        print(f"[INIT] write best: {best_path}", flush=True)

    # ensure latest exists and is finite (for PVP)
    if bool(CFG.get("SAVE_LATEST", True)):
        need_write_latest = (not os.path.isfile(latest_path))
        if not need_write_latest:
            try:
                latest_obj = torch.load(latest_path, map_location="cpu")
                sd_latest = extract_model_state_dict(latest_obj)
                if sd_latest is None:
                    raise RuntimeError("latest file does not contain a valid state_dict")
                bad_latest, first_latest = nonfinite_count_in_state_dict(sd_latest)
                if bad_latest > 0:
                    print(
                        f"[INIT] latest is non-finite; rewriting from current net "
                        f"(bad={bad_latest} first_tensor={first_latest})",
                        flush=True,
                    )
                    need_write_latest = True
            except Exception:
                need_write_latest = True
        if need_write_latest:
            latest_sd = {k: v.detach().cpu() for k, v in net.state_dict().items()}
            atomic_torch_save(latest_sd, latest_path)
            print(f"[INIT] write latest: {latest_path}", flush=True)

    # Initial weights loaded by workers at process start.
    sp_path = str(CFG.get("_SELFPLAY_PATH") or "")
    if sp_path and os.path.isfile(sp_path):
        sp_obj = torch.load(sp_path, map_location="cpu")
        init_sd = extract_model_state_dict(sp_obj)
        if init_sd is None:
            init_sd = None
        else:
            bad_sp, _ = nonfinite_count_in_state_dict(init_sd)
            if bad_sp > 0:
                print(f"[INIT] self-play init weights are non-finite: {sp_path} bad={bad_sp}", flush=True)
                init_sd = None
    else:
        init_sd = None
    if init_sd is None:
        init_sd = {k: v.detach().cpu() for k, v in net.state_dict().items()}

    out_q: mp.Queue = mp.Queue(maxsize=64)
    stop_flag = mp.Event()

    procs = []
    for r in range(int(CFG["NUM_WORKERS"])):
        p = mp.Process(target=selfplay_worker, args=(r, CFG, init_sd, out_q, stop_flag), daemon=True)
        p.start()
        procs.append(p)

    buf = ReplayBuffer(CFG["BUFFER_CAPACITY"], CFG["MAX_POLICY_MOVES"])

    stat = {"red": 0, "black": 0, "draw": 0}
    window_n = 50
    window = collections.deque(maxlen=window_n)

    loss_ema = None
    p_ema = None
    v_ema = None
    value_mse_ema = None
    policy_entropy_ema = None

    def save_weights_and_ckpt(step: int):
        sd_cpu = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        bad_save, first_save = nonfinite_count_in_state_dict(sd_cpu)
        if bad_save > 0:
            print(
                f"[SAVE] skipped non-finite model at step={step} "
                f"(bad={bad_save} first_tensor={first_save})",
                flush=True,
            )
            return

        # archive
        arch_path = os.path.join(save_dir, f"model_step{int(step):07d}.pth")
        torch.save(sd_cpu, arch_path)

        # latest weights
        if bool(CFG.get("SAVE_LATEST", True)):
            atomic_torch_save(sd_cpu, latest_path)

        # checkpoint
        ckpt = {
            "version": 11,
            "global_step": int(step),
            "game_count": int(game_count),
            "model": sd_cpu,
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "lr_runtime_state": lr_runtime_state,
            "time": float(time.time()),
        }
        atomic_torch_save(ckpt, ckpt_path)

        print(f"[SAVE] step={step} -> {arch_path} | latest={os.path.basename(latest_path)} | ckpt={os.path.basename(ckpt_path)}", flush=True)

    def maybe_run_arena(step: int):
        nonlocal best_sd
        if not bool(CFG.get("ARENA_ENABLE", True)):
            return
        every = int(CFG.get("ARENA_EVERY_TRAIN_STEPS", 0))
        if every <= 0:
            return
        if step % every != 0:
            return

        try:
            best_obj = torch.load(best_path, map_location="cpu")
            best_loaded = extract_model_state_dict(best_obj)
            if best_loaded is not None:
                bad_best, _ = nonfinite_count_in_state_dict(best_loaded)
                if bad_best == 0:
                    best_sd = best_loaded
                else:
                    print(f"[ARENA] best snapshot is non-finite; keep in-memory best (bad={bad_best})", flush=True)
        except Exception:
            best_sd = best_sd

        cur_sd = {k: v.detach().cpu() for k, v in net.state_dict().items()}

        print(f"[ARENA] start @ step={step} games={CFG.get('ARENA_GAMES')} sims={CFG.get('ARENA_SIMS')} threshold={CFG.get('ARENA_WINRATE_THRESHOLD')}", flush=True)
        res = arena_evaluate(xqcpp, CFG, XiangqiNetV10, cur_sd, best_sd, train_device)
        print(f"[ARENA] done: cur_win={res['cur_win']} best_win={res['best_win']} draw={res['draw']} winrate={res['winrate']*100:.1f}%", flush=True)

        if res["non_draw"] <= 0:
            return

        if res["winrate"] >= float(CFG.get("ARENA_WINRATE_THRESHOLD", 0.55)):
            best_sd = cur_sd
            atomic_torch_save(best_sd, best_path)
            best_step_path = os.path.join(save_dir, f"best_step{int(step):07d}.pth")
            torch.save(best_sd, best_step_path)
            print(f"[ARENA] ACCEPT -> best updated: {best_step_path}", flush=True)
        else:
            print(f"[ARENA] REJECT (keep best)", flush=True)

    nonfinite_state = {
        "streak": 0,
        "recoveries": 0,
    }

    def recover_from_nonfinite(reason: str):
        nonlocal global_step, game_count, scheduler, opt, scaler, amp_enabled
        nonlocal loss_ema, p_ema, v_ema, value_mse_ema, policy_entropy_ema

        nonfinite_state["recoveries"] += 1
        print(
            f"[RECOVER] non-finite detected (reason={reason}) "
            f"streak={nonfinite_state['streak']} recoveries={nonfinite_state['recoveries']}",
            flush=True,
        )

        restored = False
        restored_from = None
        if os.path.isfile(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                ckpt_model = ckpt.get("model")
                if isinstance(ckpt_model, dict):
                    bad_ckpt_model, _ = nonfinite_count_in_state_dict(ckpt_model)
                    if bad_ckpt_model == 0:
                        net.load_state_dict(ckpt_model, strict=True)
                        global_step = int(ckpt.get("global_step", global_step))
                        game_count = max(int(game_count), int(ckpt.get("game_count", game_count)))
                        if isinstance(ckpt.get("lr_runtime_state"), dict):
                            lr_runtime_state.update(ckpt["lr_runtime_state"])
                        loss_ema = None
                        p_ema = None
                        v_ema = None
                        value_mse_ema = None
                        policy_entropy_ema = None
                        restored = True
                        restored_from = ckpt_path
                        print(f"[RECOVER] restored checkpoint model: step={global_step} path={ckpt_path}", flush=True)
                    else:
                        print(f"[RECOVER] checkpoint model is non-finite: {ckpt_path}", flush=True)
            except Exception as e:
                print(f"[RECOVER] checkpoint restore failed: {e}", flush=True)

        if not restored:
            sd_fallback, path_fallback, step_fallback = load_latest_finite_weights(
                save_dir=save_dir,
                latest_path=latest_path,
                best_path=best_path,
                max_step=global_step,
            )
            if sd_fallback is not None and path_fallback is not None:
                net.load_state_dict(sd_fallback, strict=True)
                if step_fallback is not None:
                    global_step = min(int(global_step), int(step_fallback))
                restored = True
                restored_from = path_fallback
                loss_ema = None
                p_ema = None
                v_ema = None
                value_mse_ema = None
                policy_entropy_ema = None
                print(f"[RECOVER] restored fallback finite weights: {path_fallback}", flush=True)

        if not restored:
            print("[RECOVER] no finite snapshot available; continuing with current state.", flush=True)

        # Recreate optimizer/scheduler after recovery to drop potentially corrupted moments.
        current_lr = min(float(pg.get("lr", CFG["LR"])) for pg in opt.param_groups)
        opt = torch.optim.AdamW(net.parameters(), lr=current_lr, weight_decay=float(CFG["WEIGHT_DECAY"]))
        scheduler = build_lr_scheduler(opt, CFG)
        scaler = make_grad_scaler(scaler_enabled)


        # Reduce LR after recovery to avoid immediate divergence.
        try:
            lr_min = float(CFG.get("LR_MIN", 0.0))
            scale = float(CFG.get("NONFINITE_RECOVER_LR_SCALE", 0.5))
            for pg in opt.param_groups:
                old_lr = float(pg.get("lr", CFG["LR"]))
                pg["lr"] = max(lr_min, old_lr * scale)
            print(f"[RECOVER] scaled LR by {scale:.3f} (source={restored_from})", flush=True)
        except Exception as e:
            print(f"[RECOVER] failed to scale LR: {e}", flush=True)

        # If MPS AMP was forced on, disable it after repeated non-finite events.
        if amp_enabled and train_device.type == "mps":
            amp_enabled = False
            scaler = None
            print("[RECOVER] disabled AMP on MPS and switched train dtype to float32.", flush=True)

        nonfinite_state["streak"] = 0

    last_loss = float('nan')

    try:
        while True:
            '''
            current_time = datetime.datetime.now().time()

            # Optional power-price cutoff window.
            cutoff_time = datetime.time(6, 59, 0)
            resume_time = datetime.time(7, 5, 0)

            if cutoff_time <= current_time <= resume_time:
                print(f"\n[POWER] current time {current_time.strftime('%H:%M:%S')}", flush=True)
                print("[POWER] entering the configured pause window; save and exit.", flush=True)
                print("[POWER] hand off to finally-block checkpoint save.", flush=True)
                print("[POWER] trainer paused.", flush=True)

                # Break here so finally() handles save-and-exit.
                break
                #
            '''
            # Use a timeout so the main process can periodically re-check control conditions.
            try:
                msg = out_q.get(timeout=1.0)
            except queue.Empty:
                continue


            if msg[0] == "game":
                _, samples, outcome, reason, ply, _moves_iccs, tag, duration = msg
                game_count += 1

                if outcome > 0:
                    stat["red"] += 1
                elif outcome < 0:
                    stat["black"] += 1
                else:
                    stat["draw"] += 1

                window.append((float(outcome), str(reason), int(ply), float(duration)))

                # push buffer + mirror
                for s in samples:
                    buf.push(s)
                    st2 = flip_state_tensor(s.state)
                    idx2, pr2 = flip_sparse_policy(s.idxs, s.probs)
                    buf.push(Sample(state=st2, idxs=idx2, probs=pr2, z=s.z, w=s.w))

                # train steps
                if len(buf) >= max(1024, CFG["BATCH_SIZE"] * 4):
                    for _ in range(int(CFG["TRAIN_STEPS_PER_GAME"])):
                        states_u8, idx_mat, prob_mat, z, w = buf.sample(CFG["BATCH_SIZE"])

                        states = states_u8.to(device=train_device, dtype=torch.float32).div_(255.0)
                        idx_mat = idx_mat.to(device=train_device)
                        prob_mat = prob_mat.to(device=train_device)
                        z = z.to(device=train_device)
                        w = w.to(device=train_device)

                        # Sanitize targets/weights to prevent NaN propagation from bad samples.
                        idx_mat = idx_mat.long()
                        prob_mat = torch.nan_to_num(prob_mat.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_(min=0.0)
                        z = torch.nan_to_num(z.float(), nan=0.0, posinf=1.0, neginf=-1.0).clamp_(-1.0, 1.0)
                        w = torch.nan_to_num(w.float(), nan=1.0, posinf=1.0, neginf=1.0).clamp_(min=0.0)

                        valid_idx = (idx_mat >= 0) & (idx_mat < 8100)
                        prob_mat = prob_mat * valid_idx.float()
                        row_sum = prob_mat.sum(dim=1, keepdim=True)
                        safe_den = torch.where(row_sum > 1e-12, row_sum, torch.ones_like(row_sum))
                        prob_mat = torch.where(row_sum > 1e-12, prob_mat / safe_den, prob_mat)

                        with amp_context(train_device, amp_enabled, amp_dtype):
                            p_logits, wdl_logits, m_pred = train_net(states)

                        if (not torch.isfinite(p_logits).all()) or (not torch.isfinite(wdl_logits).all()) or (not torch.isfinite(m_pred).all()):
                            nonfinite_state["streak"] += 1
                            print(
                                f"[WARN] non-finite model output at step={global_step} "
                                f"(streak={nonfinite_state['streak']}) -> skip",
                                flush=True,
                            )
                            opt.zero_grad(set_to_none=True)
                            if nonfinite_state["streak"] >= int(CFG.get("NONFINITE_RECOVER_STREAK", 8)):
                                recover_from_nonfinite("model_output")
                            continue

                        # policy loss
                        # 1) Compute log_softmax on the full 8100 policy space.
                        log_p_full = F.log_softmax(p_logits.float(), dim=1)

                        # 2) Gather target move log-probabilities.
                        gather_idx = idx_mat.clamp(min=0, max=8099)
                        log_p_sel = log_p_full.gather(1, gather_idx)

                        # 3) Invalid padded entries already have zero target probability.
                        loss_p = -(prob_mat.float() * log_p_sel).sum(dim=1)

                        # value loss
                        z_wdl = scalar_value_to_wdl_target(z.float().clamp(-1.0, 1.0))
                        log_v = F.log_softmax(wdl_logits.float(), dim=1)
                        loss_v = -(z_wdl * log_v).sum(dim=1)

                        # aux material
                        vals = states.new_tensor([0.0, 2.0, 2.0, 4.0, 9.0, 4.5, 1.0], dtype=torch.float32)
                        red_cnt = states[:, 0:7].sum(dim=(2, 3)).float()
                        blk_cnt = states[:, 7:14].sum(dim=(2, 3)).float()
                        mat_red = (red_cnt * vals).sum(dim=1)
                        mat_blk = (blk_cnt * vals).sum(dim=1)
                        mat_bal = mat_red - mat_blk
                        stm_black = (states[:, 112].mean(dim=(1, 2)) > 0.5).float()
                        mat_stm = mat_bal * (1.0 - 2.0 * stm_black)
                        mat_tgt = torch.tanh(mat_stm / 16.0)
                        loss_m = F.mse_loss(m_pred.squeeze(1).float(), mat_tgt, reduction="none")

                        w_sum = w.sum()
                        if (not torch.isfinite(w_sum)) or float(w_sum) <= 1e-12:
                            wnorm = torch.full_like(w, 1.0 / max(1, int(w.numel())))
                        else:
                            wnorm = w / w_sum

                        p_term = (loss_p * wnorm).sum()
                        v_term = (loss_v * wnorm).sum()
                        m_term = (loss_m * wnorm).sum()
                        loss = p_term + v_term + 0.05 * m_term
                        value_mse = F.mse_loss(
                            wdl_logits_to_value(wdl_logits.float()),
                            z.float().clamp(-1.0, 1.0),
                            reduction="mean",
                        )
                        policy_entropy = -(log_p_full.exp() * log_p_full).sum(dim=1).mean()

                        if not torch.isfinite(loss):
                            nonfinite_state["streak"] += 1
                            print(
                                f"[WARN] non-finite loss at step={global_step} "
                                f"(p={float(p_term.detach()):.4f} "
                                f"v={float(v_term.detach()):.4f} m={float(m_term.detach()):.4f}) "
                                f"streak={nonfinite_state['streak']} -> skip",
                                flush=True,
                            )
                            opt.zero_grad(set_to_none=True)
                            if nonfinite_state["streak"] >= int(CFG.get("NONFINITE_RECOVER_STREAK", 8)):
                                recover_from_nonfinite("loss")
                            continue

                        opt.zero_grad(set_to_none=True)
                        if scaler is not None:
                            scaler.scale(loss).backward()
                            scaler.unscale_(opt)
                            torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
                            scaler.step(opt)
                            scaler.update()
                        else:
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
                            opt.step()

                        nonfinite_state["streak"] = 0

                        # One optimizer step completed.
                        global_step += 1

                        loss_val = float(loss.detach())
                        p_val = float(p_term.detach())
                        v_val = float(v_term.detach())
                        value_mse_val = float(value_mse.detach())
                        policy_entropy_val = float(policy_entropy.detach())
                        last_loss = loss_val
                        loss_ema = loss_val if loss_ema is None else (0.98 * loss_ema + 0.02 * loss_val)
                        p_ema = p_val if p_ema is None else (0.98 * p_ema + 0.02 * p_val)
                        v_ema = v_val if v_ema is None else (0.98 * v_ema + 0.02 * v_val)
                        value_mse_ema = (
                            value_mse_val
                            if value_mse_ema is None
                            else (0.98 * value_mse_ema + 0.02 * value_mse_val)
                        )
                        policy_entropy_ema = (
                            policy_entropy_val
                            if policy_entropy_ema is None
                            else (0.98 * policy_entropy_ema + 0.02 * policy_entropy_val)
                        )

                        # scheduler
                        if scheduler is not None:
                            every = int(CFG.get("LR_SCHED_EVERY", 0))
                            if every > 0 and (global_step % every == 0):
                                scheduler.step(float(loss_ema))
                        recover_every = int(CFG.get("LR_RUNTIME_RECOVER_CHECK_EVERY", 0))
                        if recover_every > 0 and (global_step % recover_every == 0):
                            scheduler = maybe_runtime_recover_learning_rate(
                                opt=opt,
                                scheduler=scheduler,
                                cfg=CFG,
                                global_step=global_step,
                                loss_ema=float(loss_ema),
                                state=lr_runtime_state,
                            )

                        if global_step % 50 == 0:
                            lr_now = float(opt.param_groups[0].get("lr", 0.0))
                            print(
                                f"[TRAIN] step={global_step} loss={loss_val:.4f} ema={float(loss_ema):.4f} "
                                f"(p={p_val:.4f} v={v_val:.4f} vmse={value_mse_val:.4f} pent={policy_entropy_val:.4f}) "
                                f"p_ema={float(p_ema):.4f} v_ema={float(v_ema):.4f} "
                                f"lr={lr_now:.6g} buf={len(buf)}",
                                flush=True,
                            )

                        if global_step % int(CFG["SAVE_EVERY_TRAIN_STEPS"]) == 0:
                            save_weights_and_ckpt(global_step)

                        maybe_run_arena(global_step)


                if game_count % 50 == 0 and len(window) == window_n:
                    w_red = sum(1 for o, _, _, _ in window if o > 0)
                    w_black = sum(1 for o, _, _, _ in window if o < 0)
                    w_draw = window_n - w_red - w_black
                    avg_ply = sum(p for _, _, p, _ in window) / float(window_n)
                    win_time = sum(d for _, _, _, d in window)
                    reason_counts = collections.Counter(r for _, r, _, _ in window)
                    reason_str = " ".join(f"{k}={v}" for k, v in sorted(reason_counts.items()))

                    lr_now = float(opt.param_groups[0].get("lr", 0.0))
                    print("=" * 92, flush=True)
                    print(
                        f"[WINDOW] last={window_n} games | step={global_step} loss={float(last_loss):.4f} "
                        f"ema={float(loss_ema or last_loss):.4f} lr={lr_now:.6g} buffer={len(buf)}",
                        flush=True,
                    )
                    print(
                        f"  R/B/D: {w_red}/{w_black}/{w_draw}  ("
                        f"{w_red/window_n*100:.1f}% / {w_black/window_n*100:.1f}% / {w_draw/window_n*100:.1f}%)",
                        flush=True,
                    )
                    print(f"  avg_ply={avg_ply:.1f} | window_time={win_time:.1f}s", flush=True)
                    if reason_str:
                        print(f"  reasons: {reason_str}", flush=True)
                    print("=" * 92, flush=True)

            elif msg[0] == "skip":
                _, outcome, reason, ply, tag, duration = msg
                game_count += 1
                if outcome > 0:
                    stat["red"] += 1
                elif outcome < 0:
                    stat["black"] += 1
                else:
                    stat["draw"] += 1
                window.append((float(outcome), str(reason), int(ply), float(duration)))

    except KeyboardInterrupt:
        print("\n[MAIN] stopping...", flush=True)
    finally:
        stop_flag.set()
        for p in procs:
            p.terminate()

        # exit save
        try:
            save_weights_and_ckpt(global_step)
        except Exception as e:
            print(f"[SAVE] failed at exit: {e}", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # Use filesystem-backed sharing to avoid file-descriptor pressure on large runs.
    mp.set_sharing_strategy("file_system")

    trainer_main()
