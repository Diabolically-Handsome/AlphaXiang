from __future__ import annotations

import argparse
import importlib.util
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pygame
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from xiangqi_mcts_ext import Board, canonical_action, make_gpu_evaluator, mcts_search
from xiangqi_transformer_model import (
    XiangqiPVTransformer,
    XiangqiTransformerConfig,
    load_xiangqi_model_state_dict,
)


FILE_CHARS = "abcdefghi"
SQUARE_COUNT = 90
POLICY_DIM = 8100
TERMINAL_ONGOING = -1
TERMINATION_CHECKMATE_OR_STALEMATE = 0
TERMINATION_MAX_PLIES_DRAW = 1
TERMINATION_REPETITION_DRAW = 2
TERMINATION_NO_CAPTURE_DRAW = 3
TERMINATION_PERPETUAL_CHECK_LOSS = 4
SIM_LEVELS = {
    pygame.K_1: 256,
    pygame.K_2: 800,
    pygame.K_3: 1600,
    pygame.K_4: 3200,
    pygame.K_5: 6400,
    pygame.K_6: 12800,
}

CN_RED = {1: "帅", 2: "仕", 3: "相", 4: "马", 5: "车", 6: "炮", 7: "兵"}
CN_BLACK = {1: "将", 2: "士", 3: "象", 4: "马", 5: "车", 6: "炮", 7: "卒"}
ASCII_RED = {1: "K", 2: "A", 3: "B", 4: "N", 5: "R", 6: "C", 7: "P"}
ASCII_BLACK = {1: "k", 2: "a", 3: "b", 4: "n", 5: "r", 6: "c", 7: "p"}


@dataclass(frozen=True)
class AgentSpec:
    name: str
    label: str
    evaluator: Any


@dataclass(frozen=True)
class SearchRequest:
    request_id: int
    fen: str
    turn: int
    sims: int
    plies_played: int
    no_capture_count: int
    repetition_count: int


@dataclass(frozen=True)
class SearchResult:
    request_id: int
    fen: str
    side: int
    best_move: int
    root_v: float
    elapsed_ms: float
    sims: int
    topk: list[tuple[str, float]]
    agent_label: str
    error: str | None = None


def square_to_iccs(square: int) -> str:
    y, x = divmod(int(square), 9)
    return f"{FILE_CHARS[x]}{9 - y}"


def action_to_from_to(action_id: int) -> tuple[int, int]:
    a = int(action_id)
    return a // 90, a % 90


def action_to_iccs(action_id: int) -> str:
    from_sq, to_sq = action_to_from_to(action_id)
    return square_to_iccs(from_sq) + square_to_iccs(to_sq)


def piece_label(piece_code: int, prefer_cn: bool = True) -> str:
    p = int(piece_code)
    if p == 0:
        return ""
    side = 1 if p > 0 else -1
    piece_type = abs(p)
    if prefer_cn:
        return CN_RED[piece_type] if side > 0 else CN_BLACK[piece_type]
    return ASCII_RED[piece_type] if side > 0 else ASCII_BLACK[piece_type]


def scalar_int(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.detach().reshape(-1)[0].item())
    if isinstance(value, np.ndarray):
        return int(value.reshape(-1)[0].item())
    return int(value)


def as_numpy_1d(value: Any, dtype) -> np.ndarray:
    if torch.is_tensor(value):
        array = value.detach().cpu().reshape(-1).numpy()
    elif isinstance(value, np.ndarray):
        array = value.reshape(-1)
    else:
        array = np.asarray(value).reshape(-1)
    return array.astype(dtype, copy=False)


def normalize_state_dict_keys(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    for prefix in ("module.", "_orig_mod.", "model."):
        if all(key.startswith(prefix) for key in keys):
            return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


def load_module_from_path(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def build_transformer_config(raw_config: Any) -> XiangqiTransformerConfig:
    if isinstance(raw_config, XiangqiTransformerConfig):
        return raw_config
    if isinstance(raw_config, dict):
        allowed = XiangqiTransformerConfig.__dataclass_fields__.keys()
        payload = {key: value for key, value in raw_config.items() if key in allowed}
        return XiangqiTransformerConfig(**payload)
    return XiangqiTransformerConfig()


def load_transformer_agent(checkpoint_path: Path, device: str) -> AgentSpec:
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
        raw_config = raw.get("model_config") or raw.get("config")
        step = raw.get("global_step")
    elif isinstance(raw, dict) and "state_dict" in raw:
        state_dict = raw["state_dict"]
        raw_config = raw.get("model_config") or raw.get("config")
        step = raw.get("global_step")
    elif isinstance(raw, dict) and raw and all(torch.is_tensor(value) for value in raw.values()):
        state_dict = raw
        raw_config = None
        step = None
    else:
        raise RuntimeError(f"unsupported transformer checkpoint format: {checkpoint_path}")

    config = build_transformer_config(raw_config)
    model = XiangqiPVTransformer(config)
    load_xiangqi_model_state_dict(model, normalize_state_dict_keys(state_dict))
    evaluator = make_gpu_evaluator(model=model, device=device, use_bfloat16=True)
    step_suffix = f" step {int(step)}" if step is not None else ""
    return AgentSpec(
        name="transformer",
        label=f"Transformer{step_suffix}",
        evaluator=evaluator,
    )


def pick_cnn_net_class(module: Any):
    for class_name in ("XiangqiNetV11", "XiangqiNetV10", "XiangqiNet"):
        if hasattr(module, class_name):
            return getattr(module, class_name)
    raise AttributeError("CNN engine module is missing XiangqiNetV11/XiangqiNetV10/XiangqiNet")


def wdl_logits_to_value(wdl_logits: Tensor) -> Tensor:
    probs = F.softmax(wdl_logits.float(), dim=1)
    return (probs[:, 0] - probs[:, 2]).unsqueeze(1)


class ResidualBlockCompat(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class LegacyScalarCnnNet(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        channels = int(cfg.get("MODEL_CHANNELS", 128))
        blocks = int(cfg.get("MODEL_RES_BLOCKS", 20))
        in_channels = int(cfg.get("INPUT_CHANNELS", 115))

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlockCompat(channels) for _ in range(blocks)])

        self.p_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.p_bn = nn.BatchNorm2d(2)
        self.p_fc = nn.Linear(2 * 10 * 9, POLICY_DIM)

        self.v_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.v_bn = nn.BatchNorm2d(32)
        self.v_fc1 = nn.Linear(32 * 10 * 9, 128)
        self.v_fc2 = nn.Linear(128, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.blocks(self.stem(x))

        policy = F.relu(self.p_bn(self.p_conv(x))).flatten(1)
        policy_logits = self.p_fc(policy)

        value = F.relu(self.v_bn(self.v_conv(x))).flatten(1)
        value = F.relu(self.v_fc1(value))
        value_scalar = torch.tanh(self.v_fc2(value))
        return policy_logits, value_scalar


class LegacyCnnEvaluator:
    def __init__(self, model: nn.Module, device: str | torch.device = "cuda:0", use_fp16: bool = True):
        self.device = torch.device(device)
        self.use_fp16 = bool(use_fp16) and self.device.type == "cuda"
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA evaluator requested for CNN, but CUDA is unavailable")
        self.model = model.eval().to(self.device)

    def __call__(self, batch_cpu: Tensor) -> dict[str, Tensor]:
        if not isinstance(batch_cpu, torch.Tensor):
            raise TypeError("CNN evaluator expects a torch.Tensor")

        batch_cpu = batch_cpu.detach().to(device="cpu", dtype=torch.float32).contiguous()
        batch_gpu = batch_cpu.to(self.device, non_blocking=self.device.type == "cuda")

        with torch.inference_mode():
            if self.use_fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    model_out = self.model(batch_gpu)
            else:
                model_out = self.model(batch_gpu)

        if isinstance(model_out, dict):
            policy_logits = model_out["policy_logits"]
            value_scalar = model_out["value_scalar"]
            wdl_logits = model_out.get("wdl_logits")
        elif isinstance(model_out, tuple):
            if len(model_out) == 3:
                policy_logits, wdl_logits, _ = model_out
                value_scalar = wdl_logits_to_value(wdl_logits)
            elif len(model_out) == 2:
                policy_logits, value_scalar = model_out
                wdl_logits = None
            else:
                raise RuntimeError(f"unsupported CNN model output tuple length: {len(model_out)}")
        else:
            raise RuntimeError(f"unsupported CNN model output type: {type(model_out)!r}")

        result = {
            "policy_logits": policy_logits.detach().to(device="cpu", dtype=torch.float32).contiguous(),
            "value_scalar": value_scalar.detach().to(device="cpu", dtype=torch.float32).contiguous(),
        }
        if wdl_logits is not None:
            result["wdl_logits"] = wdl_logits.detach().to(device="cpu", dtype=torch.float32).contiguous()
        return result


def load_cnn_agent(engine_path: Path, weights_path: Path, device: str) -> AgentSpec:
    module = load_module_from_path(engine_path)
    cfg = dict(getattr(module, "CFG", {})) if isinstance(getattr(module, "CFG", None), dict) else {}
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"unsupported CNN weights format: {weights_path}")
    normalized_state_dict = normalize_state_dict_keys(state_dict)

    value_head_weight = normalized_state_dict.get("v_fc2.weight")
    uses_scalar_value_head = (
        torch.is_tensor(value_head_weight)
        and value_head_weight.ndim == 2
        and int(value_head_weight.shape[0]) == 1
        and "m_conv.weight" not in normalized_state_dict
    )

    if uses_scalar_value_head:
        model = LegacyScalarCnnNet(cfg).cpu()
    else:
        net_class = pick_cnn_net_class(module)
        model = net_class(cfg).cpu()

    model.load_state_dict(normalized_state_dict, strict=True)
    evaluator = LegacyCnnEvaluator(model=model, device=device, use_fp16=True)
    return AgentSpec(
        name="cnn",
        label="CNN best",
        evaluator=evaluator,
    )


class SearchWorker(threading.Thread):
    def __init__(
        self,
        agent: AgentSpec,
        request_queue: "queue.Queue[SearchRequest | None]",
        result_queue: "queue.Queue[SearchResult]",
        c_puct: float,
        eval_batch_size: int,
        seed_base: int,
        max_plies: int,
        repeat_limit: int,
        repeat_min_ply: int,
        no_capture_limit: int,
        tactical_mate1_extension: bool,
        tactical_mate2_extension: bool,
    ) -> None:
        super().__init__(daemon=True, name=f"search-{agent.name}")
        self.agent = agent
        self.request_queue = request_queue
        self.result_queue = result_queue
        self.c_puct = float(c_puct)
        self.eval_batch_size = int(eval_batch_size)
        self.seed_base = int(seed_base)
        self.max_plies = int(max_plies)
        self.repeat_limit = int(repeat_limit)
        self.repeat_min_ply = int(repeat_min_ply)
        self.no_capture_limit = int(no_capture_limit)
        self.tactical_mate1_extension = bool(tactical_mate1_extension)
        self.tactical_mate2_extension = bool(tactical_mate2_extension)

    def run(self) -> None:
        while True:
            request = self.request_queue.get()
            if request is None:
                return

            try:
                board = Board()
                board.set_fen(request.fen)
                board.set_search_context(
                    int(request.plies_played),
                    int(request.no_capture_count),
                    max(1, int(request.repetition_count)),
                )
                start_time = time.time()
                best_move, idxs, probs, root_v = mcts_search(
                    board=board,
                    net=self.agent.evaluator,
                    num_simulations=int(request.sims),
                    c_puct=self.c_puct,
                    q_weight=1.0,
                    q_clip=1.0,
                    add_root_noise=False,
                    dirichlet_alpha=0.3,
                    dirichlet_eps=0.0,
                    temperature_move=1e-6,
                    temperature_target=1.0,
                    eval_batch_size=self.eval_batch_size,
                    seed=(self.seed_base + request.request_id * 104729) & 0x7FFFFFFF,
                    canonical_input=True,
                    canonical_policy=True,
                    max_plies=self.max_plies,
                    repeat_limit=self.repeat_limit,
                    repeat_min_ply=self.repeat_min_ply,
                    no_capture_limit=self.no_capture_limit,
                    tactical_mate1_extension=self.tactical_mate1_extension,
                    tactical_mate2_extension=self.tactical_mate2_extension,
                )

                idxs_np = as_numpy_1d(idxs, np.int64)
                probs_np = as_numpy_1d(probs, np.float32)
                stm_black = bool(request.turn == 1)
                topk: list[tuple[str, float]] = []
                if idxs_np.size > 0:
                    order = np.argsort(-probs_np)
                    for index in order[:8]:
                        action_id = int(idxs_np[index])
                        if stm_black:
                            action_id = int(canonical_action(action_id, True))
                        topk.append((action_to_iccs(action_id), float(probs_np[index])))

                self.result_queue.put(
                    SearchResult(
                        request_id=request.request_id,
                        fen=request.fen,
                        side=int(request.turn),
                        best_move=int(best_move),
                        root_v=float(root_v),
                        elapsed_ms=float((time.time() - start_time) * 1000.0),
                        sims=int(request.sims),
                        topk=topk,
                        agent_label=self.agent.label,
                    )
                )
            except Exception as exc:
                self.result_queue.put(
                    SearchResult(
                        request_id=request.request_id,
                        fen=request.fen,
                        side=int(request.turn),
                        best_move=-1,
                        root_v=0.0,
                        elapsed_ms=0.0,
                        sims=int(request.sims),
                        topk=[],
                        agent_label=self.agent.label,
                        error=repr(exc),
                    )
                )


def init_fonts(
    piece_size: int = 34,
    ui_size: int = 24,
    small_size: int = 20,
) -> tuple[pygame.font.Font, pygame.font.Font, pygame.font.Font]:
    pygame.font.init()

    cjk_candidates = [
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]

    font_path = None
    for name in cjk_candidates:
        try:
            path = pygame.font.match_font(name)
        except Exception:
            path = None
        if path:
            font_path = path
            break

    if font_path is None:
        direct_candidates = [
            "/mnt/c/Windows/Fonts/msyh.ttc",
            "/mnt/c/Windows/Fonts/msyhbd.ttc",
            "/mnt/c/Windows/Fonts/simhei.ttf",
            "/mnt/c/Windows/Fonts/simsun.ttc",
            "/mnt/c/Windows/Fonts/simkai.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        ]
        for path in direct_candidates:
            if os.path.isfile(path):
                font_path = path
                break

    try:
        if font_path:
            piece_font = pygame.font.Font(font_path, int(piece_size))
            piece_font.set_bold(True)
            ui_font = pygame.font.Font(font_path, int(ui_size))
            small_font = pygame.font.Font(font_path, int(small_size))
        else:
            piece_font = pygame.font.SysFont("consolas", int(piece_size), bold=True)
            ui_font = pygame.font.SysFont("consolas", int(ui_size))
            small_font = pygame.font.SysFont("consolas", int(small_size))
    except Exception:
        piece_font = pygame.font.SysFont("consolas", int(piece_size), bold=True)
        ui_font = pygame.font.SysFont("consolas", int(ui_size))
        small_font = pygame.font.SysFont("consolas", int(small_size))

    return piece_font, ui_font, small_font


def compute_layout(screen: pygame.Surface) -> tuple[tuple[int, int, int, int], int, int, int, int]:
    screen_w, screen_h = screen.get_size()
    margin_x = max(36, int(screen_w * 0.026))
    margin_y = max(32, int(screen_h * 0.045))
    panel_min_w = max(500, int(screen_w * 0.30))

    board_h = screen_h - 2 * margin_y
    board_w = int(board_h * 8.0 / 9.0)
    max_board_w = max(360, screen_w - 2 * margin_x - panel_min_w)
    if board_w > max_board_w:
        board_w = int(max_board_w)
        board_h = int(board_w * 9.0 / 8.0)

    board_h = max(540, min(board_h, screen_h - 2 * margin_y))
    board_w = int(board_h * 8.0 / 9.0)
    board_x = margin_x
    board_y = max(margin_y, int((screen_h - board_h) / 2))
    panel_x = board_x + board_w + max(36, int(screen_w * 0.03))
    search_x = panel_x + max(310, int(screen_w * 0.18))
    result_y = max(360, int(screen_h * 0.45))
    moves_y = result_y + 48
    return (int(board_x), int(board_y), int(board_w), int(board_h)), int(panel_x), int(search_x), int(result_y), int(moves_y)


def draw_board(screen: pygame.Surface, rect: tuple[int, int, int, int], dark_mode: bool = False) -> None:
    x0, y0, width, height = rect
    cell_w = width / 8.0
    cell_h = height / 9.0
    if dark_mode:
        line_color = (177, 137, 89)
        board_fill = (54, 42, 34)
        board_border = (160, 111, 66)
    else:
        line_color = (60, 45, 30)
        board_fill = (225, 196, 150)
        board_border = (135, 94, 53)

    board_surface = pygame.Rect(x0 - 18, y0 - 18, width + 36, height + 36)
    pygame.draw.rect(screen, board_fill, board_surface, border_radius=12)
    pygame.draw.rect(screen, board_border, board_surface, width=3, border_radius=12)

    for row in range(10):
        y = y0 + row * cell_h
        pygame.draw.line(screen, line_color, (x0, y), (x0 + width, y), 1)

    for col in range(9):
        x = x0 + col * cell_w
        if col in (0, 8):
            pygame.draw.line(screen, line_color, (x, y0), (x, y0 + height), 1)
        else:
            pygame.draw.line(screen, line_color, (x, y0), (x, y0 + 4 * cell_h), 1)
            pygame.draw.line(screen, line_color, (x, y0 + 5 * cell_h), (x, y0 + height), 1)

    pygame.draw.line(screen, line_color, (x0 + 3 * cell_w, y0), (x0 + 5 * cell_w, y0 + 2 * cell_h), 1)
    pygame.draw.line(screen, line_color, (x0 + 5 * cell_w, y0), (x0 + 3 * cell_w, y0 + 2 * cell_h), 1)
    pygame.draw.line(screen, line_color, (x0 + 3 * cell_w, y0 + 7 * cell_h), (x0 + 5 * cell_w, y0 + 9 * cell_h), 1)
    pygame.draw.line(screen, line_color, (x0 + 5 * cell_w, y0 + 7 * cell_h), (x0 + 3 * cell_w, y0 + 9 * cell_h), 1)


def board_sq_to_screen(board_rect: tuple[int, int, int, int], square: int, flip: bool) -> tuple[float, float]:
    x0, y0, width, height = board_rect
    cell_w = width / 8.0
    cell_h = height / 9.0
    row, col = divmod(int(square), 9)
    if flip:
        row = 9 - row
        col = 8 - col
    return x0 + col * cell_w, y0 + row * cell_h


def mouse_to_square(board_rect: tuple[int, int, int, int], mx: int, my: int, flip: bool) -> int | None:
    x0, y0, width, height = board_rect
    if mx < x0 - 10 or mx > x0 + width + 10 or my < y0 - 10 or my > y0 + height + 10:
        return None
    cell_w = width / 8.0
    cell_h = height / 9.0
    col = int(round((mx - x0) / cell_w))
    row = int(round((my - y0) / cell_h))
    col = max(0, min(8, col))
    row = max(0, min(9, row))
    if flip:
        row = 9 - row
        col = 8 - col
    return row * 9 + col


def legal_dest_squares(board: Board, from_sq: int) -> list[int]:
    dests: list[int] = []
    for action in board.legal_moves():
        src, dst = action_to_from_to(int(action))
        if src == int(from_sq):
            dests.append(dst)
    return dests


def try_push_move(board: Board, from_sq: int, to_sq: int) -> int | None:
    target = int(from_sq) * 90 + int(to_sq)
    for action in board.legal_moves():
        if int(action) == target:
            board.push(target)
            return target
    return None


def draw_pieces(
    screen: pygame.Surface,
    board_rect: tuple[int, int, int, int],
    board: Board,
    font_piece: pygame.font.Font,
    last_move: int | None,
    flip: bool,
    selected_sq: int | None = None,
    legal_dests: list[int] | None = None,
    dark_mode: bool = False,
) -> None:
    x0, y0, width, height = board_rect
    cell_w = width / 8.0
    cell_h = height / 9.0

    if last_move is not None:
        from_sq, to_sq = action_to_from_to(last_move)
        for square in (from_sq, to_sq):
            sx, sy = board_sq_to_screen(board_rect, square, flip)
            highlight = pygame.Rect(sx - 0.45 * cell_w, sy - 0.45 * cell_h, 0.9 * cell_w, 0.9 * cell_h)
            pygame.draw.rect(screen, (106, 87, 52) if dark_mode else (240, 225, 170), highlight)

    if selected_sq is not None:
        sx, sy = board_sq_to_screen(board_rect, selected_sq, flip)
        selected_rect = pygame.Rect(sx - 0.47 * cell_w, sy - 0.47 * cell_h, 0.94 * cell_w, 0.94 * cell_h)
        pygame.draw.rect(screen, (122, 184, 132) if dark_mode else (186, 220, 160), selected_rect, width=3)
    for square in legal_dests or []:
        sx, sy = board_sq_to_screen(board_rect, square, flip)
        pygame.draw.circle(
            screen,
            (95, 185, 124) if dark_mode else (80, 145, 80),
            (int(sx), int(sy)),
            max(5, int(min(cell_w, cell_h) * 0.08)),
        )

    for square in range(SQUARE_COUNT):
        piece = scalar_int(board.piece_at(square))
        if piece == 0:
            continue
        sx, sy = board_sq_to_screen(board_rect, square, flip)
        radius = int(min(cell_w, cell_h) * 0.34)
        if dark_mode:
            fill = (42, 38, 35)
            border = (196, 158, 104)
            text_color = (255, 105, 105) if piece > 0 else (226, 220, 205)
        else:
            fill = (248, 244, 236)
            border = (50, 40, 30)
            text_color = (180, 35, 35) if piece > 0 else (35, 35, 35)
        pygame.draw.circle(screen, fill, (int(sx), int(sy)), radius)
        pygame.draw.circle(screen, border, (int(sx), int(sy)), radius, 2)
        label = piece_label(piece, prefer_cn=True)
        surf = font_piece.render(label, True, text_color)
        if surf.get_bounding_rect().width == 0:
            surf = font_piece.render(piece_label(piece, prefer_cn=False), True, text_color)
        rect = surf.get_rect(center=(int(sx), int(sy)))
        screen.blit(surf, rect)


def draw_text_block(
    screen: pygame.Surface,
    font: pygame.font.Font,
    x: int,
    y: int,
    lines: list[str],
    color: tuple[int, int, int] = (28, 28, 28),
    line_gap: int = 28,
) -> None:
    current_y = y
    for line in lines:
        surf = font.render(line, True, color)
        screen.blit(surf, (x, current_y))
        current_y += line_gap


def build_sidebar_lines(
    board: Board,
    red_agent: AgentSpec,
    black_agent: AgentSpec,
    sims: int,
    move_history: list[int],
    paused: bool,
    pending_side: int | None,
    last_results: dict[int, SearchResult],
    error_text: str,
) -> tuple[list[str], list[str], list[str]]:
    turn = int(board.turn())
    turn_text = "Red to move" if turn == 0 else "Black to move"
    state_text = "Paused" if paused else "Running"
    pending_text = "None" if pending_side is None else ("Red" if pending_side == 0 else "Black")

    header_lines = [
        "Cyber Cricket Arena",
        f"State: {state_text}",
        f"Turn: {turn_text}",
        f"Sims: {sims}",
        f"Pending: {pending_text}",
        f"Red: {red_agent.label}",
        f"Black: {black_agent.label}",
        "SPACE pause/resume",
        "N new game  S swap sides",
        "1-6 sims: 256/800/1600/3200/6400/12800",
        "ESC quit",
    ]
    if error_text:
        header_lines.append(f"Last error: {error_text}")

    move_lines = ["Moves:"]
    if not move_history:
        move_lines.append("(none)")
    else:
        for ply_index, move in enumerate(move_history[-24:], start=max(len(move_history) - 23, 1)):
            move_lines.append(f"{ply_index:>3}. {action_to_iccs(move)}")

    search_lines = ["Search:"]
    for side, title in ((0, "Red"), (1, "Black")):
        result = last_results.get(side)
        if result is None:
            search_lines.append(f"{title}: (no search yet)")
            continue
        search_lines.append(
            f"{title}: {result.agent_label}  v={result.root_v:+.3f}  {result.elapsed_ms:.0f}ms"
        )
        if result.topk:
            for move_text, probability in result.topk[:4]:
                search_lines.append(f"  {move_text}  {probability:.3f}")
        if result.error:
            search_lines.append(f"  error: {result.error}")

    return header_lines, move_lines, search_lines


def outcome_text_from_red_view(result_red_view: int) -> str:
    if result_red_view > 0:
        return "Red wins"
    if result_red_view < 0:
        return "Black wins"
    return "Draw"


def terminal_status(
    board: Board,
    *,
    max_plies: int,
    repeat_limit: int,
    repeat_min_ply: int,
    no_capture_limit: int,
) -> tuple[int, int]:
    terminal_code = int(
        board.terminal_code(
            int(max_plies),
            int(repeat_limit),
            int(repeat_min_ply),
            int(no_capture_limit),
        )
    )
    result_red_view = int(board.terminal_result_red_view(terminal_code)) if terminal_code != TERMINAL_ONGOING else 0
    return terminal_code, result_red_view


def outcome_text_from_terminal(result_red_view: int, termination_code: int) -> str:
    if termination_code == TERMINATION_REPETITION_DRAW:
        return "Draw (repetition)"
    if termination_code == TERMINATION_PERPETUAL_CHECK_LOSS:
        return "Loss by perpetual check"
    if termination_code == TERMINATION_NO_CAPTURE_DRAW:
        return "Draw (no capture)"
    if termination_code == TERMINATION_MAX_PLIES_DRAW:
        return "Draw (max plies)"
    return outcome_text_from_red_view(result_red_view)


def validate_move_or_result(
    board: Board,
    best_move: int,
    *,
    max_plies: int,
    repeat_limit: int,
    repeat_min_ply: int,
    no_capture_limit: int,
) -> tuple[bool, str | None]:
    if best_move < 0:
        terminal_code, _ = terminal_status(
            board,
            max_plies=max_plies,
            repeat_limit=repeat_limit,
            repeat_min_ply=repeat_min_ply,
            no_capture_limit=no_capture_limit,
        )
        if terminal_code != TERMINAL_ONGOING:
            return True, None
        return False, "search returned no move on a non-terminal position"
    if not board.is_legal(int(best_move)):
        return False, f"search returned illegal move: {action_to_iccs(best_move)}"
    return True, None


def request_search(
    board: Board,
    paused: bool,
    pending_side: int | None,
    request_id: int,
    sims: int,
    side_workers: dict[int, SearchWorker],
    side_queues: dict[int, queue.Queue[SearchRequest | None]],
    human_side: int | None = None,
    *,
    max_plies: int,
    repeat_limit: int,
    repeat_min_ply: int,
    no_capture_limit: int,
) -> tuple[int, int | None, str]:
    if paused or pending_side is not None:
        return request_id, pending_side, board.fen()
    terminal_code, _ = terminal_status(
        board,
        max_plies=max_plies,
        repeat_limit=repeat_limit,
        repeat_min_ply=repeat_min_ply,
        no_capture_limit=no_capture_limit,
    )
    if terminal_code != TERMINAL_ONGOING:
        return request_id, pending_side, board.fen()
    turn = int(board.turn())
    if human_side is not None and turn == int(human_side):
        return request_id, pending_side, board.fen()
    if turn not in side_queues:
        return request_id, pending_side, board.fen()
    fen = board.fen()
    side_queues[turn].put(
        SearchRequest(
            request_id=request_id,
            fen=fen,
            turn=turn,
            sims=sims,
            plies_played=int(board.plies_played()),
            no_capture_count=int(board.no_capture_count()),
            repetition_count=max(1, int(board.current_repetition_count())),
        )
    )
    return request_id + 1, turn, fen


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual AI-vs-AI Xiangqi battle: Transformer vs CNN.")
    parser.add_argument(
        "--transformer-checkpoint",
        default=str((Path(__file__).resolve().parent / "training_runs" / "run_001" / "best.pt").resolve()),
    )
    parser.add_argument(
        "--cnn-engine",
        default=str((Path(__file__).resolve().parent / "CNN" / "Chessv11_cpp_hist8_115_mps_fp16.py").resolve()),
    )
    parser.add_argument(
        "--cnn-weights",
        default=str((Path(__file__).resolve().parent / "CNN" / "best.pth").resolve()),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--transformer-side", choices=["red", "black"], default="red")
    parser.add_argument(
        "--human-side",
        choices=["red", "black", "none"],
        default="none",
        help="Human-vs-v13 mode. Choose the human side; 'none' keeps the original AI-vs-AI battle.",
    )
    parser.add_argument("--num-simulations", type=int, default=800)
    parser.add_argument("--c-puct", type=float, default=1.25)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-plies", type=int, default=480)
    parser.add_argument("--repeat-limit", type=int, default=6)
    parser.add_argument("--repeat-min-ply", type=int, default=30)
    parser.add_argument("--no-capture-limit", type=int, default=120)
    parser.add_argument("--flip-board", action="store_true", help="Render black side at the bottom.")
    parser.add_argument("--fullscreen", action="store_true", help="Start in fullscreen mode with a larger board.")
    parser.add_argument("--dark-mode", action="store_true", help="Use a darker board and UI palette.")
    parser.add_argument(
        "--tactical-mate1-extension",
        action="store_true",
        help="Enable the mate-in-1 tactical leaf extension used by the strongest V13 search setup.",
    )
    parser.add_argument(
        "--tactical-mate2-extension",
        action="store_true",
        help="Enable the check-forced mate-in-2 tactical leaf extension used by the strongest V13 search setup.",
    )
    parser.add_argument("--auto-quit-after-moves", type=int, default=0)
    args = parser.parse_args()

    transformer_agent = load_transformer_agent(Path(args.transformer_checkpoint).resolve(), device=args.device)

    human_side: int | None
    if args.human_side == "red":
        human_side = 0
    elif args.human_side == "black":
        human_side = 1
    else:
        human_side = None

    human_agent = AgentSpec(name="human", label="Human", evaluator=None)
    if human_side is None:
        cnn_agent = load_cnn_agent(
            engine_path=Path(args.cnn_engine).resolve(),
            weights_path=Path(args.cnn_weights).resolve(),
            device=args.device,
        )
        if args.transformer_side == "red":
            red_agent = transformer_agent
            black_agent = cnn_agent
        else:
            red_agent = cnn_agent
            black_agent = transformer_agent
    else:
        red_agent = human_agent if human_side == 0 else transformer_agent
        black_agent = human_agent if human_side == 1 else transformer_agent

    pygame.init()
    if args.fullscreen:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    else:
        screen = pygame.display.set_mode((1460, 860), pygame.RESIZABLE)
    pygame.display.set_caption("Xiangqi Cyber Cricket Arena")
    board_rect, panel_x, search_x, result_y, moves_y = compute_layout(screen)
    piece_font_size = max(34, min(58, int(board_rect[3] / 17)))
    ui_font_size = max(22, min(30, int(screen.get_height() / 42)))
    small_font_size = max(18, min(24, int(screen.get_height() / 50)))
    font_piece, font_ui, font_small = init_fonts(piece_font_size, ui_font_size, small_font_size)
    board = Board()
    move_history: list[int] = []
    last_move: int | None = None
    last_results: dict[int, SearchResult] = {}
    pending_side: int | None = None
    pending_fen = board.fen()
    request_id = 1
    paused = False
    result_text = ""
    error_text = ""
    view_flip = bool(args.flip_board) if human_side is None else (human_side == 1)
    current_sims = int(args.num_simulations)
    selected_sq: int | None = None
    legal_dests: list[int] | None = None

    result_queue: "queue.Queue[SearchResult]" = queue.Queue()
    side_queues: dict[int, queue.Queue[SearchRequest | None]] = {
        0: queue.Queue(),
        1: queue.Queue(),
    }
    worker_red_agent = red_agent if human_side is None or human_side != 0 else transformer_agent
    worker_black_agent = black_agent if human_side is None or human_side != 1 else transformer_agent
    side_workers = {
        0: SearchWorker(
            worker_red_agent,
            side_queues[0],
            result_queue,
            args.c_puct,
            args.eval_batch_size,
            seed_base=7001,
            max_plies=args.max_plies,
            repeat_limit=args.repeat_limit,
            repeat_min_ply=args.repeat_min_ply,
            no_capture_limit=args.no_capture_limit,
            tactical_mate1_extension=args.tactical_mate1_extension,
            tactical_mate2_extension=args.tactical_mate2_extension,
        ),
        1: SearchWorker(
            worker_black_agent,
            side_queues[1],
            result_queue,
            args.c_puct,
            args.eval_batch_size,
            seed_base=9001,
            max_plies=args.max_plies,
            repeat_limit=args.repeat_limit,
            repeat_min_ply=args.repeat_min_ply,
            no_capture_limit=args.no_capture_limit,
            tactical_mate1_extension=args.tactical_mate1_extension,
            tactical_mate2_extension=args.tactical_mate2_extension,
        ),
    }
    for worker in side_workers.values():
        worker.start()

    clock = pygame.time.Clock()
    request_id, pending_side, pending_fen = request_search(
        board=board,
        paused=paused,
        pending_side=pending_side,
        request_id=request_id,
        sims=current_sims,
        side_workers=side_workers,
        side_queues=side_queues,
        human_side=human_side,
        max_plies=args.max_plies,
        repeat_limit=args.repeat_limit,
        repeat_min_ply=args.repeat_min_ply,
        no_capture_limit=args.no_capture_limit,
    )

    running = True
    try:
        while running:
            clock.tick(60)

            try:
                while True:
                    result = result_queue.get_nowait()
                    side = int(result.side)
                    last_results[side] = result
                    if result.error:
                        error_text = result.error
                    if pending_side is None or result.fen != pending_fen:
                        continue
                    if side != pending_side:
                        continue
                    pending_side = None

                    ok, move_error = validate_move_or_result(
                        board,
                        result.best_move,
                        max_plies=args.max_plies,
                        repeat_limit=args.repeat_limit,
                        repeat_min_ply=args.repeat_min_ply,
                        no_capture_limit=args.no_capture_limit,
                    )
                    if not ok:
                        error_text = move_error or "unknown search error"
                        result_text = "Search error"
                        paused = True
                        continue

                    if result.best_move >= 0:
                        board.push(int(result.best_move))
                        move_history.append(int(result.best_move))
                        last_move = int(result.best_move)

                    terminal_code, terminal_result = terminal_status(
                        board,
                        max_plies=args.max_plies,
                        repeat_limit=args.repeat_limit,
                        repeat_min_ply=args.repeat_min_ply,
                        no_capture_limit=args.no_capture_limit,
                    )
                    if terminal_code != TERMINAL_ONGOING:
                        result_text = outcome_text_from_terminal(terminal_result, terminal_code)
                    elif args.auto_quit_after_moves > 0 and len(move_history) >= args.auto_quit_after_moves:
                        result_text = f"Stopped after {len(move_history)} moves"
                        paused = True
                        running = False

                    request_id, pending_side, pending_fen = request_search(
                        board=board,
                        paused=paused,
                        pending_side=pending_side,
                        request_id=request_id,
                        sims=current_sims,
                        side_workers=side_workers,
                        side_queues=side_queues,
                        human_side=human_side,
                        max_plies=args.max_plies,
                        repeat_limit=args.repeat_limit,
                        repeat_min_ply=args.repeat_min_ply,
                        no_capture_limit=args.no_capture_limit,
                    )
            except queue.Empty:
                pass

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        if not paused and not result_text:
                            request_id, pending_side, pending_fen = request_search(
                                board=board,
                                paused=paused,
                                pending_side=pending_side,
                                request_id=request_id,
                                sims=current_sims,
                                side_workers=side_workers,
                                side_queues=side_queues,
                                human_side=human_side,
                                max_plies=args.max_plies,
                                repeat_limit=args.repeat_limit,
                                repeat_min_ply=args.repeat_min_ply,
                                no_capture_limit=args.no_capture_limit,
                            )
                    elif event.key == pygame.K_n:
                        board = Board()
                        move_history.clear()
                        last_move = None
                        selected_sq = None
                        legal_dests = None
                        result_text = ""
                        error_text = ""
                        pending_side = None
                        pending_fen = board.fen()
                        request_id += 1000
                        request_id, pending_side, pending_fen = request_search(
                            board=board,
                            paused=paused,
                            pending_side=pending_side,
                            request_id=request_id,
                            sims=current_sims,
                            side_workers=side_workers,
                            side_queues=side_queues,
                            human_side=human_side,
                            max_plies=args.max_plies,
                            repeat_limit=args.repeat_limit,
                            repeat_min_ply=args.repeat_min_ply,
                            no_capture_limit=args.no_capture_limit,
                        )
                    elif event.key == pygame.K_s:
                        if human_side is None:
                            red_agent, black_agent = black_agent, red_agent
                            side_workers[0].agent = red_agent
                            side_workers[1].agent = black_agent
                        else:
                            human_side = 1 - human_side
                            red_agent = human_agent if human_side == 0 else transformer_agent
                            black_agent = human_agent if human_side == 1 else transformer_agent
                            view_flip = (human_side == 1)
                        board = Board()
                        move_history.clear()
                        last_move = None
                        selected_sq = None
                        legal_dests = None
                        result_text = ""
                        error_text = ""
                        pending_side = None
                        pending_fen = board.fen()
                        request_id += 1000
                        request_id, pending_side, pending_fen = request_search(
                            board=board,
                            paused=paused,
                            pending_side=pending_side,
                            request_id=request_id,
                            sims=current_sims,
                            side_workers=side_workers,
                            side_queues=side_queues,
                            human_side=human_side,
                            max_plies=args.max_plies,
                            repeat_limit=args.repeat_limit,
                            repeat_min_ply=args.repeat_min_ply,
                            no_capture_limit=args.no_capture_limit,
                        )
                    elif event.key in SIM_LEVELS:
                        current_sims = SIM_LEVELS[event.key]
                        error_text = ""
                        if pending_side is None and not paused and not result_text:
                            request_id, pending_side, pending_fen = request_search(
                                board=board,
                                paused=paused,
                                pending_side=pending_side,
                                request_id=request_id,
                                sims=current_sims,
                                side_workers=side_workers,
                                side_queues=side_queues,
                                human_side=human_side,
                                max_plies=args.max_plies,
                                repeat_limit=args.repeat_limit,
                                repeat_min_ply=args.repeat_min_ply,
                                no_capture_limit=args.no_capture_limit,
                            )

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and human_side is not None:
                    if paused or result_text or pending_side is not None:
                        continue
                    if int(board.turn()) != int(human_side):
                        continue
                    square = mouse_to_square(board_rect, *event.pos, flip=view_flip)
                    if square is None:
                        continue
                    piece = scalar_int(board.piece_at(square))
                    if selected_sq is None:
                        if piece == 0:
                            continue
                        if (human_side == 0 and piece < 0) or (human_side == 1 and piece > 0):
                            continue
                        selected_sq = square
                        legal_dests = legal_dest_squares(board, square)
                        continue
                    if piece != 0 and ((human_side == 0 and piece > 0) or (human_side == 1 and piece < 0)):
                        selected_sq = square
                        legal_dests = legal_dest_squares(board, square)
                        continue
                    move = try_push_move(board, selected_sq, square)
                    selected_sq = None
                    legal_dests = None
                    if move is None:
                        continue
                    move_history.append(int(move))
                    last_move = int(move)
                    terminal_code, terminal_result = terminal_status(
                        board,
                        max_plies=args.max_plies,
                        repeat_limit=args.repeat_limit,
                        repeat_min_ply=args.repeat_min_ply,
                        no_capture_limit=args.no_capture_limit,
                    )
                    if terminal_code != TERMINAL_ONGOING:
                        result_text = outcome_text_from_terminal(terminal_result, terminal_code)
                    request_id, pending_side, pending_fen = request_search(
                        board=board,
                        paused=paused,
                        pending_side=pending_side,
                        request_id=request_id,
                        sims=current_sims,
                        side_workers=side_workers,
                        side_queues=side_queues,
                        human_side=human_side,
                        max_plies=args.max_plies,
                        repeat_limit=args.repeat_limit,
                        repeat_min_ply=args.repeat_min_ply,
                        no_capture_limit=args.no_capture_limit,
                    )

                elif event.type == pygame.VIDEORESIZE and not args.fullscreen:
                    screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                    board_rect, panel_x, search_x, result_y, moves_y = compute_layout(screen)
                    piece_font_size = max(34, min(58, int(board_rect[3] / 17)))
                    ui_font_size = max(22, min(30, int(screen.get_height() / 42)))
                    small_font_size = max(18, min(24, int(screen.get_height() / 50)))
                    font_piece, font_ui, font_small = init_fonts(
                        piece_font_size, ui_font_size, small_font_size
                    )

            if result_text and pending_side is None:
                paused = True

            screen.fill((18, 20, 22) if args.dark_mode else (244, 240, 232))
            draw_board(screen, board_rect, dark_mode=bool(args.dark_mode))
            draw_pieces(
                screen,
                board_rect,
                board,
                font_piece,
                last_move=last_move,
                flip=view_flip,
                selected_sq=selected_sq,
                legal_dests=legal_dests,
                dark_mode=bool(args.dark_mode),
            )

            header_lines, move_lines, search_lines = build_sidebar_lines(
                board=board,
                red_agent=red_agent,
                black_agent=black_agent,
                sims=current_sims,
                move_history=move_history,
                paused=paused,
                pending_side=pending_side,
                last_results=last_results,
                error_text=error_text,
            )

            text_color = (225, 221, 211) if args.dark_mode else (28, 28, 28)
            small_text_color = (204, 198, 188) if args.dark_mode else (28, 28, 28)
            result_color = (255, 118, 118) if args.dark_mode else (160, 20, 20)
            draw_text_block(screen, font_ui, panel_x, board_rect[1], header_lines, color=text_color)
            if result_text:
                draw_text_block(screen, font_ui, panel_x, result_y, [f"Result: {result_text}"], color=result_color)
            draw_text_block(
                screen,
                font_small,
                panel_x,
                moves_y,
                move_lines,
                color=small_text_color,
                line_gap=max(22, small_font_size + 4),
            )
            draw_text_block(
                screen,
                font_small,
                search_x,
                board_rect[1],
                search_lines,
                color=small_text_color,
                line_gap=max(22, small_font_size + 4),
            )

            pygame.display.flip()
    finally:
        for side_queue in side_queues.values():
            side_queue.put(None)
        for worker in side_workers.values():
            worker.join(timeout=2.0)
        pygame.quit()


if __name__ == "__main__":
    main()
