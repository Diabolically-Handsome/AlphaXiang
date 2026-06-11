from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils import cpp_extension
from torch.utils.cpp_extension import load


_THIS_DIR = Path(__file__).resolve().parent
_SOURCE_PATH = _THIS_DIR / "xqcpp_ext_hist8_115.cpp"
_BUILD_DIR = _THIS_DIR / "__pycache__" / "torch_extensions"
_EXTENSION_NAME = "xqcpp_ext_hist8_115_ext"
_EXTENSION = None


def _load_extension():
    global _EXTENSION
    if _EXTENSION is None:
        _BUILD_DIR.mkdir(parents=True, exist_ok=True)
        _patch_torch_lib_path_for_spaces()
        _EXTENSION = load(
            name=_EXTENSION_NAME,
            sources=[str(_SOURCE_PATH)],
            build_directory=str(_BUILD_DIR),
            extra_cflags=["-O3", "-std=c++17"],
            verbose=False,
        )
    return _EXTENSION


def _patch_torch_lib_path_for_spaces() -> None:
    torch_lib_path = Path(cpp_extension.TORCH_LIB_PATH)
    if " " not in str(torch_lib_path):
        return

    safe_root = Path.home() / ".cache" / "xiangqi_mcts_torch_lib"
    safe_root.parent.mkdir(parents=True, exist_ok=True)
    if not (safe_root.exists() or safe_root.is_symlink()):
        safe_root.symlink_to(torch_lib_path, target_is_directory=True)
    cpp_extension.TORCH_LIB_PATH = str(safe_root)


class _GpuModelEvaluator:
    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "cuda:0",
        use_bfloat16: bool = True,
    ) -> None:
        self.model = model.eval()
        self.device = torch.device(device)
        self.use_bfloat16 = use_bfloat16 and self.device.type == "cuda"

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, but a CUDA evaluator was requested")

        self.model.to(self.device)

    def __call__(self, batch_cpu: Tensor) -> dict[str, Tensor]:
        if not isinstance(batch_cpu, torch.Tensor):
            raise TypeError("net(batch) expects a torch.Tensor")

        batch_cpu = batch_cpu.detach().to(device="cpu", dtype=torch.float32).contiguous()
        model_input = batch_cpu.to(
            self.device,
            non_blocking=self.device.type == "cuda",
        )

        with torch.inference_mode():
            if self.use_bfloat16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    model_out = self.model(model_input)
            else:
                model_out = self.model(model_input)

        if not isinstance(model_out, dict):
            raise TypeError("model(batch) must return a dict")

        result: dict[str, Tensor] = {}
        for key in ("policy_logits", "value_scalar", "wdl_logits"):
            tensor = model_out.get(key)
            if tensor is None:
                continue
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"model(batch)['{key}'] must be a torch.Tensor")
            result[key] = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()

        for required_key in ("policy_logits", "value_scalar"):
            if required_key not in result:
                raise KeyError(f"model(batch) is missing required key '{required_key}'")

        return result


def make_gpu_evaluator(
    model: nn.Module,
    device: str | torch.device = "cuda:0",
    use_bfloat16: bool = True,
):
    return _GpuModelEvaluator(model=model, device=device, use_bfloat16=use_bfloat16)


def mcts_search(*args, **kwargs):
    return _call_mcts("mcts_search", *args, **kwargs)


def mcts_search_with_root_stats(*args, **kwargs):
    return _call_mcts("mcts_search_with_root_stats", *args, **kwargs)


def _call_mcts(function_name: str, *args, **kwargs):
    if not kwargs:
        return getattr(_load_extension(), function_name)(*args)

    # Some pybind11 builds of the C++ extension expose mcts_search as
    # positional-only even though older builds accepted keyword arguments.
    # Keep the Python API stable by canonicalizing kwargs here.
    order = [
        "board",
        "net",
        "num_simulations",
        "c_puct",
        "q_weight",
        "q_clip",
        "add_root_noise",
        "dirichlet_alpha",
        "dirichlet_eps",
        "temperature_move",
        "temperature_target",
        "eval_batch_size",
        "seed",
        "canonical_input",
        "canonical_policy",
        "max_plies",
        "repeat_limit",
        "repeat_min_ply",
        "no_capture_limit",
        "tactical_mate1_extension",
        "tactical_mate2_extension",
        "c_puct_base",
        "c_puct_factor",
        "fpu_reduction_root",
        "fpu_reduction_tree",
    ]
    defaults = {
        "q_weight": 1.0,
        "q_clip": 1.0,
        "add_root_noise": False,
        "dirichlet_alpha": 0.3,
        "dirichlet_eps": 0.1,
        "temperature_move": 1.0,
        "temperature_target": 1.0,
        "eval_batch_size": 16,
        "seed": 0,
        "canonical_input": True,
        "canonical_policy": True,
        "max_plies": 0,
        "repeat_limit": 0,
        "repeat_min_ply": 0,
        "no_capture_limit": 0,
        "tactical_mate1_extension": False,
        "tactical_mate2_extension": False,
        "c_puct_base": 1.0,
        "c_puct_factor": 0.0,
        "fpu_reduction_root": -1.0,
        "fpu_reduction_tree": -1.0,
    }
    unknown = set(kwargs) - set(order)
    if unknown:
        raise TypeError(f"unknown mcts_search keyword argument(s): {sorted(unknown)}")

    values = list(args)
    for name in order[len(values):]:
        if name in kwargs:
            values.append(kwargs.pop(name))
        elif name in defaults:
            values.append(defaults[name])
        else:
            raise TypeError(f"missing required mcts_search argument: {name}")
    if kwargs:
        raise TypeError(f"duplicate mcts_search keyword argument(s): {sorted(kwargs)}")
    return getattr(_load_extension(), function_name)(*values)


def __getattr__(name: str):
    if name in {"Board", "canonical_square", "canonical_action"}:
        return getattr(_load_extension(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Board",
    "canonical_square",
    "canonical_action",
    "mcts_search",
    "mcts_search_with_root_stats",
    "make_gpu_evaluator",
]
