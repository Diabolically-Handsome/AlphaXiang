"""Cross-game leaf-batching evaluator.

Drop-in replacement for `_GpuModelEvaluator` from `xiangqi_mcts_ext` that
aggregates calls from multiple MCTS threads into a single GPU forward pass.

Why this matters
----------------
Each MCTS thread (e.g. one per concurrent game in `pikafish_selfplay.py` with
`--parallel-games N`) calls its evaluator with up to `eval_batch_size=16`
leaves at a time.  With per-thread evaluators, the GPU sees N independent
small calls — each pays Python→C++→cudaLaunchKernel overhead.

This wrapper accepts calls from any thread, queues them, and a single worker
thread coalesces requests into one big batch (default cap 256 leaves) which
then runs as ONE forward pass.  Results are sliced and returned to each
caller.  At parallel_games=16, that's a ~256-leaf batch instead of 16x16,
and the GPU stays continuously busy instead of doing 16 small bursts.

Interface contract (mirrors `xiangqi_mcts_ext._GpuModelEvaluator`):
    batcher = CrossGameBatcher(model, device='cuda:0', use_bfloat16=True)
    out_dict = batcher(batch_cpu_tensor)
        # out_dict = {"policy_logits": ..., "value_scalar": ..., optionally "wdl_logits": ...}
        # all on CPU, float32, contiguous
    batcher.close()  # graceful shutdown of worker thread

Concurrency model:
    - Multiple producer threads call __call__() and block on a per-call Event.
    - One consumer thread (the "worker") drains the queue, batches, runs
      model.forward, slices outputs, signals each waiter.
    - Model.forward is called from exactly one thread, so we don't need to
      worry about concurrent CUDA-context safety inside the model.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Any

import torch
from torch import Tensor, nn


_REQUIRED_OUT_KEYS = ("policy_logits", "value_scalar")
_OPTIONAL_OUT_KEYS = ("wdl_logits",)


class CrossGameBatcher:
    """Multi-thread aggregator for MCTS leaf evaluation requests."""

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "cuda:0",
        use_bfloat16: bool = True,
        max_batch_size: int = 256,
        coalesce_timeout_ms: float = 2.0,
    ) -> None:
        self.model = model.eval()
        self.device = torch.device(device)
        self.use_bfloat16 = bool(use_bfloat16) and self.device.type == "cuda"
        self.max_batch_size = max(1, int(max_batch_size))
        self.coalesce_timeout_s = max(0.0, float(coalesce_timeout_ms) / 1000.0)

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, but a CUDA batcher was requested")
        self.model.to(self.device)

        # Internal state
        self._queue: queue.Queue = queue.Queue()
        self._shutdown = threading.Event()
        self._closed = False

        # Telemetry (updated lock-free; readers may see momentarily inconsistent values, fine for stats)
        self._calls_received = 0           # total __call__ invocations
        self._batches_run = 0              # total GPU forward passes
        self._max_observed_batch = 0       # largest batch we ever ran
        self._total_leaves_processed = 0   # sum of all batch sizes

        self._worker = threading.Thread(
            target=self._run, daemon=True, name="cross-game-batcher",
        )
        self._worker.start()

    # ------------------------------------------------------------------
    # Public interface — matches xiangqi_mcts_ext._GpuModelEvaluator.__call__
    # ------------------------------------------------------------------

    def __call__(self, batch_cpu: Tensor) -> dict[str, Tensor]:
        """Submit a batch (one MCTS thread's worth of leaves) and block for result."""
        if self._closed:
            raise RuntimeError("CrossGameBatcher is closed")
        if not isinstance(batch_cpu, torch.Tensor):
            raise TypeError("batcher(batch) expects a torch.Tensor")

        # Same dtype/device normalisation as _GpuModelEvaluator does inline.
        batch_cpu = batch_cpu.detach().to(device="cpu", dtype=torch.float32).contiguous()

        result_event = threading.Event()
        slot: list[Any] = [None]
        self._queue.put((batch_cpu, result_event, slot))
        self._calls_received += 1
        result_event.wait()
        out = slot[0]
        if isinstance(out, BaseException):
            raise out
        return out  # type: ignore[return-value]

    def close(self) -> None:
        """Stop the worker thread.  Pending requests will get RuntimeError."""
        if self._closed:
            return
        self._closed = True
        self._shutdown.set()
        # Drain pending callers with a clear error
        while True:
            try:
                _, event, slot = self._queue.get_nowait()
                slot[0] = RuntimeError("CrossGameBatcher closed before request handled")
                event.set()
            except queue.Empty:
                break
        self._worker.join(timeout=5.0)

    def __enter__(self) -> "CrossGameBatcher":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, float]:
        """Return throughput stats.  Useful to verify cross-game batching is
        actually happening (avg batch size should be much larger than the
        per-thread eval_batch_size of 16)."""
        avg = (self._total_leaves_processed / self._batches_run) if self._batches_run > 0 else 0.0
        return {
            "calls_received": self._calls_received,
            "batches_run": self._batches_run,
            "max_observed_batch": self._max_observed_batch,
            "total_leaves_processed": self._total_leaves_processed,
            "avg_leaves_per_batch": avg,
            "coalesce_ratio": (self._calls_received / self._batches_run) if self._batches_run > 0 else 0.0,
        }

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._shutdown.is_set():
            requests: list[tuple[Tensor, threading.Event, list[Any]]] = []
            try:
                first = self._queue.get(timeout=0.1)
                requests.append(first)
            except queue.Empty:
                continue

            # Coalesce additional requests up to max_batch_size or timeout.
            sizes = [requests[0][0].shape[0]]
            total = sizes[0]
            deadline = time.monotonic() + self.coalesce_timeout_s
            while total < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                try:
                    req = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                requests.append(req)
                sz = req[0].shape[0]
                sizes.append(sz)
                total += sz

            try:
                self._process_batch(requests, sizes, total)
            except BaseException as exc:  # noqa: BLE001 — propagate to all waiters
                for _batch, event, slot in requests:
                    slot[0] = exc
                    event.set()

    def _process_batch(
        self,
        requests: list[tuple[Tensor, threading.Event, list[Any]]],
        sizes: list[int],
        total: int,
    ) -> None:
        """Stack inputs into one tensor, run model, slice outputs back."""
        big_batch_cpu = torch.cat([r[0] for r in requests], dim=0)
        big_batch_gpu = big_batch_cpu.to(
            self.device,
            non_blocking=self.device.type == "cuda",
        )

        with torch.inference_mode():
            if self.use_bfloat16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    model_out = self.model(big_batch_gpu)
            else:
                model_out = self.model(big_batch_gpu)

        if not isinstance(model_out, dict):
            raise TypeError("model(batch) must return a dict")

        # Move outputs to CPU once for the whole big batch, then slice.
        out_cpu: dict[str, Tensor] = {}
        for key in _REQUIRED_OUT_KEYS:
            tensor = model_out.get(key)
            if tensor is None:
                raise KeyError(f"model(batch) is missing required key {key!r}")
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"model(batch)[{key!r}] must be a torch.Tensor")
            out_cpu[key] = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        for key in _OPTIONAL_OUT_KEYS:
            tensor = model_out.get(key)
            if tensor is not None and isinstance(tensor, torch.Tensor):
                out_cpu[key] = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()

        # Telemetry
        self._batches_run += 1
        self._total_leaves_processed += total
        if total > self._max_observed_batch:
            self._max_observed_batch = total

        # Slice outputs back to each caller (preserve order: first request gets first chunk).
        offset = 0
        for (_batch, event, slot), sz in zip(requests, sizes):
            sliced = {k: v[offset:offset + sz] for k, v in out_cpu.items()}
            slot[0] = sliced
            offset += sz
            event.set()
