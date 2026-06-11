"""Lightweight live power dashboard: GPU0 + GPU1 (measured) + CPU (estimated).

- GPU power is read from nvidia-smi (accurate, NVIDIA sensor).
- CPU power on Windows is not directly readable without a 3rd-party service.
  Here we estimate from CPU utilization: idle baseline + linear up to TDP.
  The scaling constants below are tuned for the AMD Threadripper 7970X
  (350W TDP, 32 cores).  Adjust `CPU_TDP_W` / `CPU_IDLE_W` for other chips.

Usage:
    python tools/power_dashboard.py                  # default 2s refresh
    python tools/power_dashboard.py --interval 1.0   # faster refresh
    python tools/power_dashboard.py --cpu-tdp 170    # Intel i9 etc.

Works on both Windows PowerShell and WSL. Requires `nvidia-smi` in PATH and,
optionally, `psutil` (pip install psutil) — if missing we fall back to zero.

Ctrl-C to exit.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time


# --- Tunable constants for CPU power estimate ---
# AMD Threadripper 7970X defaults.  Override with CLI flags if needed.
DEFAULT_CPU_TDP_W = 350.0
DEFAULT_CPU_IDLE_W = 100.0  # socket + IO die + uncore baseline

# Max width of the bar we draw.  Also caps max-displayed watts per line.
BAR_CELLS = 30
BAR_MAX_W_CPU = 400.0
BAR_MAX_W_GPU = 600.0
BAR_MAX_W_TOTAL = 1600.0


def _get_gpu_power() -> list[tuple[int, str, float, float, int, int]]:
    """Returns [(index, short_name, power_w, util_pct, mem_used_mib, mem_total_mib), ...]."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,power.draw,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=3,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return []
    rows: list[tuple[int, str, float, float, int, int]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            idx = int(parts[0])
            name = parts[1].replace("NVIDIA GeForce ", "")
            power = float(parts[2])
            util = float(parts[3])
            mem_u = int(float(parts[4]))
            mem_t = int(float(parts[5]))
        except ValueError:
            continue
        rows.append((idx, name, power, util, mem_u, mem_t))
    return rows


def _get_cpu_pct() -> float:
    """Returns aggregate CPU util 0..100, or 0.0 if psutil missing."""
    try:
        import psutil  # type: ignore
    except ImportError:
        return 0.0
    # interval=None uses non-blocking sample vs the last call
    return float(psutil.cpu_percent(interval=None))


def _estimate_cpu_power_w(cpu_pct: float, idle_w: float, tdp_w: float) -> float:
    dynamic = (tdp_w - idle_w) * max(0.0, min(cpu_pct, 100.0)) / 100.0
    return idle_w + dynamic


def _bar(watts: float, cap: float, cells: int = BAR_CELLS) -> str:
    filled = max(0, min(cells, int(round(watts / cap * cells))))
    return "█" * filled + "·" * (cells - filled)


def _clear_screen() -> None:
    # Simple cross-platform cursor-home + clear.  Works in Windows Terminal,
    # conhost.exe (Windows 10+ with VT sequences), and Linux terminals.
    sys.stdout.write("\033[2J\033[H")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--interval", type=float, default=2.0, help="refresh seconds (default 2.0)")
    ap.add_argument("--cpu-tdp", type=float, default=DEFAULT_CPU_TDP_W,
                    help=f"CPU TDP watts (default {DEFAULT_CPU_TDP_W}, tuned for TR 7970X)")
    ap.add_argument("--cpu-idle", type=float, default=DEFAULT_CPU_IDLE_W,
                    help=f"CPU idle-baseline watts (default {DEFAULT_CPU_IDLE_W})")
    ap.add_argument("--no-clear", action="store_true",
                    help="don't clear screen between frames (useful for piping to a log)")
    args = ap.parse_args()

    # Prime psutil so the first cpu_percent() call returns a non-zero value.
    _get_cpu_pct()
    time.sleep(0.1)

    term_cols = shutil.get_terminal_size((80, 24)).columns
    hr = "─" * max(50, min(term_cols - 2, 66))

    try:
        while True:
            gpus = _get_gpu_power()
            cpu_pct = _get_cpu_pct()
            cpu_w = _estimate_cpu_power_w(cpu_pct, args.cpu_idle, args.cpu_tdp)
            gpu_total = sum(row[2] for row in gpus)
            total = gpu_total + cpu_w

            if not args.no_clear:
                _clear_screen()
            print(f"Power Dashboard  {time.strftime('%H:%M:%S')}")
            print(hr)
            print(
                f"  CPU (est) : {cpu_w:6.1f} W  [{_bar(cpu_w, BAR_MAX_W_CPU)}]  {cpu_pct:5.1f}% util"
            )
            for idx, name, w, util, mem_u, mem_t in gpus:
                mem_pct = 100.0 * mem_u / mem_t if mem_t else 0.0
                print(
                    f"  GPU{idx} {name:<8}: {w:6.1f} W  [{_bar(w, BAR_MAX_W_GPU)}]  "
                    f"{util:5.1f}% util  mem {mem_u//1024}/{mem_t//1024}GB ({mem_pct:4.1f}%)"
                )
            print(hr)
            print(f"  TOTAL     : {total:6.1f} W  [{_bar(total, BAR_MAX_W_TOTAL)}]")
            print()
            print(
                f"  Refresh: {args.interval:.1f}s  |  "
                f"CPU power is estimated (idle {args.cpu_idle:.0f}W + util × {(args.cpu_tdp - args.cpu_idle):.0f}W)  |  "
                "Ctrl-C to exit"
            )
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nexiting.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
