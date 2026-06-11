"""Analyze the 10-min GPU log to find stutter events + per-GPU patterns."""
from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp/game_stutter")
    ap.add_argument("--stutter-drop-threshold", type=float, default=30.0,
                    help="Util drop in 1s greater than this (in pp) flagged as stutter event.")
    args = ap.parse_args()

    base = Path(args.dir)
    gpu_log = base / "gpu_state.csv"
    proc_log = base / "processes.csv"
    display_log = base / "display.txt"

    # Header: ts_ms,gpu_idx,util_gpu,util_mem,mem_used_mib,power_w,temp_c,fan_pct,clk_gfx_mhz,clk_mem_mhz
    samples_per_gpu: dict[int, list[dict]] = defaultdict(list)
    with gpu_log.open() as f:
        next(f)  # skip header
        for line in f:
            parts = [p.strip() for p in line.rstrip("\n").split(",")]
            if len(parts) < 10:
                continue
            try:
                row = {
                    "ts_ms": int(parts[0]),
                    "gpu_idx": int(parts[1]),
                    "util_gpu": float(parts[2]),
                    "util_mem": float(parts[3]),
                    "mem_used_mib": float(parts[4]),
                    "power_w": float(parts[5]),
                    "temp_c": float(parts[6]),
                    "fan_pct": float(parts[7]) if parts[7].replace(".", "").isdigit() else 0.0,
                    "clk_gfx_mhz": int(parts[8]),
                    "clk_mem_mhz": int(parts[9]),
                }
            except (ValueError, IndexError):
                continue
            samples_per_gpu[row["gpu_idx"]].append(row)

    # Display info
    print(display_log.read_text(encoding="utf-8"))

    # Per-GPU summary
    for gpu_idx in sorted(samples_per_gpu.keys()):
        rows = samples_per_gpu[gpu_idx]
        if len(rows) < 5:
            continue
        utils = [r["util_gpu"] for r in rows]
        powers = [r["power_w"] for r in rows]
        temps = [r["temp_c"] for r in rows]
        fans = [r["fan_pct"] for r in rows]
        gfx_clks = [r["clk_gfx_mhz"] for r in rows]

        print(f"\n========== GPU {gpu_idx} ({len(rows)} samples over "
              f"{(rows[-1]['ts_ms'] - rows[0]['ts_ms']) / 1000:.0f}s) ==========")
        print(f"  util:     mean={statistics.mean(utils):5.1f}%  median={statistics.median(utils):5.1f}%  "
              f"min={min(utils):5.1f}%  max={max(utils):5.1f}%  stdev={statistics.pstdev(utils):5.1f}")
        print(f"  power:    mean={statistics.mean(powers):5.0f}W  max={max(powers):5.0f}W")
        print(f"  temp:     mean={statistics.mean(temps):5.1f}C  max={max(temps):5.1f}C")
        print(f"  fan:      mean={statistics.mean(fans):5.1f}%  max={max(fans):5.1f}%")
        print(f"  gfx clk:  mean={statistics.mean(gfx_clks):5.0f}MHz  min={min(gfx_clks)}  max={max(gfx_clks)}")

        # Stutter detection: util drop > threshold between consecutive samples
        stutter_events = []
        for i in range(1, len(rows)):
            drop = rows[i - 1]["util_gpu"] - rows[i]["util_gpu"]
            if drop >= args.stutter_drop_threshold:
                stutter_events.append({
                    "ts_ms": rows[i]["ts_ms"],
                    "from_util": rows[i - 1]["util_gpu"],
                    "to_util": rows[i]["util_gpu"],
                    "drop_pp": drop,
                    "from_clk": rows[i - 1]["clk_gfx_mhz"],
                    "to_clk": rows[i]["clk_gfx_mhz"],
                    "from_power": rows[i - 1]["power_w"],
                    "to_power": rows[i]["power_w"],
                })
        print(f"  stutter events (util drop > {args.stutter_drop_threshold}pp in 1s): "
              f"{len(stutter_events)}")
        if stutter_events:
            for ev in stutter_events[:15]:  # first 15
                rel_s = (ev["ts_ms"] - rows[0]["ts_ms"]) / 1000
                clk_change = ev["to_clk"] - ev["from_clk"]
                print(f"    t+{rel_s:6.1f}s: util {ev['from_util']:5.1f}% -> {ev['to_util']:5.1f}% "
                      f"({ev['drop_pp']:+5.1f}pp)  power {ev['from_power']:.0f}->{ev['to_power']:.0f}W  "
                      f"clk {ev['from_clk']}->{ev['to_clk']} ({clk_change:+}MHz)")
            if len(stutter_events) > 15:
                print(f"    ... and {len(stutter_events) - 15} more")

        # Clock-state segments (look for PCIe/clock throttling pattern)
        clock_low = sum(1 for c in gfx_clks if c < 1500)
        clock_mid = sum(1 for c in gfx_clks if 1500 <= c < 2200)
        clock_high = sum(1 for c in gfx_clks if c >= 2200)
        total = len(gfx_clks)
        print(f"  clock states: low(<1500)={100*clock_low/total:4.1f}%  "
              f"mid(1500-2200)={100*clock_mid/total:4.1f}%  high(>=2200)={100*clock_high/total:4.1f}%")

    # Process summary — what was running on each GPU?
    print("\n========== PROCESSES OBSERVED (sampled every 5s) ==========")
    proc_seen: dict[tuple, list[float]] = defaultdict(list)
    bus_to_gpu: dict[str, int] = {}
    with proc_log.open() as f:
        next(f)
        for line in f:
            parts = [p.strip() for p in line.rstrip("\n").split(",")]
            if len(parts) < 5:
                continue
            try:
                pid = int(parts[1])
                pname = parts[2]
                bus_id = parts[3]
                mem = float(parts[4])
            except ValueError:
                continue
            # Map bus_id to gpu_idx
            if bus_id not in bus_to_gpu:
                bus_to_gpu[bus_id] = len(bus_to_gpu)
            gpu_idx = bus_to_gpu[bus_id]
            proc_seen[(gpu_idx, pname, pid)].append(mem)

    if not proc_seen:
        print("  (no compute processes recorded — note: graphics processes don't show here)")
    else:
        for (gpu_idx, pname, pid), mems in sorted(proc_seen.items()):
            print(f"  gpu{gpu_idx}  {pname}  pid={pid}  "
                  f"mem mean={statistics.mean(mems):.0f}MiB max={max(mems):.0f}MiB  "
                  f"appearances={len(mems)}")


if __name__ == "__main__":
    main()
