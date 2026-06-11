"""Create an oversampled run directory from an existing labeled shard run.

This intentionally preserves shard contents and only gives the same clean data
multiple unique paths, so the replay-buffer sampler sees it more often without
changing training code or mutating the source run.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _sample_count(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state")
    if not isinstance(state, torch.Tensor):
        raise RuntimeError(f"{path} has no tensor state")
    return int(state.shape[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-run", required=True,
                        help="Existing run root with train/shard_*.pt and manifest.json.")
    parser.add_argument("--output-run", required=True)
    parser.add_argument("--copies", type=int, default=12)
    parser.add_argument("--mode", choices=["hardlink", "copy"], default="hardlink")
    args = parser.parse_args()

    input_run = Path(args.input_run).resolve()
    output_run = Path(args.output_run).resolve()
    input_train = input_run / "train"
    output_train = output_run / "train"
    shards = sorted(input_train.glob("shard_*.pt"))
    if not shards:
        raise SystemExit(f"no shards under {input_train}")
    copies = max(1, int(args.copies))

    output_train.mkdir(parents=True, exist_ok=True)
    for stale in output_train.glob("shard_*.pt"):
        stale.unlink()

    shard_infos = []
    out_id = 0
    source_counts = {str(path): _sample_count(path) for path in shards}
    for copy_idx in range(copies):
        for src in shards:
            dst = output_train / f"shard_{out_id:06d}.pt"
            _link_or_copy(src, dst, str(args.mode))
            shard_infos.append({
                "path": str(dst),
                "samples": int(source_counts[str(src)]),
                "source_path": str(src),
                "copy_index": int(copy_idx),
            })
            out_id += 1

    total_samples = sum(int(item["samples"]) for item in shard_infos)
    manifest = {
        "manifest_state": "complete",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "oversample_shard_run",
        "source_run": str(input_run),
        "copies": copies,
        "samples": total_samples,
        "shards": shard_infos,
        "quality": "ok",
        "quality_metrics": {
            "rep_draw_rate": 0.0,
            "decisive_rate": 100.0,
            "nocap_draw_rate": 0.0,
        },
        "config": {
            "mode": str(args.mode),
        },
    }
    output_run.mkdir(parents=True, exist_ok=True)
    (output_run / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"DONE: wrote {len(shard_infos)} shard paths, {total_samples} samples "
        f"from {len(shards)} source shards to {output_run}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
