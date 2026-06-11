#!/usr/bin/env python3
"""Verify checkpoint integrity and print steps."""
import os
import sys
import torch


def main():
    paths = [
        "/home/laure/alphaxiang/training_runs/run_001/best.pt",
        "/home/laure/alphaxiang/training_runs/run_001/latest.pt",
        "/home/laure/alphaxiang/training_runs/run_001/snapshots/latest_step108000.pt",
        "/home/laure/alphaxiang/training_runs/run_001/snapshots/latest_step106000.pt",
        "/home/laure/alphaxiang/training_runs/run_001/snapshots/latest_step104000.pt",
        "/home/laure/alphaxiang/training_runs/run_001/snapshots/best_step71358.pt",
    ]
    for p in paths:
        if not os.path.exists(p):
            print(f"{p}: MISSING")
            continue
        size = os.path.getsize(p)
        try:
            s = torch.load(p, map_location="cpu", weights_only=False)
            step = s.get("global_step", "?")
            print(f"{p}: step={step} size={size} OK")
        except Exception as exc:
            print(f"{p}: CORRUPT  size={size}  err={type(exc).__name__}: {str(exc)[:80]}")


if __name__ == "__main__":
    main()
