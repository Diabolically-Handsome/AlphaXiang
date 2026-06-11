"""Audit v12+ training shards before finetuning.

Checks the invariants that matter for legal-masked policy training:
- legal move CSR exists and has one row per sample
- oracle_policy indices, when present, are contained in the legal set
- teacher_q indices, when present, are contained in the legal set
- stm_is_black and FEN metadata exist for downstream canonical re-labeling

This is intentionally read-only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch


def _iter_shards(root: Path, pattern: str) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob(pattern) if p.is_file())


def _row_set(offsets: torch.Tensor, idxs: torch.Tensor, i: int) -> set[int]:
    start = int(offsets[i].item())
    end = int(offsets[i + 1].item())
    if end <= start:
        return set()
    return {int(x) for x in idxs[start:end].tolist()}


def _audit_csr_against_legal(
    *,
    n: int,
    payload: dict[str, Any],
    prefix: str,
    legal_offsets: torch.Tensor | None,
    legal_idxs: torch.Tensor | None,
    max_examples: int,
) -> dict[str, Any]:
    offsets = payload.get(f"{prefix}_offsets")
    idxs = payload.get(f"{prefix}_idxs")
    if offsets is None or idxs is None:
        return {
            "present": False,
            "rows": 0,
            "entries": 0,
            "rows_with_entries": 0,
            "illegal_entries": 0,
            "bad_rows": 0,
            "examples": [],
        }
    offsets = offsets.to(torch.int64)
    idxs = idxs.to(torch.int64)
    rows_with_entries = 0
    illegal_entries = 0
    bad_rows = 0
    examples: list[dict[str, Any]] = []
    for i in range(n):
        start = int(offsets[i].item())
        end = int(offsets[i + 1].item())
        if end <= start:
            continue
        rows_with_entries += 1
        if legal_offsets is None or legal_idxs is None:
            continue
        legal = _row_set(legal_offsets, legal_idxs, i)
        if not legal:
            continue
        row_bad = [int(x) for x in idxs[start:end].tolist() if int(x) not in legal]
        if row_bad:
            bad_rows += 1
            illegal_entries += len(row_bad)
            if len(examples) < max_examples:
                examples.append({
                    "row": i,
                    "bad": row_bad[:16],
                    "legal_count": len(legal),
                })
    return {
        "present": True,
        "rows": int(offsets.numel() - 1),
        "entries": int(idxs.numel()),
        "rows_with_entries": rows_with_entries,
        "illegal_entries": illegal_entries,
        "bad_rows": bad_rows,
        "examples": examples,
    }


def audit_one(path: Path, max_examples: int) -> dict[str, Any]:
    t0 = time.monotonic()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "state" not in payload:
        return {"path": str(path), "error": "missing state"}
    n = int(payload["state"].shape[0])

    legal_offsets = payload.get("legal_offsets")
    legal_idxs = payload.get("legal_idxs")
    if legal_offsets is not None:
        legal_offsets = legal_offsets.to(torch.int64)
    if legal_idxs is not None:
        legal_idxs = legal_idxs.to(torch.int64)
    legal_present = legal_offsets is not None and legal_idxs is not None
    legal_rows = int(legal_offsets.numel() - 1) if legal_offsets is not None else 0
    legal_entries = int(legal_idxs.numel()) if legal_idxs is not None else 0
    empty_legal_rows = 0
    if legal_present:
        counts = legal_offsets[1:] - legal_offsets[:-1]
        empty_legal_rows = int((counts == 0).sum().item())

    oracle_meta = payload.get("oracle_policy_meta") or {}
    if not isinstance(oracle_meta, dict):
        oracle_meta = {"raw_meta": str(oracle_meta)}
    teacher_meta = payload.get("teacher_q_meta") or {}
    if not isinstance(teacher_meta, dict):
        teacher_meta = {"raw_meta": str(teacher_meta)}

    oracle = _audit_csr_against_legal(
        n=n,
        payload=payload,
        prefix="oracle_policy",
        legal_offsets=legal_offsets,
        legal_idxs=legal_idxs,
        max_examples=max_examples,
    )
    teacher_q = _audit_csr_against_legal(
        n=n,
        payload=payload,
        prefix="teacher_q",
        legal_offsets=legal_offsets,
        legal_idxs=legal_idxs,
        max_examples=max_examples,
    )

    fens = payload.get("fens")
    stm = payload.get("stm_is_black")
    dirty = (
        not legal_present
        or legal_rows != n
        or oracle["illegal_entries"] > 0
        or teacher_q["illegal_entries"] > 0
        or (oracle["present"] and not bool(oracle_meta.get("canonical_action", False)))
    )
    return {
        "path": str(path),
        "samples": n,
        "has_legal": bool(legal_present),
        "legal_rows": legal_rows,
        "legal_entries": legal_entries,
        "empty_legal_rows": empty_legal_rows,
        "has_fens": isinstance(fens, list) and len(fens) == n,
        "has_stm_is_black": isinstance(stm, torch.Tensor) and int(stm.numel()) == n,
        "oracle_policy": oracle,
        "oracle_policy_meta": oracle_meta,
        "teacher_q": teacher_q,
        "teacher_q_meta": teacher_meta,
        "dirty": bool(dirty),
        "seconds": time.monotonic() - t0,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", help="Shard files or directories to scan.")
    p.add_argument("--pattern", default="shard_*.pt")
    p.add_argument("--max-examples", type=int, default=5)
    p.add_argument("--json-out", default=None)
    p.add_argument("--fail-on-dirty", action="store_true")
    args = p.parse_args()

    shard_paths: list[Path] = []
    for raw in args.paths:
        shard_paths.extend(_iter_shards(Path(raw), args.pattern))
    shard_paths = sorted(dict.fromkeys(shard_paths))
    if not shard_paths:
        raise SystemExit("no shards found")

    results = []
    totals = {
        "shards": 0,
        "dirty_shards": 0,
        "samples": 0,
        "oracle_illegal_entries": 0,
        "teacher_q_illegal_entries": 0,
        "missing_legal_shards": 0,
        "missing_fen_shards": 0,
        "missing_stm_shards": 0,
        "oracle_without_canonical_meta": 0,
    }
    for path in shard_paths:
        row = audit_one(path, max_examples=int(args.max_examples))
        results.append(row)
        totals["shards"] += 1
        totals["samples"] += int(row.get("samples", 0))
        if row.get("dirty"):
            totals["dirty_shards"] += 1
        if not row.get("has_legal"):
            totals["missing_legal_shards"] += 1
        if not row.get("has_fens"):
            totals["missing_fen_shards"] += 1
        if not row.get("has_stm_is_black"):
            totals["missing_stm_shards"] += 1
        oracle = row.get("oracle_policy") or {}
        teacher_q = row.get("teacher_q") or {}
        totals["oracle_illegal_entries"] += int(oracle.get("illegal_entries", 0))
        totals["teacher_q_illegal_entries"] += int(teacher_q.get("illegal_entries", 0))
        meta = row.get("oracle_policy_meta") or {}
        if oracle.get("present") and not bool(meta.get("canonical_action", False)):
            totals["oracle_without_canonical_meta"] += 1
        print(
            f"{'DIRTY' if row.get('dirty') else 'OK   '} {path} "
            f"samples={row.get('samples')} legal={row.get('has_legal')} "
            f"oracle_bad={oracle.get('illegal_entries', 0)} "
            f"teacher_q_bad={teacher_q.get('illegal_entries', 0)}",
            flush=True,
        )

    payload = {"totals": totals, "shards": results}
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {out}", flush=True)

    print(json.dumps(totals, indent=2, ensure_ascii=False), flush=True)
    if args.fail_on_dirty and totals["dirty_shards"] > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
