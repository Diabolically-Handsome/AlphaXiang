"""Add `canonical_action: True` flag to existing v12 shards' oracle_policy_meta.

Post-canonical-fix v12 shards have correct canonical indices in
`oracle_policy_idxs`, but the meta flag indicating this was never written.
shard_hygiene_audit flags them DIRTY for that reason. This script writes
the flag in place (atomic via .tmp move) without modifying any data.

Idempotent — safe to re-run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def fix_one(shard_path: Path, dry_run: bool = False) -> dict:
    payload = torch.load(shard_path, map_location="cpu", weights_only=False)
    meta = payload.get("oracle_policy_meta")
    if meta is None:
        return {"path": str(shard_path), "status": "no_oracle_meta", "changed": False}
    if not isinstance(meta, dict):
        return {"path": str(shard_path), "status": "non_dict_meta", "changed": False}
    if meta.get("canonical_action") is True:
        return {"path": str(shard_path), "status": "already_canonical", "changed": False}
    meta["canonical_action"] = True
    payload["oracle_policy_meta"] = meta
    if dry_run:
        return {"path": str(shard_path), "status": "would_fix", "changed": False}
    tmp = shard_path.with_suffix(shard_path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(shard_path)
    return {"path": str(shard_path), "status": "fixed", "changed": True}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", help="Shard files or directories to scan.")
    p.add_argument("--pattern", default="shard_*.pt")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    shards: list[Path] = []
    for raw in args.paths:
        rp = Path(raw)
        if rp.is_file():
            shards.append(rp)
        elif rp.is_dir():
            shards.extend(sorted(p for p in rp.rglob(args.pattern) if p.is_file()))
    shards = sorted(dict.fromkeys(shards))
    if not shards:
        raise SystemExit("no shards found")

    n_changed = n_already = n_skipped = 0
    for s in shards:
        r = fix_one(s, dry_run=bool(args.dry_run))
        status = r["status"]
        if status == "fixed":
            n_changed += 1
        elif status == "already_canonical":
            n_already += 1
        else:
            n_skipped += 1
        print(f"  {status:>22} {r['path']}", flush=True)
    print(f"DONE: changed={n_changed} already_canonical={n_already} skipped={n_skipped}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
