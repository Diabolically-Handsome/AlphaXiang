#!/usr/bin/env python3
"""Inventory V12.8 formal FullPika d20/d20 root-regret data.

This script is deliberately read-only.  It separates formal root d20 + child d20
audits from provisional shallow runs, counts JSONL/shard outputs, and reports
whether the current dataset is large enough to justify training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _root_key(record: dict[str, Any]) -> str:
    pos = record.get("position", {}) if isinstance(record.get("position"), dict) else {}
    return "\n".join(
        [
            str(pos.get("fen", "")),
            str(pos.get("game_index", "")),
            str(pos.get("ply", "")),
            str(pos.get("opening_id", "")),
            str(pos.get("chosen_uci", "")),
        ]
    )


def _jsonl_stats(path: Path) -> dict[str, Any]:
    rows = 0
    roots: set[str] = set()
    selected = 0
    refuted = 0
    bad_selected = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            try:
                row = json.loads(line)
            except Exception:
                continue
            roots.add(
                "\n".join(
                    [
                        str(row.get("audit_json", "")),
                        str(row.get("fen", "")),
                        str(row.get("game_index", "")),
                        str(row.get("ply", "")),
                    ]
                )
            )
            if bool(row.get("is_selected")):
                selected += 1
                if float(row.get("regret_cp", 0.0) or 0.0) >= 150.0:
                    bad_selected += 1
            if bool(row.get("is_refuted")):
                refuted += 1
    return {
        "path": str(path),
        "rows": rows,
        "roots": len(roots),
        "selected_rows": selected,
        "refuted_rows": refuted,
        "bad_selected_rows": bad_selected,
    }


def _is_audit(payload: dict[str, Any]) -> bool:
    summary = payload.get("summary", {})
    return (
        isinstance(summary, dict)
        and "pika_root_depth" in summary
        and "pika_child_depth" in summary
        and isinstance(payload.get("records"), list)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/laure/alphaxiang/v128_fullpika_root_retune")
    parser.add_argument("--target-roots", type=int, default=10000)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    args = parser.parse_args()

    root = Path(args.root)
    audit_rows: list[dict[str, Any]] = []
    formal_keys: set[str] = set()
    provisional_keys: set[str] = set()

    for path in sorted(root.rglob("*.json")):
        payload = _load_json(path)
        if payload is None or not _is_audit(payload):
            continue
        summary = payload.get("summary", {})
        records = payload.get("records", []) or []
        root_depth = summary.get("pika_root_depth")
        child_depth = summary.get("pika_child_depth")
        formal = isinstance(root_depth, int) and isinstance(child_depth, int) and root_depth >= 20 and child_depth >= 20
        keys = {_root_key(record) for record in records}
        if formal:
            formal_keys.update(keys)
        else:
            provisional_keys.update(keys)
        elapsed_s = summary.get("elapsed_s")
        roots_per_hour = None
        if isinstance(elapsed_s, (int, float)) and elapsed_s > 0:
            roots_per_hour = len(records) / float(elapsed_s) * 3600.0
        audit_rows.append(
            {
                "path": str(path),
                "records": len(records),
                "unique_roots": len(keys),
                "formal_d20d20": formal,
                "pika_root_depth": root_depth,
                "pika_child_depth": child_depth,
                "mcts_sims": summary.get("mcts_sims"),
                "elapsed_s": elapsed_s,
                "roots_per_hour": roots_per_hour,
                "bad_root_count": (summary.get("counts", {}) or {}).get("bad_root"),
                "catastrophic_count": (summary.get("counts", {}) or {}).get("catastrophic"),
            }
        )

    jsonl_rows = [_jsonl_stats(path) for path in sorted(root.rglob("*.jsonl"))]

    shard_rows: list[dict[str, Any]] = []
    for manifest_path in sorted(root.rglob("manifest.json")):
        manifest = _load_json(manifest_path) or {}
        shard_rows.append(
            {
                "path": str(manifest_path),
                "dir": str(manifest_path.parent),
                "total_samples_written": manifest.get("total_samples_written"),
                "fullpika_ok": manifest.get("fullpika_ok"),
                "fullpika_depths": manifest.get("fullpika_depths"),
                "has_do_not_train": any(child.name.startswith("DO_NOT_TRAIN") for child in manifest_path.parent.iterdir()),
            }
        )

    formal_audits = [row for row in audit_rows if row["formal_d20d20"]]
    provisional_audits = [row for row in audit_rows if not row["formal_d20d20"]]
    speed_samples = [row["roots_per_hour"] for row in formal_audits if isinstance(row.get("roots_per_hour"), float)]
    avg_roots_per_hour = sum(speed_samples) / len(speed_samples) if speed_samples else None
    target = int(args.target_roots)
    remaining = max(0, target - len(formal_keys))
    eta_hours = remaining / avg_roots_per_hour if avg_roots_per_hour and avg_roots_per_hour > 0 else None

    report = {
        "root": str(root),
        "target_roots": target,
        "formal_d20d20": {
            "audit_files": len(formal_audits),
            "records_sum": sum(int(row["records"]) for row in formal_audits),
            "unique_roots": len(formal_keys),
        },
        "provisional": {
            "audit_files": len(provisional_audits),
            "records_sum": sum(int(row["records"]) for row in provisional_audits),
            "unique_roots": len(provisional_keys),
        },
        "remaining_to_target": remaining,
        "avg_roots_per_hour_observed": avg_roots_per_hour,
        "eta_hours_at_observed_rate": eta_hours,
        "training_allowed": len(formal_keys) >= target,
        "audit_files": audit_rows,
        "jsonl_files": jsonl_rows,
        "shards": shard_rows,
    }

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# V12.8 d20 Root Data Inventory",
        "",
        f"- formal d20/d20 audit files: {len(formal_audits)}",
        f"- formal d20/d20 records: {sum(int(row['records']) for row in formal_audits)}",
        f"- formal d20/d20 unique roots: {len(formal_keys)} / {target}",
        f"- provisional records excluded: {sum(int(row['records']) for row in provisional_audits)}",
        f"- remaining to target: {remaining}",
    ]
    if avg_roots_per_hour:
        lines.append(f"- observed d20/d20 labeling rate: {avg_roots_per_hour:.1f} roots/hour")
    if eta_hours is not None:
        lines.append(f"- projected time to {target}: {eta_hours:.1f} hours at observed rate")
    lines.append(f"- training allowed: {'yes' if report['training_allowed'] else 'no'}")
    lines.append("")
    lines.append("## Formal Audits")
    for row in formal_audits:
        rate = row.get("roots_per_hour")
        rate_text = f", {rate:.1f} roots/hour" if isinstance(rate, float) else ""
        lines.append(
            f"- `{Path(row['path']).name}`: {row['records']} roots, "
            f"root d{row['pika_root_depth']} child d{row['pika_child_depth']}{rate_text}"
        )
    if provisional_audits:
        lines.append("")
        lines.append("## Provisional Excluded")
        for row in provisional_audits:
            lines.append(
                f"- `{Path(row['path']).name}`: {row['records']} roots, "
                f"root d{row['pika_root_depth']} child d{row['pika_child_depth']}"
            )
    lines.append("")
    lines.append("## Shards")
    for row in shard_rows:
        lines.append(
            f"- `{Path(row['dir']).name}`: samples={row['total_samples_written']} "
            f"fullpika_ok={row['fullpika_ok']} DO_NOT_TRAIN={row['has_do_not_train']}"
        )
    text = "\n".join(lines) + "\n"
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(text, encoding="utf-8")

    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
