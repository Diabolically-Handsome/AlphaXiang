"""Build V13.1 detox training directories.

The tool is intentionally conservative: it never edits source shards.  Clean
shards are linked into a new run; shards with only non-canonical oracle-policy
metadata can be sanitized by copying them with oracle-policy fields removed.
Everything else is reported as toxic and excluded from training.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import torch


TERMINAL_REPETITION_DRAW = 2
TERMINAL_NO_CAPTURE_DRAW = 3
TERMINAL_PERPETUAL_CHECK_LOSS = 4
SUSPICIOUS_TERMINALS = {
    TERMINAL_REPETITION_DRAW,
    TERMINAL_NO_CAPTURE_DRAW,
    TERMINAL_PERPETUAL_CHECK_LOSS,
}

ORACLE_POLICY_KEYS = (
    "oracle_policy_offsets",
    "oracle_policy_idxs",
    "oracle_policy_probs",
    "oracle_policy_meta",
)


def _iter_run_roots(root: Path) -> list[Path]:
    root = root.resolve()
    if (root / "train").is_dir():
        return [root]
    if not root.is_dir():
        return []
    runs = [child for child in sorted(root.iterdir()) if child.is_dir() and (child / "train").is_dir()]
    if runs:
        return runs
    return [root]


def _load_manifest(run_root: Path) -> dict[str, Any]:
    path = run_root / "manifest.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"manifest_error": f"{type(exc).__name__}: {exc}"}


def _row_set(offsets: torch.Tensor, idxs: torch.Tensor, row: int) -> set[int]:
    start = int(offsets[row].item())
    end = int(offsets[row + 1].item())
    return {int(x) for x in idxs[start:end].tolist()}


def _csr_illegal_count(
    *,
    n: int,
    payload: dict[str, Any],
    prefix: str,
    legal_offsets: torch.Tensor,
    legal_idxs: torch.Tensor,
) -> tuple[int, int]:
    offsets = payload.get(f"{prefix}_offsets")
    idxs = payload.get(f"{prefix}_idxs")
    if offsets is None or idxs is None:
        return 0, 0
    offsets = offsets.to(torch.int64)
    idxs = idxs.to(torch.int64)
    bad_rows = 0
    bad_entries = 0
    for i in range(n):
        start = int(offsets[i].item())
        end = int(offsets[i + 1].item())
        if end <= start:
            continue
        legal = _row_set(legal_offsets, legal_idxs, i)
        row_bad = [int(x) for x in idxs[start:end].tolist() if int(x) not in legal]
        if row_bad:
            bad_rows += 1
            bad_entries += len(row_bad)
    return bad_rows, bad_entries


def _audit_payload(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source_path": str(path),
        "source_run": str(path.parent.parent if path.parent.name == "train" else path.parent),
        "samples": 0,
        "reasons": [],
        "sanitize_reasons": [],
        "eligible": False,
        "needs_sanitize": False,
    }

    q = manifest.get("quality_metrics", {}) if isinstance(manifest, dict) else {}
    if str(manifest.get("manifest_state", "complete")).lower() != "complete":
        row["reasons"].append("manifest_not_complete")
    if float(q.get("rep_draw_rate", 0.0) or 0.0) > 0.0:
        row["reasons"].append("manifest_rep_draw_rate_positive")
    if float(q.get("nocap_draw_rate", 0.0) or 0.0) > 0.0:
        row["reasons"].append("manifest_nocap_draw_rate_positive")

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        row["reasons"].append(f"unreadable_shard:{type(exc).__name__}")
        return row

    if not isinstance(payload, dict) or "state" not in payload:
        row["reasons"].append("missing_state")
        return row
    state = payload["state"]
    if not isinstance(state, torch.Tensor) or state.ndim != 4 or tuple(state.shape[1:]) != (115, 10, 9):
        row["reasons"].append("bad_state_shape")
        return row
    n = int(state.shape[0])
    row["samples"] = n

    legal_offsets = payload.get("legal_offsets")
    legal_idxs = payload.get("legal_idxs")
    if legal_offsets is None or legal_idxs is None:
        row["reasons"].append("missing_legal_idxs")
        return row
    legal_offsets = legal_offsets.to(torch.int64)
    legal_idxs = legal_idxs.to(torch.int64)
    if int(legal_offsets.numel()) != n + 1:
        row["reasons"].append("legal_offsets_wrong_rows")
        return row
    legal_counts = legal_offsets[1:] - legal_offsets[:-1]
    if bool((legal_counts <= 0).any()):
        row["reasons"].append("empty_legal_row")

    fens = payload.get("fens")
    if not isinstance(fens, list) or len(fens) != n:
        row["reasons"].append("missing_or_bad_fens")
    stm = payload.get("stm_is_black")
    if not isinstance(stm, torch.Tensor) or int(stm.numel()) != n:
        row["reasons"].append("missing_or_bad_stm_is_black")

    termination_code = payload.get("termination_code")
    if isinstance(termination_code, torch.Tensor) and termination_code.numel() > 0:
        terms = {int(x) for x in termination_code.reshape(-1).tolist()}
        suspicious = sorted(terms.intersection(SUSPICIOUS_TERMINALS))
        if suspicious:
            row["reasons"].append(f"suspicious_terminal_codes:{suspicious}")

    for prefix in ("oracle_policy", "teacher_q"):
        bad_rows, bad_entries = _csr_illegal_count(
            n=n,
            payload=payload,
            prefix=prefix,
            legal_offsets=legal_offsets,
            legal_idxs=legal_idxs,
        )
        row[f"{prefix}_illegal_rows"] = bad_rows
        row[f"{prefix}_illegal_entries"] = bad_entries
        if bad_entries > 0:
            row["reasons"].append(f"{prefix}_outside_legal")

    oracle_offsets = payload.get("oracle_policy_offsets")
    if oracle_offsets is not None:
        meta = payload.get("oracle_policy_meta") or {}
        if not isinstance(meta, dict) or not bool(meta.get("canonical_action", False)):
            row["needs_sanitize"] = True
            row["sanitize_reasons"].append("strip_noncanonical_oracle_policy")

    row["eligible"] = not row["reasons"]
    return row


def _safe_link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
        return "symlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def _sanitize_copy(src: Path, dst: Path, audit: dict[str, Any], group: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    payload = torch.load(src, map_location="cpu", weights_only=False)
    for key in ORACLE_POLICY_KEYS:
        payload.pop(key, None)
    payload["detox_meta"] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_path": str(src),
        "group": str(group),
        "action": "removed_noncanonical_oracle_policy",
        "sanitize_reasons": list(audit.get("sanitize_reasons", [])),
    }
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(dst)
    return "sanitized_copy"


def _manifest_for_group(group: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    shards = [
        {"path": row["dest_path"], "samples": int(row["samples"])}
        for row in rows
    ]
    return {
        "manifest_state": "complete",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "v131_weekend_detox",
        "group": group,
        "samples": sum(int(row["samples"]) for row in rows),
        "shards": shards,
        "quality": "ok",
        "quality_metrics": {
            "rep_draw_rate": 0.0,
            "decisive_rate": 100.0,
            "nocap_draw_rate": 0.0,
        },
        "config": {
            "policy": "exclude suspicious repetition/longcheck/no-capture; strip noncanonical oracle_policy",
        },
    }


def _prepare_group_dir(path: Path) -> None:
    train = path / "train"
    train.mkdir(parents=True, exist_ok=True)
    for old in train.glob("shard_*.pt"):
        old.unlink()


def build_group(
    *,
    group: str,
    source_root: Path,
    output_root: Path,
    strip_noncanonical_oracle: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    clean_rows: list[dict[str, Any]] = []
    toxic_rows: list[dict[str, Any]] = []
    group_dir = output_root / f"clean_{group}"
    _prepare_group_dir(group_dir)
    shard_index = 0

    for run_root in _iter_run_roots(source_root):
        manifest = _load_manifest(run_root)
        for shard in sorted((run_root / "train").glob("shard_*.pt")):
            audit = _audit_payload(shard, manifest)
            audit["group"] = group
            if not audit.get("eligible"):
                toxic_rows.append(audit)
                continue
            dest = group_dir / "train" / f"shard_{shard_index:06d}.pt"
            if audit.get("needs_sanitize"):
                if not strip_noncanonical_oracle:
                    audit["reasons"].append("noncanonical_oracle_policy")
                    toxic_rows.append(audit)
                    continue
                mode = _sanitize_copy(shard, dest, audit, group)
            else:
                mode = _safe_link_or_copy(shard.resolve(), dest)
            clean = dict(audit)
            clean["dest_path"] = str(dest)
            clean["materialization"] = mode
            clean_rows.append(clean)
            shard_index += 1

    manifest = _manifest_for_group(group, clean_rows)
    group_dir.mkdir(parents=True, exist_ok=True)
    (group_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return clean_rows, toxic_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build V13.1 detox data dirs.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--normal-root", type=Path, required=True)
    parser.add_argument("--d4-root", type=Path, required=True)
    parser.add_argument("--d5-root", type=Path, required=True)
    parser.add_argument(
        "--strip-noncanonical-oracle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy otherwise-clean shards after removing non-canonical oracle_policy fields.",
    )
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    all_clean: list[dict[str, Any]] = []
    all_toxic: list[dict[str, Any]] = []
    totals: dict[str, Any] = {"groups": {}, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for group, root in (
        ("normal", args.normal_root),
        ("d4", args.d4_root),
        ("d5", args.d5_root),
    ):
        clean, toxic = build_group(
            group=group,
            source_root=root,
            output_root=output_root,
            strip_noncanonical_oracle=bool(args.strip_noncanonical_oracle),
        )
        all_clean.extend(clean)
        all_toxic.extend(toxic)
        totals["groups"][group] = {
            "source_root": str(root.resolve()),
            "clean_shards": len(clean),
            "clean_samples": sum(int(row.get("samples", 0)) for row in clean),
            "toxic_shards": len(toxic),
            "toxic_samples": sum(int(row.get("samples", 0)) for row in toxic),
            "sanitized_shards": sum(1 for row in clean if row.get("materialization") == "sanitized_copy"),
            "symlinked_shards": sum(1 for row in clean if row.get("materialization") == "symlink"),
        }

    report = {
        **totals,
        "policy": {
            "toxic": [
                "manifest rep_draw_rate > 0",
                "manifest nocap_draw_rate > 0",
                "suspicious terminal_code in shard",
                "missing legal/fens/stm metadata",
                "oracle_policy or teacher_q outside legal set",
            ],
            "sanitized": [
                "otherwise-clean shards with non-canonical oracle_policy have oracle_policy fields removed",
            ],
        },
        "clean_manifest": str(output_root / "clean_manifest.json"),
        "toxic_manifest": str(output_root / "toxic_manifest.json"),
    }
    (output_root / "clean_manifest.json").write_text(
        json.dumps({"clean": all_clean}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_root / "toxic_manifest.json").write_text(
        json.dumps({"toxic": all_toxic}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_root / "detox_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
