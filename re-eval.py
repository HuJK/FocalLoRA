#!/usr/bin/env python3
"""
Recompute evaluation metrics from saved detail logs.

Given a fine-tune output directory (containing `training_log.csv` and
`batch_{epoch}_{batch_idx}/detail_log.pkl` folders), this script:
  1) Re-evaluates each saved sample using the latest `evallib._eval_constraint`
  2) Updates every `detail_log.pkl` with refreshed metrics
  3) Rewrites `training_log.csv` with the new success rates

Usage:
    python re-eval.py /path/to/finetune_out_dir
"""

import argparse
import csv
import glob
import os
import pickle
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure local evallib is importable
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "code"))
import evallib  # noqa: E402


def _to_float(val: Any) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _parse_batch_key(path: str) -> Optional[Tuple[int, int]]:
    m = re.search(r"batch_(\d+)_(\d+)", path)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _rate(bucket: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    return {
        k: (v["pass"] / v["total"] if v["total"] else 0.0)
        for k, v in bucket.items()
    }


def recompute_eval(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Recalculate success metrics from stored raw outputs.
    """
    normal_pass = normal_total = 0
    conflict_pass = conflict_total = 0
    both_pass = 0
    per_constraint_normal: Dict[str, Dict[str, int]] = {}
    per_constraint_conflict: Dict[str, Dict[str, int]] = {}
    updated_samples: List[Dict[str, Any]] = []

    for sample in samples:
        s = dict(sample)  # shallow copy before we mutate
        constraint = s.get("constraint_type") or s.get("constraint") or "unknown"

        normal_output = s.get("normal_output")
        if normal_output is None:
            normal_output = s.get("valid_output")
        conflict_output = s.get("conflict_output")
        if conflict_output is None:
            conflict_output = s.get("asr_output")

        normal_cond = (
            s.get("normal_condition_used")
            or s.get("valid_condition_used")
            or s.get("normal_prompt")
            or ""
        )
        conflict_cond = (
            s.get("conflict_condition_used")
            or s.get("asr_condition_used")
            or s.get("conflict_prompt")
            or normal_cond
        )

        n_ok = bool(evallib._eval_constraint(normal_cond, constraint, normal_output or ""))
        c_ok = bool(evallib._eval_constraint(conflict_cond, constraint, conflict_output or ""))

        # Store both the new and legacy keys for compatibility
        s["normal_pass"] = n_ok
        s["conflict_pass"] = c_ok
        s["valid_pass"] = n_ok if "valid_pass" in s or "valid_output" in s else s.get("valid_pass", n_ok)
        s["asr_pass"] = c_ok if "asr_pass" in s or "asr_output" in s else s.get("asr_pass", c_ok)
        updated_samples.append(s)

        normal_total += 1
        conflict_total += 1
        normal_pass += int(n_ok)
        conflict_pass += int(c_ok)
        both_pass += int(n_ok and c_ok)

        n_stats = per_constraint_normal.setdefault(constraint, {"pass": 0, "total": 0})
        n_stats["total"] += 1
        n_stats["pass"] += int(n_ok)
        c_stats = per_constraint_conflict.setdefault(constraint, {"pass": 0, "total": 0})
        c_stats["total"] += 1
        c_stats["pass"] += int(c_ok)

    normal_success = normal_pass / normal_total if normal_total else 0.0
    conflict_success = conflict_pass / conflict_total if conflict_total else 0.0
    both_success = both_pass / normal_total if normal_total else 0.0

    per_constraint_normal_rate = _rate(per_constraint_normal)
    per_constraint_conflict_rate = _rate(per_constraint_conflict)

    eval_asr = {
        "status": "ok",
        "normal_success": normal_success,
        "conflict_success": conflict_success,
        "both_success": both_success,
        "evaluated_pairs": normal_total,
        "per_constraint_normal": per_constraint_normal_rate,
        "per_constraint_conflict": per_constraint_conflict_rate,
        "samples": updated_samples,
        # Legacy keys preserved for older notebooks/scripts
        "valid_rate": normal_success,
        "asr_rate": conflict_success,
        "valid_asr_rate": both_success,
        "per_constraint_valid": per_constraint_normal_rate,
        "per_constraint_asr": per_constraint_conflict_rate,
    }
    return eval_asr


def process_detail_log(path: str, dry_run: bool = False) -> Optional[Dict[str, Any]]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    eval_asr_old = payload.get("eval_asr", {}) or {}
    samples = eval_asr_old.get("samples", [])
    if not samples:
        print(f"[skip] No samples in {path}")
        return None

    eval_asr_new = recompute_eval(samples)
    if "attn" in eval_asr_old:
        eval_asr_new["attn"] = eval_asr_old["attn"]

    payload["eval_asr"] = eval_asr_new

    if not dry_run:
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    print(
        f"[updated] {path}: normal={eval_asr_new['normal_success']:.4f}, "
        f"conflict={eval_asr_new['conflict_success']:.4f}, "
        f"both={eval_asr_new['both_success']:.4f}, pairs={eval_asr_new['evaluated_pairs']}"
    )
    return {
        "eval_asr": eval_asr_new,
        "eval_mmlu": payload.get("eval_mmlu", {}),
    }


def rewrite_training_log(
    log_path: str, metrics_map: Dict[Tuple[int, int], Dict[str, Any]], dry_run: bool = False
):
    if not os.path.exists(log_path):
        print(f"[warn] training_log.csv not found at {log_path}, skip rewrite.")
        return

    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    fieldnames = [
        "epoch",
        "batch_idx",
        "current_ratio",
        "normal_success",
        "conflict_success",
        "both_success",
        "mmlu_acc",
    ]
    new_rows: List[Dict[str, Any]] = []
    for row in rows:
        epoch_str = row.get("epoch") or row.get("ep") or ""
        batch_str = row.get("batch_idx") or row.get("batch") or ""
        key = None
        try:
            key = (int(epoch_str), int(batch_str))
        except ValueError:
            pass

        metrics = metrics_map.get(key, {})
        normal_success = metrics.get("normal_success")
        conflict_success = metrics.get("conflict_success")
        both_success = metrics.get("both_success")
        mmlu_acc = metrics.get("mmlu_acc")

        def pick(*names: str) -> str:
            for name in names:
                if name in row and row[name] not in (None, ""):
                    return row[name]
            return ""

        new_rows.append(
            {
                "epoch": epoch_str,
                "batch_idx": batch_str,
                "current_ratio": pick("current_ratio", "ratio"),
                "normal_success": f"{normal_success:.6f}" if normal_success is not None else pick("normal_success", "valid_rate"),
                "conflict_success": f"{conflict_success:.6f}" if conflict_success is not None else pick("conflict_success", "asr_rate"),
                "both_success": f"{both_success:.6f}" if both_success is not None else pick("both_success", "valid_asr_rate"),
                "mmlu_acc": (
                    f"{mmlu_acc:.6f}" if mmlu_acc is not None else pick("mmlu_acc", "accuracy", "mmlu")
                ),
            }
        )

    if dry_run:
        print(f"[dry-run] Would rewrite {log_path} with {len(new_rows)} rows.")
        return

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)
    print(f"[done] Rewrote training_log.csv with {len(new_rows)} rows.")


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate saved checkpoints using stored outputs.")
    parser.add_argument("finetune_out_dir", help="Directory containing batch_* folders and training_log.csv")
    parser.add_argument("--dry-run", action="store_true", help="Compute metrics without writing files")
    args = parser.parse_args()

    detail_paths = sorted(
        glob.glob(os.path.join(args.finetune_out_dir, "batch_*_*", "detail_log.pkl"))
    )
    if not detail_paths:
        print(f"No detail_log.pkl files found under {args.finetune_out_dir}")
        return

    metrics_map: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for path in detail_paths:
        key = _parse_batch_key(path)
        result = process_detail_log(path, dry_run=args.dry_run)
        if not result or not key:
            continue
        eval_asr = result.get("eval_asr", {})
        eval_mmlu = result.get("eval_mmlu", {}) or {}
        metrics_map[key] = {
            "normal_success": eval_asr.get("normal_success"),
            "conflict_success": eval_asr.get("conflict_success"),
            "both_success": eval_asr.get("both_success"),
            "mmlu_acc": eval_mmlu.get("accuracy"),
        }

    training_log_path = os.path.join(args.finetune_out_dir, "training_log.csv")
    rewrite_training_log(training_log_path, metrics_map, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
