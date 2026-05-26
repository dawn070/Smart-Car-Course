"""Logging helpers for training scripts.

This module is intentionally self-contained so that the main training script
stays small and readable.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
import io
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


@dataclass(frozen=True)
class SplitCount:
    images: int
    instances: int
    missing_labels: int


@dataclass(frozen=True)
class DatasetInfo:
    yaml_abs_path: Path
    dataset_root: Path
    train_images_dir: Path
    val_images_dir: Path
    test_images_dir: Path
    train_labels_dir: Path
    val_labels_dir: Path
    test_labels_dir: Path
    counts_train: SplitCount
    counts_val: SplitCount
    counts_test: SplitCount


def _labels_rel_from_images_rel(images_rel: str) -> str:
    # e.g. images/train -> labels/train
    parts = images_rel.replace("\\", "/").split("/")
    if parts and parts[0] == "images":
        parts[0] = "labels"
    return "/".join(parts)


def _count_images_and_instances(images_dir: Path, labels_dir: Path) -> SplitCount:
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [p for p in images_dir.glob("*") if p.is_file() and p.suffix.lower() in img_exts]

    instances = 0
    missing_labels = 0
    for img in images:
        lab = labels_dir / f"{img.stem}.txt"
        if not lab.exists():
            missing_labels += 1
            continue
        with lab.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip():
                    instances += 1

    return SplitCount(images=len(images), instances=instances, missing_labels=missing_labels)


def collect_dataset_info(base_dir: str, yaml_rel_path: str) -> DatasetInfo:
    """Parse dataset yaml and count train/val/test samples.

    Assumes yaml has keys: path, train, val, test (test optional).
    """

    base = Path(base_dir)
    yaml_abs_path = base / yaml_rel_path

    # defaults
    dataset_root = base / "datasets" / "kitti"
    train_images_rel = "images/train"
    val_images_rel = "images/val"
    test_images_rel = "images/test"

    if yaml is not None and yaml_abs_path.exists():
        try:
            cfg = yaml.safe_load(yaml_abs_path.read_text(encoding="utf-8")) or {}
            path_field = cfg.get("path", "datasets/kitti")
            path_p = Path(str(path_field))
            dataset_root = path_p if path_p.is_absolute() else (base / path_p)
            train_images_rel = str(cfg.get("train", train_images_rel))
            val_images_rel = str(cfg.get("val", val_images_rel))
            test_images_rel = str(cfg.get("test", test_images_rel))
        except Exception:
            pass

    train_images_dir = dataset_root / Path(train_images_rel)
    val_images_dir = dataset_root / Path(val_images_rel)
    test_images_dir = dataset_root / Path(test_images_rel)

    train_labels_dir = dataset_root / Path(_labels_rel_from_images_rel(train_images_rel))
    val_labels_dir = dataset_root / Path(_labels_rel_from_images_rel(val_images_rel))
    test_labels_dir = dataset_root / Path(_labels_rel_from_images_rel(test_images_rel))

    counts_train = (
        _count_images_and_instances(train_images_dir, train_labels_dir)
        if train_images_dir.exists()
        else SplitCount(0, 0, 0)
    )
    counts_val = (
        _count_images_and_instances(val_images_dir, val_labels_dir)
        if val_images_dir.exists()
        else SplitCount(0, 0, 0)
    )
    counts_test = (
        _count_images_and_instances(test_images_dir, test_labels_dir)
        if test_images_dir.exists()
        else SplitCount(0, 0, 0)
    )

    return DatasetInfo(
        yaml_abs_path=yaml_abs_path,
        dataset_root=dataset_root,
        train_images_dir=train_images_dir,
        val_images_dir=val_images_dir,
        test_images_dir=test_images_dir,
        train_labels_dir=train_labels_dir,
        val_labels_dir=val_labels_dir,
        test_labels_dir=test_labels_dir,
        counts_train=counts_train,
        counts_val=counts_val,
        counts_test=counts_test,
    )


def capture_val_output(fn, *args, **kwargs) -> str:
    """Capture stdout/stderr produced by a function call."""

    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        fn(*args, **kwargs)
    return buf.getvalue()


def extract_metrics_table(text: str) -> str:
    """Extract the Class/Images/Instances/mAP table block from Ultralytics output."""

    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if "Class" in line and "Images" in line and "Instances" in line and "mAP" in line:
            start = i
            break
    if start is None:
        return ""

    collected = []
    for line in lines[start:]:
        if not line.strip() and collected:
            break
        if line.lstrip().startswith("Speed:"):
            break
        if "Results saved to" in line:
            break
        collected.append(line.rstrip())

    return "\n".join(collected).rstrip()


def _format_kv_block(title: str, items: Dict[str, Any]) -> str:
    keys = sorted(items.keys())
    max_k = max((len(k) for k in keys), default=0)
    lines = [f"{title}:"]
    for k in keys:
        lines.append(f"  {k.ljust(max_k)} : {items[k]}")
    return "\n".join(lines)


def append_training_log(
    *,
    log_path: Path,
    start_time: datetime,
    end_time: datetime,
    dataset: DatasetInfo,
    args: Dict[str, Any],
    project_dir: str,
    run_name: str,
    best_weights: str,
    eval_test_enabled: bool,
    test_metrics_table: str,
    test_raw_output: str,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    sep = "=" * 92
    meta = {
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": str(end_time - start_time),
        "dataset_yaml": str(dataset.yaml_abs_path),
        "dataset_root": str(dataset.dataset_root),
        "project_dir": str(project_dir),
        "run_name": str(run_name),
        "best_weights": str(best_weights),
    }

    counts_lines = [
        "Split Counts:",
        f"  train : images={dataset.counts_train.images:<5} instances={dataset.counts_train.instances:<6} missing_labels={dataset.counts_train.missing_labels}",
        f"  val   : images={dataset.counts_val.images:<5} instances={dataset.counts_val.instances:<6} missing_labels={dataset.counts_val.missing_labels}",
        f"  test  : images={dataset.counts_test.images:<5} instances={dataset.counts_test.instances:<6} missing_labels={dataset.counts_test.missing_labels}",
    ]

    if not eval_test_enabled:
        metrics_block = "Test Metrics: (skipped)"
    else:
        if test_metrics_table:
            metrics_block = "Test Metrics:\n" + test_metrics_table
        else:
            tail = "\n".join(test_raw_output.splitlines()[-80:]).rstrip()
            metrics_block = "Test Metrics: (raw output tail)\n" + tail

    content = "\n".join(
        [
            sep,
            _format_kv_block("Run Meta", meta),
            "\n".join(counts_lines),
            _format_kv_block("Hyperparameters", {k: str(v) for k, v in args.items()}),
            metrics_block,
            sep,
            "",
        ]
    )

    with log_path.open("a", encoding="utf-8") as f:
        f.write(content)
