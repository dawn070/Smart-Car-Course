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
from typing import Any, Dict, Optional, Tuple, List
import re
import csv

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


def write_test_results_csv(
    *,
    results_csv: Path,
    metrics: Any,
    total_images: int,
    total_instances: int,
) -> None:
    """Write a stable CSV for test metrics so logs can read it reliably.

    The CSV schema is independent from Ultralytics internal files.
    """

    # class names
    names = getattr(metrics, "names", None)
    if isinstance(names, dict):
        class_names = [names[i] for i in sorted(names.keys())]
    elif isinstance(names, (list, tuple)):
        class_names = list(names)
    else:
        # fallback
        nc = int(getattr(getattr(metrics, "box", None), "nc", 0) or 0)
        class_names = [f"class{i}" for i in range(nc)]

    nc = len(class_names)

    # per-class counts (Ultralytics already computes these during val)
    nt_per_class = getattr(metrics, "nt_per_class", None)
    nt_per_image = getattr(metrics, "nt_per_image", None)

    def _get_count(arr, i: int) -> int:
        try:
            return int(arr[i])
        except Exception:
            return 0

    # metrics accessors
    def _mean_results() -> Tuple[float, float, float, float]:
        mr = getattr(metrics, "mean_results", None)
        if mr is None:
            mr = getattr(getattr(metrics, "box", None), "mean_results", None)
        if callable(mr):
            mr = mr()
        if isinstance(mr, (list, tuple)) and len(mr) >= 4:
            return float(mr[0]), float(mr[1]), float(mr[2]), float(mr[3])
        return 0.0, 0.0, 0.0, 0.0

    def _class_result(i: int) -> Tuple[float, float, float, float]:
        cr = getattr(metrics, "class_result", None)
        if callable(cr):
            r = cr(i)
            if isinstance(r, (list, tuple)) and len(r) >= 4:
                return float(r[0]), float(r[1]), float(r[2]), float(r[3])
        return 0.0, 0.0, 0.0, 0.0

    results_csv.parent.mkdir(parents=True, exist_ok=True)
    with results_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Class", "Images", "Instances", "Box(P)", "R", "mAP50", "mAP50-95"])

        mp, mr, map50, map5095 = _mean_results()
        w.writerow([
            "all",
            int(total_images),
            int(total_instances),
            mp,
            mr,
            map50,
            map5095,
        ])

        for i, name in enumerate(class_names):
            p, r, ap50, ap = _class_result(i)
            w.writerow(
                [
                    name,
                    _get_count(nt_per_image, i) if nt_per_image is not None else 0,
                    _get_count(nt_per_class, i) if nt_per_class is not None else 0,
                    p,
                    r,
                    ap50,
                    ap,
                ]
            )


def format_metrics_table_from_results_csv(results_csv: Path) -> str:
    """Read the stable CSV produced by write_test_results_csv() and format a neat table."""

    rows: List[Dict[str, str]] = []
    with results_csv.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Class"):
                rows.append(row)

    if not rows:
        return ""

    def f_int(s: str) -> int:
        try:
            return int(float(s))
        except Exception:
            return 0

    def f_float(s: str) -> float:
        try:
            return float(s)
        except Exception:
            return 0.0

    def fmt_float_stripped(value: float, width: int = 11, decimals: int = 3) -> str:
        s = f"{value:.{decimals}f}"
        # strip trailing zeros to mimic Ultralytics console style (e.g. 0.730 -> 0.73)
        s = s.rstrip("0").rstrip(".")
        return s.rjust(width)

    # Match Ultralytics column widths: %22s + %11s*6
    class_w = max(22, max(len(r["Class"]) for r in rows))
    img_w = 11
    inst_w = 11
    p_w = 11
    r_w = 11
    map50_w = 11
    map5095_w = 11

    header = (
        f"{'Class':>{class_w}}"
        f"{'Images':>{img_w}}"
        f"{'Instances':>{inst_w}}"
        f"{'Box(P':>{p_w}}"
        f"{'R':>{r_w}}"
        f"{'mAP50':>{map50_w}}"
        f"{'mAP50-95)':>{map5095_w}}"
    )

    out_lines = [header]
    for r in rows:
        out_lines.append(
            f"{r['Class']:>{class_w}}"
            f"{f_int(r.get('Images', '0')):>{img_w}}"
            f"{f_int(r.get('Instances', '0')):>{inst_w}}"
            f"{fmt_float_stripped(f_float(r.get('Box(P)', '0')), width=p_w)}"
            f"{fmt_float_stripped(f_float(r.get('R', '0')), width=r_w)}"
            f"{fmt_float_stripped(f_float(r.get('mAP50', '0')), width=map50_w)}"
            f"{fmt_float_stripped(f_float(r.get('mAP50-95', '0')), width=map5095_w)}"
        )

    return "\n".join(out_lines).rstrip()


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


def extract_metrics_table(text: str) -> str:
    """Extract the final Class/Images/Instances/mAP table block from Ultralytics output."""

    raw_lines = text.splitlines()
    lines = [_strip_ansi(line) for line in raw_lines]

    header_indices: List[int] = []
    for i, line in enumerate(lines):
        if "Class" in line and "Images" in line and "Instances" in line and "mAP" in line:
            header_indices.append(i)

    def is_progress(line: str) -> bool:
        return "%" in line or "it/s" in line or "s/it" in line

    blocks: List[str] = []
    for start in header_indices:
        header = lines[start].rstrip()
        # remove progress suffix if present
        if ")" in header and is_progress(header):
            header = header.split(")", 1)[0] + ")"

        collected = [header]
        for line in lines[start + 1 :]:
            if not line.strip() and len(collected) > 1:
                break
            if line.lstrip().startswith("Speed:"):
                break
            if "Results saved to" in line:
                break
            if is_progress(line):
                continue
            if "Class" in line and "Images" in line and "Instances" in line and "mAP" in line:
                break
            collected.append(line.rstrip())

        # keep only blocks with data rows
        if len(collected) > 1:
            blocks.append("\n".join(collected).rstrip())

    return blocks[-1] if blocks else ""


def _format_kv_block(title: str, items: List[Tuple[str, Any]]) -> str:
    max_k = max((len(k) for k, _ in items), default=0)
    lines = [f"{title}:"]
    for k, v in items:
        lines.append(f"  {k.ljust(max_k)} :  {v}")
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
    test_results_csv: Optional[Path],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    sep = "=" * 92
    meta_items = [
        ("start_time", start_time.strftime("%Y-%m-%d %H:%M:%S")),
        ("duration", str(end_time - start_time)),
        ("end_time", end_time.strftime("%Y-%m-%d %H:%M:%S")),
        ("dataset_yaml", str(dataset.yaml_abs_path)),
        ("dataset_root", str(dataset.dataset_root)),
        ("project_dir", str(project_dir)),
        ("run_name", str(run_name)),
        ("best_weights", str(best_weights)),
    ]

    counts_lines = [
        "Split Counts:",
        f"  train : images={dataset.counts_train.images:<5} instances={dataset.counts_train.instances:<6} missing_labels={dataset.counts_train.missing_labels}",
        f"  val   : images={dataset.counts_val.images:<5} instances={dataset.counts_val.instances:<6} missing_labels={dataset.counts_val.missing_labels}",
        f"  test  : images={dataset.counts_test.images:<5} instances={dataset.counts_test.instances:<6} missing_labels={dataset.counts_test.missing_labels}",
    ]

    if not eval_test_enabled:
        metrics_block = "Test Metrics: (skipped)"
    else:
        if test_results_csv and test_results_csv.exists():
            table = format_metrics_table_from_results_csv(test_results_csv)
            metrics_block = "Test Metrics:\n" + table if table else "Test Metrics: (empty results.csv)"
        else:
            metrics_block = "Test Metrics: (results.csv not found)"

    args_items = [(k, str(v)) for k, v in sorted(args.items())]

    content = "\n".join(
        [
            sep,
            _format_kv_block("Run Meta", meta_items),
            "\n".join(counts_lines),
            _format_kv_block("Hyperparameters", args_items),
            metrics_block,
            sep,
            "",
        ]
    )

    with log_path.open("a", encoding="utf-8") as f:
        f.write(content)
