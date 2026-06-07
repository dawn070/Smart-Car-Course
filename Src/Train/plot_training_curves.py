"""Plot training curves from Ultralytics results.csv.

Creates a 2x2 figure:
1) train/val box-like loss (YOLO: box_loss; RT-DETR: giou_loss)
2) train/val cls loss
3) train/val dfl-like loss (YOLO: dfl_loss; RT-DETR: l1_loss)
4) learning rate curve

Usage examples:
  python Src\Train\plot_training_curves.py --project-dir runs --run-name yolo26_kitti_baseline
  python Src\Train\plot_training_curves.py --run-dir runs\yolo26_kitti_baseline
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def configure_matplotlib_chinese_font() -> None:
    """Configure Matplotlib to render Chinese characters.

    On Windows, Microsoft YaHei is usually available. We provide a fallback list.
    """

    try:
        import matplotlib

        matplotlib.rcParams["font.sans-serif"] = [
            "Microsoft YaHei",
            "SimHei",
            "SimSun",
            "Noto Sans CJK SC",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        # If matplotlib isn't available or config fails, skip silently.
        pass


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot 2x2 training curves from results.csv")
    parser.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="训练输出目录（包含 results.csv）。如 runs/yolo26_kitti_baseline",
    )
    parser.add_argument(
        "--project-dir",
        type=str,
        default="runs",
        help="runs 根目录（当不提供 --run-dir 时使用）",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="训练输出目录名（当不提供 --run-dir 时使用）",
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        default="",
        help="results.csv 的路径（优先级最高）",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="输出图片路径（默认保存到 run-dir/curves_2x2.png）",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="输出图片 DPI",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="弹窗显示图片（在本机有 GUI 时有效）",
    )
    return parser


def _resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    # returns (run_dir, results_csv, out_path)
    if args.results_csv:
        results_csv = Path(args.results_csv)
        run_dir = results_csv.parent
    else:
        if args.run_dir:
            run_dir = Path(args.run_dir)
        else:
            if not args.run_name:
                raise SystemExit("请提供 --run-dir 或同时提供 --project-dir 与 --run-name")
            run_dir = Path(args.project_dir) / args.run_name

        results_csv = run_dir / "results.csv"

    if not results_csv.exists():
        raise SystemExit(f"未找到 results.csv: {results_csv}")

    out_path = Path(args.out) if args.out else (run_dir / "curves_2x2.png")
    return run_dir, results_csv, out_path


def _read_results_csv(path: Path) -> Tuple[List[int], Dict[str, List[float]]]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"results.csv 没有表头: {path}")

        epochs: List[int] = []
        series: Dict[str, List[float]] = {k: [] for k in reader.fieldnames}

        for row in reader:
            # epoch may be missing in some versions; fallback to incremental index
            ep_raw = row.get("epoch", "")
            if ep_raw.strip() == "":
                epochs.append(len(epochs))
            else:
                try:
                    epochs.append(int(float(ep_raw)))
                except Exception:
                    epochs.append(len(epochs))

            for k in series.keys():
                v = row.get(k, "")
                try:
                    series[k].append(float(v))
                except Exception:
                    series[k].append(float("nan"))

    return epochs, series


def _find_col(all_cols: List[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in all_cols:
            return c
    return None


def _pretty_loss_name(col: Optional[str], fallback: str) -> str:
    if not col:
        return fallback

    mapping = {
        "train/box_loss": "Box Loss",
        "val/box_loss": "Box Loss",
        "train/giou_loss": "GIoU Loss",
        "val/giou_loss": "GIoU Loss",
        "train/dfl_loss": "DFL Loss",
        "val/dfl_loss": "DFL Loss",
        "train/l1_loss": "L1 Loss",
        "val/l1_loss": "L1 Loss",
        "train/cls_loss": "Class Loss",
        "val/cls_loss": "Class Loss",
    }
    return mapping.get(col, fallback)


def _mean_lr(all_cols: List[str], series: Dict[str, List[float]]) -> Tuple[str, List[float]]:
    lr_cols = [c for c in all_cols if c.startswith("lr/")]
    if not lr_cols:
        # try older naming
        lr_cols = [c for c in all_cols if c.lower() in {"lr", "learning_rate"}]

    if not lr_cols:
        return "(lr not found)", []

    # mean across param groups
    n = len(series[lr_cols[0]])
    lr_values: List[float] = []
    for i in range(n):
        vals = []
        for c in lr_cols:
            v = series[c][i]
            if v == v:  # not nan
                vals.append(v)
        lr_values.append(sum(vals) / len(vals) if vals else float("nan"))

    label = "+".join(lr_cols) if len(lr_cols) <= 2 else f"mean({len(lr_cols)} lr groups)"
    return label, lr_values


def plot_2x2_from_results_csv(
    *,
    results_csv: Path,
    out_path: Path,
    dpi: int = 150,
    show: bool = False,
    title: str = "",
) -> None:
    """Plot a 2x2 figure from an Ultralytics results.csv."""

    configure_matplotlib_chinese_font()

    epochs, series = _read_results_csv(results_csv)
    cols = list(series.keys())

    # Resolve loss columns
    # YOLO: box_loss/cls_loss/dfl_loss
    # RT-DETR: giou_loss/cls_loss/l1_loss
    train_box = _find_col(cols, ["train/box_loss", "train/box", "box_loss", "train/giou_loss", "train/giou"])
    val_box = _find_col(cols, ["val/box_loss", "val/box", "val/giou_loss", "val/giou"])

    train_cls = _find_col(cols, ["train/cls_loss", "train/cls", "cls_loss"])
    val_cls = _find_col(cols, ["val/cls_loss", "val/cls"])

    train_dfl = _find_col(cols, ["train/dfl_loss", "train/dfl", "dfl_loss", "train/l1_loss", "train/l1"])
    val_dfl = _find_col(cols, ["val/dfl_loss", "val/dfl", "val/l1_loss", "val/l1"])

    lr_label, lr_values = _mean_lr(cols, series)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax1, ax2, ax3, ax4 = axes[0][0], axes[0][1], axes[1][0], axes[1][1]

    def plot_pair(ax, train_col: Optional[str], val_col: Optional[str], title_: str) -> None:
        plotted_any = False
        if train_col and train_col in series:
            ax.plot(epochs, series[train_col], label=f"train ({train_col})")
            plotted_any = True
        if val_col and val_col in series:
            ax.plot(epochs, series[val_col], label=f"val ({val_col})")
            plotted_any = True

        ax.set_title(title_)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        if plotted_any:
            ax.legend()
        else:
            ax.text(0.5, 0.5, "Loss column not found in results.csv", ha="center", va="center")

    box_title = _pretty_loss_name(train_box or val_box, "Box/GIoU Loss")
    cls_title = _pretty_loss_name(train_cls or val_cls, "Class Loss")
    dfl_title = _pretty_loss_name(train_dfl or val_dfl, "DFL/L1 Loss")

    plot_pair(ax1, train_box, val_box, box_title)
    plot_pair(ax2, train_cls, val_cls, cls_title)
    plot_pair(ax3, train_dfl, val_dfl, dfl_title)

    ax4.set_title("Learning Rate")
    ax4.set_xlabel("epoch")
    ax4.grid(True, alpha=0.3)
    if lr_values:
        ax4.plot(epochs, lr_values, label=lr_label)
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, "LR column not found in results.csv", ha="center", va="center")

    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    print(f"Saved: {out_path}")

    if show:
        plt.show()


def main() -> None:
    args = _build_arg_parser().parse_args()
    run_dir, results_csv, out_path = _resolve_paths(args)

    plot_2x2_from_results_csv(
        results_csv=results_csv,
        out_path=out_path,
        dpi=args.dpi,
        show=args.show,
        title=f"Training Curves: {run_dir.as_posix()}",
    )


if __name__ == "__main__":
    main()
