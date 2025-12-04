import argparse
import glob
import os
from typing import Any, List, Dict, Tuple

import numpy as np
import torch
from medpy.metric import hd
from medpy.metric.binary import hd95 as medpy_hd95
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryRecall,
    BinaryPrecision,
    BinarySpecificity,
    JaccardIndex,
)


def _ensure_5d(x: np.ndarray) -> np.ndarray:
    """Normalize an array to shape (N, 1, D, H, W).

    Allowed input shapes are:
    - (D, H, W)
    - (1, D, H, W)
    - (N, D, H, W)
    - already (N, 1, D, H, W)

    Parameters
    ----------
    x: Input array to be reshaped.

    Returns
    -------
    x_5d: Array with shape (N, 1, D, H, W).
    """
    x = np.asarray(x)
    if x.ndim == 3:
        x = x[None, None, ...]
    elif x.ndim == 4:
        if x.shape[0] != 1:
            x = x[:, None, ...]
        else:
            x = x[None, ...]
    elif x.ndim == 5:
        pass
    else:
        raise ValueError(f"Unexpected array ndim={x.ndim}, expected 3/4/5.")
    return x


def _split_first_dim_to_list(a5d: np.ndarray) -> List[np.ndarray]:
    """Split (N, 1, D, H, W) into a list of (1, 1, D, H, W) samples.

    Parameters
    ----------
    a5d: Batched array of shape (N, 1, D, H, W).

    Returns
    -------
    samples: List of arrays, each with shape (1, 1, D, H, W).
    """
    return [a5d[i:i+1, ...] for i in range(a5d.shape[0])]


def _load_pairs(
    data_path: str,
    split: str | None = "test",
    tested_dir: str = "prediction",
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Load (prediction, ground truth) pairs from disk.

    Each pair is returned as NumPy arrays shaped (1, 1, D, H, W).

    Parameters
    ----------
    data_path: Root directory containing predictions and ground truths.
    split: Dataset split name. Use "test" or None.
    tested_dir: Subdirectory name containing prediction volumes.

    Returns
    -------
    pairs: List of (pred, gt) array pairs, each with shape (1, 1, D, H, W).
    """
    if not os.path.isdir(data_path):
        raise RuntimeError(f"'{data_path}' must be a directory.")

    if split is None:
        pred_dir = os.path.join(data_path, tested_dir or "prediction")
        gt_dir = os.path.join(data_path, "gt")
    elif split == "test":
        pred_dir = os.path.join(data_path, "test", tested_dir)
        gt_dir = os.path.join(data_path, "test", "gt")
    else:
        raise RuntimeError(f"Unsupported split='{split}'. Use None or 'test'.")

    if not (os.path.isdir(pred_dir) and os.path.isdir(gt_dir)):
        raise RuntimeError(f"Expected '{pred_dir}' and '{gt_dir}' to exist.")

    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.npy")))
    gt_files = sorted(glob.glob(os.path.join(gt_dir,   "*.npy")))

    if len(pred_files) == 0 or len(pred_files) != len(gt_files):
        raise RuntimeError(f"Mismatched counts: {len(pred_files)} preds vs {len(gt_files)} gts in {data_path}")

    pairs: List[Tuple[np.ndarray, np.ndarray]] = []

    for pf, gf in zip(pred_files, gt_files):
        p = _ensure_5d(np.load(pf))
        g = _ensure_5d(np.load(gf))
        if p.shape[0] != g.shape[0]:
            raise RuntimeError(f"Batch count mismatch: {pf} has {p.shape[0]}, {gf} has {g.shape[0]}.")
        for pi, gi in zip(_split_first_dim_to_list(p), _split_first_dim_to_list(g)):
            pairs.append((pi, gi))

    return pairs


def _nanmeanstd_tensor(t: torch.Tensor) -> Tuple[float, float]:
    """Compute mean and std of a tensor, ignoring NaNs.

    Uses population standard deviation (ddof=0) and is compatible with
    older PyTorch versions.

    Parameters
    ----------
    t: Input tensor.

    Returns
    -------
    mean: Mean value ignoring NaNs, or NaN if undefined.
    std: Standard deviation ignoring NaNs, or NaN if undefined.
    """
    if t.numel() == 0:
        return float("nan"), float("nan")

    x = t.detach().float().reshape(-1)
    mask = ~torch.isnan(x)

    if not torch.any(mask):
        return float("nan"), float("nan")

    x = x[mask]
    return float(x.mean().item()), float(x.std(unbiased=False).item())


def _nanmeanstd_list(vals: List[float]) -> Tuple[float, float]:
    """Compute mean and std over a list of floats, ignoring NaNs.

    Parameters
    ----------
    vals: List of float values.

    Returns
    -------
    mean: Mean value ignoring NaNs, or NaN if undefined.
    std: Standard deviation ignoring NaNs, or NaN if undefined.
    """
    if not vals:
        return float("nan"), float("nan")
    a = np.asarray(vals, dtype=float)
    return float(np.nanmean(a)), float(np.nanstd(a, ddof=0))


def _safe_medpy_hd(pred_bin: np.ndarray, gt_bin: np.ndarray, spacing=(9, 1, 1)) -> Tuple[float, float]:
    """Compute Hausdorff distance and HD95 using MedPy.

    For empty masks, both values are returned as NaN. Inputs are expected
    as 3D arrays with values in {0, 1}.

    Parameters
    ----------
    pred_bin: Binarized prediction volume of shape (D, H, W).
    gt_bin: Binarized ground-truth volume of shape (D, H, W).
    spacing: Voxel spacing in (D, H, W) order.

    Returns
    -------
    hd_val: Hausdorff distance, or NaN if undefined.
    hd95_val: 95th percentile Hausdorff distance, or NaN if undefined.
    """
    try:
        if pred_bin.sum() == 0 or gt_bin.sum() == 0:
            return float("nan"), float("nan")
        return (float(hd(pred_bin, gt_bin, voxelspacing=spacing)),
                float(medpy_hd95(pred_bin, gt_bin, voxelspacing=spacing)))
    except Exception:
        return float("nan"), float("nan")


def eval_from_file(
    data_path: str,
    tested_dir: str = "prediction",
    threshold: float = 0.5,
    split: str | None = "test",
    device: str | None = None,
    print_all: bool = False,
) -> Dict[str, Any]:
    """Evaluate predictions in a directory against corresponding ground truths.

    This function loads prediction and ground-truth volumes from disk,
    computes classification metrics and Hausdorff distances per sample,
    and aggregates them into mean ± std statistics.

    Parameters
    ----------
    data_path: Root directory containing predictions and ground truths.
    tested_dir: Subdirectory containing predictions (e.g. "prediction").
    threshold: Threshold used for binarizing predictions.
    split: Dataset split name. Use "test" or None to control directory layout.
    device: Device string (e.g. "cuda" or "cpu"). If None, chosen automatically.
    print_all: If True, print a summary of all metrics.

    Returns
    -------
    results: Dictionary with keys like "mean_acc", "std_acc", "mean_hd_95", etc.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pairs = _load_pairs(data_path, split=split, tested_dir=tested_dir)

    per_sample: List[Dict[str, torch.Tensor | float]] = []
    for p_np, g_np in pairs:
        p = torch.from_numpy(p_np.astype(np.float32)).to(device)
        g = torch.from_numpy(g_np.astype(np.float32)).to(device)

        acc = BinaryAccuracy(threshold=threshold).to(device)(p, g)
        sens = BinaryRecall(threshold=threshold).to(device)(p, g)
        prec = BinaryPrecision(threshold=threshold).to(device)(p, g)
        spec = BinarySpecificity(threshold=threshold).to(device)(p, g)
        jacc = JaccardIndex(task="binary", threshold=threshold).to(device)(p, g)

        bin_p = (p > threshold).detach().cpu().numpy().astype(np.uint8)[0, 0]
        bin_g = (g > 0.5).detach().cpu().numpy().astype(np.uint8)[0, 0]
        hd_val, hd95_val = _safe_medpy_hd(bin_p, bin_g, spacing=(9, 1, 1))

        per_sample.append({
            "acc": acc,
            "sens": sens,
            "prec": prec,
            "spec": spec,
            "jacc": jacc,
            "hd": hd_val,
            "hd95": torch.tensor(float(hd95_val), device=device),
        })

    def _stack(key: str) -> torch.Tensor:
        return torch.stack([s[key] for s in per_sample]) if per_sample else torch.tensor([])

    mean_acc, std_acc = _nanmeanstd_tensor(_stack("acc"))
    mean_sens, std_sens = _nanmeanstd_tensor(_stack("sens"))
    mean_prec, std_prec = _nanmeanstd_tensor(_stack("prec"))
    mean_spec, std_spec = _nanmeanstd_tensor(_stack("spec"))
    mean_jacc, std_jacc = _nanmeanstd_tensor(_stack("jacc"))
    mean_hd95, std_hd95 = _nanmeanstd_tensor(_stack("hd95"))
    mean_hd, std_hd = _nanmeanstd_list([float(s["hd"]) for s in per_sample])

    if print_all:
        print("----------------------------------------")
        print("RESULTS (mean ± std across samples)")
        print(f"    Accuracy:    {mean_acc:.3f} ± {std_acc:.3f}")
        print(f"    Sensitivity: {mean_sens:.3f} ± {std_sens:.3f}")
        print(f"    Specificity: {mean_spec:.3f} ± {std_spec:.3f}")
        print(f"    Precision:   {mean_prec:.3f} ± {std_prec:.3f}")
        print(f"    Jaccard:     {mean_jacc:.3f} ± {std_jacc:.3f}")
        print(f"    HD:          {mean_hd:.3f} ± {std_hd:.3f}")
        print(f"    HD95:        {mean_hd95:.3f} ± {std_hd95:.3f}")
        print("----------------------------------------")

    return {
        "mean_acc": mean_acc * 100, "std_acc": std_acc * 100,
        "mean_sens": mean_sens * 100, "std_sens": std_sens * 100,
        "mean_spec": mean_spec * 100, "std_spec": std_spec * 100,
        "mean_prec": mean_prec * 100, "std_prec": std_prec * 100,
        "mean_jacc": mean_jacc * 100, "std_jacc": std_jacc * 100,
        "mean_hd": mean_hd, "std_hd": std_hd,
        "mean_hd_95": mean_hd95, "std_hd_95": std_hd95,
    }


def aggregate_runs(
    root_dir: str,
    run_names: List[str],
    tested_dir: str = "prediction",
    split: str | None = None,
    threshold: float = 0.5,
    device: str | None = None,
    print_table: bool = True,
    display_names: List[str] | None = None,
) -> Dict[str, Any]:
    """Aggregate metrics across multiple run directories.

    This function evaluates each run directory using ``eval_from_file``,
    aggregates metrics across runs, and optionally prints a formatted table.

    Parameters
    ----------
    root_dir: Root directory containing all run subdirectories.
    run_names: List of run subdirectory names to evaluate.
    tested_dir: Subdirectory name for predictions inside each run.
    split: Dataset split name passed to ``eval_from_file`` (e.g. "test" or None).
    threshold: Threshold used for binarizing predictions.
    device: Device string (e.g. "cuda" or "cpu"). If None, chosen automatically.
    print_table: If True, print a formatted aggregation table.
    display_names: Optional list of column labels for the printed table.

    Returns
    -------
    results: Dictionary with keys:
        - "per_run": mapping run_name → metrics dict from ``eval_from_file``.
        - "aggregate": mapping metric → {"mean", "std", "n_runs"}.
        - "table_rows": list of rows used in the printed table.
        - "dataframe": pandas DataFrame or None (if pandas is not available).
    """
    # Evaluate each run directory
    per_run: Dict[str, Dict[str, float]] = {}
    for rn in run_names:
        run_path = os.path.join(str(root_dir), rn)
        per_run[rn] = eval_from_file(
            data_path=run_path,
            tested_dir=tested_dir,
            threshold=threshold,
            split=split,
            device=device,
            print_all=False,
        )
    if display_names is None:
        display_names = [str(i) for i in range(len(run_names))]
    if len(display_names) != len(run_names):
        raise ValueError("display_names must be the same length as run_names.")

    first_metrics = next(iter(per_run.values()))
    base_metrics = [k[len("mean_"):] for k in first_metrics.keys() if k.startswith("mean_")]

    aggregate: Dict[str, Dict[str, float]] = {}
    for metric in base_metrics:
        vals = np.array([per_run[rn][f"mean_{metric}"] for rn in run_names], dtype=float)
        aggregate[metric] = {
            "mean": float(np.nanmean(vals)),
            "std":  float(np.nanstd(vals, ddof=0)),
            "n_runs": int(vals.size),
        }

    table_rows: List[Dict[str, str]] = []
    for metric in base_metrics:
        row = {"metric": metric}
        for label, rn in zip(display_names, run_names):
            m = per_run[rn].get(f"mean_{metric}", float("nan"))
            s = per_run[rn].get(f"std_{metric}", float("nan"))
            row[label] = f"{m:.1f} ± {s:.1f}"
        row["Overall"] = f"{aggregate[metric]['mean']:.1f} ± {aggregate[metric]['std']:.1f}"
        table_rows.append(row)

    if print_table:
        headers = ["metric"] + list(display_names) + ["Overall"]
        colw = max(14, max(len(h) for h in headers))
        print("\n========== Cross-run aggregation ==========")
        print(" ".join(h.ljust(colw) for h in headers))
        print("-" * ((colw + 1) * len(headers)))
        for r in table_rows:
            print(" ".join(str(r.get(h, "")).ljust(colw) for h in headers))
        print("===========================================\n")

    return {"per_run": per_run, "aggregate": aggregate, "table_rows": table_rows}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TNT segmentation predictions from saved .npy files."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: eval
    p_eval = subparsers.add_parser("eval", help="Evaluate a single directory of predictions against ground truth.")
    p_eval.add_argument("--path", type=str, required=True, help="Root directory.")
    p_eval.add_argument("--tested_dir", type=str, default="prediction", help="Subdirectory containing preds and GTs")
    p_eval.add_argument("--threshold", type=float, default=0.5, help="Threshold for binarizing predictions")
    p_eval.add_argument("--split", type=str, default="none", help="Dataset split to use")
    p_eval.add_argument("--device", type=str, default=None, help="Device to use")
    p_eval.add_argument("--print_all", action="store_true", default=False, help="Print a detailed summary of metrics.")

    # Subcommand: aggregate
    p_agg = subparsers.add_parser("aggregate", help="Aggregate metrics across multiple run directories.")
    p_agg.add_argument("--root_dir", type=str, required=True, help="Root directory containing all run subdirectories.")
    p_agg.add_argument("--sub_dirs", type=str, nargs="+", required=True, help="List of run subdirectories to evaluate")
    p_agg.add_argument(
        "--tested_dir", type=str, default="prediction",
        help="Subdirectory name for predictions inside each run (default: 'prediction')."
    )
    p_agg.add_argument("--split", type=str, default="none")
    p_agg.add_argument("--threshold", type=float, default=0.5, help="Threshold for binarizing predictions")
    p_agg.add_argument("--device", type=str, default=None, help="Device to use")
    p_agg.add_argument(
        "--display_names", type=str, nargs="*",
        help="Optional list of display names for columns (same length as run_names)."
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.command == "eval":
        split = None if args.split.lower() == "none" else args.split
        eval_from_file(
            data_path=args.path,
            tested_dir=args.tested_dir,
            threshold=args.threshold,
            split=split,
            device=args.device,
            print_all=args.print_all,
        )

    elif args.command == "aggregate":
        split = None if args.split.lower() == "none" else args.split

        display_names = args.display_names
        if display_names is not None and len(display_names) != len(args.run_names):
            raise ValueError("display_names must be the same length as run_names.")

        aggregate_runs(
            root_dir=args.root_dir,
            run_names=args.sub_dirs,
            tested_dir=args.tested_dir,
            split=split,
            threshold=args.threshold,
            device=args.device,
            print_table=True,
            display_names=display_names,
        )


if __name__ == "__main__":
    main()
