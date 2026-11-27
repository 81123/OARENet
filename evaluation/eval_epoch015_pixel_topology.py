"""Detailed pixel-level and topological evaluation for the epoch_015 checkpoint.

Pixel metrics (all test images, 512x512, identical to the training pipeline):
    confusion matrix (micro), OA, precision, recall, F1, road/background IoU, mIoU,
    per-image (macro) distributions.

Topological metrics:
    - beta0 / beta1 (connected components / loops) errors, per image and aggregated
    - GT-component preservation (topology recall) and pred-component purity (topology precision)
    - mask-geodesic APLS (Average Path Length Similarity) on a seeded random subset

GT is loaded exactly like datasets.py: convert("RGB") -> NEAREST resize to 512 -> channel 0 > 127.
Predictions: predictions/test_result/masks/{id}_pred.png (epoch_015 run, threshold 0.5).
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.measure import euler_number
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parents[1]
GT_DIR = Path(r"D:\project\pythonProject\Road_Identification\spin4\dataset\test\seg")
PRED_DIR = ROOT / "predictions" / "test_result" / "masks"
OUT_DIR = ROOT / "evaluation" / "epoch_015"

IMAGE_SIZE = 512
STRUCT8 = np.ones((3, 3), dtype=bool)

# APLS parameters (mask-geodesic approximation of the official metric)
APLS_SUBSET = 300
APLS_SOURCES_PER_IMG = 10
APLS_TARGETS_PER_SRC = 2
APLS_MIN_DIST = 50.0       # px, ignore trivially short paths
APLS_SNAP_TOL = 8.0        # px, max snap distance of an endpoint to the pred mask


def load_pair(sample_id: str) -> tuple[np.ndarray, np.ndarray]:
    gt = Image.open(GT_DIR / f"{sample_id}_mask.png").convert("RGB")
    gt = gt.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
    gt_np = np.array(gt, dtype=np.uint8)[..., 0] > 127

    pred = Image.open(PRED_DIR / f"{sample_id}_pred.png")
    pred_np = np.array(pred.convert("L")) > 127
    return gt_np, pred_np


# ---------------------------------------------------------------- pixel metrics

def pixel_metrics(sample_ids: list[str]) -> tuple[dict, list[dict]]:
    tp = fp = fn = tn = 0
    per_image: list[dict] = []
    t0 = time.time()
    for i, sid in enumerate(sample_ids):
        gt, pred = load_pair(sid)
        p, g = pred.ravel(), gt.ravel()
        tp_i = int(np.sum(p & g))
        fp_i = int(np.sum(p & ~g))
        fn_i = int(np.sum(~p & g))
        tn_i = int(np.sum(~p & ~g))
        tp += tp_i; fp += fp_i; fn += fn_i; tn += tn_i

        eps = 1e-9
        per_image.append({
            "sample_id": sid,
            "gt_road_px": int(tp_i + fn_i),
            "pred_road_px": int(tp_i + fp_i),
            "tp": tp_i, "fp": fp_i, "fn": fn_i, "tn": tn_i,
            "oa": (tp_i + tn_i) / max(tp_i + fp_i + fn_i + tn_i, 1),
            "precision": tp_i / max(tp_i + fp_i, eps),
            "recall": tp_i / max(tp_i + fn_i, eps),
            "f1": 2 * tp_i / max(2 * tp_i + fp_i + fn_i, eps),
            "road_iou": tp_i / max(tp_i + fp_i + fn_i, eps),
        })
        if (i + 1) % 200 == 0:
            print(f"  pixel {i+1}/{len(sample_ids)}  ({time.time()-t0:.0f}s)", flush=True)

    eps = 1e-9
    agg = {
        "num_images": len(sample_ids),
        "confusion_matrix_micro": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "oa": (tp + tn) / max(tp + fp + fn + tn, 1),
        "precision": tp / max(tp + fp, eps),
        "recall": tp / max(tp + fn, eps),
        "f1": 2 * tp / max(2 * tp + fp + fn, eps),
        "road_iou": tp / max(tp + fp + fn, eps),
        "background_iou": tn / max(tn + fp + fn, eps),
        "miou": 0.5 * (tp / max(tp + fp + fn, eps) + tn / max(tn + fp + fn, eps)),
        "gt_road_ratio": (tp + fn) / max(tp + fp + fn + tn, 1),
        "pred_road_ratio": (tp + fp) / max(tp + fp + fn + tn, 1),
    }

    def macro(key: str) -> dict:
        vals = np.array([row[key] for row in per_image], dtype=np.float64)
        return {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "p05": float(np.percentile(vals, 5)),
            "p25": float(np.percentile(vals, 25)),
            "median": float(np.percentile(vals, 50)),
            "p75": float(np.percentile(vals, 75)),
            "p95": float(np.percentile(vals, 95)),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }

    agg["macro"] = {k: macro(k) for k in ("oa", "precision", "recall", "f1", "road_iou")}
    ious = np.array([row["road_iou"] for row in per_image])
    agg["road_iou_hist"] = {
        "iou<0.3": int(np.sum(ious < 0.3)),
        "0.3<=iou<0.5": int(np.sum((ious >= 0.3) & (ious < 0.5))),
        "0.5<=iou<0.7": int(np.sum((ious >= 0.5) & (ious < 0.7))),
        "0.7<=iou<0.9": int(np.sum((ious >= 0.7) & (ious < 0.9))),
        "iou>=0.9": int(np.sum(ious >= 0.9)),
    }
    return agg, per_image


# ------------------------------------------------------- component-level topology

def component_stats(gt_lab: np.ndarray, n_gt: int, pred_lab: np.ndarray, n_pred: int) -> tuple[int, int]:
    """(gt components preserved, pred components pure) via a joint label histogram."""
    enc = gt_lab.ravel().astype(np.int64) * (n_pred + 1) + pred_lab.ravel()
    pairs, counts = np.unique(enc, return_counts=True)
    g_of = pairs // (n_pred + 1)
    p_of = pairs % (n_pred + 1)

    # totals per label (row/col sums of the joint histogram)
    g_tot = np.bincount(g_of, weights=counts, minlength=n_gt + 1)
    p_tot = np.bincount(p_of, weights=counts, minlength=n_pred + 1)

    road = (g_of > 0) & (p_of > 0)
    max_gp = np.zeros(n_gt + 1)          # best matching pred component size per gt component
    np.maximum.at(max_gp, g_of[road], counts[road])
    preserved = int(np.sum(max_gp[1:] >= 0.9 * np.maximum(g_tot[1:], 1)))

    max_pg = np.zeros(n_pred + 1)        # dominant gt component size per pred component
    np.maximum.at(max_pg, p_of[road], counts[road])
    pure = int(np.sum(max_pg[1:] >= 0.9 * np.maximum(p_tot[1:], 1)))
    return preserved, pure


def component_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    gt_lab, n_gt = ndimage.label(gt, structure=STRUCT8)
    pred_lab, n_pred = ndimage.label(pred, structure=STRUCT8)

    b0_gt, b0_pred = n_gt, n_pred
    chi_gt = int(euler_number(gt, connectivity=2))
    chi_pred = int(euler_number(pred, connectivity=2))
    b1_gt = b0_gt - chi_gt
    b1_pred = b0_pred - chi_pred

    preserved, pure = component_stats(gt_lab, n_gt, pred_lab, n_pred)

    return {
        "b0_gt": b0_gt, "b0_pred": b0_pred,
        "b1_gt": b1_gt, "b1_pred": b1_pred,
        "delta_b0": b0_pred - b0_gt,
        "abs_delta_b0": abs(b0_pred - b0_gt),
        "delta_b1": b1_pred - b1_gt,
        "abs_delta_b1": abs(b1_pred - b1_gt),
        "gt_components": n_gt,
        "pred_components": n_pred,
        "gt_components_preserved": preserved,
        "pred_components_pure": pure,
    }


def topology_metrics(sample_ids: list[str]) -> tuple[dict, list[dict]]:
    per_image: list[dict] = []
    skipped_no_gt = 0
    t0 = time.time()
    for i, sid in enumerate(sample_ids):
        gt, pred = load_pair(sid)
        if gt.sum() == 0:
            skipped_no_gt += 1
            continue
        row = {"sample_id": sid}
        row.update(component_metrics(gt, pred))
        row["gt_skeleton_px"] = int(skeletonize(gt).sum())
        row["pred_skeleton_px"] = int(skeletonize(pred).sum())
        per_image.append(row)
        if (i + 1) % 200 == 0:
            print(f"  topo  {i+1}/{len(sample_ids)}  ({time.time()-t0:.0f}s)", flush=True)

    def stat(key: str) -> dict:
        vals = np.array([r[key] for r in per_image], dtype=np.float64)
        return {
            "mean": float(vals.mean()), "std": float(vals.std()),
            "median": float(np.median(vals)),
            "p95": float(np.percentile(vals, 95)),
            "min": float(vals.min()), "max": float(vals.max()),
        }

    n_gt_total = sum(r["gt_components"] for r in per_image)
    n_pred_total = sum(r["pred_components"] for r in per_image)
    agg = {
        "num_images": len(per_image),
        "skipped_no_gt": skipped_no_gt,
        "b0_gt_mean": stat("b0_gt"), "b0_pred_mean": stat("b0_pred"),
        "b1_gt_mean": stat("b1_gt"), "b1_pred_mean": stat("b1_pred"),
        "delta_b0": stat("delta_b0"), "abs_delta_b0": stat("abs_delta_b0"),
        "delta_b1": stat("delta_b1"), "abs_delta_b1": stat("abs_delta_b1"),
        "exact_b0_match_rate": float(np.mean([r["abs_delta_b0"] == 0 for r in per_image])),
        "exact_b1_match_rate": float(np.mean([r["abs_delta_b1"] == 0 for r in per_image])),
        "gt_components_total": n_gt_total,
        "pred_components_total": n_pred_total,
        "component_fragmentation_ratio": n_pred_total / max(n_gt_total, 1),
        "gt_components_preserved_total": sum(r["gt_components_preserved"] for r in per_image),
        "gt_component_preservation_rate": sum(r["gt_components_preserved"] for r in per_image) / max(n_gt_total, 1),
        "pred_components_pure_total": sum(r["pred_components_pure"] for r in per_image),
        "pred_component_purity_rate": sum(r["pred_components_pure"] for r in per_image) / max(n_pred_total, 1),
        "gt_skeleton_px_total": sum(r["gt_skeleton_px"] for r in per_image),
        "pred_skeleton_px_total": sum(r["pred_skeleton_px"] for r in per_image),
        "skeleton_length_ratio": sum(r["pred_skeleton_px"] for r in per_image) / max(sum(r["gt_skeleton_px"] for r in per_image), 1),
    }
    return agg, per_image


# ------------------------------------------------------------------- mask APLS

def build_mask_graph(mask: np.ndarray) -> tuple[csr_matrix, np.ndarray]:
    """Sparse 8-neighbor pixel graph of a binary mask; edge weight = Euclidean px.

    Returns (graph, index_map) where index_map[y, x] is the node id (or -1).
    """
    H, W = mask.shape
    idx = -np.ones((H, W), dtype=np.int64)
    ys, xs = np.nonzero(mask)
    n = len(ys)
    idx[ys, xs] = np.arange(n)
    if n == 0:
        return csr_matrix((0, 0)), idx
    rows, cols, data = [], [], []
    for dr, dc, w in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, np.sqrt(2)), (1, -1, np.sqrt(2))):
        r0a, r1a = max(0, -dr), min(H, H - dr)
        c0a, c1a = max(0, -dc), min(W, W - dc)
        A = idx[r0a:r1a, c0a:c1a]
        B = idx[r0a + dr:r1a + dr, c0a + dc:c1a + dc]
        both = (A >= 0) & (B >= 0)
        rows.append(A[both]); cols.append(B[both]); data.append(np.full(int(both.sum()), w))
    graph = csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
    )
    return graph, idx


def apls_one_image(gt: np.ndarray, pred: np.ndarray, rng: np.random.Generator) -> tuple[float, int]:
    """Mask-geodesic APLS for one image. Returns (apls, num_pairs)."""
    skel = skeletonize(gt)
    skel_pts = np.argwhere(skel)
    if len(skel_pts) < 50:
        return float("nan"), 0

    gt_graph, gt_idx = build_mask_graph(gt)
    pred_graph, pred_idx = build_mask_graph(pred)
    if pred_graph.shape[0] == 0:
        return 0.0, APLS_SOURCES_PER_IMG * APLS_TARGETS_PER_SRC  # empty prediction: all pairs worst

    # distance transform of the pred mask to snap GT endpoints onto it
    snap_dist, (ny, nx) = ndimage.distance_transform_edt(~pred, return_indices=True)

    costs: list[float] = []
    n_sources = min(APLS_SOURCES_PER_IMG, len(skel_pts))
    src_choice = rng.choice(len(skel_pts), size=n_sources, replace=False)

    for s in src_choice:
        sy, sx = skel_pts[s]
        d_gt = dijkstra(gt_graph, directed=False, indices=int(gt_idx[sy, sx]))
        d_sk = d_gt[gt_idx[skel_pts[:, 0], skel_pts[:, 1]]]
        ok = np.isfinite(d_sk) & (d_sk >= APLS_MIN_DIST)
        if ok.sum() == 0:
            continue
        targets = rng.choice(
            np.nonzero(ok)[0], size=min(APLS_TARGETS_PER_SRC, int(ok.sum())), replace=False
        )

        snap_s = float(snap_dist[sy, sx])
        ps = int(pred_idx[ny[sy, sx], nx[sy, sx]])
        if snap_s > APLS_SNAP_TOL or ps < 0:
            costs.extend([1.0] * len(targets))   # source missing in pred: all pairs worst
            continue
        d_pred = dijkstra(pred_graph, directed=False, indices=ps)

        for t in targets:
            ty, tx = skel_pts[t]
            d_gt_pair = float(d_sk[t])
            snap_t = float(snap_dist[ty, tx])
            if snap_t > APLS_SNAP_TOL:
                costs.append(1.0)
                continue
            pt = int(pred_idx[ny[ty, tx], nx[ty, tx]])
            if pt < 0:
                costs.append(1.0)
                continue
            dp = d_pred[pt]
            if not np.isfinite(dp):
                costs.append(1.0)  # endpoints disconnected inside the prediction
            else:
                costs.append(min(1.0, abs(dp - d_gt_pair) / d_gt_pair))

    if not costs:
        return float("nan"), 0
    return 1.0 - float(np.mean(costs)), len(costs)


def apls_metrics(sample_ids: list[str]) -> tuple[dict, list[dict]]:
    rng_master = np.random.default_rng(42)
    subset_idx = rng_master.choice(len(sample_ids), size=min(APLS_SUBSET, len(sample_ids)), replace=False)
    subset_idx.sort()
    per_image: list[dict] = []
    t0 = time.time()
    for i, si in enumerate(subset_idx):
        sid = sample_ids[si]
        gt, pred = load_pair(sid)
        rng = np.random.default_rng(1000 + int(si))
        apls, n_pairs = apls_one_image(gt, pred, rng)
        per_image.append({"sample_id": sid, "apls": apls, "num_pairs": n_pairs})
        if (i + 1) % 25 == 0:
            print(f"  apls {i+1}/{len(subset_idx)}  ({time.time()-t0:.0f}s)", flush=True)

    valid = [r for r in per_image if np.isfinite(r["apls"])]
    vals = np.array([r["apls"] for r in valid])
    total_pairs = sum(r["num_pairs"] for r in valid)
    pair_weighted = (
        float(np.average([r["apls"] for r in valid], weights=[r["num_pairs"] for r in valid]))
        if valid else float("nan")
    )
    agg = {
        "subset_size": len(subset_idx),
        "valid_images": len(valid),
        "skipped_small_gt": len(subset_idx) - len(valid),
        "total_pairs": total_pairs,
        "apls_mean": float(vals.mean()) if len(vals) else float("nan"),
        "apls_std": float(vals.std()) if len(vals) else float("nan"),
        "apls_median": float(np.median(vals)) if len(vals) else float("nan"),
        "apls_p05": float(np.percentile(vals, 5)) if len(vals) else float("nan"),
        "apls_p95": float(np.percentile(vals, 95)) if len(vals) else float("nan"),
        "apls_min": float(vals.min()) if len(vals) else float("nan"),
        "apls_max": float(vals.max()) if len(vals) else float("nan"),
        "apls_pair_weighted": pair_weighted,
        "apls_below_0.5_rate": float(np.mean(vals < 0.5)) if len(vals) else float("nan"),
        "note": (
            "mask-geodesic approximation: node pairs sampled on the GT skeleton, path lengths "
            "are geodesic distances inside the binary masks (8-neighbor pixel graph), snap tol "
            f"{APLS_SNAP_TOL}px, min path {APLS_MIN_DIST}px"
        ),
    }
    return agg, per_image


# ------------------------------------------------------------------------ main

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_ids = sorted(p.name[: -len("_pred.png")] for p in PRED_DIR.glob("*_pred.png"))
    print(f"Found {len(sample_ids)} predicted masks in {PRED_DIR}")

    print("== Pixel metrics ==", flush=True)
    pixel_agg, pixel_rows = pixel_metrics(sample_ids)
    (OUT_DIR / "pixel_metrics.json").write_text(
        json.dumps(pixel_agg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with open(OUT_DIR / "pixel_metrics_per_image.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pixel_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pixel_rows)

    print("== Topology (Betti / components) ==", flush=True)
    topo_agg, topo_rows = topology_metrics(sample_ids)
    (OUT_DIR / "topology_metrics.json").write_text(
        json.dumps(topo_agg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with open(OUT_DIR / "topology_metrics_per_image.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(topo_rows[0].keys()))
        writer.writeheader()
        writer.writerows(topo_rows)

    print(f"== APLS (subset of {APLS_SUBSET}) ==", flush=True)
    apls_agg, apls_rows = apls_metrics(sample_ids)
    (OUT_DIR / "apls_metrics.json").write_text(
        json.dumps(apls_agg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with open(OUT_DIR / "apls_metrics_per_image.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(apls_rows[0].keys()))
        writer.writeheader()
        writer.writerows(apls_rows)

    print("\n===== SUMMARY (epoch_015) =====")
    print(f"[pixel  micro] OA={pixel_agg['oa']:.4f}  P={pixel_agg['precision']:.4f}  "
          f"R={pixel_agg['recall']:.4f}  F1={pixel_agg['f1']:.4f}  "
          f"roadIoU={pixel_agg['road_iou']:.4f}  bgIoU={pixel_agg['background_iou']:.4f}  "
          f"mIoU={pixel_agg['miou']:.4f}")
    print(f"[pixel  macro] roadIoU mean={pixel_agg['macro']['road_iou']['mean']:.4f} "
          f"median={pixel_agg['macro']['road_iou']['median']:.4f} "
          f"p05={pixel_agg['macro']['road_iou']['p05']:.4f}")
    print(f"[topology] |db0| mean={topo_agg['abs_delta_b0']['mean']:.3f}  "
          f"|db1| mean={topo_agg['abs_delta_b1']['mean']:.3f}  "
          f"b0 gt/pred = {topo_agg['b0_gt_mean']['mean']:.2f}/{topo_agg['b0_pred_mean']['mean']:.2f}  "
          f"b1 gt/pred = {topo_agg['b1_gt_mean']['mean']:.2f}/{topo_agg['b1_pred_mean']['mean']:.2f}")
    print(f"[topology] GT comp preserved = {topo_agg['gt_component_preservation_rate']:.4f}  "
          f"pred comp purity = {topo_agg['pred_component_purity_rate']:.4f}  "
          f"fragmentation = {topo_agg['component_fragmentation_ratio']:.3f}")
    print(f"[APLS] mean={apls_agg['apls_mean']:.4f}  median={apls_agg['apls_median']:.4f}  "
          f"pair-weighted={apls_agg['apls_pair_weighted']:.4f}  (<0.5: {apls_agg['apls_below_0.5_rate']:.3f})")
    print(f"\nSaved results to {OUT_DIR}")


if __name__ == "__main__":
    main()
