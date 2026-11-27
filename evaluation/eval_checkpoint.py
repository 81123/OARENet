"""Run the full pixel + topology evaluation for any checkpoint's predicted masks.

Usage:
    python eval_checkpoint.py --pred-dir ../predictions/epoch_040_test/masks \
                              --out-dir ../evaluation/epoch_040 --label epoch_040

Reuses the metric implementations from eval_epoch015_pixel_topology.py and
eval_epoch015_connectivity.py (their module globals are patched, code unchanged).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

import eval_epoch015_pixel_topology as core
import eval_epoch015_connectivity as conn


def dump_json_and_csv(out_dir: Path, stem: str, agg: dict, rows: list[dict]) -> None:
    (out_dir / f"{stem}.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with open(out_dir / f"{stem}_per_image.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not pred_dir.is_dir():
        raise SystemExit(f"pred dir not found: {pred_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # patch module globals so all metric functions read the new dirs
    core.PRED_DIR = pred_dir
    core.OUT_DIR = out_dir
    conn.PRED_DIR = pred_dir
    conn.OUT_DIR = out_dir

    sample_ids = sorted(p.name[: -len("_pred.png")] for p in pred_dir.glob("*_pred.png"))
    print(f"[{args.label}] {len(sample_ids)} predicted masks in {pred_dir}", flush=True)

    t0 = time.time()
    pixel_agg, pixel_rows = core.pixel_metrics(sample_ids)
    dump_json_and_csv(out_dir, "pixel_metrics", pixel_agg, pixel_rows)

    topo_agg, topo_rows = core.topology_metrics(sample_ids)
    dump_json_and_csv(out_dir, "topology_metrics", topo_agg, topo_rows)

    apls_agg, apls_rows = core.apls_metrics(sample_ids)
    dump_json_and_csv(out_dir, "apls_metrics", apls_agg, apls_rows)

    # connectivity (same tau=0.4 metric, inline to reuse patched globals)
    conn_rows = []
    for sid in sample_ids:
        gt, pred = core.load_pair(sid)
        res = conn.connectivity_one_image(gt, pred)
        if res["status"] == "no_gt":
            continue
        conn_rows.append({"sample_id": sid, **{k: v for k, v in res.items() if k != "status"}})
    c = np.array([r["connectivity"] for r in conn_rows])
    conn_agg = {
        "tau": conn.TAU,
        "num_images": len(conn_rows),
        "connectivity_mean": float(c.mean()),
        "connectivity_std": float(c.std()),
        "connectivity_median": float(np.median(c)),
        "connectivity_p05": float(np.percentile(c, 5)),
        "connectivity_p95": float(np.percentile(c, 95)),
        "soft_connectivity_mean": float(np.mean([r["soft_connectivity"] for r in conn_rows])),
        "eps_gt_mean": float(np.mean([r["eps_gt"] for r in conn_rows])),
        "eps_pred_mean": float(np.mean([r["eps_pred"] for r in conn_rows])),
        "definition": conn.connectivity_one_image.__doc__ or "Batra et al. 2019, tau=0.4",
    }
    dump_json_and_csv(out_dir, "connectivity_metrics", conn_agg, conn_rows)

    print(f"\n===== SUMMARY ({args.label}) =====")
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
    print(f"[connect] mean={conn_agg['connectivity_mean']:.4f}  "
          f"median={conn_agg['connectivity_median']:.4f}  "
          f"eps_gt={conn_agg['eps_gt_mean']:.4f}  eps_pred={conn_agg['eps_pred_mean']:.4f}")
    print(f"[APLS] mean={apls_agg['apls_mean']:.4f}  median={apls_agg['apls_median']:.4f}  "
          f"pair-weighted={apls_agg['apls_pair_weighted']:.4f}  (<0.5: {apls_agg['apls_below_0.5_rate']:.3f})")
    print(f"\n[{args.label}] total {time.time()-t0:.0f}s, results in {out_dir}")


if __name__ == "__main__":
    main()
