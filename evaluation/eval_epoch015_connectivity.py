"""Standard road Connectivity metric (Batra et al. 2019, tau=0.4) for epoch_015.

connectivity = 1 - (eps(S_pred -> S_gt) + eps(S_gt -> S_pred)) / 2
eps(A -> B)  = fraction of components in A whose best overlap |a ∩ b|/|a| < tau
Computed per image (8-connectivity, 512x512, same pipeline as the other metrics),
then averaged over images with non-empty GT.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from eval_epoch015_pixel_topology import GT_DIR, IMAGE_SIZE, OUT_DIR, PRED_DIR, STRUCT8, load_pair

TAU = 0.4


def connectivity_one_image(gt: np.ndarray, pred: np.ndarray) -> dict:
    gt_lab, n_gt = ndimage.label(gt, structure=STRUCT8)
    pred_lab, n_pred = ndimage.label(pred, structure=STRUCT8)

    if n_gt == 0:
        return {"status": "no_gt"}

    enc = gt_lab.ravel().astype(np.int64) * (n_pred + 1) + pred_lab.ravel()
    pairs, counts = np.unique(enc, return_counts=True)
    g_of = pairs // (n_pred + 1)
    p_of = pairs % (n_pred + 1)
    road = (g_of > 0) & (p_of > 0)

    # best overlap fraction per component, both directions
    g_tot = np.bincount(g_of, weights=counts, minlength=n_gt + 1)
    p_tot = np.bincount(p_of, weights=counts, minlength=n_pred + 1)
    best_g = np.zeros(n_gt + 1)
    best_p = np.zeros(n_pred + 1)
    if road.any():
        np.maximum.at(best_g, g_of[road], counts[road])
        np.maximum.at(best_p, p_of[road], counts[road])
    frac_g = best_g[1:] / np.maximum(g_tot[1:], 1)
    eps_gt = float(np.mean(frac_g < TAU))                      # gt components left unmatched
    soft_gt = float(np.mean(np.minimum(frac_g, 1.0)))

    if n_pred == 0:
        eps_pred = 1.0                                          # empty prediction: total failure
        soft_pred = 0.0
    else:
        frac_p = best_p[1:] / np.maximum(p_tot[1:], 1)
        eps_pred = float(np.mean(frac_p < TAU))
        soft_pred = float(np.mean(np.minimum(frac_p, 1.0)))

    error = (eps_gt + eps_pred) / 2.0
    return {
        "status": "ok",
        "connectivity": 1.0 - error,
        "eps_gt": eps_gt,
        "eps_pred": eps_pred,
        "soft_connectivity": 1.0 - (1.0 - soft_gt + 1.0 - soft_pred) / 2.0,
    }


def main() -> None:
    sample_ids = sorted(p.name[: -len("_pred.png")] for p in PRED_DIR.glob("*_pred.png"))
    rows: list[dict] = []
    for sid in sample_ids:
        gt, pred = load_pair(sid)
        res = connectivity_one_image(gt, pred)
        if res["status"] == "no_gt":
            continue
        rows.append({"sample_id": sid, **{k: v for k, v in res.items() if k != "status"}})

    conn = np.array([r["connectivity"] for r in rows])
    soft = np.array([r["soft_connectivity"] for r in rows])
    agg = {
        "tau": TAU,
        "num_images": len(rows),
        "connectivity_mean": float(conn.mean()),
        "connectivity_std": float(conn.std()),
        "connectivity_median": float(np.median(conn)),
        "connectivity_p05": float(np.percentile(conn, 5)),
        "connectivity_p95": float(np.percentile(conn, 95)),
        "soft_connectivity_mean": float(soft.mean()),
        "eps_gt_mean": float(np.mean([r["eps_gt"] for r in rows])),
        "eps_pred_mean": float(np.mean([r["eps_pred"] for r in rows])),
        "definition": (
            "Batra et al. 2019 (road connectivity, tau=0.4): per 8-connected component, best "
            "overlap fraction |a∩b|/|a| to the other mask; a component fails when < tau. "
            "connectivity = 1 - (eps_gt + eps_pred)/2, averaged over images"
        ),
    }
    (OUT_DIR / "connectivity_metrics.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with open(OUT_DIR / "connectivity_metrics_per_image.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Connectivity (tau={TAU}) over {len(rows)} images: "
          f"mean={agg['connectivity_mean']:.4f}  median={agg['connectivity_median']:.4f}  "
          f"soft={agg['soft_connectivity_mean']:.4f}")
    print(f"  eps_gt={agg['eps_gt_mean']:.4f}  eps_pred={agg['eps_pred_mean']:.4f}")


if __name__ == "__main__":
    main()
