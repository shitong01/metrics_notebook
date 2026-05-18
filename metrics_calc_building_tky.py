#!/usr/bin/env python3
"""
metrics_calc_building_tky.py
============================
Building-damage validation pipeline for the Turkiye (Basarsoft) dataset.

Unlike the generic metrics_calc.py (which spatial-joins damage *points* onto
footprints), the Turkiye GeoPackage already carries a per-building damage label
in Turkish (column ACIKLAMA).  Step 1 here therefore translates those labels to
binary Damaged / Undamaged and writes an annotated GeoPackage.  Steps 2-3 are
identical in logic to metrics_calc.py.

Usage
-----
    python metrics_calc_building_tky.py                          # uses config_building_tky.yaml in CWD
    python metrics_calc_building_tky.py --config my_cfg.yaml

Steps (each toggled in the config file):
  1. label_ground_truth  – Translate ACIKLAMA labels → binary Damaged/Undamaged
  2. threshold_sweep     – Pixel-overlap raster sweep; overlap table is cached
  3. plot                – Save threshold-sweep plots as PNG
"""

import argparse
import importlib.util
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import yaml
from shapely.prepared import prep
from metrics_calc import _compute_pixel_overlap_table, _raster_cache_key
try:
    from shapely import make_valid          # Shapely >= 2.0
except ImportError:
    from shapely.validation import make_valid  # Shapely 1.8 – 1.x

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Script location — used to find the sibling Turkiye/translations.py
_HERE = Path(__file__).parent


# ===========================================================================
# Config helpers
# ===========================================================================

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    log.info("Loaded config from %s", config_path)
    return cfg


def _load_translations():
    """
    Load Turkiye/translations.py relative to this script.
    Returns the module object so callers can access grading_translate,
    grading_score, damaged_categories, inaccessible_categories, etc.
    """
    tpath = _HERE / "Turkiye" / "translations.py"
    if not tpath.exists():
        raise FileNotFoundError(f"translations.py not found at {tpath}")
    spec = importlib.util.spec_from_file_location("translations", tpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ===========================================================================
# STEP 1 – Label ground-truth GeoPackage
# ===========================================================================

def run_step1(cfg: dict) -> None:
    """
    Translate per-building Turkish damage labels (ACIKLAMA) to English and
    to a binary Damaged / Undamaged column, compute building area in m², and
    write a filtered GeoPackage.

    Differs from the generic metrics_calc.py Step 1 because the Turkiye
    dataset is already joined — no spatial join with damage points needed.
    """
    s = cfg["step1"]
    log.info("=== STEP 1: Labelling Turkiye ground-truth GeoPackage ===")

    trans = _load_translations()
    grading_translate = trans.grading_translate
    grading_score     = trans.grading_score

    pts_class_col    = s["pts_class_col"]
    bld_id_col       = s["bld_id_col"]
    damaged_classes  = set(s.get("damaged_classes",   []))
    undamaged_classes= set(s.get("undamaged_classes", []))
    out_gpkg         = s["out_gpkg"]

    log.info("Loading GeoPackage: %s", s["valid_polygon"])
    bld = gpd.read_file(s["valid_polygon"])
    log.info("  Loaded %d buildings", len(bld))

    # Map Turkish label → English name and severity score
    bld["damage_class"] = bld[pts_class_col].map(grading_translate)
    bld["damage_score"] = bld[pts_class_col].map(grading_score)

    # Log any unmapped labels so the user can extend translations.py
    unmapped = bld.loc[bld["damage_class"].isna(), pts_class_col].unique()
    if len(unmapped):
        log.warning("  %d unmapped label(s) in %s — they become 'Unknown':\n    %s",
                    len(unmapped), pts_class_col, list(unmapped))

    # Binary classification driven by config lists
    def to_binary(en_label):
        if pd.isna(en_label):
            return "Unknown"
        if en_label in damaged_classes:
            return "Damaged"
        if en_label in undamaged_classes:
            return "Undamaged"
        return "Unknown"

    bld["damage_binary"] = bld["damage_class"].apply(to_binary)

    # Drop buildings with no usable label
    bld_out = bld[bld["damage_binary"] != "Unknown"].copy()
    log.info("  %d buildings kept after filtering Unknowns (dropped %d)",
             len(bld_out), len(bld) - len(bld_out))

    # Area in m² (reproject to UTM, keep polygon CRS for the output file)
    bld_m = bld_out.to_crs(bld_out.estimate_utm_crs())
    bld_out["area_m2"] = bld_m.geometry.area.values

    # Class-balance summary (informational)
    for grp_col in ("damage_class", "damage_binary"):
        metrics = (
            bld_out.groupby(grp_col)
            .agg(num_buildings=(grp_col, "count"),
                 total_area_m2=("area_m2", "sum"))
        )
        metrics["pct_area"] = (
            metrics["total_area_m2"] / metrics["total_area_m2"].sum() * 100
        ).round(2)
        log.info("  Balance (%s):\n%s", grp_col, metrics.to_string())

    Path(out_gpkg).parent.mkdir(parents=True, exist_ok=True)
    bld_out.to_file(out_gpkg, driver="GPKG")
    log.info("  Saved to %s", out_gpkg)


# ===========================================================================
# STEP 2 – Raster threshold sweep (with overlap-table caching)
# ===========================================================================

def _metrics_from_confusion(TP, FP, FN, TN):
    """
    Returns (accuracy, f1, csi, precision, recall, fpr) from confusion matrix
    values.  Any metric that would divide by zero is returned as NaN.
    """
    total = TP + FP + FN + TN
    acc   = (TP + TN) / total                if total              > 0 else np.nan
    f1    = (2 * TP)  / (2 * TP + FP + FN)  if (2*TP + FP + FN)  > 0 else np.nan
    csi   = TP        / (TP + FP + FN)       if (TP + FP + FN)    > 0 else np.nan
    prec  = TP        / (TP + FP)            if (TP + FP)         > 0 else np.nan
    rec   = TP        / (TP + FN)            if (TP + FN)         > 0 else np.nan
    fpr   = FP        / (FP + TN)            if (FP + TN)         > 0 else np.nan
    return acc, f1, csi, prec, rec, fpr


def _sweep_thresholds(
    bld_m, ov, thresholds,
    gt_col: str, id_col: str, area_col: str,
    area_frac_threshold: float,
    raster_path: str = "",
    save_per_threshold: bool = False,
    thresholded_dir: str = "./thresholded",
):
    """
    Sweep probability thresholds and compute count- and area-weighted metrics.

    For each threshold T:
      - A pixel is 'damaged' if raster_value >= T
      - A building is 'Damaged' if sum(damaged_pixel_area) / building_area
        >= area_frac_threshold

    Returns a DataFrame with one row per threshold.
    """
    ALLOWED = {"Damaged", "Undamaged"}

    base = bld_m[[id_col, area_col, gt_col]].dropna().copy()
    base = base[base[gt_col].isin(ALLOWED) & (base[area_col] > 0)].copy()
    base["y_true"] = (base[gt_col] == "Damaged").astype(int)

    ov2 = ov[ov[id_col].isin(base[id_col])].copy()
    ov2 = ov2.sort_values("value")

    rows = []
    for t in thresholds:
        log.info("      threshold %.2f", t)

        dmg_area = (
            ov2.loc[ov2["value"] >= t]
            .groupby(id_col)["overlap_m2"]
            .sum()
            .rename("damage_area_m2")
        )

        df = base.merge(dmg_area, on=id_col, how="left")
        df["damage_area_m2"] = df["damage_area_m2"].fillna(0.0)
        df["damage_frac"]    = df["damage_area_m2"] / df[area_col]
        df["y_pred"]         = (df["damage_frac"] >= area_frac_threshold).astype(int)

        # Count-based confusion
        TP_c = int(((df.y_true == 1) & (df.y_pred == 1)).sum())
        FP_c = int(((df.y_true == 0) & (df.y_pred == 1)).sum())
        FN_c = int(((df.y_true == 1) & (df.y_pred == 0)).sum())
        TN_c = int(((df.y_true == 0) & (df.y_pred == 0)).sum())
        acc_c, f1_c, csi_c, prec_c, rec_c, fpr_c = _metrics_from_confusion(
            TP_c, FP_c, FN_c, TN_c)

        # Area-based confusion
        TP_a = df.loc[(df.y_true == 1) & (df.y_pred == 1), area_col].sum()
        FP_a = df.loc[(df.y_true == 0) & (df.y_pred == 1), area_col].sum()
        FN_a = df.loc[(df.y_true == 1) & (df.y_pred == 0), area_col].sum()
        TN_a = df.loc[(df.y_true == 0) & (df.y_pred == 0), area_col].sum()
        acc_a, f1_a, csi_a, prec_a, rec_a, fpr_a = _metrics_from_confusion(
            TP_a, FP_a, FN_a, TN_a)

        rows.append({
            "threshold":  t,
            # Count-based
            "acc_count":  acc_c,  "f1_count":  f1_c,  "csi_count":  csi_c,
            "prec_count": prec_c, "rec_count": rec_c, "fpr_count":  fpr_c,
            # Area-based
            "acc_area":   acc_a,  "f1_area":   f1_a,  "csi_area":   csi_a,
            "prec_area":  prec_a, "rec_area":  rec_a, "fpr_area":   fpr_a,
            # Raw counts
            "TP_count": TP_c, "FP_count": FP_c, "FN_count": FN_c, "TN_count": TN_c,
        })

        if save_per_threshold and raster_path:
            Path(thresholded_dir).mkdir(parents=True, exist_ok=True)
            out_path = Path(thresholded_dir) / f"{Path(raster_path).stem}_t{t:.2f}.gpkg"
            df.drop(columns=["y_true", "y_pred"], errors="ignore").to_file(
                str(out_path), driver="GPKG")

    return pd.DataFrame(rows)


def run_step2(cfg: dict) -> None:
    s = cfg["step2"]

    log.info("=== STEP 2: Raster threshold sweep ===")

    # Decide input: Step 1 output or override
    if cfg["steps"].get("label_ground_truth"):
        input_gpkg = cfg["step1"]["out_gpkg"]
        log.info("  Using Step 1 output: %s", input_gpkg)
    else:
        input_gpkg = s["input_gpkg"]
        log.info("  Step 1 skipped — using: %s", input_gpkg)

    bld_out = gpd.read_file(input_gpkg)
    log.info("  Loaded %d buildings", len(bld_out))

    gt_col             = s["gt_col"]
    id_col             = s["id_col"]
    area_col           = s["area_col"]
    area_frac_thresh   = float(s["area_frac_threshold"])
    band               = int(s.get("band", 1))
    cache_dir          = s.get("cache_dir")
    raster_paths       = s["target_raster_paths"]
    save_per_threshold = bool(s.get("save_per_threshold", False))
    thresholded_dir    = s.get("thresholded_dir", "./thresholded")

    t_cfg = s["thresholds"]
    thresholds = np.round(
        np.arange(
            float(t_cfg["start"]),
            float(t_cfg["stop"]) + 1e-9,
            float(t_cfg["step"]),
        ),
        2,
    )
    log.info("  Area fraction threshold: %.2f", area_frac_thresh)
    log.info("  Raster thresholds: %s", thresholds.tolist())
    log.info("  Rasters to process: %d", len(raster_paths))

    results = []
    for i, rp in enumerate(raster_paths, 1):
        log.info("  [%d/%d] %s", i, len(raster_paths), rp)
        try:
            bld_m, ov = _compute_pixel_overlap_table(
                bld_out, rp, id_col, area_col, band, cache_dir)
            df_metrics = _sweep_thresholds(
                bld_m, ov, thresholds,
                gt_col, id_col, area_col, area_frac_thresh,
                raster_path=rp,
                save_per_threshold=save_per_threshold,
                thresholded_dir=thresholded_dir,
            )
            df_metrics["raster"] = rp
            results.append(df_metrics)
        except Exception as exc:
            log.error("  FAILED for %s: %s", rp, exc, exc_info=True)

    if not results:
        log.error("  No results produced — check raster paths and input GeoPackage.")
        return

    sweep_df = pd.concat(results, ignore_index=True)
    out_csv = s["out_csv"]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(out_csv, index=False)
    log.info("  Sweep results saved to %s  (%d rows)", out_csv, len(sweep_df))


# ===========================================================================
# STEP 3 – Plotting
# ===========================================================================

def run_step3(cfg: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = cfg["step3"]
    log.info("=== STEP 3: Saving threshold-sweep plots ===")

    if cfg["steps"].get("threshold_sweep"):
        input_csv = cfg["step2"]["out_csv"]
    else:
        input_csv = s["input_csv"]
        log.info("  Step 2 skipped — using: %s", input_csv)

    sweep_df = pd.read_csv(input_csv)
    log.info("  Loaded sweep CSV: %d rows", len(sweep_df))

    selected_rasters = cfg["step2"]["target_raster_paths"]
    before = len(sweep_df)
    sweep_df = sweep_df[sweep_df["raster"].isin(selected_rasters)].copy()
    log.info("  Filtered to step2 rasters: %d → %d rows (%d rasters)",
             before, len(sweep_df), sweep_df["raster"].nunique())

    out_dir   = Path(s["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    color_map = s.get("color_map", {})
    xlim      = s.get("xlim", [0.5, 1.0])
    ylim      = s.get("ylim", [0.3, 0.9])

    grouped = {
        r: df.sort_values("threshold")
        for r, df in sweep_df.groupby("raster")
    }

    raster_names = list(grouped.keys())
    _cmap = plt.cm.get_cmap("tab10", max(len(raster_names), 1))
    auto_colors = {name: _cmap(i) for i, name in enumerate(raster_names)}

    # Count-based and area-based variants for each metric
    plots = [
        ("csi_count",  "CSI vs Threshold (Count)"),
        ("f1_count",   "F1 vs Threshold (Count)"),
        ("csi_area",   "CSI vs Threshold (Area)"),
        ("f1_area",    "F1 vs Threshold (Area)"),
        ("prec_count", "Precision vs Threshold (Count)"),
        ("rec_count",  "Recall vs Threshold (Count)"),
        ("prec_area",  "Precision vs Threshold (Area)"),
        ("rec_area",   "Recall vs Threshold (Area)"),
    ]

    BG_COLOR    = "#000000"
    GRID_COLOR  = "#333333"
    TEXT_COLOR  = "#ffffff"
    SPINE_COLOR = "#555555"

    for col, title in plots:
        if col not in sweep_df.columns:
            log.warning("  Column %r not in CSV — skipping plot %r", col, title)
            continue

        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor(BG_COLOR)

        for raster_name, df_r in grouped.items():
            color = next(
                (c for k, c in color_map.items() if k in raster_name),
                auto_colors[raster_name],
            )
            marker = "x" if "mask" in raster_name else "o"
            ax.plot(df_r["threshold"], df_r[col],
                    marker=marker, markersize=3, color=color, label=raster_name)

        ax.set_xlabel("Raster threshold", color=TEXT_COLOR)
        ax.set_ylabel("Score",            color=TEXT_COLOR)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(title, color=TEXT_COLOR)
        ax.tick_params(colors=TEXT_COLOR)
        ax.grid(True, color=GRID_COLOR, alpha=0.8)
        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE_COLOR)

        legend = ax.legend(fontsize=7, facecolor=BG_COLOR, edgecolor=SPINE_COLOR)
        for text in legend.get_texts():
            text.set_color(TEXT_COLOR)

        fname = col + ".png"
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight",
                    facecolor=BG_COLOR)
        plt.close(fig)
        log.info("  Saved %s", out_dir / fname)

    log.info("  All plots saved to %s/", out_dir)


# ===========================================================================
# Entry point
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Turkiye building damage metrics pipeline."
    )
    parser.add_argument(
        "--config", "-c",
        default="config_building_tky.yaml",
        help="Path to YAML config file (default: config_building_tky.yaml in CWD)",
    )
    return parser.parse_args()


def main():
    args  = parse_args()
    cfg   = load_config(args.config)
    steps = cfg.get("steps", {})

    if steps.get("label_ground_truth"):
        run_step1(cfg)
    else:
        log.info("Step 1 (label_ground_truth) is disabled — skipping.")

    if steps.get("threshold_sweep"):
        run_step2(cfg)
    else:
        log.info("Step 2 (threshold_sweep) is disabled — skipping.")

    if steps.get("plot"):
        run_step3(cfg)
    else:
        log.info("Step 3 (plot) is disabled — skipping.")

    log.info("Done.")


if __name__ == "__main__":
    main()
