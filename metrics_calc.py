#!/usr/bin/env python3
"""
metrics_calc.py
===============
Command-line script version of metrics_calc_building.ipynb.

Usage
-----
    python metrics_calc.py                       # uses config.yaml in CWD
    python metrics_calc.py --config my_cfg.yaml  # explicit config path

Steps (each can be toggled in the config file):
  1. Merge ground-truth damage points onto building footprints
  2. Sweep raster probability thresholds and compute Accuracy / F1 / CSI
  3. Save threshold-sweep plots as PNG files
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import yaml
from rasterio.features import shapes
from rasterio.vrt import WarpedVRT
from shapely.geometry import shape

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# Config helpers
# ===========================================================================

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    log.info("Loaded config from %s", config_path)
    return cfg


# ===========================================================================
# STEP 1 – Merge ground-truth damage points with building footprints
# ===========================================================================

def run_step1(cfg: dict) -> None:
    s = cfg["step1"]

    log.info("=== STEP 1: Merging building footprints & ground-truth points ===")

    # --- Load damage points ---
    log.info("Loading %d point file(s)...", len(s["pts_shp"]))
    gdfs = [gpd.read_file(f) for f in s["pts_shp"]]
    pts = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
    log.info("  Total points loaded: %d", len(pts))

    # Optionally save merged points
    if s.get("merged_pts_out"):
        out_pts = s["merged_pts_out"]
        Path(out_pts).parent.mkdir(parents=True, exist_ok=True)
        pts.to_file(out_pts, driver="GeoJSON")
        log.info("  Merged points saved to %s", out_pts)

    # --- Load building footprints ---
    log.info("Loading %d building parquet file(s)...", len(s["bld_gpq"]))
    bld_gdfs = [gpd.read_parquet(f) for f in s["bld_gpq"]]
    bld = gpd.GeoDataFrame(pd.concat(bld_gdfs, ignore_index=True), crs=bld_gdfs[0].crs)
    bld["geometry"] = bld.geometry.buffer(0)
    log.info("  Total buildings loaded: %d", len(bld))

    # Reproject points to match buildings CRS
    if pts.crs != bld.crs:
        log.info("  Reprojecting points from %s to %s", pts.crs, bld.crs)
        pts = pts.to_crs(bld.crs)

    # --- Spatial join: points within buildings ---
    bld_id_col     = s["bld_id_col"]
    pts_class_col  = s["pts_class_col"]
    severity_order = s["severity_order"]
    rule           = s.get("rule", "majority")

    bld_small = bld[[bld_id_col, "geometry"]].copy()
    j = gpd.sjoin(
        pts[[pts_class_col, "geometry"]],
        bld_small,
        how="inner",
        predicate="within",
    )
    log.info("  Spatial join produced %d point-building matches", len(j))

    # --- Aggregation ---
    def majority_vote(series: pd.Series):
        vc = series.value_counts()
        return vc.index[0]

    def max_severity(series: pd.Series, order=severity_order):
        rank = {k: i for i, k in enumerate(order)}
        return max(series, key=lambda x: rank.get(x, -1))

    if rule == "majority":
        bld_class = j.groupby(bld_id_col)[pts_class_col].apply(majority_vote)
    elif rule == "max_severity":
        bld_class = j.groupby(bld_id_col)[pts_class_col].apply(max_severity)
    else:
        raise ValueError(f"Unknown rule: {rule!r}. Choose 'majority' or 'max_severity'.")

    # --- Attach back to buildings ---
    bld_out = bld.merge(
        bld_class.rename("damage_class_inherited_gnd"),
        on=bld_id_col,
        how="left",
    )

    # Compute area in metres
    bld_m = bld_out.to_crs(bld_out.estimate_utm_crs())
    bld_m["area_m2"] = bld_m.geometry.area
    bld_out["area_m2"] = bld_m["area_m2"].values

    # Drop buildings with no matched points
    bld_out = bld_out[bld_out["damage_class_inherited_gnd"].notna()].copy()
    log.info("  Buildings with damage class: %d", len(bld_out))

    # --- Binary damage label ---
    damaged_classes   = set(s.get("damaged_classes",   []))
    undamaged_classes = set(s.get("undamaged_classes", []))

    def to_binary(x):
        if x in damaged_classes:
            return "Damaged"
        if x in undamaged_classes:
            return "Undamaged"
        return "Unknown"

    bld_out["damage_binary_gnd"] = bld_out["damage_class_inherited_gnd"].apply(to_binary)

    # --- Class balance summary ---
    log.info("  Category balance (all classes):")
    metrics_cat = (
        bld_out.groupby("damage_class_inherited_gnd")
        .agg(num_buildings=("damage_class_inherited_gnd", "count"),
             total_area_m2=("area_m2", "sum"))
    )
    metrics_cat["pct_area"] = (
        metrics_cat["total_area_m2"] / metrics_cat["total_area_m2"].sum() * 100
    ).round(2)
    log.info("\n%s", metrics_cat.to_string())

    log.info("  Binary balance:")
    metrics_bin = (
        bld_out.groupby("damage_binary_gnd")
        .agg(num_buildings=("damage_binary_gnd", "count"),
             total_area_m2=("area_m2", "sum"))
    )
    metrics_bin["pct_area"] = (
        metrics_bin["total_area_m2"] / metrics_bin["total_area_m2"].sum() * 100
    ).round(2)
    log.info("\n%s", metrics_bin.to_string())

    # --- Save ---
    out_pqt = s["out_parquet"]
    Path(out_pqt).parent.mkdir(parents=True, exist_ok=True)
    bld_out.to_parquet(out_pqt, index=False)
    log.info("  Output saved to %s", out_pqt)


# ===========================================================================
# STEP 2 – Raster threshold sweep
# ===========================================================================

def _metrics_from_confusion(TP, FP, FN, TN):
    acc = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else np.nan
    f1  = (2 * TP) / (2 * TP + FP + FN)   if (2 * TP + FP + FN) > 0 else np.nan
    csi = TP / (TP + FP + FN)             if (TP + FP + FN) > 0 else np.nan
    return acc, f1, csi


def _compute_pixel_overlap_table(bld_gdf, raster_path: str, id_col: str, area_col: str, band: int = 1):
    """Intersect raster pixels with building polygons, returning an overlap table."""
    utm_crs = bld_gdf.estimate_utm_crs()
    bld_m = bld_gdf.to_crs(utm_crs).copy()
    bld_m["geometry"] = bld_m.geometry.buffer(0)
    bld_m[area_col] = bld_m.geometry.area

    with rasterio.open(raster_path) as src:
        with WarpedVRT(src, crs=utm_crs) as src_m:
            arr = src_m.read(band).astype(np.float32)
            nodata = src_m.nodata
            valid = np.isfinite(arr)
            if nodata is not None:
                valid &= (arr != nodata)

            geoms, vals = [], []
            for geom, val in shapes(arr, mask=valid, transform=src_m.transform):
                geoms.append(shape(geom))
                vals.append(float(val))

    grid = gpd.GeoDataFrame({"value": vals, "geometry": geoms}, crs=utm_crs)

    inter = gpd.overlay(
        bld_m[[id_col, area_col, "geometry"]],
        grid[["value", "geometry"]],
        how="intersection",
        keep_geom_type=True,
    )
    inter["overlap_m2"] = inter.geometry.area

    ov = inter[[id_col, "value", "overlap_m2"]].copy()
    return bld_m, ov


def _sweep_thresholds(bld_m, ov, thresholds, gt_col, id_col, area_col, area_frac_threshold):
    """Compute count- and area-weighted Accuracy / F1 / CSI for each threshold."""
    ALLOWED = {"Damaged", "Undamaged"}

    base = bld_m[[id_col, area_col, gt_col]].dropna().copy()
    base = base[base[gt_col].isin(ALLOWED) & (base[area_col] > 0)].copy()
    base["y_true"] = (base[gt_col] == "Damaged").astype(int)

    ov2 = ov[ov[id_col].isin(base[id_col])].copy()
    ov2 = ov2.sort_values("value")

    rows = []
    for t in thresholds:
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
        TP_c = ((df.y_true == 1) & (df.y_pred == 1)).sum()
        FP_c = ((df.y_true == 0) & (df.y_pred == 1)).sum()
        FN_c = ((df.y_true == 1) & (df.y_pred == 0)).sum()
        TN_c = ((df.y_true == 0) & (df.y_pred == 0)).sum()
        acc_c, f1_c, csi_c = _metrics_from_confusion(TP_c, FP_c, FN_c, TN_c)

        # Area-based confusion
        TP_a = df.loc[(df.y_true == 1) & (df.y_pred == 1), area_col].sum()
        FP_a = df.loc[(df.y_true == 0) & (df.y_pred == 1), area_col].sum()
        FN_a = df.loc[(df.y_true == 1) & (df.y_pred == 0), area_col].sum()
        TN_a = df.loc[(df.y_true == 0) & (df.y_pred == 0), area_col].sum()
        acc_a, f1_a, csi_a = _metrics_from_confusion(TP_a, FP_a, FN_a, TN_a)

        rows.append({
            "threshold": t,
            "acc_count": acc_c, "f1_count": f1_c, "csi_count": csi_c,
            "acc_area":  acc_a, "f1_area":  f1_a, "csi_area":  csi_a,
            "TP_count": TP_c, "FP_count": FP_c, "FN_count": FN_c, "TN_count": TN_c,
        })

    return pd.DataFrame(rows)


def run_step2(cfg: dict) -> None:
    s = cfg["step2"]

    log.info("=== STEP 2: Raster threshold sweep ===")

    # Decide which parquet to use: Step 1 output or override
    if cfg["steps"].get("merge_ground_truth"):
        input_parquet = cfg["step1"]["out_parquet"]
        log.info("  Using Step 1 output: %s", input_parquet)
    else:
        input_parquet = s["input_parquet"]
        log.info("  Step 1 skipped — using: %s", input_parquet)

    bld_out = gpd.read_parquet(input_parquet)
    log.info("  Loaded %d buildings", len(bld_out))

    gt_col            = s["gt_col"]
    id_col            = s["id_col"]
    area_col          = s["area_col"]
    area_frac_thresh  = float(s["area_frac_threshold"])
    band              = int(s.get("band", 1))
    raster_paths      = s["target_raster_paths"]

    t_cfg = s["thresholds"]
    thresholds = np.round(
        np.arange(
            float(t_cfg["start"]),
            float(t_cfg["stop"]) + 1e-9,
            float(t_cfg["step"]),
        ),
        2,
    )
    log.info("  Thresholds: %s", thresholds.tolist())
    log.info("  Rasters to process: %d", len(raster_paths))

    results = []
    for i, rp in enumerate(raster_paths, 1):
        log.info("  [%d/%d] Processing: %s", i, len(raster_paths), rp)
        try:
            bld_m, ov = _compute_pixel_overlap_table(bld_out, rp, id_col, area_col, band)
            df_metrics = _sweep_thresholds(bld_m, ov, thresholds, gt_col, id_col, area_col, area_frac_thresh)
            df_metrics["raster"] = rp
            results.append(df_metrics)
        except Exception as exc:
            log.error("  FAILED for %s: %s", rp, exc)

    if not results:
        log.error("  No results produced — check raster paths and input parquet.")
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
    matplotlib.use("Agg")          # headless / no display required
    import matplotlib.pyplot as plt

    s = cfg["step3"]

    log.info("=== STEP 3: Saving threshold-sweep plots ===")

    # Decide which CSV to read
    if cfg["steps"].get("threshold_sweep"):
        input_csv = cfg["step2"]["out_csv"]
    else:
        input_csv = s["input_csv"]
        log.info("  Step 2 skipped — using: %s", input_csv)

    sweep_df = pd.read_csv(input_csv)
    log.info("  Loaded sweep CSV: %d rows", len(sweep_df))

    # Filter to only the rasters defined in step2
    selected_rasters = cfg["step2"]["target_raster_paths"]
    before = len(sweep_df)
    sweep_df = sweep_df[sweep_df["raster"].isin(selected_rasters)].copy()
    log.info("  Filtered to step2 rasters: %d → %d rows (%d rasters)",
             before, len(sweep_df), sweep_df["raster"].nunique())

    out_dir = Path(s["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    color_map = s.get("color_map", {})
    xlim      = s.get("xlim", [0.3, 1.0])
    ylim      = s.get("ylim", [0.4, 0.9])

    grouped = {
        r: df.sort_values("threshold")
        for r, df in sweep_df.groupby("raster")
    }

    raster_names = list(grouped.keys())
    _cmap = plt.cm.get_cmap("tab10", max(len(raster_names), 1))
    auto_colors = {name: _cmap(i) for i, name in enumerate(raster_names)}

    plots = [
        ("csi_count", "CSI vs Threshold (Count)"),
        ("f1_count",  "F1 vs Threshold (Count)"),
        ("csi_area",  "CSI vs Threshold (Area)"),
        ("f1_area",   "F1 vs Threshold (Area)"),
    ]

    BG_COLOR    = "#000000"
    GRID_COLOR  = "#333333"
    TEXT_COLOR  = "#ffffff"
    SPINE_COLOR = "#555555"

    for col, title in plots:
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor(BG_COLOR)

        for raster_name, df_r in grouped.items():
            color = next(
                (c for k, c in color_map.items() if k in raster_name),
                auto_colors[raster_name],
            )
            marker = "x" if "mask" in raster_name else "o"
            ax.plot(df_r["threshold"], df_r[col], marker=marker,
                    markersize=3, color=color, label=raster_name)

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
        description="Building damage metrics pipeline (PBS-compatible)."
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml in CWD)",
    )
    return parser.parse_args()


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    steps  = cfg.get("steps", {})

    if steps.get("merge_ground_truth"):
        run_step1(cfg)
    else:
        log.info("Step 1 (merge_ground_truth) is disabled — skipping.")

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
