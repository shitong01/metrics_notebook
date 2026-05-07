# Building Damage Validation Pipeline

Evaluates SAR-derived damage-probability rasters against ground-truth building labels.
The pipeline sweeps a range of probability thresholds, computes per-building predictions
by comparing raster-pixel overlap area to each building footprint, and reports
Accuracy / F1 / CSI (and Precision / Recall / FPR for the Turkiye variant) across all
thresholds. Results are saved as a CSV and a set of threshold-curve PNG plots.

---

## Repository structure

```
metrics_calc.py                  # Generic pipeline (damage points → buildings → raster sweep)
metrics_calc_building_tky.py     # Turkiye pipeline (pre-labelled footprints → raster sweep)
config.yaml                      # Config for metrics_calc.py  (LA Wildfires example)
config_building_tky.yaml         # Config for metrics_calc_building_tky.py  (Turkiye TD021)
run_metrics.pbs                  # PBS job script for metrics_calc.py
run_metrics_tky.pbs              # PBS job script for metrics_calc_building_tky.py
Turkiye/translations.py          # Turkish → English damage-label look-up tables
Turkiye/validation/              # Input GeoPackage with per-building ACIKLAMA labels
Turkiye/TD021/                   # SAR damage probability rasters (Turkiye track TD021)
```

---

## Environment setup

```bash
conda install -c conda-forge \
  jupyterlab notebook \
  gdal rasterio rioxarray \
  geopandas shapely fiona pyproj \
  xarray dask netcdf4 h5py \
  numpy scipy pandas scikit-learn \
  matplotlib seaborn \
  tqdm rich \
  -y
```

---

## Quick start

### Run locally

```bash
# Generic pipeline (LA Wildfires) — reads config.yaml by default
python metrics_calc.py
python metrics_calc.py --config my_config.yaml

# Turkiye pipeline — reads config_building_tky.yaml by default
python metrics_calc_building_tky.py
python metrics_calc_building_tky.py --config my_config.yaml
```

### Run on a PBS cluster

```bash
qsub run_metrics.pbs
qsub run_metrics_tky.pbs

# Override the config file at submit time
qsub -v CONFIG=my_config.yaml run_metrics_tky.pbs
```

Logs land in `logs/metrics_calc.log` and `logs/metrics_tky.log` respectively.

---

## Script reference

### `metrics_calc.py` — generic pipeline

Ground-truth comes as **damage point files** (GeoJSON / Shapefile) that must first be
spatially joined onto building footprints. Use this script when you have per-point
field assessments (e.g. DINS data for LA Wildfires).

Three steps, each toggled independently in `config.yaml` under `steps:`.

**Step 1 — `merge_ground_truth`**

| | |
|---|---|
| Input | Damage point files + building footprint GeoParquet(s) |
| Output | GeoParquet with `damage_class_inherited_gnd` and `damage_binary_gnd` columns |
| Logic | Spatial join (points within building polygons); aggregate multiple points per building by `majority` vote or `max_severity` rule; map categories to binary Damaged / Undamaged |

**Step 2 — `threshold_sweep`**

| | |
|---|---|
| Input | Parquet from Step 1 (or `input_parquet` override) + one or more rasters |
| Output | CSV with Accuracy / F1 / CSI for every (raster, threshold) combination |
| Logic | Polygonise raster pixels; intersect with buildings; label building as Damaged if `damaged_pixel_area / building_area >= area_frac_threshold` |

**Step 3 — `plot`**

| | |
|---|---|
| Input | CSV from Step 2 (or `input_csv` override) |
| Output | 4 PNG files — `csi_count`, `f1_count`, `csi_area`, `f1_area` |

---

### `metrics_calc_building_tky.py` — Turkiye pipeline

Ground-truth is **already per building** in a GeoPackage (Basarsoft / Microsoft
footprints), with a Turkish damage label in the `ACIKLAMA` column. Step 1 translates
those labels rather than running a spatial join.

**Step 1 — `label_ground_truth`**

| | |
|---|---|
| Input | GeoPackage with `ACIKLAMA` column |
| Output | GeoPackage with `damage_class`, `damage_score`, `damage_binary`, `area_m2` |
| Logic | Look up each Turkish label in `Turkiye/translations.py`; map to binary Damaged / Undamaged using the lists in `step1.damaged_classes` / `step1.undamaged_classes`; drop buildings with unmapped labels |

**Steps 2 & 3** follow the same logic as the generic script with two additions:
- **Overlap-table caching** (`cache_dir`) — expensive pixel/polygon intersections are
  written to disk on the first run and reloaded on subsequent runs.
- **Windowed raster reads** — only the portion of the raster that overlaps the building
  extent is loaded, giving a large speed-up on full-scene rasters.
- **Extra metrics** — Precision, Recall, FPR are also computed → 8 plots instead of 4.

---

## Configuration reference

Both scripts share the same three-section YAML structure.

### `steps:` block

```yaml
steps:
  merge_ground_truth: true   # (metrics_calc.py only)    Step 1 on/off
  label_ground_truth: true   # (tky script only)         Step 1 on/off
  threshold_sweep:    true   # Step 2 on/off
  plot:               true   # Step 3 on/off
```

Set a step to `false` to skip it. When Step 1 is skipped, Step 2 reads from
`step2.input_parquet` (generic) or `step2.input_gpkg` (Turkiye) instead.

### `step1:` keys

| Key | Description |
|---|---|
| `pts_shp` | (generic) List of point-file paths |
| `bld_gpq` | (generic) List of building GeoParquet paths |
| `valid_polygon` | (Turkiye) Path to input GeoPackage |
| `pts_class_col` | Column holding the raw damage label (`DAMAGE` or `ACIKLAMA`) |
| `bld_id_col` | Unique building ID column |
| `rule` | (generic) Aggregation rule: `majority` or `max_severity` |
| `severity_order` | (generic) List of classes from least to most severe, for `max_severity` |
| `damaged_classes` | English class names that map to `"Damaged"` |
| `undamaged_classes` | English class names that map to `"Undamaged"` |
| `out_parquet` / `out_gpkg` | Output file path |

### `step2:` keys

| Key | Description |
|---|---|
| `input_parquet` / `input_gpkg` | Input file when Step 1 is disabled |
| `target_raster_paths` | List of damage-probability raster paths to evaluate |
| `gt_col` | Ground-truth binary column name |
| `id_col` | Building ID column name |
| `area_col` | Building area column name |
| `area_frac_threshold` | Fraction of building area that must be "damaged pixels" for a building to be classified Damaged (0.0–1.0) |
| `thresholds.start/stop/step` | Raster probability sweep range (inclusive, rounded to 2 d.p.) |
| `band` | Raster band index (1-based) |
| `cache_dir` | (Turkiye) Directory for cached overlap tables; delete to force re-computation |
| `save_per_threshold` | (Turkiye) Write a per-threshold GeoPackage of predictions |
| `thresholded_dir` | (Turkiye) Output directory for per-threshold GeoPackages |
| `out_csv` | Output CSV path |

**Dual-threshold logic:**

```
pixel is "damaged"  if  raster_value >= raster_threshold   (swept over thresholds.start → stop)
building is "Damaged" if  damaged_pixel_area / building_area >= area_frac_threshold
```

### `step3:` keys

| Key | Description |
|---|---|
| `input_csv` | CSV when Step 2 is disabled |
| `out_dir` | Directory where PNGs are saved |
| `xlim` / `ylim` | Plot axis limits `[min, max]` |
| `color_map` | Dict mapping substring of raster filename → matplotlib color |

---

## Outputs

| Path | Produced by | Description |
|---|---|---|
| `*_w_gnd_damage.parquet` | Step 1 (generic) | Building footprints with inherited ground-truth class |
| `Turkiye/validation/*_Binary.gpkg` | Step 1 (Turkiye) | Buildings labelled Damaged / Undamaged |
| `threshold_sweep_metrics*.csv` | Step 2 | One row per (raster, threshold) with all metrics |
| `plots/*.png` | Step 3 (generic) | CSI and F1 threshold curves |
| `Turkiye/plots/**/*.png` | Step 3 (Turkiye) | CSI, F1, Precision, Recall threshold curves |
| `Turkiye/validation_overlap/` | Step 2 (Turkiye) | Cached pixel-overlap tables (GPKG + CSV) |

---

## Adding a new event

1. Copy `config.yaml` (generic) or `config_building_tky.yaml` (Turkiye) and edit paths.
2. Set all three `steps:` flags to `true` for a fresh run.
3. Point `step1` at your ground-truth data and `step2.target_raster_paths` at your rasters.
4. Run locally or submit via PBS with `qsub -v CONFIG=your_new_config.yaml <script.pbs>`.
5. Check `logs/` for progress; results appear in the CSV and `out_dir` from `step3`.
