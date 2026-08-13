"""Run manifests, QA scorecards, and preflight checks."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

from . import __version__
from .paths import output_path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute a SHA256 checksum for a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def raster_info(path: Path) -> Dict[str, Any]:
    """Return lightweight raster metadata."""
    with rasterio.open(path) as src:
        return {
            "path": str(path),
            "exists": True,
            "width": int(src.width),
            "height": int(src.height),
            "crs": str(src.crs) if src.crs else None,
            "transform": list(src.transform)[:6],
            "nodata": src.nodata,
            "dtype": src.dtypes[0],
            "bounds": [float(v) for v in src.bounds],
        }


def safe_file_info(path: Path, checksum_max_bytes: int = 128 * 1024 * 1024) -> Dict[str, Any]:
    """Return file metadata, with checksum for reasonably small files."""
    info: Dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return info
    size = path.stat().st_size
    info["size_bytes"] = int(size)
    info["mtime_utc"] = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    if size <= checksum_max_bytes:
        info["sha256"] = sha256_file(path)
    else:
        info["sha256"] = None
        info["sha256_skipped_reason"] = f"file larger than {checksum_max_bytes} bytes"
    if path.suffix.lower() in {".tif", ".tiff"}:
        try:
            info["raster"] = raster_info(path)
        except Exception as exc:  # pragma: no cover - defensive metadata path.
            info["raster_error"] = str(exc)
    return info


def dataframe_first_row(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    out: Dict[str, Any] = {}
    for key, value in df.iloc[0].to_dict().items():
        if isinstance(value, (np.integer,)):
            out[key] = int(value)
        elif isinstance(value, (np.floating,)):
            out[key] = float(value)
        else:
            out[key] = value
    return out


def raster_stats(path: Path, positive_only: bool = False) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True).filled(np.nan).astype("float64")
    valid = np.isfinite(arr)
    if positive_only:
        valid &= arr > 0
    vals = arr[valid]
    if vals.size == 0:
        return {"exists": True, "valid_cells": 0}
    return {
        "exists": True,
        "valid_cells": int(vals.size),
        "min": float(np.nanmin(vals)),
        "mean": float(np.nanmean(vals)),
        "median": float(np.nanmedian(vals)),
        "max": float(np.nanmax(vals)),
        "p95": float(np.nanpercentile(vals, 95)),
        "p99": float(np.nanpercentile(vals, 99)),
    }


def d4_connectivity_stats(mask_path: Path) -> Dict[str, Any]:
    if not mask_path.exists():
        return {"exists": False}
    with rasterio.open(mask_path) as src:
        mask = src.read(1, masked=True).filled(0) > 0
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    labels, n_components = ndimage.label(mask, structure=structure)
    sizes = np.bincount(labels.ravel())[1:]
    if sizes.size:
        sizes_sorted = np.sort(sizes)[::-1]
        largest = int(sizes_sorted[0])
        top10 = int(sizes_sorted[:10].sum())
    else:
        largest = 0
        top10 = 0
    river_cells = int(mask.sum())
    return {
        "exists": True,
        "river_cells": river_cells,
        "d4_connected_components": int(n_components),
        "largest_component_cells": largest,
        "largest_component_percent_of_river_cells": 100.0 * largest / max(river_cells, 1),
        "top10_components_cells": top10,
        "top10_components_percent_of_river_cells": 100.0 * top10 / max(river_cells, 1),
    }


def write_run_manifest(
    out_dir: Path,
    cfg: Any,
    key_outputs: Optional[Iterable[Path]] = None,
    config_path: Optional[Path] = None,
) -> Path:
    """Write a reproducibility manifest for the current run."""
    out_dir = Path(out_dir)
    if key_outputs is None:
        key_outputs = [
            output_path(out_dir, "DEM_hydraulic_conditioned.tif"),
            output_path(out_dir, "DEM_modification_final_minus_cleaned.tif"),
            output_path(out_dir, "D4_idx_facc.tif"),
            output_path(out_dir, "D4_H_abg_m.tif"),
            output_path(out_dir, "D4_HydroPol2D_creek_reduction_summary.csv"),
            output_path(out_dir, "modification_summary.csv"),
        ]

    cfg_dict = asdict(cfg) if is_dataclass(cfg) else dict(cfg)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "config_path": str(config_path) if config_path else None,
        "configuration": cfg_dict,
        "inputs": {
            "original_dem": safe_file_info(Path(cfg_dict.get("original_dem") or cfg_dict.get("dem", ""))),
        },
        "outputs": {path.name: safe_file_info(Path(path)) for path in key_outputs},
    }
    path = output_path(out_dir, "run_manifest.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def build_qa_scorecard(out_dir: Path, thresholds: Optional[Mapping[str, float]] = None) -> pd.DataFrame:
    """Build a compact run QA scorecard."""
    out_dir = Path(out_dir)
    thresholds = dict(thresholds or {})

    mod = dataframe_first_row(output_path(out_dir, "modification_summary.csv"))
    creek = dataframe_first_row(output_path(out_dir, "D4_HydroPol2D_creek_reduction_summary.csv"))
    corridor = dataframe_first_row(output_path(out_dir, "D4_external_river_profile_QA_summary.csv"))
    conn = d4_connectivity_stats(output_path(out_dir, "D4_idx_facc.tif"))
    habg = raster_stats(output_path(out_dir, "D4_H_abg_m.tif"), positive_only=True)

    checks = []

    def add(metric: str, value: Any, warn_if: bool = False, fail_if: bool = False, note: str = "") -> None:
        status = "fail" if fail_if else "warn" if warn_if else "ok"
        checks.append({"metric": metric, "value": value, "status": status, "note": note})

    add(
        "changed_percent",
        mod.get("changed_percent"),
        warn_if=float(mod.get("changed_percent", 0) or 0) > thresholds.get("changed_percent_warn", 25.0),
        note="Percent of valid cells changed relative to cleaned DEM.",
    )
    add(
        "max_lowering_m",
        mod.get("max_lowering_m"),
        warn_if=float(mod.get("max_lowering_m", 0) or 0) > thresholds.get("max_lowering_warn_m", 100.0),
        note="Large values can come from breaching, not only river bathymetry.",
    )
    add("river_cells", creek.get("river_cells"), note="D4 river-mask cells.")
    add(
        "power_law_fallback_cells",
        creek.get("power_law_fallback_cells"),
        warn_if=float(creek.get("power_law_fallback_cells", 0) or 0) > thresholds.get("fallback_cells_warn", 1000.0),
        note="Cells using base HydroPol2D fallback instead of spatial coefficients.",
    )
    add("H_abg_mean_m", habg.get("mean"), note="Mean positive D4 bathymetry lowering.")
    add(
        "H_abg_max_m",
        habg.get("max"),
        warn_if=float(habg.get("max", 0) or 0) > thresholds.get("habg_max_warn_m", 45.0),
        note="Near-cap values deserve inspection.",
    )
    add(
        "channel_bed_receiver_sill_cells_gt_0p01m",
        creek.get("channel_bed_receiver_sill_cells_gt_0p01m"),
        warn_if=float(creek.get("channel_bed_receiver_sill_cells_gt_0p01m", 0) or 0) > 0,
        note="Neal channel-bed links that become uphill after RiverDepth is applied.",
    )
    add(
        "channel_bed_receiver_sill_max_m",
        creek.get("channel_bed_receiver_sill_max_m"),
        warn_if=float(creek.get("channel_bed_receiver_sill_max_m", 0) or 0) > thresholds.get("channel_bed_sill_warn_m", 0.1),
        note="Inspect D4_Neal_channel_receiver_sill_m.tif before using geometry in HydroPol2D.",
    )
    if corridor:
        add(
            "profile_lowering_max_m",
            corridor.get("profile_lowering_max_m"),
            warn_if=float(corridor.get("profile_lowering_max_m", 0) or 0) > thresholds.get("profile_lowering_warn_m", 100.0),
            note="Maximum terrain lowering used to create the external river-corridor profile.",
        )
    add("d4_connected_components", conn.get("d4_connected_components"), note="4-neighbour river-mask components.")
    add(
        "largest_component_percent",
        conn.get("largest_component_percent_of_river_cells"),
        warn_if=float(conn.get("largest_component_percent_of_river_cells", 100) or 0) < thresholds.get("largest_component_min_percent", 50.0),
        note="Low values indicate a fragmented D4 river mask.",
    )

    df = pd.DataFrame(checks)
    csv_path = output_path(out_dir, "qa_scorecard.csv")
    json_path = output_path(out_dir, "qa_scorecard.json")
    md_path = output_path(out_dir, "qa_scorecard.md")
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    md_path.write_text(_simple_markdown_table(df), encoding="utf-8")
    return df


def _simple_markdown_table(df: pd.DataFrame) -> str:
    """Render a small dataframe as Markdown without optional dependencies."""
    headers = [str(c) for c in df.columns]
    rows = [[str(v) for v in row] for row in df.to_numpy()]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    header_line = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows]
    return "\n".join([header_line, sep_line, *body]) + "\n"


def preflight_checks(
    dem: str | Path,
    out_dir: str | Path,
    required_paths: Optional[Iterable[str | Path]] = None,
) -> pd.DataFrame:
    """Run lightweight preflight checks for a configured workflow."""
    checks = []
    dem_path = Path(dem).expanduser()
    out_path = Path(out_dir).expanduser()

    def add(name: str, status: str, detail: str = "") -> None:
        checks.append({"check": name, "status": status, "detail": detail})

    add("input_dem_exists", "ok" if dem_path.exists() else "fail", str(dem_path))
    if dem_path.exists():
        try:
            info = raster_info(dem_path)
            crs = info.get("crs")
            with rasterio.open(dem_path) as src:
                is_projected = bool(src.crs and src.crs.is_projected)
            add("input_dem_projected_crs", "ok" if is_projected else "fail", str(crs))
            add("input_dem_size", "ok", f"{info['width']} x {info['height']}")
        except Exception as exc:
            add("input_dem_readable", "fail", str(exc))

    try:
        out_path.mkdir(parents=True, exist_ok=True)
        test = out_path / ".preflight_write_test"
        test.write_text("ok", encoding="utf-8")
        test.unlink()
        add("output_dir_writable", "ok", str(out_path))
    except Exception as exc:
        add("output_dir_writable", "fail", str(exc))

    for path in required_paths or []:
        p = Path(path).expanduser()
        add(f"required_path_exists:{p.name}", "ok" if p.exists() else "fail", str(p))

    return pd.DataFrame(checks)
