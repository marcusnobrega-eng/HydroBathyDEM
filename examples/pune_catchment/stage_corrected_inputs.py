#!/usr/bin/env python3
"""Stage the supplied Pune inputs on one explicit 30 m reference grid.

The corrected DEM is delivered on a 29.6148728 m grid whereas the static
hydrologic layers are on the canonical 30 m Pune grid.  HydroBathyDEM must
not mix those grids.  This utility copies the supplied static layers and
resamples the DEM once, with an auditable provenance manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


LAYER_FILES = {
    "soil": "SOIL (6).tif",
    "lulc_esa": "LULC_ESA (1).tif",
    "lai": "LAI (5).tif",
    "dtb": "DTB (6).tif",
    "albedo": "Albedo (2).tif",
}
DEM_FILE = "DEM_30m_corrected (1).tif"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def grid_metadata(dataset: rasterio.io.DatasetReader) -> dict[str, object]:
    return {
        "width": dataset.width,
        "height": dataset.height,
        "crs": str(dataset.crs),
        "transform": list(dataset.transform)[:6],
        "nodata": dataset.nodata,
    }


def same_grid(left: rasterio.io.DatasetReader, right: rasterio.io.DatasetReader) -> bool:
    """Compare spatial geometry only; valid NoData conventions may differ."""
    return (
        left.width == right.width
        and left.height == right.height
        and left.crs == right.crs
        and left.transform.almost_equals(right.transform)
    )


def stage(source_dir: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    reference_path = source_dir / LAYER_FILES["soil"]
    if not reference_path.exists():
        raise FileNotFoundError(f"Missing reference SOIL raster: {reference_path}")

    manifest: dict[str, object] = {"source_dir": str(source_dir), "layers": {}}
    with rasterio.open(reference_path) as reference:
        reference_meta = reference.meta.copy()
        reference_grid = grid_metadata(reference)
        manifest["reference_grid"] = reference_grid

        for name, filename in LAYER_FILES.items():
            source = source_dir / filename
            if not source.exists():
                raise FileNotFoundError(f"Missing {name} raster: {source}")
            with rasterio.open(source) as dataset:
                if not same_grid(dataset, reference):
                    raise ValueError(f"{name} is not aligned to the canonical Pune 30 m grid.")
            target = destination / f"pune_{name}_30m.tif"
            shutil.copy2(source, target)
            manifest["layers"][name] = {"source": str(source), "sha256": sha256(source), "staged": str(target)}

        # The mesh builder expects percent imperviousness.  ESA WorldCover is
        # categorical, so make the semantic conversion explicit rather than
        # passing class identifiers into a percent threshold.  Class 50 is
        # built-up land in ESA WorldCover.
        with rasterio.open(destination / "pune_lulc_esa_30m.tif") as lulc_ds:
            lulc = lulc_ds.read(1)
            impervious = np.where(lulc == 50, 100, 0).astype(np.uint8)
            impervious_profile = lulc_ds.profile | {"dtype": "uint8", "nodata": 255, "compress": "deflate"}
            impervious_path = destination / "pune_impervious_from_esa_lulc_30m.tif"
            with rasterio.open(impervious_path, "w", **impervious_profile) as output_ds:
                output_ds.write(impervious, 1)
        manifest["layers"]["impervious_from_lulc_esa"] = {
            "staged": str(impervious_path),
            "rule": "100 where ESA WorldCover class equals 50 (built-up), else 0",
        }

        dem_source = source_dir / DEM_FILE
        if not dem_source.exists():
            raise FileNotFoundError(f"Missing corrected DEM: {dem_source}")
        dem_target = destination / "pune_dem_corrected_aligned_30m.tif"
        with rasterio.open(dem_source) as source:
            output = np.full((reference.height, reference.width), np.nan, dtype=np.float32)
            reproject(
                rasterio.band(source, 1), output,
                src_transform=source.transform, src_crs=source.crs, src_nodata=source.nodata,
                dst_transform=reference.transform, dst_crs=reference.crs, dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
            profile = reference_meta | {"dtype": "float32", "nodata": np.nan, "compress": "deflate", "predictor": 3}
            with rasterio.open(dem_target, "w", **profile) as destination_ds:
                destination_ds.write(output, 1)
            manifest["layers"]["dem"] = {
                "source": str(dem_source), "sha256": sha256(dem_source), "staged": str(dem_target),
                "source_grid": grid_metadata(source), "resampling": "bilinear to canonical 30 m grid",
            }

    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("/Users/mngomes/Downloads"))
    parser.add_argument("--destination", type=Path, default=Path("examples/pune_catchment/data/pune_corrected_30m"))
    args = parser.parse_args()
    stage(args.source_dir, args.destination)


if __name__ == "__main__":
    main()
