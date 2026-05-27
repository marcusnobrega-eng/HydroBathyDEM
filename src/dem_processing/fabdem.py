"""Build analysis-ready DEMs from FABDEM data and catchment boundaries."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.fill import fillnodata
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, transform_bounds
from shapely.geometry import box
from shapely.ops import unary_union

try:
    from shapely.validation import make_valid
except ImportError:  # pragma: no cover - for older Shapely versions.
    make_valid = None


PathLike = Union[str, Path]
logger = logging.getLogger(__name__)

DEFAULT_NODATA = -9999.0
DEFAULT_RASTER_OPTIONS = {
    "compress": "deflate",
    "tiled": True,
    "BIGTIFF": "IF_SAFER",
}


@dataclass(frozen=True)
class GapFillResult:
    """Summary returned by :func:`fill_dem_nodata`."""

    input_dem: Path
    output_dem: Path
    nodata_pixels_before: int
    nodata_pixels_after: int
    max_search_distance: float
    smoothing_iterations: int

    @property
    def filled_pixels(self) -> int:
        return self.nodata_pixels_before - self.nodata_pixels_after


@dataclass(frozen=True)
class DemBuildResult:
    """Paths and metadata from a FABDEM DEM build."""

    final_dem: Path
    unfilled_dem: Path
    mosaic_native: Path
    clipped_native: Path
    filled_dem: Optional[Path]
    target_crs: CRS
    target_resolution: Union[float, tuple[float, float]]
    input_tiles: int
    overlapping_tiles: int
    source_mode: str


def ensure_directory(path: PathLike) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def remove_file(path: PathLike) -> None:
    file_path = Path(path)
    if file_path.exists():
        file_path.unlink()


def find_rasters(
    folder: PathLike,
    *,
    extensions: Sequence[str] = (".tif", ".tiff"),
    recursive: bool = False,
) -> list[Path]:
    """Find raster files in a folder."""

    root = Path(folder)
    if not root.exists():
        raise FileNotFoundError(f"Raster folder does not exist: {root}")

    pattern = "**/*" if recursive else "*"
    allowed = {ext.lower() for ext in extensions}
    rasters = [p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in allowed]
    return sorted(rasters)


def _repair_geometry(geometry):
    if geometry is None or geometry.is_empty or geometry.is_valid:
        return geometry
    if make_valid is not None:
        return make_valid(geometry)
    return geometry.buffer(0)


def read_aoi(aoi_vector_path: PathLike) -> gpd.GeoDataFrame:
    """Read and lightly clean an area-of-interest vector file."""

    aoi = gpd.read_file(aoi_vector_path)
    if aoi.empty:
        raise ValueError(f"AOI vector contains no features: {aoi_vector_path}")
    if aoi.crs is None:
        raise ValueError(f"AOI vector has no CRS. Define a CRS before processing: {aoi_vector_path}")

    aoi = aoi[aoi.geometry.notnull() & ~aoi.geometry.is_empty].copy()
    if aoi.empty:
        raise ValueError(f"AOI vector contains no usable geometries: {aoi_vector_path}")

    aoi["geometry"] = aoi.geometry.apply(_repair_geometry)
    aoi = aoi[aoi.geometry.notnull() & ~aoi.geometry.is_empty].copy()
    if aoi.empty:
        raise ValueError(f"AOI geometries could not be repaired: {aoi_vector_path}")

    return aoi


def choose_target_crs(
    aoi: gpd.GeoDataFrame,
    target_crs: Optional[Union[str, CRS]] = None,
    *,
    fallback_crs: Union[str, CRS] = "EPSG:6933",
) -> CRS:
    """Choose the output CRS from user input, AOI CRS, or a global fallback."""

    if target_crs is not None:
        return CRS.from_user_input(target_crs)

    if aoi.crs is not None:
        aoi_crs = CRS.from_user_input(aoi.crs)
        if aoi_crs.is_projected:
            return aoi_crs

    return CRS.from_user_input(fallback_crs)


def aoi_bounds_epsg4326(aoi: gpd.GeoDataFrame, *, pad_degrees: float = 0.0) -> tuple[float, float, float, float]:
    """Return AOI bounds as ``west, south, east, north`` in EPSG:4326."""

    west, south, east, north = aoi.to_crs("EPSG:4326").total_bounds
    west -= pad_degrees
    south -= pad_degrees
    east += pad_degrees
    north += pad_degrees
    return (float(max(west, -180.0)), float(max(south, -90.0)), float(min(east, 180.0)), float(min(north, 90.0)))


def raster_overlaps_bounds(
    raster_path: PathLike,
    bounds: tuple[float, float, float, float],
    bounds_crs: Union[str, CRS],
    *,
    densify_pts: int = 21,
) -> bool:
    """Check whether a raster overlaps a bounding box in a specified CRS."""

    bounds_crs = CRS.from_user_input(bounds_crs)
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            logger.warning("Skipping raster with undefined CRS: %s", raster_path)
            return False
        try:
            raster_bounds = transform_bounds(src.crs, bounds_crs, *src.bounds, densify_pts=densify_pts)
        except Exception as exc:  # pragma: no cover - defensive logging.
            logger.warning("Could not transform bounds for %s: %s", raster_path, exc)
            return False

    return box(*raster_bounds).intersects(box(*bounds))


def filter_rasters_by_aoi(
    raster_paths: Iterable[PathLike],
    aoi: gpd.GeoDataFrame,
    target_crs: Union[str, CRS],
) -> list[Path]:
    """Return rasters whose bounding boxes overlap an AOI."""

    target_crs = CRS.from_user_input(target_crs)
    aoi_target = aoi.to_crs(target_crs)
    aoi_union = unary_union(aoi_target.geometry)
    aoi_bounds = aoi_union.bounds
    overlapping = [
        Path(raster_path)
        for raster_path in raster_paths
        if raster_overlaps_bounds(raster_path, aoi_bounds, target_crs)
    ]
    return sorted(overlapping)


def _should_skip_output(path: Path, overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        logger.info("Using existing output because overwrite=False: %s", path)
        return True
    if path.exists() and overwrite:
        remove_file(path)
    return False


def _with_raster_options(profile: dict, **updates) -> dict:
    out = profile.copy()
    out.update(DEFAULT_RASTER_OPTIONS)
    out.update(updates)
    return out


def mosaic_rasters(
    raster_paths: Sequence[PathLike],
    output_path: PathLike,
    *,
    nodata: Optional[float] = None,
    overwrite: bool = False,
) -> Path:
    """Mosaic raster tiles into a single GeoTIFF."""

    if not raster_paths:
        raise ValueError("No raster paths were provided for mosaicking.")

    output_path = Path(output_path)
    ensure_directory(output_path.parent)
    if _should_skip_output(output_path, overwrite):
        return output_path

    srcs = [rasterio.open(path) for path in raster_paths]
    try:
        mosaic_nodata = nodata if nodata is not None else srcs[0].nodata
        if mosaic_nodata is None:
            mosaic_nodata = DEFAULT_NODATA

        logger.info("Mosaicking %d raster tiles into %s", len(srcs), output_path)
        mosaic, mosaic_transform = merge(srcs, nodata=mosaic_nodata)
        profile = _with_raster_options(
            srcs[0].meta,
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=mosaic_transform,
            crs=srcs[0].crs,
            nodata=mosaic_nodata,
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mosaic)
    finally:
        for src in srcs:
            src.close()

    return output_path


def clip_raster_to_aoi(
    input_raster: PathLike,
    aoi: gpd.GeoDataFrame,
    output_path: PathLike,
    *,
    all_touched: bool = False,
    crop: bool = True,
    nodata: Optional[float] = None,
    overwrite: bool = False,
) -> Path:
    """Clip a raster to an area-of-interest polygon."""

    output_path = Path(output_path)
    ensure_directory(output_path.parent)
    if _should_skip_output(output_path, overwrite):
        return output_path

    with rasterio.open(input_raster) as src:
        if src.crs is None:
            raise ValueError(f"Input raster has no CRS: {input_raster}")

        output_nodata = nodata if nodata is not None else src.nodata
        if output_nodata is None:
            output_nodata = DEFAULT_NODATA

        aoi_src = aoi.to_crs(src.crs)
        aoi_union = unary_union(aoi_src.geometry)

        logger.info("Clipping %s to AOI", input_raster)
        clipped, clipped_transform = mask(
            src,
            [aoi_union],
            crop=crop,
            nodata=output_nodata,
            filled=True,
            all_touched=all_touched,
        )

        profile = _with_raster_options(
            src.meta,
            height=clipped.shape[1],
            width=clipped.shape[2],
            transform=clipped_transform,
            nodata=output_nodata,
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(clipped)

    return output_path


def reproject_resample_raster(
    input_raster: PathLike,
    output_path: PathLike,
    target_crs: Union[str, CRS],
    target_resolution: Union[float, tuple[float, float]],
    *,
    resampling: Resampling = Resampling.bilinear,
    dst_dtype: Optional[str] = "float32",
    nodata: Optional[float] = None,
    overwrite: bool = False,
) -> Path:
    """Reproject and resample a raster to a target CRS and resolution."""

    output_path = Path(output_path)
    ensure_directory(output_path.parent)
    if _should_skip_output(output_path, overwrite):
        return output_path

    target_crs = CRS.from_user_input(target_crs)
    with rasterio.open(input_raster) as src:
        if src.crs is None:
            raise ValueError(f"Input raster has no CRS: {input_raster}")

        src_nodata = src.nodata if src.nodata is not None else DEFAULT_NODATA
        dst_nodata = nodata if nodata is not None else src_nodata
        transform, width, height = calculate_default_transform(
            src.crs,
            target_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=target_resolution,
        )

        profile_updates = {
            "crs": target_crs,
            "transform": transform,
            "width": width,
            "height": height,
            "nodata": dst_nodata,
        }
        if dst_dtype is not None:
            profile_updates["dtype"] = dst_dtype

        profile = _with_raster_options(src.profile, **profile_updates)

        logger.info("Reprojecting %s to %s at resolution %s", input_raster, target_crs, target_resolution)
        with rasterio.open(output_path, "w", **profile) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src_nodata,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    dst_nodata=dst_nodata,
                    resampling=resampling,
                )

    return output_path


def _nodata_mask(array: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    invalid = ~np.isfinite(array)
    if nodata is not None:
        try:
            if np.isnan(nodata):
                invalid |= np.isnan(array)
            else:
                invalid |= array == nodata
        except TypeError:
            invalid |= array == nodata
    return invalid


def _rasterize_aoi_mask(
    aoi: gpd.GeoDataFrame,
    *,
    out_shape: tuple[int, int],
    transform,
    raster_crs: Union[str, CRS],
    all_touched: bool = False,
) -> np.ndarray:
    aoi_src = aoi.to_crs(raster_crs)
    geometry = unary_union(aoi_src.geometry)
    return rasterize(
        [(geometry, 1)],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=all_touched,
        dtype="uint8",
    ).astype(bool)


def count_nodata_pixels(
    input_raster: PathLike,
    *,
    aoi_vector_path: Optional[PathLike] = None,
    all_touched: bool = False,
) -> int:
    """Count NoData pixels in a single-band raster."""

    with rasterio.open(input_raster) as src:
        array = src.read(1)
        invalid = _nodata_mask(array, src.nodata) | (src.read_masks(1) == 0)

        if aoi_vector_path is not None:
            aoi = read_aoi(aoi_vector_path)
            inside = _rasterize_aoi_mask(
                aoi,
                out_shape=array.shape,
                transform=src.transform,
                raster_crs=src.crs,
                all_touched=all_touched,
            )
            invalid &= inside

    return int(np.count_nonzero(invalid))


def fill_dem_nodata(
    input_dem: PathLike,
    output_dem: PathLike,
    *,
    max_search_distance: float = 20,
    smoothing_iterations: int = 0,
    aoi_vector_path: Optional[PathLike] = None,
    fill_only_inside_aoi: bool = True,
    all_touched: bool = False,
    nodata: Optional[float] = None,
    overwrite: bool = False,
) -> GapFillResult:
    """Fill small NoData gaps in a single-band DEM using interpolation."""

    if max_search_distance <= 0:
        raise ValueError("max_search_distance must be greater than zero.")
    if smoothing_iterations < 0:
        raise ValueError("smoothing_iterations must be zero or greater.")

    input_dem = Path(input_dem)
    output_dem = Path(output_dem)
    ensure_directory(output_dem.parent)

    if output_dem.exists() and not overwrite:
        before = count_nodata_pixels(input_dem, aoi_vector_path=aoi_vector_path)
        after = count_nodata_pixels(output_dem, aoi_vector_path=aoi_vector_path)
        logger.info("Using existing filled DEM because overwrite=False: %s", output_dem)
        return GapFillResult(input_dem, output_dem, before, after, max_search_distance, smoothing_iterations)
    if output_dem.exists() and overwrite:
        remove_file(output_dem)

    with rasterio.open(input_dem) as src:
        if src.count != 1:
            raise ValueError("fill_dem_nodata currently expects a single-band DEM.")
        if src.crs is None:
            raise ValueError(f"Input DEM has no CRS: {input_dem}")

        output_nodata = nodata if nodata is not None else src.nodata
        if output_nodata is None:
            output_nodata = DEFAULT_NODATA

        dem = src.read(1).astype("float32")
        raster_valid_mask = src.read_masks(1) > 0
        invalid = _nodata_mask(dem, output_nodata) | ~raster_valid_mask

        inside_aoi: Optional[np.ndarray] = None
        if aoi_vector_path is not None and fill_only_inside_aoi:
            aoi = read_aoi(aoi_vector_path)
            inside_aoi = _rasterize_aoi_mask(
                aoi,
                out_shape=dem.shape,
                transform=src.transform,
                raster_crs=src.crs,
                all_touched=all_touched,
            )
            fill_targets = invalid & inside_aoi
            valid_donors = (~invalid) & inside_aoi
        else:
            fill_targets = invalid
            valid_donors = ~invalid

        nodata_before = int(np.count_nonzero(fill_targets))
        logger.info("NoData pixels targeted for filling: %d", nodata_before)

        dem_for_fill = dem.copy()
        dem_for_fill[invalid] = output_nodata
        filled = fillnodata(
            dem_for_fill,
            mask=valid_donors.astype("uint8"),
            max_search_distance=max_search_distance,
            smoothing_iterations=smoothing_iterations,
        ).astype("float32")

        output = dem.copy()
        output[fill_targets] = filled[fill_targets]
        output[~fill_targets & invalid] = output_nodata
        if inside_aoi is not None:
            output[~inside_aoi] = output_nodata

        remaining_invalid = _nodata_mask(output, output_nodata) & fill_targets
        nodata_after = int(np.count_nonzero(remaining_invalid))
        profile = _with_raster_options(src.profile, dtype="float32", count=1, nodata=output_nodata)

        with rasterio.open(output_dem, "w", **profile) as dst:
            dst.write(output, 1)

    logger.info("Filled %d DEM pixels; %d targeted pixels remain NoData", nodata_before - nodata_after, nodata_after)
    return GapFillResult(input_dem, output_dem, nodata_before, nodata_after, max_search_distance, smoothing_iterations)


def download_fabdem_bounds(
    bounds: tuple[float, float, float, float],
    output_path: PathLike,
    *,
    cache: Optional[PathLike] = None,
    overwrite: bool = False,
    show_progress: bool = True,
) -> Path:
    """Download and merge FABDEM data for EPSG:4326 bounds using optional dependency ``fabdem``."""

    output_path = Path(output_path)
    ensure_directory(output_path.parent)
    if _should_skip_output(output_path, overwrite):
        return output_path

    try:
        import fabdem as fabdem_downloader
    except ImportError as exc:  # pragma: no cover - depends on optional package.
        raise RuntimeError(
            "Automatic FABDEM download requires the optional 'fabdem' package. "
            "Install it with: python3 -m pip install 'hydrobathydem[fabdem]'"
        ) from exc

    logger.info("Downloading FABDEM for bounds %s into %s", bounds, output_path)
    kwargs = {"output_path": str(output_path), "show_progress": show_progress}
    if cache is not None:
        kwargs["cache"] = Path(cache)
    fabdem_downloader.download(bounds, **kwargs)
    return output_path


def _resolution_label(target_resolution: Union[float, tuple[float, float]]) -> str:
    if isinstance(target_resolution, (int, float)):
        return f"{target_resolution:g}m"
    return f"{target_resolution[0]:g}x{target_resolution[1]:g}"


def build_dem_from_fabdem(
    aoi_vector_path: PathLike,
    output_dir: PathLike,
    *,
    fabdem_dir: Optional[PathLike] = None,
    download: bool = False,
    download_cache: Optional[PathLike] = None,
    download_pad_degrees: float = 0.02,
    output_prefix: str = "DEM_fabdem",
    target_resolution: Union[float, tuple[float, float]] = 30,
    target_crs: Optional[Union[str, CRS]] = None,
    fallback_crs: Union[str, CRS] = "EPSG:6933",
    recursive: bool = False,
    resampling: Resampling = Resampling.bilinear,
    all_touched: bool = False,
    overwrite: bool = False,
    fill_gaps: bool = False,
    fill_max_search_distance: float = 20,
    fill_smoothing_iterations: int = 0,
    keep_intermediate: bool = True,
) -> DemBuildResult:
    """Build a clipped, projected DEM from downloaded FABDEM or local FABDEM tiles."""

    if download == (fabdem_dir is not None):
        raise ValueError("Choose exactly one FABDEM source: set download=True or provide fabdem_dir.")

    output_dir = ensure_directory(output_dir)
    aoi = read_aoi(aoi_vector_path)
    selected_crs = choose_target_crs(aoi, target_crs, fallback_crs=fallback_crs)
    resolution_label = _resolution_label(target_resolution)

    mosaic_native = output_dir / f"{output_prefix}_mosaic_native_unclipped.tif"
    clipped_native = output_dir / f"{output_prefix}_native_clipped.tif"
    unfilled_dem = output_dir / f"{output_prefix}_{resolution_label}.tif"
    filled_dem = output_dir / f"{output_prefix}_{resolution_label}_filled.tif"

    input_tiles = 0
    overlapping_tiles = 0
    source_mode = "download" if download else "local_tiles"

    if download:
        bounds = aoi_bounds_epsg4326(aoi, pad_degrees=download_pad_degrees)
        download_fabdem_bounds(
            bounds,
            mosaic_native,
            cache=download_cache,
            overwrite=overwrite,
        )
        input_tiles = 1
        overlapping_tiles = 1
    else:
        assert fabdem_dir is not None
        rasters = find_rasters(fabdem_dir, recursive=recursive)
        if not rasters:
            raise FileNotFoundError(f"No GeoTIFF rasters found in: {fabdem_dir}")

        logger.info("Found %d raster tiles. Checking AOI overlap...", len(rasters))
        overlapping = filter_rasters_by_aoi(rasters, aoi, selected_crs)
        if not overlapping:
            raise RuntimeError("No raster tiles overlap the AOI.")
        logger.info("%d tiles overlap the AOI.", len(overlapping))

        input_tiles = len(rasters)
        overlapping_tiles = len(overlapping)
        mosaic_rasters(overlapping, mosaic_native, overwrite=overwrite)

    clip_raster_to_aoi(
        mosaic_native,
        aoi,
        clipped_native,
        all_touched=all_touched,
        overwrite=overwrite,
    )
    reproject_resample_raster(
        clipped_native,
        unfilled_dem,
        selected_crs,
        target_resolution,
        resampling=resampling,
        dst_dtype="float32",
        overwrite=overwrite,
    )

    final_dem = unfilled_dem
    final_filled_dem: Optional[Path] = None
    if fill_gaps:
        gap_result = fill_dem_nodata(
            unfilled_dem,
            filled_dem,
            max_search_distance=fill_max_search_distance,
            smoothing_iterations=fill_smoothing_iterations,
            aoi_vector_path=aoi_vector_path,
            fill_only_inside_aoi=True,
            all_touched=all_touched,
            overwrite=overwrite,
        )
        logger.info("Gap-fill summary: %s", gap_result)
        final_dem = filled_dem
        final_filled_dem = filled_dem

    if not keep_intermediate:
        remove_file(mosaic_native)
        remove_file(clipped_native)

    return DemBuildResult(
        final_dem=final_dem,
        unfilled_dem=unfilled_dem,
        mosaic_native=mosaic_native,
        clipped_native=clipped_native,
        filled_dem=final_filled_dem,
        target_crs=selected_crs,
        target_resolution=target_resolution,
        input_tiles=input_tiles,
        overlapping_tiles=overlapping_tiles,
        source_mode=source_mode,
    )


def build_dem_from_fabdem_tiles(
    fabdem_dir: PathLike,
    aoi_vector_path: PathLike,
    output_dir: PathLike,
    **kwargs,
) -> DemBuildResult:
    """Backward-compatible helper for building from local unzipped FABDEM tiles."""

    return build_dem_from_fabdem(aoi_vector_path, output_dir, fabdem_dir=fabdem_dir, download=False, **kwargs)


def write_build_manifest(result: DemBuildResult, output_dir: PathLike) -> Path:
    path = Path(output_dir) / "fabdem_build_manifest.json"
    data = asdict(result)
    data["target_crs"] = str(result.target_crs)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def parse_resolution(value: str) -> Union[float, tuple[float, float]]:
    parts = [part.strip() for part in value.replace(",", "x").split("x") if part.strip()]
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise argparse.ArgumentTypeError("Resolution must be a number or X,Y pair, such as 30 or 30,30.")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download or mosaic FABDEM and create a clipped, projected DEM from an AOI vector.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--download", action="store_true", help="Download FABDEM for the AOI bounds using the optional fabdem package.")
    source.add_argument("--fabdem-dir", type=Path, help="Directory containing local unzipped FABDEM GeoTIFF tiles.")
    parser.add_argument("--aoi", required=True, type=Path, help="Catchment/AOI vector file readable by GeoPandas.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for DEM outputs.")
    parser.add_argument("--output-prefix", default="DEM_fabdem", help="Prefix for generated DEM filenames.")
    parser.add_argument("--target-resolution", type=parse_resolution, default=30.0, help="Output resolution in target CRS units, e.g. 30 or 30,30.")
    parser.add_argument("--target-crs", default=None, help="Output CRS, e.g. EPSG:32643. Defaults to projected AOI CRS or EPSG:6933.")
    parser.add_argument("--fallback-crs", default="EPSG:6933", help="CRS used when AOI CRS is geographic and --target-crs is omitted.")
    parser.add_argument("--recursive", action="store_true", help="Search local FABDEM directory recursively.")
    parser.add_argument("--all-touched", action="store_true", help="Include pixels touched by AOI geometry during clipping.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--fill-gaps", action="store_true", help="Fill small internal NoData gaps after reprojection.")
    parser.add_argument("--fill-max-search-distance", type=float, default=20.0, help="Gap-fill search distance in pixels.")
    parser.add_argument("--fill-smoothing-iterations", type=int, default=0, help="Gap-fill smoothing iterations.")
    parser.add_argument("--download-cache", type=Path, default=None, help="Optional cache directory for downloaded FABDEM archives.")
    parser.add_argument("--download-pad-degrees", type=float, default=0.02, help="Padding added to AOI bounds before FABDEM download.")
    parser.add_argument("--drop-intermediate", action="store_true", help="Remove native mosaic and native clipped rasters after success.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s:%(name)s:%(message)s")
    result = build_dem_from_fabdem(
        aoi_vector_path=args.aoi,
        output_dir=args.output_dir,
        fabdem_dir=args.fabdem_dir,
        download=args.download,
        download_cache=args.download_cache,
        download_pad_degrees=args.download_pad_degrees,
        output_prefix=args.output_prefix,
        target_resolution=args.target_resolution,
        target_crs=args.target_crs,
        fallback_crs=args.fallback_crs,
        recursive=args.recursive,
        all_touched=args.all_touched,
        overwrite=args.overwrite,
        fill_gaps=args.fill_gaps,
        fill_max_search_distance=args.fill_max_search_distance,
        fill_smoothing_iterations=args.fill_smoothing_iterations,
        keep_intermediate=not args.drop_intermediate,
    )
    manifest = write_build_manifest(result, args.output_dir)
    print(f"[SAVED] final DEM: {result.final_dem}")
    print(f"[SAVED] manifest : {manifest}")


if __name__ == "__main__":
    main()
