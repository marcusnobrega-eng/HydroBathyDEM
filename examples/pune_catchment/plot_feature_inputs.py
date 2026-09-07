"""Plot the exact DEM, mapped rivers, widths, and HAND floodplain used by a mesh case."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from rasterio.features import geometry_mask

from dem_processing.config import load_config_file
from dem_processing.hybrid_mesh import (
    HybridMeshConfig,
    _mask_geometry,
    _physical_river_geometry,
    _river_segment_records,
    _vector_river_records,
    connected_hand_with_reach,
    receiver_from_d4_direction,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = HybridMeshConfig.from_mapping(load_config_file(args.config))
    out_dir = config.out_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(config.dem) as source:
        dem = source.read(1, masked=True).filled(np.nan).astype(float)
        transform, crs = source.transform, source.crs
        west, south, east, north = rasterio.transform.array_bounds(*dem.shape, transform)
        extent = (west, east, south, north)
    domain = gpd.read_file(config.domain_vector).to_crs(crs).geometry.union_all()
    xmin, ymin, xmax, ymax = domain.bounds
    active = np.isfinite(dem) & geometry_mask([domain], out_shape=dem.shape, transform=transform, invert=True)
    with rasterio.open(config.river_mask) as source:
        river = active & (source.read(1, masked=True).filled(0) > 0)
    with rasterio.open(config.river_direction) as source:
        receiver = receiver_from_d4_direction(source.read(1, masked=True).filled(0))
    with rasterio.open(config.river_width) as source:
        width = source.read(1, masked=True).filled(np.nan).astype(float)

    hand, reach = connected_hand_with_reach(dem, receiver, river)
    if config.river_source == "hydrobathydem_d4":
        records = _river_segment_records(river, receiver, transform, width)
    else:
        records = _vector_river_records(
            config.river_network, config.river_width_field, crs, domain, dem, transform
        )
    river_polygon = _physical_river_geometry(records)
    if config.floodplain_vector is not None:
        floodplain = gpd.read_file(config.floodplain_vector).to_crs(crs).geometry.union_all().intersection(domain).difference(river_polygon)
        floodplain_label = "Design-flow compound-Manning corridor"
        floodplain_title = "Mapped rivers and design-flow mesh corridor"
        output_name = "mesh_input_river_width_design_corridor.png"
    elif config.floodplain_mask is not None:
        with rasterio.open(config.floodplain_mask) as source:
            floodplain_mask = active & (source.read(1, masked=True).filled(0) > 0) & ~river
        floodplain_label = "Design-flow compound-Manning corridor"
        floodplain_title = "Mapped rivers and design-flow mesh corridor"
        output_name = "mesh_input_river_width_design_corridor.png"
    else:
        floodplain_mask = active & (reach >= 0) & np.isfinite(hand) & (hand <= config.floodplain_hand_stage_m) & ~river
        floodplain_label = f"D4-connected HAND ≤ {config.floodplain_hand_stage_m:g} m"
        floodplain_title = "Mapped rivers and potential HAND floodplain"
        output_name = f"mesh_input_river_width_hand{config.floodplain_hand_stage_m:g}m.png"
    if config.floodplain_vector is None:
        floodplain = _mask_geometry(floodplain_mask, transform).difference(river_polygon)

    feature_file = out_dir / "mesh_input_features.gpkg"
    if feature_file.exists():
        feature_file.unlink()
    gpd.GeoDataFrame(
        {
            "segment_id": range(len(records)),
            "source_cell": [item[0] for item in records],
            "width_m": [item[2] for item in records],
            "river_source": config.river_source,
        },
        geometry=[item[1] for item in records],
        crs=crs,
    ).to_file(feature_file, layer="river_centerlines", driver="GPKG")
    gpd.GeoDataFrame(
        {"feature": ["mapped_bankfull_river"]}, geometry=[river_polygon], crs=crs
    ).to_file(feature_file, layer="river_footprint", driver="GPKG", mode="a")
    gpd.GeoDataFrame(
        {"feature": ["design_flow_floodplain"]}, geometry=[floodplain], crs=crs
    ).to_file(feature_file, layer="floodplain_corridor", driver="GPKG", mode="a")

    masked_dem = np.where(active, dem, np.nan)
    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    image = ax.imshow(masked_dem, extent=extent, origin="upper", cmap="terrain")
    gpd.GeoSeries([domain.boundary], crs=crs).plot(ax=ax, color="#262626", linewidth=0.7)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.038, pad=0.015)
    colorbar.set_label("Elevation (m)", fontname="Helvetica")
    ax.set_title("Pune mesh window: conditioned DEM", fontname="Helvetica", fontsize=14)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal"); ax.set_axis_off()
    fig.savefig(out_dir / "mesh_input_dem.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    ax.imshow(masked_dem, extent=extent, origin="upper", cmap="Greys", alpha=0.28)
    if not floodplain.is_empty:
        gpd.GeoSeries([floodplain], crs=crs).plot(ax=ax, color="#8CC5E3", edgecolor="#3594CC", linewidth=0.6, alpha=0.68)
    if not river_polygon.is_empty:
        gpd.GeoSeries([river_polygon], crs=crs).plot(ax=ax, color="#0B81A2", edgecolor="#082A54", linewidth=0.7, alpha=0.70)
    widths = np.asarray([item[2] for item in records])
    lines = [np.asarray(item[1].coords) for item in records]
    collection = LineCollection(lines, cmap="viridis", norm=Normalize(widths.min(), widths.max()), linewidths=1.5, zorder=4)
    collection.set_array(widths)
    ax.add_collection(collection)
    colorbar = fig.colorbar(collection, ax=ax, fraction=0.038, pad=0.015)
    colorbar.set_label("Mapped bankfull width (m)", fontname="Helvetica")
    gpd.GeoSeries([domain.boundary], crs=crs).plot(ax=ax, color="#262626", linewidth=0.8, zorder=5)
    ax.legend(
        handles=[
            Patch(facecolor="#0B81A2", edgecolor="#082A54", label="Mapped bankfull river footprint"),
            Patch(facecolor="#8CC5E3", edgecolor="#3594CC", label=floodplain_label),
        ],
        loc="lower left", frameon=True, fontsize=9,
    )
    ax.set_title(floodplain_title, fontname="Helvetica", fontsize=14)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal"); ax.set_axis_off()
    fig.savefig(out_dir / output_name, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if config.floodplain_direction is not None:
        axis_path = config.floodplain_direction.with_name("D4_design_floodplain_axis_mask.tif")
        if axis_path.exists():
            with rasterio.open(axis_path) as source:
                axis_mask = active & (source.read(1, masked=True).filled(0) > 0)
            axes = _mask_geometry(axis_mask, transform)
            fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
            ax.imshow(masked_dem, extent=extent, origin="upper", cmap="Greys", alpha=0.28)
            if not floodplain.is_empty:
                gpd.GeoSeries([floodplain], crs=crs).plot(ax=ax, color="#8CC5E3", edgecolor="none", alpha=0.45)
            if not axes.is_empty:
                gpd.GeoSeries([axes], crs=crs).plot(ax=ax, color="#0B81A2", edgecolor="none", alpha=0.85)
            if not river_polygon.is_empty:
                gpd.GeoSeries([river_polygon], crs=crs).plot(ax=ax, color="#082A54", edgecolor="none", alpha=0.85)
            gpd.GeoSeries([domain.boundary], crs=crs).plot(ax=ax, color="#262626", linewidth=0.8)
            ax.set_title("Floodplain drainage axes conditioned to mapped rivers", fontname="Helvetica", fontsize=14)
            ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
            ax.set_aspect("equal"); ax.set_axis_off()
            fig.savefig(out_dir / "mesh_input_floodplain_drainage_axes.png", dpi=300, bbox_inches="tight")
            plt.close(fig)


if __name__ == "__main__":
    main()
