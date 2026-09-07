"""Versioned file contract shared by HydroPol2D mesh producers and consumers."""

from __future__ import annotations

MESH_CONTRACT_VERSION = "1.0"
MESH_PRODUCT_SCHEMA = "hydrobathydem-mesh-1.0"
FILE_INDEX_BASE = 0
FILE_BOUNDARY_NEIGHBOR = -1
COORDINATE_UNITS = "m"
EDGE_NORMAL_CONVENTION = "unit normal points outward from edge_owner"
OVERLAP_RASTER_ORDER = "south_up_row_major"
SUBGRID_CONVEYANCE_CONVENTION = "K=sum((A/n)*(A/P)^(2/3))"


def apply_contract_attributes(dataset, *, product: str) -> None:
    """Attach the non-negotiable exchange conventions to a NetCDF dataset."""
    dataset.mesh_contract_version = MESH_CONTRACT_VERSION
    dataset.schema_version = MESH_PRODUCT_SCHEMA
    dataset.product_type = product
    dataset.file_index_base = FILE_INDEX_BASE
    dataset.coordinate_units = COORDINATE_UNITS
    if product == "hydraulic_mesh":
        dataset.boundary_neighbor_sentinel = FILE_BOUNDARY_NEIGHBOR
