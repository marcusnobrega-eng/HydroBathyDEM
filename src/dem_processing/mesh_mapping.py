"""Conservative data transfer using a HydroBathyDEM overlap product."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import netCDF4
import numpy as np
from scipy import sparse

from .mesh_contract import MESH_CONTRACT_VERSION, OVERLAP_RASTER_ORDER


@dataclass(frozen=True)
class RasterMeshOverlap:
    mesh_index: np.ndarray
    raster_index: np.ndarray
    area_m2: np.ndarray
    mesh_area_m2: np.ndarray
    raster_area_m2: np.ndarray
    raster_shape: tuple[int, int]
    x_edges: np.ndarray
    y_edges: np.ndarray

    def raster_to_mesh_matrix(self) -> sparse.csr_matrix:
        weights = self.area_m2 / self.mesh_area_m2[self.mesh_index]
        return sparse.csr_matrix(
            (weights, (self.mesh_index, self.raster_index)),
            shape=(self.mesh_area_m2.size, self.raster_area_m2.size),
        )

    def mesh_to_raster_matrix(self) -> sparse.csr_matrix:
        weights = self.area_m2 / self.raster_area_m2[self.raster_index]
        return sparse.csr_matrix(
            (weights, (self.raster_index, self.mesh_index)),
            shape=(self.raster_area_m2.size, self.mesh_area_m2.size),
        )


def read_overlap(path: str | Path) -> RasterMeshOverlap:
    """Read and validate a conservative-overlap file from a mesh product."""
    with netCDF4.Dataset(path) as dataset:
        if str(getattr(dataset, "mesh_contract_version", "")) != MESH_CONTRACT_VERSION:
            raise ValueError("Unsupported or missing mesh_contract_version in overlap product.")
        if str(getattr(dataset, "raster_index_order", "")) != OVERLAP_RASTER_ORDER:
            raise ValueError(f"Overlap raster order must be {OVERLAP_RASTER_ORDER!r}.")
        overlap = RasterMeshOverlap(
            mesh_index=np.asarray(dataset["overlap_mesh_index"][:], dtype=np.int64),
            raster_index=np.asarray(dataset["overlap_raster_index"][:], dtype=np.int64),
            area_m2=np.asarray(dataset["overlap_area_m2"][:], dtype=np.float64),
            mesh_area_m2=np.asarray(dataset["mesh_area_m2"][:], dtype=np.float64),
            raster_area_m2=np.asarray(dataset["raster_area_m2"][:], dtype=np.float64),
            raster_shape=(int(dataset.raster_rows), int(dataset.raster_cols)),
            x_edges=np.asarray(dataset["x_edges"][:], dtype=np.float64),
            y_edges=np.asarray(dataset["y_edges"][:], dtype=np.float64),
        )
    _validate_overlap(overlap)
    return overlap


def aggregate_continuous(overlap: RasterMeshOverlap, raster_values: np.ndarray) -> np.ndarray:
    """Area-average a south-up raster onto mesh cells."""
    values = np.asarray(raster_values, dtype=np.float64)
    if values.shape != overlap.raster_shape:
        raise ValueError("Raster shape does not match the overlap product.")
    return np.asarray(overlap.raster_to_mesh_matrix() @ values.reshape(-1)).reshape(-1)


def aggregate_class_fractions(
    overlap: RasterMeshOverlap, raster_labels: np.ndarray, classes: np.ndarray
) -> np.ndarray:
    """Return one area fraction per mesh cell and requested raster class."""
    labels = np.asarray(raster_labels)
    if labels.shape != overlap.raster_shape:
        raise ValueError("Raster shape does not match the overlap product.")
    matrix = overlap.raster_to_mesh_matrix()
    flat = labels.reshape(-1)
    return np.column_stack(
        [np.asarray(matrix @ (flat == value).astype(np.float64)).reshape(-1) for value in classes]
    )


def remap_mesh_depth_to_raster(overlap: RasterMeshOverlap, mesh_depth_m: np.ndarray) -> np.ndarray:
    """Conservatively remap mesh depth to the overlap's south-up raster grid."""
    depth = np.asarray(mesh_depth_m, dtype=np.float64).reshape(-1)
    if depth.size != overlap.mesh_area_m2.size:
        raise ValueError("Mesh depth does not match the overlap product.")
    return np.asarray(overlap.mesh_to_raster_matrix() @ depth).reshape(overlap.raster_shape)


def _validate_overlap(overlap: RasterMeshOverlap) -> None:
    count = overlap.area_m2.size
    if overlap.mesh_index.size != count or overlap.raster_index.size != count:
        raise ValueError("Overlap index and area arrays must have equal lengths.")
    if np.any(overlap.area_m2 <= 0.0) or not np.all(np.isfinite(overlap.area_m2)):
        raise ValueError("Overlap areas must be finite and positive.")
    if np.any(overlap.mesh_index < 0) or np.any(overlap.mesh_index >= overlap.mesh_area_m2.size):
        raise ValueError("Overlap contains an invalid zero-based mesh index.")
    if np.any(overlap.raster_index < 0) or np.any(overlap.raster_index >= overlap.raster_area_m2.size):
        raise ValueError("Overlap contains an invalid zero-based raster index.")
    covered = np.bincount(
        overlap.mesh_index, weights=overlap.area_m2, minlength=overlap.mesh_area_m2.size
    )
    error = np.abs(covered - overlap.mesh_area_m2) / np.maximum(overlap.mesh_area_m2, 1e-12)
    if float(error.max(initial=0.0)) > 1e-8:
        raise ValueError("Overlap is not conservative for every mesh cell.")
