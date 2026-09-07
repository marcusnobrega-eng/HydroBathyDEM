# HydroBathyDEM Mesh Exchange Contract 1.0

HydroBathyDEM is the production mesh generator. HydroPol2D-Python and
HydroPol2D-MATLAB are consumers of the files described here; they must not
silently rebuild connectivity, geometry, overlap, or subgrid properties.

## Package

A published hydraulic package contains:

- a UGRID NetCDF mesh;
- a conservative raster-to-cell overlap NetCDF file;
- optional cell-level and face-level subgrid tables;
- a GeoPackage for inspection;
- a quality report;
- `mesh_manifest.json` with source version, Git revision, configuration hash,
  file sizes, and SHA-256 checksums.

Every NetCDF product declares:

| Attribute | Required value |
|---|---|
| `mesh_contract_version` | `1.0` |
| `schema_version` | `hydrobathydem-mesh-1.0` |
| `file_index_base` | `0` |
| `coordinate_units` | `m` |
| `crs_wkt` | non-empty projected CRS WKT |

MATLAB converts file indices to one-based indices only after reading. Python
keeps zero-based indices. No other implicit index conversion is permitted.

## Hydraulic Mesh

The mesh product type is `hydraulic_mesh`. Required cell variables include
cell centers, node connectivity, `cell_area_m2`, `cell_bed_elevation_m`,
`cell_hydraulic_roughness`, and `cell_cfl_width_m`. The CFL width is the
minimum center-to-center separation normal to a face; it is not an equivalent
cell diameter.

Required face variables include owner and neighbor indices, length,
center-to-center distance, midpoint, and unit-normal components. Normals follow
`unit normal points outward from edge_owner`. A boundary face has neighbor
`-1`. Areas, lengths, coordinates, elevations, roughness, and CFL widths use
`m2`, `m`, `m`, `m`, `s m-1/3`, and `m`, respectively.

## Conservative Overlap

The overlap product type is `conservative_raster_overlap` and uses
`south_up_row_major` raster indexing. It provides mesh index, raster index,
intersection area, mesh area, raster area, and raster x/y edges. Intersection
areas for each mesh cell must sum to its declared area within the published
tolerance.

The raster used for hydrologic states remains distinct from the hydraulic mesh.
The fine DEM used to construct subgrid relations is not a third simulation
grid.

## Subgrid Tables

The subgrid product type is `hydraulic_subgrid_tables`. Cell tables provide
datum, stage, volume, wet area, point count, and plan area. Face tables provide
datum, stage, flow area, wetted perimeter, conveyance, point count, and face
length.

Conveyance is stored as

```text
K = sum((A/n) * (A/P)^(2/3))
```

It is not `g*n^2` and must not be squared again by a solver. Stages are
monotonic, volumes and wet areas are nonnegative, and face data cannot use a
datum below the connected cell data without an explicit, validated reason.

## Validation

Consumers reject zero-length faces, invalid ownership, inconsistent normals,
nonpositive areas or CFL widths, absent CRS/units, nonconservative overlap, and
subgrid tables whose dimensions or conventions disagree with the mesh.
Compatibility mode may read historical products for diagnosis, but production
and release validation require the strict 1.0 contract.
