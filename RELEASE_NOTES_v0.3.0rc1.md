# HydroBathyDEM v0.3.0rc1

## Purpose

HydroBathyDEM is the production generator for HydroPol2D regular and Voronoi mesh packages.
This release candidate introduces mesh contract `1.0` for coordinated use by
HydroPol2D-Python `0.8.0rc1` and HydroPol2D-MATLAB `1.18.0-rc1`.

## Main changes

- Versioned UGRID topology with explicit zero-based indexing, orientation, CRS, and SI units.
- Cell areas, centers, face lengths, connectivity, and face-normal CFL widths.
- Conservative raster-to-cell overlap products.
- Structured and Voronoi subgrid cell-level and face-level tables.
- River corridors, infrastructure refinement, GeoPackage inspection layers, and QA reports.
- Generation manifests with source, configuration, version, commit, and checksum provenance.

## Verification status

The complete HydroBathyDEM suite passes with 129 tests. MATLAB and Python consumed the same Pune
mesh and conservative overlap bundle for baseline and Voronoi-subgrid runs at 2 h, 6 h, and 12 h.
The largest cumulative outlet-volume difference was `0.03779%` of total model input, below the
integrated-domain threshold of `0.1%`; both solvers independently closed mass to numerical
precision. Stable publication remains coordinated with clean-clone verification, Studio
regression, and final release-candidate packaging.
