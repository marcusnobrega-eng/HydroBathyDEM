# Lin et al. 2020 Bankfull Geometry

Source: https://zenodo.org/records/3552776

This folder stores the project-local copy and processed subset of the Lin et al. global reach-level bankfull-width dataset.
The Zenodo shapefile does not include a `.prj` file, but its bounds are global longitude/latitude degrees, so the prep script assigns EPSG:4326 before projecting to the DEM grid CRS.

## Raw Files

The raw `rivers_ge30m` shapefile parts were read from `Data/Lin2020_bankfull_width/raw`.
They are not copied into this processed-data folder, which keeps small examples lightweight.

- `Data/Lin2020_bankfull_width/raw/rivers_ge30m.cpg` (10 bytes)
- `Data/Lin2020_bankfull_width/raw/rivers_ge30m.shx` (5,790,676 bytes)
- `Data/Lin2020_bankfull_width/raw/rivers_ge30m.dbf` (361,187,884 bytes)
- `Data/Lin2020_bankfull_width/raw/rivers_ge30m.shp` (1,421,971,140 bytes)

## Processed Files

- `processed/lin2020_dem_domain_width_depth.gpkg`: DEM-domain reaches with Manning Q2 depth.
- `processed/lin2020_width_depth_summary.csv`: field statistics and processing counts.
- `processed/lin2020_processing_metadata.json`: source, grid, and parameter metadata.
- `processed/diagnostic_lin2020_width_depth.png`: width/depth/H_abg diagnostics and current D4 overlap.

Raster products on the model grid:

- `processed/Lin2020_width_m_30m.tif`
- `processed/Lin2020_depth_Q2_m_30m.tif`
- `processed/Lin2020_Q2_m3s_30m.tif`
- `processed/Lin2020_slope_used_30m.tif`
- `processed/Lin2020_H_abg_m_30m.tif`
- `processed/Lin2020_mask_30m.tif`

## Manning Depth Calculation

Depth is solved from the rectangular Manning equation:

```text
Q2 = (1/n) * A * R^(2/3) * S^(1/2)
A = width_m * depth
R = A / (width_m + 2 * depth)
```

Current settings: `n=0.035`, `min_slope=1e-05`, `max_slope=0.05`, `depth_cap_m=20.0`.
