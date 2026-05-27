# Spatial Hydraulic Geometry Calibration

This folder contains FABDEM-D4 subcatchment coefficient rasters calibrated from Lin et al. 2020 data.

Selected threshold: `20 km2`
Thresholds tested: `5.0, 10.0, 20.0, 50.0 km2`
Fit area source: `d4`

The coefficient rasters can be used by the DEM-conditioning workflow with:

```bash
--river-geometry-source spatial_coefficients_or_power_law \
--spatial-beta-1-raster examples/pune_catchment/data/lin2020/calibration/D4_beta_1_width_20km2.tif \
--spatial-beta-2-raster examples/pune_catchment/data/lin2020/calibration/D4_beta_2_width_20km2.tif \
--spatial-alfa-1-raster examples/pune_catchment/data/lin2020/calibration/D4_alfa_1_depth_20km2.tif \
--spatial-alfa-2-raster examples/pune_catchment/data/lin2020/calibration/D4_alfa_2_depth_20km2.tif
```

Diagnostics:

- `diagnostic_spatial_calibration_thresholds.png`: validation metrics across tested subcatchment thresholds.
- `diagnostic_spatial_calibration_20km2.png`: Lin-area width/depth fits, sample counts, and local/global fallback map.
- `diagnostic_spatial_calibration_d4_area_20km2.png`: companion plots using matched FABDEM-D4 drainage area.
