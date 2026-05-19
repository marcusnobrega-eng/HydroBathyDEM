# Spatial Hydraulic Geometry Calibration

This folder contains FABDEM-D4 subcatchment coefficient rasters calibrated from Lin et al. 2020 data.

Selected threshold: `5000 km2`
Thresholds tested: `500.0, 1000.0, 2500.0, 5000.0 km2`

The coefficient rasters can be used by the DEM-conditioning workflow with:

```bash
--river-geometry-source spatial_coefficients_or_power_law \
--spatial-beta-1-raster /Users/mngomes/Documents/GitHub/DEM_Processing/Data/Lin2020_bankfull_width/calibration/D4_beta_1_width_5000km2.tif \
--spatial-beta-2-raster /Users/mngomes/Documents/GitHub/DEM_Processing/Data/Lin2020_bankfull_width/calibration/D4_beta_2_width_5000km2.tif \
--spatial-alfa-1-raster /Users/mngomes/Documents/GitHub/DEM_Processing/Data/Lin2020_bankfull_width/calibration/D4_alfa_1_depth_5000km2.tif \
--spatial-alfa-2-raster /Users/mngomes/Documents/GitHub/DEM_Processing/Data/Lin2020_bankfull_width/calibration/D4_alfa_2_depth_5000km2.tif
```

Diagnostics:

- `diagnostic_spatial_calibration_thresholds.png`: validation metrics across tested subcatchment thresholds.
- `diagnostic_spatial_calibration_5000km2.png`: Lin-area width/depth fits, sample counts, and local/global fallback map.
- `diagnostic_spatial_calibration_d4_area_5000km2.png`: companion plots using matched FABDEM-D4 drainage area.
