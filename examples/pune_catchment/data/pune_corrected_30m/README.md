# Pune corrected 30 m inputs

This local, Git-ignored directory is built from the six user-supplied Pune
rasters with `../stage_corrected_inputs.py`.

`pune_dem_corrected_aligned_30m.tif` is the supplied corrected DEM resampled
once with bilinear interpolation to the exact 30 m grid shared by soil, ESA
LULC, LAI, depth to bedrock, and albedo.  `manifest.json` records the source
checksums and alignment operation.  Do not mix the original 29.6148728 m DEM
grid with the 30 m static layers directly.
