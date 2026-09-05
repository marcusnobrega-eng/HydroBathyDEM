# HydroLAKES Pune Subset

Source: HydroLAKES polygons v1.0 from HydroSHEDS.

Downloaded global archive:

```text
https://data.hydrosheds.org/file/hydrolakes/HydroLAKES_polys_v10_shp.zip
```

Local files:

- `HydroLAKES_polys_v10_shp.zip`: original global HydroLAKES shapefile archive.
- `hydrolakes_pune_bbox.gpkg`: padded Pune bounding-box clip from the global archive.
- `hydrolakes_pune_domain.gpkg`: exact Pune model-domain subset in the model CRS, with local metric area and shoreline fields.

HydroLAKES provides shoreline polygons for global lakes/reservoirs with surface area of at least 10 ha. Use this subset as the preferred waterbody source for reservoir/lake shoreline breakline candidates.
