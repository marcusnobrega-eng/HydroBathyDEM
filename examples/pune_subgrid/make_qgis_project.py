"""Build a QGIS project for visual inspection of the Pune corridor mesh.

Numerical QA is not enough on its own -- the 40-piece river fragmentation on the
previous uniform lattice passed every mesh check (one connected mesh, A/P floor
met, overlap exact) and was only obvious once someone looked at a map. So the mesh
goes over satellite imagery, with river cells picked out and the mapped
centrelines on top, so corridor alignment can be judged by eye.
"""
from __future__ import annotations

import sys
from pathlib import Path

from qgis.core import (
    QgsApplication, QgsCoordinateReferenceSystem, QgsFillSymbol, QgsLineSymbol,
    QgsProject, QgsRasterLayer, QgsRendererCategory, QgsCategorizedSymbolRenderer,
    QgsSingleSymbolRenderer, QgsVectorLayer,
)
from PyQt5.QtGui import QColor

HERE = Path(__file__).resolve().parent
QgsApplication.setPrefixPath("/Applications/QGIS.app/Contents/MacOS", True)
app = QgsApplication([], False)
app.initQgis()

project = QgsProject.instance()
project.setCrs(QgsCoordinateReferenceSystem("EPSG:32643"))

# satellite basemap first so it sits at the bottom
esri = ("type=xyz&url=https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/%7Bz%7D/%7By%7D/%7Bx%7D&zmax=19&zmin=0")
basemap = QgsRasterLayer(esri, "Esri World Imagery", "wms")
if basemap.isValid():
    project.addMapLayer(basemap)

mesh = QgsVectorLayer(f"{HERE}/mesh_bl90m_r1.gpkg|layername=mesh", "mesh 90 m (corridor)", "ogr")
cats = []
for value, colour, label in [
    ("river", "#2b7bba", "river cell (sub-grid tables)"),
    ("rural", "#f0e6d2", "rural cell (flat prism)"),
]:
    symbol = QgsFillSymbol.createSimple({
        "color": QColor(colour).name(),
        "outline_color": "#444444",
        "outline_width": "0.08",
        "style": "solid",
    })
    symbol.setOpacity(0.55)
    cats.append(QgsRendererCategory(value, symbol, label))
mesh.setRenderer(QgsCategorizedSymbolRenderer("feature_class", cats))
project.addMapLayer(mesh)

lines = QgsVectorLayer(f"{HERE}/river_centerlines.gpkg|layername=reaches",
                       "mapped river centrelines", "ogr")
lines.setRenderer(QgsSingleSymbolRenderer(QgsLineSymbol.createSimple(
    {"color": "#e8452c", "width": "0.7"})))
project.addMapLayer(lines)

out = HERE / "pune_corridor_mesh.qgz"
project.write(str(out))
print(f"wrote {out}")
print(f"  layers: {[l.name() for l in project.mapLayers().values()]}")
print(f"  mesh features: {mesh.featureCount()}   centrelines: {lines.featureCount()}")
app.exitQgis()
