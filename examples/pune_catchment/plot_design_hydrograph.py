"""Compact QA figure for the ready-CN SCS-HUT design-flow preprocessor."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LogNorm
from scipy.ndimage import maximum_filter


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "outputs" / "pune_design_hydrograph_gcn250"


def read(name: str) -> tuple[np.ndarray, object]:
    with rasterio.open(RESULTS / name) as source:
        return source.read(1, masked=True).astype(float).filled(np.nan), source.bounds


def draw_scale_bar(axis, bounds, length_m: float = 5_000.0) -> None:
    x0 = bounds.left + 0.06 * (bounds.right - bounds.left)
    y0 = bounds.bottom + 0.06 * (bounds.top - bounds.bottom)
    axis.plot([x0, x0 + length_m], [y0, y0], color="#262626", linewidth=2.0)
    axis.text(x0 + length_m / 2, y0 + 0.018 * (bounds.top - bounds.bottom), "5 km", ha="center", va="bottom", fontsize=8)


def main() -> None:
    cn, bounds = read("D4_composite_CN_AMCII.tif")
    coverage, _ = read("D4_CN_coverage_fraction.tif")
    qp, _ = read("D4_SCS_HUT_Qp_100yr_m3s.tif")
    # A 30 m raster river disappears at full-domain scale.  This is a display
    # dilation only; the design-flow GeoTIFF retains its native one-cell width.
    qp_display = maximum_filter(np.nan_to_num(qp, nan=0.0), size=5)
    qp_display[qp_display <= 0.0] = np.nan
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), layout="constrained")
    panels = [
        (cn, "YlGnBu", None, 72, 92, "Composite CN (AMC-II)"),
        (coverage, "YlOrRd", None, 0.80, 1.00, "CN coverage of contributing area"),
        (qp_display, "magma_r", LogNorm(vmin=0.01, vmax=max(1.0, float(np.nanquantile(qp, 0.99)))), None, None, "$Q_p$, 100-year SCS-HUT (m³ s⁻¹)\n(display widened only)"),
    ]
    for label, axis, (data, cmap, norm, vmin, vmax, colorbar_label) in zip("abc", axes, panels, strict=True):
        image = axis.imshow(data, extent=extent, origin="upper", cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, interpolation="none")
        axis.text(0.02, 0.98, f"({label})", transform=axis.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")
        axis.set_aspect("equal"); axis.set_axis_off(); draw_scale_bar(axis, bounds)
        bar = fig.colorbar(image, ax=axis, shrink=0.78, pad=0.02)
        bar.set_label(colorbar_label, fontsize=9)
        bar.outline.set_linewidth(1.2)
    fig.savefig(RESULTS / "design_hydrograph_qa.png", dpi=300, bbox_inches="tight")
    fig.savefig(RESULTS / "design_hydrograph_qa.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
