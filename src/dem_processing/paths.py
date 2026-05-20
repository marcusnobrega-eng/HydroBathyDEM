"""Shared project and output path helpers."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Outputs"


REPORT_FILES = {
    "conditioning_config.json",
    "DEM_conditioning_README.md",
    "modification_summary.csv",
    "D4_HydroPol2D_creek_reduction_summary.csv",
    "D4_river_connectivity_summary.csv",
    "run_manifest.json",
    "qa_scorecard.csv",
    "qa_scorecard.json",
    "qa_scorecard.md",
}


def output_theme(filename: str) -> str:
    """Return the themed output subfolder for a pipeline artifact."""
    name = Path(filename).name
    if name in REPORT_FILES:
        return "reports"
    if name.startswith("D4_"):
        return "d4"
    if name.startswith("DEM_"):
        return "dem"
    if (
        name.startswith("diagnostic_")
        or name.startswith("quicklook_")
        or name.startswith("slope_")
        or name.startswith("mask_")
    ):
        return "diagnostics"
    return "misc"


def themed_output_path(out_dir: Path | str, filename: str) -> Path:
    """Build an output path under a themed subfolder."""
    root = Path(out_dir).expanduser().resolve()
    return root / output_theme(filename) / filename


def output_path(out_dir: Path | str, filename: str) -> Path:
    """Alias for themed_output_path used by reporting utilities."""
    return themed_output_path(out_dir, filename)


def ensure_output_layout(out_dir: Path | str) -> None:
    """Create the standard themed output folders."""
    root = Path(out_dir).expanduser().resolve()
    for theme in ["dem", "d4", "diagnostics", "reports", "misc"]:
        (root / theme).mkdir(parents=True, exist_ok=True)
