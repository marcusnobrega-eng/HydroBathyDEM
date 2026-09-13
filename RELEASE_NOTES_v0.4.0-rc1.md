# HydroBathyDEM 0.4.0-rc1

This release candidate makes HydroBathyDEM the authoritative computational-mesh
builder used by HydroPol2D Studio and both solvers.

## Highlights

- public computation-point, edit, compile, and validation APIs;
- persistent mesh recipes, editor layers, and atomic current/previous revisions;
- regular and editable Voronoi products using the shared UGRID contract;
- river-bank breaklines and refinement-region support;
- faster conformity repair and topology reconstruction around edited regions;
- optional intermediate diagnostics for responsive interactive compilation;
- final global QA and contract validation remain mandatory.

## Validation

- all 158 tests pass locally;
- CI passes on Python 3.10, 3.11, 3.12, and 3.13;
- Ruff passes without exclusions;
- the Stanford reference mesh compiles in about 6.82 seconds with unchanged
  finite topology values and QA metrics relative to the saved baseline.

This is a release candidate. Stable promotion remains conditional on the
coordinated HydroPol2D Python, MATLAB, Studio, and GPU acceptance gates.
