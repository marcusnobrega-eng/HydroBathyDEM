# HydroBathyDEM 0.4.0-rc2

Faster hydraulic (sub-grid) tables for HydroPol2D's urban-only and quadtree cases.

## Highlights

- Detailed cell and face tables are built only for sub-grid cells (river policy), plus
  the boundary faces of those cells, which the solver reads for its stage,
  normal-depth and critical outlets.
- The face-curve builder tests each bisection interval once, computes each level's
  conveyance once, and builds faces with identical panels once. The curves are
  bit-identical to rc1.
- Recife urban-only at 90 m (166,621 faces): 67 minutes to 8 seconds. Every table value
  the solver reads, and every result of the 6-hour run, is identical to rc1.

## Fixed (in the unreleased river-cell change)

- Boundary faces of sub-grid cells got an empty table row, which closed a river's
  inflow and outflow edges. They keep their flat-profile curves, as in rc1.

## Validation

- all 163 tests pass locally (including an exact comparison of the new face builder
  with the rc1 one on 1,200 faces of five kinds);
- Ruff passes on the changed files.
