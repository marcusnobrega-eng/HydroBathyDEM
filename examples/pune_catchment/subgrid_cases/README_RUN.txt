Reproduce the two sub-grid A/B test subcatchments (deterministic, ~15 s total):

  cd /Users/mngomes/Documents/GitHub/HydroBathyDEM/examples/pune_catchment/subgrid_cases
  python3 candidates.py     # D8 topology + enumerate maximal 150-300 km2 sub-basins
  python3 finalise.py       # characterise, verify, write clipped products

candidates.py caches d8_topology.npz / masks.npz / cand_labels.npy next to itself.
Outputs land in ../data/pune_caseA/, ../data/pune_caseB/,
../data/pune_case{A,B}_domain.geojson, ../data/pune_subcatchment_cases.json.
No file in src/ or tests/ is read-modified; hybrid_mesh.receiver_from_d8_direction
is imported unchanged.
