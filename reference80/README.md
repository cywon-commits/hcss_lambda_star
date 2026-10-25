# reference80 — generalised Python reference (cfg: lam12 | z18)

`cfg.py` selects the tile system (`HCSS_CFG=lam12|z18` or `cfg.select(name)`): lam12 = lambda* tiling (Z[zeta12], regression), z18 = triangle + 80-degree rhombus (Z[zeta18]).
`front.py` (region counting, class vectors, lifted-area solve), `cyl.py`, `cyl_ref.py` (transfer-matrix builder), `tilt.py` (entropy vs perpendicular tilt fields) are the generalisations of `reference/python/`.
Verification: `python3 -m pytest -q test_phase0.py` (7 tests) and `bash run_regression.sh` (five lambda* cylinders vs `reference/data`, bit-identical).
Independent checks: `indep_count.py` (shapely, different tie-break), `patches.py` (random edge-glued patches), `sigma_phase.py` + `follow_tiling.py` (10-degree-tilted sigma phase in exact coordinates followed through the transfer matrix), `enum_tilings.py`.
Usage: `HCSS_CFG=z18 python3 cyl_ref.py a0 a1 a2 a3 a4 a5 out.npz`.
