# tm80 — C++ cylinder transfer-matrix generator for lam12 and z18

Generalisation of `tm_cpp/tm_gen.cpp`; the tile system is chosen at run time (`lam12`: lambda* tiling, `z18`: triangle + 80-degree rhombus).
```
make -C tm80
tm80/tm80 regions z18                     # 18-gon: 6 fillings, class vector (6,6,6,2^9); lam12: the 8 region counts
tm80/tm80 d0 z18 a0..a5                   # start front (same permutation rule as reference80)
tm80/tm80 build z18 a0..a5 runs/raw_X [d0]
python3 tm80/bin2npz80.py runs/raw_X runs/X.npz       # npz in the layout of reference80/cyl_ref.py
bash tm80/test_ref.sh                      # Phase-A validation (lam12: 5 reference cylinders; z18: 5 cylinders vs Python)
```
Output per transition: fr, to, c (uint64), w, na, ar, off (rank ints), cls (12 int16, z18), q5.c16 [q7.c16] (complex128, lateral part removed); meta.json carries cfg, P, d0, Ps_m, class-vector violations and the
area/vertex-weight consistency check of the hole tile counts.
Canonical start vertices: exact tie tests (x == conj x in Z[zeta_n]) and doubles for the order; crossing tests with the reference tolerances (compile with -ffp-contract=off).
