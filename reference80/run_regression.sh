#!/bin/bash
# lambda* regression: reference80 (cfg lam12) vs reference/data/ref_*.npz via tools/compare_tm.py  (about 6 minutes)
cd "$(dirname "$0")"
for a in "2 2 0 -1" "2 3 0 -1" "2 3 1 -1" "3 3 0 -1" "3 4 0 -2"; do
  n=$(echo $a | tr ' ' '_'); HCSS_CFG=lam12 python3 cyl_ref.py $a /tmp/rg_$n.npz | tail -1
  python3 ../tools/compare_tm.py ../reference/data/ref_$n.npz /tmp/rg_$n.npz | tail -1
done
