#!/bin/bash
# Phase B: tilt scan (ranges 0.3 and 0.2, 5x5), K fit and chi for the given cylinders.
# Usage: bash tm_cpp/run_phaseB.sh 3_4_1_-1 4_4_0_-2 ...   (needs runs/cpp_<name>.npz from test_ref.sh / tm_gen)
cd "$(dirname "$0")/../reference/python"
for n in "$@"; do
  f=../../runs/cpp_$n.npz
  for r in 0.3 0.2; do
    python3 tiltscan.py $f $r 5 > ../../runs/scan_${n}_$r.log
    mv ../../runs/cpp_${n}_scan.npy ../../runs/scan_${n}_$r.npy; rm -f ../../runs/cpp_${n}_fit.npy
    python3 kfit.py ../../runs/scan_${n}_$r.npy >> ../../runs/scan_${n}_$r.log
  done
  python3 chi.py $f > ../../runs/chi_$n.log
  echo "done $n"
done
