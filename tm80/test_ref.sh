#!/bin/bash
# Phase A validation: tm80 vs Python references.  lam12: 5 reference cylinders (reference/data); z18: cylinders built by reference80 (Python).
# Usage: bash tm80/test_ref.sh [quick]      (run from the repository root; needs numpy)
set -e
cd "$(dirname "$0")/.."
make -C tm80 -s
mkdir -p runs/t80; fail=0
./tm80/tm80 regions lam12 | tail -2 | head -1; ./tm80/tm80 regions z18 | grep -E "ALL|FAIL"
for a in "2 2 0 -1" "2 3 0 -1" "2 3 1 -1" "3 3 0 -1" "3 4 0 -2"; do
  [ "$1" = quick ] && [ "$a" != "2 2 0 -1" ] && [ "$a" != "2 3 0 -1" ] && continue
  n=$(echo $a | tr ' ' '_')
  d0=$(python3 -c "import json,sys;a=sys.argv[1].split();print(','.join(map(str,json.load(open('reference/expected.json'))['cylinders']['(%s)'%', '.join(a)]['d0'])))" "$a")
  [ "$(./tm80/tm80 d0 lam12 $a)" = "$d0" ] || { echo "d0 differs for lam12 $a"; fail=1; }
  ./tm80/tm80 build lam12 $a runs/t80/raw_l12_$n $d0 > /dev/null 2>&1
  python3 tm80/bin2npz80.py runs/t80/raw_l12_$n runs/t80/l12_$n.npz > /dev/null
  python3 tm80/compare80.py reference/data/ref_$n.npz runs/t80/l12_$n.npz | tail -1 | grep -q PASS && echo "lam12 ($a) PASS" || { echo "lam12 ($a) FAIL"; fail=1; }
done
for a in "1 1 2 0 0 -1" "2 1 2 0 0 -1" "1 2 2 0 0 -1" "1 1 1 1 1 0" "2 2 1 0 0 0"; do
  [ "$1" = quick ] && [ "$a" != "1 1 2 0 0 -1" ] && continue
  n=$(echo $a | tr ' ' '_')
  (cd reference80 && HCSS_CFG=z18 python3 cyl_ref.py $a ../runs/t80/ref_z18_$n.npz | tail -1)
  d0=$(python3 -c "import numpy as np;print(','.join(map(str,np.load('runs/t80/ref_z18_$n.npz')['d0'])))")
  [ "$(./tm80/tm80 d0 z18 $a)" = "$d0" ] || { echo "d0 differs for z18 $a"; fail=1; }
  ./tm80/tm80 build z18 $a runs/t80/raw_z18_$n > /dev/null 2>&1
  python3 tm80/bin2npz80.py runs/t80/raw_z18_$n runs/t80/z18_$n.npz | tail -1
  python3 tm80/compare80.py runs/t80/ref_z18_$n.npz runs/t80/z18_$n.npz | tail -1 | grep -q PASS && echo "z18 ($a) PASS" || { echo "z18 ($a) FAIL"; fail=1; }
done
[ $fail = 0 ] && echo "ALL PASS" || { echo "SOME FAILED"; exit 1; }
