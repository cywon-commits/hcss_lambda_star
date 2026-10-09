#!/bin/bash
# Phase-A validation: build all reference cylinders with tm_gen and compare with the Python reference.
# Usage: bash tm_cpp/test_ref.sh   (run from the repository root; needs numpy)
set -e
cd "$(dirname "$0")/.."
make -C tm_cpp -s
mkdir -p runs
fail=0
for a in "2 2 0 -1" "2 3 0 -1" "2 3 1 -1" "3 3 0 -1" "3 4 0 -2"; do
  n=$(echo $a | tr ' ' '_')
  d0=$(python3 -c "import json,sys;a=sys.argv[1].split();print(','.join(map(str,json.load(open('reference/expected.json'))['cylinders']['(%s)'%', '.join(a)]['d0'])))" "$a")
  [ "$(tm_cpp/tm_gen d0 $a)" = "$d0" ] || { echo "d0 search differs for $a"; fail=1; }
  SECONDS=0
  tm_cpp/tm_gen build $a runs/raw_$n $d0 2>/dev/null
  echo "($a): ${SECONDS}s"
  python3 tools/bin2npz.py runs/raw_$n runs/cpp_$n.npz > /dev/null
  python3 tools/compare_tm.py reference/data/ref_$n.npz runs/cpp_$n.npz | tail -1 | grep -q PASS || { echo "FAIL $a"; fail=1; }
done
tm_cpp/tm_gen regions | tail -1
[ $fail = 0 ] && echo "ALL PASS" || { echo "SOME FAILED"; exit 1; }
