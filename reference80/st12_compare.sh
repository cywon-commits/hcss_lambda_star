#!/bin/bash
# usage: bash st12_compare.sh "a0 a1 a2 a3 a4 a5" ...   (z18 two-family periods); outputs ../runs/cmp_*.json and prints the comparison
cd "$(dirname "$0")"
for a in "$@"; do n=$(echo $a | tr ' -' '_m')
  HCSS_CFG=z18  python3 st12_cyl.py $a ../runs/cmp_${n}_z18.json  > /dev/null
  HCSS_CFG=st12 python3 st12_cyl.py $a ../runs/cmp_${n}_st12.json ../runs/cmp_${n}_z18.json > /dev/null
done
python3 st12_report.py "$@"
