#!/usr/bin/env bash
# TASK3 Step 3: Frenkel-Ladd at all points.  Usage: NPROC=4 bash sim/run_task3.sh   (from the repository root)
# Needs runs/samples/{r12_s1..4,r312_s1..2}.npz (python3 sim/make_samples.py ...).  Output: runs/fe/*.json, logs runs/fe/logs/
set -euo pipefail
cd "$(dirname "$0")"
NPROC=${NPROC:-$(nproc)}; LAM=1.93; OUT=${OUT:-../runs/fe}; EXTRA=${EXTRA:-}; ONLY=${ONLY:-}; mkdir -p $OUT/logs
jobs=()
add(){ # name args...
  local name=$1; shift
  if [ -n "$ONLY" ] && ! grep -qx "$name" "$ONLY"; then return 0; fi
  [ -f $OUT/$name.json ] || jobs+=("python3 fl_free_energy.py --sites mean --lam $LAM $* $EXTRA --out $OUT/$name.json > $OUT/logs/$name.log 2>&1")
}
pt(){ # P T  kinds...
  local P=$1 T=$2; shift 2
  for k in "$@"; do
    case $k in
      B931)  add B_N931_P${P}_T${T}  --kind B --N 931 --P $P --T $T --seed 1;;
      B1600) add B_N1600_P${P}_T${T} --kind B --N 1600 --P $P --T $T --seed 1;;
      A931)  add A_N931_P${P}_T${T}  --kind A --N 931 --P $P --T $T --seed 1;;
      A1600) add A_N1600_P${P}_T${T} --kind A --N 1600 --P $P --T $T --seed 1;;
      dod931) for s in 1 2; do add dodeca_N931_s${s}_P${P}_T${T} --kind dodeca --N 931 --P $P --T $T --seed $s; done;;
      dod1539) for s in 1 2; do add dodeca_N1539_s${s}_P${P}_T${T} --kind dodeca --N 1539 --P $P --T $T --seed $s; done;;
      r12) for s in 1 2 3 4; do add r12_s${s}_P${P}_T${T} --kind file --config ../runs/samples/r12_s${s}.npz --P $P --T $T --seed $s; done;;
      r312) for s in 1 2; do add r312_s${s}_P${P}_T${T} --kind file --config ../runs/samples/r312_s${s}.npz --P $P --T $T --seed $s; done;;
    esac
  done
}
# Rerun of flagged points with longer NPT:  OUT=../runs/fe_long EXTRA="--npt-sweeps 200000" ONLY=../runs/fe/flagged.txt bash sim/run_task3.sh
# prediction check first
pt 0.760 0.06 B931 B1600 dod931 dod1539 r12 r312
pt 0.770 0.08 B931 B1600 dod931 dod1539 r12 r312
# references / Delta_vib
pt 0.735 0.06 B931 B1600 A931 A1600 dod931 dod1539 r12 r312
pt 0.735 0.08 B931 B1600 A931 A1600 dod931 dod1539 r12 r312
# beyond the boundary, A side
pt 0.780 0.08 B1600 r12
pt 0.790 0.08 B1600 r12
pt 0.700 0.06 A1600 r12
printf '%s\n' "${jobs[@]}" | xargs -P "$NPROC" -I{} sh -c "{}"
echo ALL_DONE
