"""Step 2: random-tiling samples.  python3 make_samples.py <r12|r312> <seed> <nsweep> <outdir>
r12 : approximant n1=2, n2=3 (N=1590, |beta|=(2-sqrt3)^4=0.0052) -> E ~ 0 sector;  r312: periodic 3.12.12, N=931 (n=7), beta=0.0718.
Writes <outdir>/<kind>_s<seed>.npz (s, box) and <kind>_s<seed>.json (history, sector start/end)."""
import sys, json, os, numpy as np
import hcss_mc as mc, sampling as S, build_approx as BA

kind, seed, nsweep, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
os.makedirs(out, exist_ok=True)
if kind == "r12":
    s0, box = BA.build(2, 3)
elif kind == "r312":
    s0, box = mc.lattice_state("dodeca", S.LS, 931, scale=1 + 1e-7, seed=seed)
else:
    raise SystemExit("kind")
s, hist, info = S.randomize(s0, box, nsweep, 1000 + seed, check_every=100, tag="%s_s%d" % (kind, seed))
np.savez(f"{out}/{kind}_s{seed}.npz", s=s, box=box, s_start=s0)
def conv(o):
    if isinstance(o, dict): return {str(k): conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [conv(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    return o
json.dump(conv(dict(kind=kind, seed=seed, nsweep=nsweep, history=hist, info=info)), open(f"{out}/{kind}_s{seed}.json", "w"))
print("done", kind, seed, info["end"]["eta"], info["end"]["abs_beta"])
