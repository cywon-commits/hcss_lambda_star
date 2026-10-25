"""Local statistics of a sampled cylinder tiling (sampler.py output): vertex types (t,a,o) with t = #triangle corners (60 deg), a = #80-deg and
o = #100-deg rhombus corners, arrangement a.a.o.o vs a.o.a.o of (0,2,2) vertices, vortex ((1,0,3), (2,3,0)) density and net charge, rhombus-label
order M_rh.  Only vertices whose corner angles close up to 360 degrees are counted.  Usage: analyze_sample.py sample.npz [P a0..a5]"""
import sys, os, json, math, collections
import numpy as np
z = np.exp(2j * np.pi / 18)
def stats(f, a, margin=0):
    S = np.load(f); V = S['V'].astype(int); names = S['names']; cls = S['cls']; P = sum(a[i] * z ** i for i in range(6)); ph = P / abs(P); a = np.array(a)
    cx = lambda v: v @ z ** np.arange(6)
    corners = collections.defaultdict(list); seen = set(); ntile = 0; labs = []; ntri = collections.Counter(); dup = 0
    for t in range(len(names)):
        nv = 3 if names[t] == 'T' else 4; v = V[t][:nv]; c = np.array([cx(x) for x in v]); cen = c.mean()
        k = math.floor((cen * ph.conjugate()).real / abs(P)); v = v - k * a
        key = tuple(sorted(tuple(x) for x in v))
        if key in seen: dup += 1; continue
        seen.add(key); ntile += 1; c = np.array([cx(x) for x in v])
        if names[t] != 'T': labs.append((int(cls[t]) - 3 + 2) % 3)
        else: ntri[int(cls[t])] += 1
        for i in range(nv):
            p, nx, pv = c[i], c[(i + 1) % nv], c[i - 1]
            an = round(math.degrees(math.atan2((nx - p).imag, (nx - p).real)) / 20) * 20 % 360
            ap = round(math.degrees(math.atan2((pv - p).imag, (pv - p).real)) / 20) * 20 % 360
            vk = tuple(v[i] - math.floor((cx(v[i]) * ph.conjugate()).real / abs(P)) * a)
            corners[vk].append((an, (ap - an) % 360, cen.imag * ph.real - cen.real * ph.imag))
    types = collections.Counter(); arr = collections.Counter(); bad = 0; vort = collections.Counter()
    for vk, cs in corners.items():
        cs.sort(); tot = sum(w for _, w, _ in cs)
        if tot != 360: continue
        if any((cs[(i + 1) % len(cs)][0] - cs[i][0]) % 360 != cs[i][1] for i in range(len(cs))): bad += 1; continue
        t = sum(w == 60 for _, w, _ in cs); aa = sum(w == 80 for _, w, _ in cs); oo = sum(w == 100 for _, w, _ in cs)
        types[(t, aa, oo)] += 1
        if (t, aa, oo) == (0, 2, 2):
            ws = [w for _, w, _ in cs]; arr['aooa' if any(ws[i] == ws[(i + 1) % 4] for i in range(4)) is False else 'aaoo'] += 1
    N = sum(types.values()); lab = np.array(labs)
    out = dict(tiles=ntile, dup=dup, bad=bad, vertices=N, types={'%d%d%d' % k: v for k, v in sorted(types.items())},
               frac={'%d%d%d' % k: v / N for k, v in sorted(types.items())}, aaoo=arr['aaoo'], aoao=arr['aooa'],
               vort_pos=types[(2, 3, 0)], vort_neg=types[(1, 0, 3)], vort_density=(types[(2, 3, 0)] + types[(1, 0, 3)]) / N,
               Mrh=float(abs(np.exp(2j * np.pi * lab / 3).sum()) / len(lab)) if len(lab) else None, rh_labels=np.bincount(lab, minlength=3).tolist(),
               tri_fam=[ntri[i] for i in range(3)])
    return out
if __name__ == '__main__':
    print(json.dumps(stats(sys.argv[1], [int(x) for x in sys.argv[2:8]]), indent=1))
