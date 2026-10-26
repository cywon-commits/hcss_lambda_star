# REFERENCE cylinder transfer-matrix builder (Python), generalised over cfg.  Usage:
#   HCSS_CFG=lam12 python3 cyl_ref.py a0 a1 a2 a3 [out.npz]            (regression: lambda* tiling)
#   HCSS_CFG=z18   python3 cyl_ref.py a0 a1 a2 a3 a4 a5 [out.npz]      (triangle + 80-degree rhombus)
# Records per transition: fr, to, c (hole multiplicity), w (=2*vertices), ar (physical area), q (perp displacement(s), lateral part removed),
# na (#triangles incl. holes); for z18 additionally cls (12 tile-class counts incl. holes) and the two perpendicular displacements q5, q7.
import sys, os, itertools, time, math
import numpy as np
import cfg as C
from front import *
from zz import star, add, cplx
import cyl
from cyl import lift, PHAT, canon_front, path_for, find_d0
def cancel_shift(d):
    d = list(d); st = tuple(0 for _ in range(C.r)); ch = True; n, h = C.n, C.half
    while ch and len(d) >= 2:
        ch = False
        for i in range(len(d)):
            j = (i + 1) % len(d)
            if (d[i] - d[j]) % n == h:
                if j == 0: st = add(st, C.E[d[0]])
                d = [x for k, x in enumerate(d) if k not in (i, j)]; ch = True; break
    return d, st
def split_shift(d, Pv, negP):
    d, st = cancel_shift(d)
    P = lift(d); idx = {}
    for i, p in enumerate(P[:-1]): idx.setdefault(p, []).append(i)
    z0 = tuple(0 for _ in range(C.r))
    for i, p in enumerate(P[:-1]):
        for q, typ in ((p, 0), (add(p, Pv), 1)):
            for j in idx.get(q, []):
                if j <= i: continue
                if typ == 0: hole = d[i:j]; rest = d[:i] + d[j:]; rst = z0
                else: hole = d[:i] + d[j:]; rest = d[i:j]; rst = P[i]
                hp = closed_pieces(hole)
                if hp is None: return None
                r = split_shift(rest, Pv, negP)
                if r is None: return None
                return (r[0], r[1] + hp, add(st, add(rst, r[2])))
        if add(p, negP) in idx: return None
    Cc = [cplx(p) for p in P]; pc = cplx(Pv); n = len(d); segs = []
    for sh in (-pc, 0, pc): segs += [(Cc[k] + sh, Cc[k + 1] + sh) for k in range(n)]
    if crosses(segs): return None
    Q = P[:-1]; ys = [round((cplx(p) * PHAT[0].conjugate()).imag, 9) for p in Q]; m = min(ys)
    cands = [(tuple(d[i:] + d[:i]), i) for i in range(len(d)) if ys[i] == m]
    f, i = min(cands)
    return (f, [], add(st, Q[i]))
AREA_MISMATCH = []
def build(Pv, d0):
    negP = tuple(-x for x in Pv)
    s0 = canon_front(d0); states = {s0: 0}; order = [s0]; tr = []
    k = 0; n, h = C.n, C.half
    while k < len(order):
        d = list(order[k]); din, dout = d[-1], d[0]
        turn = (dout - din) % n; turn = turn - n if turn > h else turn
        for t, td, ang in C.TILES:
            if ang > h - turn: continue
            tdir = [(dout + x) % n for x in td]; rep = [(x + h) % n for x in reversed(tdir[1:])]
            r = split_shift(rep + d[1:], Pv, negP)
            if r is None: continue
            f, holes, off = r; c = 1; w = C.VW[t]; ar = C.AREA[t]
            cl = tile_class(t, dout)
            cls = tuple(1 if x == cl else 0 for x in range(12)) if C.name == 'z18' else ()
            for hh in holes:
                cc, ww, vv = fill_info(hh); c *= cc; w += ww; ar += area(list(hh))
                if C.name == 'z18': cls = tuple(x + y for x, y in zip(cls, vv))
            if c == 0: continue
            if f not in states: states[f] = len(order); order.append(f)
            if C.name == 'st12':
                na = (1 if t == 'T' else 0) + sum(int(round((area(list(hh)) - fill_count(hh)[1]) / (math.sqrt(3) / 4 - 0.5))) for hh in holes)
            elif C.name == 'lam12':
                na = (1 if t == 'A' else 0) + sum(int(round((area(list(hh)) - fill_count(hh)[1] / 2) / ((math.sqrt(3) - 1) / 4))) for hh in holes)
            else:
                na = sum(cls[:3])
                # independent check of the triangle count from the area / vertex-weight identity (holes + tile)
                sr = math.sin(math.radians(80)); nt_h = 0
                for hh in holes: nt_h += (area(list(hh)) - fill_count(hh)[1] * sr) / (math.sqrt(3) / 4 - sr / 2)
                if abs(nt_h + (1 if t == 'T' else 0) - na) > 1e-6: AREA_MISMATCH.append((k, t, nt_h, na))
            tr.append((k, states[f], c, int(round(2 * w)), ar, off, na, cls))
        k += 1
    return order, tr
def main():
    a = tuple(int(x) for x in sys.argv[1:1 + C.r]); Pv = a
    outp = sys.argv[1 + C.r] if len(sys.argv) > 1 + C.r else 'ref_%s_%s.npz' % (C.name, '_'.join(map(str, a)))
    PHAT[0] = cplx(Pv) / abs(cplx(Pv)); d0 = find_d0(a)
    t0 = time.time(); order, tr = build(Pv, d0)
    fr, to, c, w, ar, off, na, cls = zip(*tr)
    off = np.array(off); dphys = np.array([cplx(o) for o in off]); Pc = cplx(Pv)
    a_ = (dphys * np.conj(Pc / abs(Pc))).real
    qs = []; Pss = []
    for m in C.PERP:
        dperp = np.array([star(tuple(o), m) for o in off]); Ps = star(Pv, m)
        qs.append(dperp - (a_ / abs(Pc)) * Ps); Pss.append(Ps)
    kw = dict(fr=fr, to=to, c=c, w=w, ar=ar, na=na, n=len(order), P=Pc, d0=np.array(d0), off=off)
    if C.name in ('lam12', 'st12'): kw.update(q=qs[0], Ps=Pss[0])
    else: kw.update(q5=qs[0], q7=qs[1], Ps5=Pss[0], Ps7=Pss[1], cls=np.array(cls))
    np.savez_compressed(outp, **kw)
    print('saved', a, len(order), '%.0fs' % (time.time() - t0), 'violations', len(VIOLATIONS), 'area mismatches', len(AREA_MISMATCH), flush=True)
if __name__ == '__main__':
    sys.setrecursionlimit(100000); main()
