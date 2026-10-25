# exact front-growth counting for unit-edge tilings (cfg decides the tiles).  Generalisation of reference/python/front.py:
# n directions (half = n/2 is the reversal), tiles from cfg.TILES; fill counts also carry the class vector of the tiles in the region.
import numpy as np
from functools import lru_cache
import cfg as C
from zz import add, sub, cplx
_BASIS = {}
VIOLATIONS = []        # regions in which two fillings had different class vectors (theorem 1.3.1 would be violated)
def E(k): return C.E[k]
def cancel(d):
    d = list(d); ch = True; n, h = C.n, C.half
    while ch and len(d) >= 2:
        ch = False
        for i in range(len(d)):
            j = (i + 1) % len(d)
            if (d[i] - d[j]) % n == h:
                d = [x for k, x in enumerate(d) if k not in (i, j)]; ch = True; break
    return d
def pos(d):
    P = [tuple(0 for _ in range(C.r))]
    for x in d: P.append(add(P[-1], C.E[x]))
    return P
def crosses(segs):
    cr = lambda a, b: (a.conjugate() * b).imag
    n = len(segs); eps = 1e-9
    for i in range(n):
        a, b = segs[i]
        for j in range(i + 1, n):
            c, dd = segs[j]
            if min(abs(a - c), abs(a - dd), abs(b - c), abs(b - dd)) > 2.01: continue
            r, s = b - a, dd - c; den = cr(r, s)
            if abs(den) < 1e-12:
                if abs(cr(c - a, r)) < 1e-9:
                    t0 = ((c - a) * r.conjugate()).real; t1 = ((dd - a) * r.conjugate()).real
                    if min(1, max(t0, t1)) - max(0, min(t0, t1)) > eps: return True
                continue
            t = cr(c - a, s) / den; u = cr(c - a, r) / den
            if eps < t < 1 - eps and -eps < u < 1 + eps: return True
            if eps < u < 1 - eps and -eps < t < 1 + eps: return True
    return False
def area(d):
    P = [cplx(p) for p in pos(d)]
    return sum((P[i].conjugate() * P[i + 1]).imag for i in range(len(d))) / 2
def canon_closed(d):
    P = pos(d)[:-1]; Cc = [cplx(p) for p in P]
    i = min(range(len(P)), key=lambda k: (round(Cc[k].imag, 9), round(Cc[k].real, 9)))
    return tuple(d[i:] + d[:i])
# ---- classes of tiles: triangles by family f = d mod 3 (3 classes), rhombi by the unordered pair of edge directions mod n/2 (9 classes)
def tile_class(name, d):
    """class index of the tile placed with start direction d (offsets from cfg).  Only defined for cfg 'z18'."""
    if C.name != 'z18': return None
    if name == 'T': return d % 3
    if name == 'R80': return 3 + d % 9                 # edge directions d, d+4 (mod 9)
    return 3 + (d + 5) % 9                              # R100: directions d, d+5 -> pair {d+5, d+9=d}: class (d+5) mod 9
def ncls(): return 12 if C.name == 'z18' else 0
def _zero(): return (0,) * ncls()
def _addv(a, b): return tuple(x + y for x, y in zip(a, b))
# ---------- finite regions (closed loops, CCW) ----------
@lru_cache(maxsize=None)
def _fill(d):
    if not d: return (1, 0.0, _zero())
    d = list(d); din, dout = d[-1], d[0]; n, h = C.n, C.half
    turn = (dout - din) % n; turn = turn - n if turn > h else turn
    interior = h - turn; tot = 0; vw = None; vec = None
    for t, td, ang in C.TILES:
        if ang > interior: continue
        tdir = [(dout + x) % n for x in td]
        rep = [(x + h) % n for x in reversed(tdir[1:])]
        nd = cancel(rep + d[1:])
        r = closed_pieces(nd)
        if r is None: continue
        c = 1; w = C.VW[t]; v = ()
        v = tile_class(t, dout) if C.name == 'z18' else None
        v = _zero() if v is None else tuple(1 if k == v else 0 for k in range(ncls()))
        for piece in r:
            cc, ww, vv = _fill(piece); c *= cc; w += ww; v = _addv(v, vv)
            if c == 0: break
        if c:
            tot += c; vw = w
            if vec is None: vec = v
            elif vec != v: VIOLATIONS.append((tuple(d), vec, v))
    return (tot, vw if vw is not None else 0.0, vec if vec is not None else _zero())
C.HOOKS.append(lambda: (_fill.cache_clear(), _BASIS.clear(), VIOLATIONS.clear()))
def fill_count(d):
    """(#fillings, vertex-weight sum of the tiles)"""
    r = _fill(d); return (r[0], r[1])
def fill_info(d):
    """(#fillings, vertex weight, class vector of the tiles in the region)"""
    return _fill(d)
def closed_pieces(nd):
    nd = cancel(nd)
    if not nd: return []
    P = pos(nd); seen = {}
    for i, p in enumerate(P[:-1]):
        if p in seen:
            i0 = seen[p]
            a = closed_pieces(nd[i0:i]); b = closed_pieces(nd[:i0] + nd[i:])
            return None if a is None or b is None else a + b
        seen[p] = i
    Cc = [cplx(p) for p in P]
    if crosses([(Cc[i], Cc[i + 1]) for i in range(len(nd))]): return None
    if area(nd) < 1e-9: return None
    return [canon_closed(nd)]
# ---------- lifted areas (theorem 1.3.1): the 2-form  sum_k P_k ^ P_{k+1} in Lambda^2 Z^r fixes the class counts of any region ----------
def lifted_area(d):
    P = pos(d); r = C.r; out = {}
    for k in range(len(d)):
        a, b = P[k], P[k + 1]
        for i in range(r):
            for j in range(i + 1, r):
                out[(i, j)] = out.get((i, j), 0) + a[i] * b[j] - a[j] * b[i]
    return np.array([out.get((i, j), 0) for i in range(r) for j in range(i + 1, r)], float)
def lifted_cls(d):
    """class counts of the tiles in the region bounded by the closed loop d, from the lifted-area linear system (z18 only).
    Returns (vector, residual, integral?)"""
    if C.name != 'z18': return None
    if 'M' not in _BASIS:
        cols = []
        for cl in range(12):
            if cl < 3: name, dd = 'T', cl
            else:
                a = cl - 3; name, dd = 'R80', a
            offs = {t: td for t, td, _ in C.TILES}[name]
            loop = [(dd + x) % C.n for x in offs[:-1]]
            # closed loop of the tile: edge directions d, d+off..., edge k has direction (d + offs[k])
            loop = [(dd + o) % C.n for o in offs]
            cols.append(lifted_area(loop))
        _BASIS['M'] = np.array(cols).T
    M = _BASIS['M']; b = lifted_area(list(d))
    x, res, rk, _ = np.linalg.lstsq(M, b, rcond=None)
    r = float(np.abs(M @ x - b).max()); integral = bool(np.allclose(x, np.round(x), atol=1e-6))
    return np.round(x).astype(int), r, integral
