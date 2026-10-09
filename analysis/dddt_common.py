"""Shared geometry / classification for the DDDT crossing-set analyses, reading the committed JSON (no pickles).
Mirrors the definitions of vclass3x.py (same C3v maps, segments, primitive library, independent-combination classification)."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.setrecursionlimit(100000)
import numpy as np
from collections import defaultdict, Counter
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
from shapely.geometry.polygon import orient
from zz import cplx, add, sub, E, rot
from front import closed_pieces, fill_count
from load_crossing_sets import load
R3, M3 = load('DDDT'); R2, M2 = load('DDT')
tup = lambda p: tuple(p)
SEG3 = {k: (tup(a), tup(b)) for k, (a, b) in M3['segments'].items()}; SEG2 = {k: (tup(a), tup(b)) for k, (a, b) in M2['segments'].items()}
DaI, DbI, DcI, TI = [[tup(p) for p in c] for c in M3['cells']]
O = (0, 0, 0, 0); X = E[0]; Y = add(E[0], E[4]); tot3 = sum(R3.values())
def conj(a): return tuple(sum(a[i]*E[(12-i) % 12][j] for i in range(4)) for j in range(4))
def Rrot(z): return add(rot(z, 4), X)                 # O->X->Y->O
def Mir(z): return add(X, rot(conj(sub(z, X)), 10))   # fixes X, swaps Y<->O
assert [Rrot(O), Rrot(X), Rrot(Y)] == [X, Y, O] and Mir(Y) == O and Mir(O) == Y and Mir(X) == X
GROUP = {'e': lambda z: z, 'C3': Rrot, 'C3^2': lambda z: Rrot(Rrot(z)), 's_X': Mir,
         's_Y': lambda z: Rrot(Mir(Rrot(Rrot(z)))), 's_O': lambda z: Rrot(Rrot(Mir(Rrot(z))))}
def tmap(f, S): return frozenset((t, frozenset(f(v) for v in vs)) for t, vs in S)
shp = lambda P: Polygon([(cplx(p).real, cplx(p).imag) for p in P])
def ordered(vs):
    C = [cplx(v) for v in vs]; c = sum(C)/len(C); C.sort(key=lambda z: np.angle(z-c)); return [(z.real, z.imag) for z in C]
tpoly = lambda t: Polygon(ordered(t[1]))
CELLS = [shp(DaI), shp(DbI), shp(DcI), shp(TI)]
SEGL = {k: LineString([(cplx(a).real, cplx(a).imag), (cplx(b).real, cplx(b).imag)]) for k, (a, b) in SEG3.items()}
JUNC = {'X': ('DaDb', 'TDa', 'TDb'), 'Y': ('DbDc', 'TDb', 'TDc'), 'O': ('DcDa', 'TDc', 'TDa')}
JPT = {'X': X, 'Y': Y, 'O': O}
def segs_of(t):
    P = tpoly(t).buffer(-1e-7); return frozenset(k for k in SEGL if P.intersection(SEGL[k]).length > 1e-7)
def ring_dirs(g):
    g = orient(g, 1.0); cs0 = list(g.exterior.coords)[:-1]; cs = []; out = []
    for p in cs0:
        if not cs or abs(complex(*p)-complex(*cs[-1])) > 1e-6: cs.append(p)
    if len(cs) > 1 and abs(complex(*cs[0])-complex(*cs[-1])) <= 1e-6: cs.pop()
    for k in range(len(cs)):
        a = complex(*cs[k]); b = complex(*cs[(k+1) % len(cs)]); L = abs(b-a); dd = int(round(np.angle(b-a)/(np.pi/6))) % 12
        assert abs(L-round(L)) < 1e-6 and L > 0.5
        out += [dd]*int(round(L))
    return out
def cnt(g):
    if g.is_empty or g.area < 1e-6: return 1
    c = 1
    for x in (list(g.geoms) if g.geom_type == 'MultiPolygon' else [g]):
        if x.area < 1e-6: continue
        if len(x.interiors): raise RuntimeError('hole in remainder region')
        pcs = closed_pieces(ring_dirs(x))
        if pcs is None: return 0
        for p in pcs: c *= fill_count(p)[0]
    return c
def weight(S):
    """product of fillings of the four cells minus the tiles of S (exact filling count of crossing set S if S is a valid partial tiling)"""
    U = unary_union([tpoly(t) for t in S]) if S else None
    c = 1
    for cell in CELLS:
        rem = cell if U is None else cell.difference(U)
        rem = rem.buffer(-1e-9).buffer(1e-9) if not rem.is_empty else rem
        c *= cnt(rem)
        if c == 0: return 0
    return c
# ---- primitive library from DDT, placed by rotations at X, Y, O ----
def seg_groups(S, segf):
    g = defaultdict(list)
    for t in S: g[segf(t)].append(t)
    return g
SEGL2 = {k: LineString([(cplx(a).real, cplx(a).imag), (cplx(b).real, cplx(b).imag)]) for k, (a, b) in SEG2.items()}
def segs2(t):
    P = tpoly(t).buffer(-1e-7); return frozenset(k for k in SEGL2 if P.intersection(SEGL2[k]).length > 1e-7)
bond2 = []; vert2 = []
for S in R2:
    if not S: continue
    g = seg_groups(S, segs2); multi = any(len(k) > 1 for k in g)
    if not multi and len(g) == 1 and len(S) == 2 and all(t[0] == 'R' for t in S): bond2.append(S)
    elif multi or any(len(v) != 2 or any(t[0] != 'R' for t in v) for v in g.values()): vert2.append(S)
assert len(vert2) == 22 and len(bond2) == 4
Rn = [lambda z: z, Rrot, lambda z: Rrot(Rrot(z))]
PRIM = {}
for j, (jn, f) in enumerate(zip('XYO', Rn)):
    for i, S in enumerate(bond2): PRIM.setdefault(tmap(f, S), ('bond', jn, i))
    for i, S in enumerate(vert2): PRIM[tmap(f, S)] = ('vertex', jn, i)
prims = list(PRIM); polys = [unary_union([tpoly(t) for t in S]) for S in prims]
ovl = [[(i != j and polys[i].intersection(polys[j]).area > 1e-7) or bool(prims[i] & prims[j] and i != j) for j in range(len(prims))] for i in range(len(prims))]
ICL = set()
def _dfs(chosen, start):
    if chosen: ICL.add(frozenset().union(*[prims[i] for i in chosen]))
    for i in range(start, len(prims)):
        if all(not ovl[i][j] for j in chosen): _dfs(chosen+[i], i+1)
_dfs([], 0); ICL.add(frozenset())
def klass(S):
    """a,b,c,c2 as in results02 for independent sets (by primitive decomposition), d for the rest"""
    if S not in ICL: return 'd'
    return 'a' if not S else None   # finer a/b/c split not needed here; independent sets return None
COOP = [S for S in R3 if S not in ICL]
def orbit_key(S):
    return min(tuple(sorted((t, tuple(sorted(vs))) for t, vs in tmap(f, S))) for f in GROUP.values())
