"""Independent region-filling counter, written separately from front.fill_count: geometry based (shapely), different tie-break and
different bookkeeping.  The remainder is a LIST of simple polygons (pieces that touch at a point stay separate); at each step the
HIGHEST-RIGHT vertex v of the first piece is taken, w = v + unit vector along the next counter-clockwise boundary edge, and every tile that
has (v,w) as a side, lies inside the piece and is edge-to-edge with the remainder is tried.  Each tiling is therefore counted once.
Lattice (integer) coordinates of all vertices are carried exactly in a dictionary keyed by rounded planar coordinates.
Usage: count(d)  (d = closed ccw loop of directions; cfg selected).  count(d, listing) appends the tilings as lists of vertex tuples."""
import sys, math
import numpy as np
import shapely, shapely.geometry.polygon
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
import cfg as C
from zz import cplx, add
def _xy(v): z = cplx(v); return (z.real, z.imag)
def _key(c): return (round(c[0], 5) + 0.0, round(c[1], 5) + 0.0)
def _dedupe(coords):
    out = []
    for c in coords:
        if not out or math.hypot(c[0] - out[-1][0], c[1] - out[-1][1]) > 1e-6: out.append(c)
    if len(out) > 1 and math.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) <= 1e-6: out.pop()
    return out
def _parts(g):
    if g.is_empty: return []
    gs = list(g.geoms) if g.geom_type == 'MultiPolygon' else [g]
    out = []
    for p in gs:
        if p.area > 1e-7:
            q = Polygon(_dedupe(list(p.exterior.coords)))      # merge the near-duplicate vertices created by snapping
            if q.area > 1e-7: out.append(q)
    return out
def count(d, listing=None):
    sys.setrecursionlimit(100000)
    P = [tuple(0 for _ in range(C.r))]
    for x in d: P.append(add(P[-1], C.E[x]))
    P = P[:-1]; known = {}
    def reg(v):
        k = _key(_xy(v))
        if k in known and known[k] != v: raise RuntimeError('rounded key collision')
        known[k] = v
    for v in P: reg(v)
    memo = {}; stack = []
    def mkey(parts):
        return tuple(sorted(tuple(_key(c) for c in p.exterior.coords) for p in parts))
    def tiles_at(v):
        out = {}
        for name, offs, ang in C.TILES:
            for dd in range(C.n):
                pts = [v]
                for k in offs[:-1]: pts.append(add(pts[-1], C.E[(dd + k) % C.n]))
                out.setdefault(frozenset(pts), pts)
        return list(out.values())
    def rec(parts):
        if not parts:
            if listing is not None: listing.append(tuple(stack))
            return 1
        k = mkey(parts)
        if k in memo and listing is None: return memo[k]
        parts = sorted(parts, key=lambda p: -p.area); g, others = parts[0], parts[1:]
        g = shapely.geometry.polygon.orient(g, 1.0)
        gc = list(g.exterior.coords)[:-1]
        i0 = max(range(len(gc)), key=lambda q: (round(gc[q][0], 7), round(gc[q][1], 7)))
        v = known[_key(gc[i0])]; c1 = gc[(i0 + 1) % len(gc)]
        ang = math.atan2(c1[1] - gc[i0][1], c1[0] - gc[i0][0]); kk = int(round(ang / (2 * math.pi / C.n))) % C.n
        w = add(v, C.E[kk]); mid = ((_xy(v)[0] + _xy(w)[0]) / 2, (_xy(v)[1] + _xy(w)[1]) / 2)
        allp = MultiPolygon([g] + others) if others else g
        ring = allp.boundary; gverts = [Point(c) for p in ([g] + others) for c in list(p.exterior.coords)[:-1]]
        gb = g.buffer(1e-7); tot = 0
        for pts in tiles_at(v):
            if w not in pts: continue
            T = Polygon([_xy(p) for p in pts])
            if not (T.is_valid and T.difference(gb).area < 1e-7): continue
            Tring = LineString(list(T.exterior.coords))
            if Tring.distance(Point(mid)) > 1e-7: continue
            tv = [Point(_xy(p)) for p in pts]
            if any(ring.distance(q) < 1e-7 and min(q.distance(x) for x in gverts) > 1e-7 for q in tv): continue
            if any(Tring.distance(x) < 1e-7 and min(x.distance(q) for q in tv) > 1e-7 for x in gverts): continue
            for p in pts: reg(p)
            new = _parts(g.difference(T)) + others
            stack.append(tuple(sorted(pts))); tot += rec(new); stack.pop()
        memo[k] = tot; return tot
    return rec([Polygon([_xy(v) for v in P])])
