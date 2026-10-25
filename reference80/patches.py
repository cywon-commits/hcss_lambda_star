"""Random edge-glued patches of k tiles (exact lattice coordinates) and their boundary loops (direction lists), for region tests."""
import math, random
import numpy as np
import shapely
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import cfg as C
from zz import cplx, add
def _xy(v): z = cplx(v); return (z.real, z.imag)
def _tile_polys(v):
    out = {}
    for name, offs, ang in C.TILES:
        for dd in range(C.n):
            pts = [v]
            for k in offs[:-1]: pts.append(add(pts[-1], C.E[(dd + k) % C.n]))
            out.setdefault(frozenset(pts), pts)
    return list(out.values())
def _t_junction(tiles):
    """True if some tile vertex lies in the interior of an edge of another tile"""
    from shapely.geometry import LineString
    allv = set(p for t in tiles for p in t)
    for t in tiles:
        c = sum(cplx(q) for q in t) / len(t); pts = sorted(t, key=lambda p: math.atan2((cplx(p) - c).imag, (cplx(p) - c).real))
        for i in range(len(pts)):
            a, b = pts[i], pts[(i + 1) % len(pts)]; seg = LineString([_xy(a), _xy(b)])
            for v in allv:
                if v in (a, b): continue
                if seg.distance(Point(_xy(v))) < 1e-7: return True
    return False
def random_patch(k, rng):
    z0 = tuple(0 for _ in range(C.r))
    first = rng.choice(_tile_polys(z0)); tiles = [frozenset(first)]
    polys = [Polygon([_xy(p) for p in first])]
    for _ in range(k - 1):
        U = unary_union(polys); done = False
        for attempt in range(60):
            t = rng.choice(tiles); vs = list(t)
            # boundary edge: pick a vertex of a random tile and a random unit direction; attach a tile having that edge as a side
            u = rng.choice(vs); e = rng.choice(C.E); v = add(u, e)
            cand = [pts for pts in _tile_polys(u) if v in pts]
            rng.shuffle(cand)
            for pts in cand:
                T = Polygon([_xy(p) for p in pts])
                if frozenset(pts) in tiles: continue
                if U.intersection(T).area > 1e-7: continue
                if U.intersection(T.buffer(1e-7)).length < 1 - 1e-6 and not U.buffer(1e-7).intersection(T).area > 0: continue
                # must share a full unit edge with the patch
                if U.boundary.intersection(T.boundary).length < 1 - 1e-6: continue
                if _t_junction(tiles + [frozenset(pts)]): continue          # keep the patch edge-to-edge
                tiles.append(frozenset(pts)); polys.append(T); done = True; break
            if done: break
        if not done: return None
    U = unary_union(polys)
    if U.geom_type != 'Polygon' or len(U.interiors) > 0 or not U.is_valid: return None
    return tiles, U
def boundary_loop(tiles, U):
    """closed ccw loop of unit-direction indices of the polygon U; None if an edge is not a lattice direction"""
    pts = {}
    for t in tiles:
        for p in t: pts[(round(_xy(p)[0], 5) + 0.0, round(_xy(p)[1], 5) + 0.0)] = p
    g = shapely.geometry.polygon.orient(U, 1.0); cs = list(g.exterior.coords)[:-1]
    dirs = []
    for i in range(len(cs)):
        a = cs[i]; b = cs[(i + 1) % len(cs)]
        L = math.hypot(b[0] - a[0], b[1] - a[1]); nstep = int(round(L))
        if abs(L - nstep) > 1e-5 or nstep < 1: return None
        ang = math.atan2(b[1] - a[1], b[0] - a[0]); kk = int(round(ang / (2 * math.pi / C.n))) % C.n
        dd = ang - kk * 2 * math.pi / C.n; dd -= 2 * math.pi * round(dd / (2 * math.pi))
        if abs(dd) > 1e-6: return None
        dirs += [kk] * nstep
    # closure check
    z = tuple(0 for _ in range(C.r))
    for x in dirs: z = add(z, C.E[x])
    if not all(c == 0 for c in z): return None
    P = [tuple(0 for _ in range(C.r))]
    for x in dirs: P.append(add(P[-1], C.E[x]))
    if len(set(P[:-1])) != len(dirs): return None          # pinched boundary (touching itself at a vertex): not a simply connected filling region
    return dirs
