"""Search for local retilings ('flips') in tilings by unit equilateral triangles and 80° unit rhombi.
Directions are integers mod 18 (units of 20°). A tile is a polygon given by start vertex and edge directions.
We enumerate connected edge-glued patches of k tiles, compute their boundary, and group patches with the
same boundary region (same oriented boundary edge set after translation) but different interior tilings."""
import cmath, math, itertools, sys
from collections import defaultdict

import numpy as np
_Z = [complex(math.cos(math.pi * k / 9), math.sin(math.pi * k / 9)) for k in range(6)]
def _zeta(k):
    k %= 18; sgn = 1
    if k >= 9: k -= 9; sgn = -1
    v = [0] * 6
    if k < 6: v[k] = 1
    else:  # zeta^6 = zeta^3 - 1, zeta^7 = zeta^4 - zeta, zeta^8 = zeta^5 - zeta^2
        v[k - 3] += 1; v[k - 6] -= 1
    return tuple(sgn * x for x in v)
UI = [_zeta(k) for k in range(18)]
class Pt(tuple):
    def __add__(self, o): return Pt(a + b for a, b in zip(self, o))
    def __sub__(self, o): return Pt(a - b for a, b in zip(self, o))
    @property
    def c(self): return sum(x * z for x, z in zip(self, _Z))
U = [Pt(u) for u in UI]
ORIGIN = Pt((0,) * 6)
TRI = (0, 6, 12)            # edge directions relative to d
RHO_A = (0, 5, 9, 14)       # rhombus starting at an 80° corner... (turns 5,4,5,4)
RHO_B = (0, 4, 9, 13)       # starting at a 100° corner (turns 4,5,4,5)
SHAPES = [("t", TRI), ("r", RHO_A), ("r", RHO_B)]


def key(z):
    return tuple(z)


def tile_vertices(p0, d, shape):
    pts = [p0]; z = p0
    for k in shape[:-1]:
        z = z + U[(d + k) % 18]; pts.append(z)
    return pts


def canon_tile(kind, pts):
    return (kind, tuple(sorted(key(p) for p in pts)))


def edges_of(pts):
    n = len(pts)
    return [(key(pts[i]), key(pts[(i + 1) % n])) for i in range(n)]


def area(pts):
    cs = [p.c for p in pts]
    return 0.5 * sum((cs[i].conjugate() * cs[(i + 1) % len(cs)]).imag for i in range(len(cs)))


def inside(pt, poly):
    # strict interior test for convex polygons (tiles are convex), poly ccw
    pt = pt.c if isinstance(pt, Pt) else pt
    poly = [p.c if isinstance(p, Pt) else p for p in poly]
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        if ((b - a).conjugate() * (pt - a)).imag <= 1e-7:
            return False
    return True


def overlaps(newpts, tiles):
    newpts = [p.c for p in newpts]
    c = sum(newpts) / len(newpts)
    for kind, pts in tiles:
        pts = [p.c for p in pts]
        cc = sum(pts) / len(pts)
        if abs(c - cc) > 2.5:
            continue
        if inside(c, pts) or inside(cc, newpts):
            return True
        if any(inside(p, pts) for p in newpts) or any(inside(p, newpts) for p in pts):
            return True
        # edge midpoints
        for i in range(len(newpts)):
            m = (newpts[i] + newpts[(i + 1) % len(newpts)]) / 2
            if inside(m, pts):
                return True
    return False


def attach_options(tiles):
    """All tiles that can be glued (ccw, on the outside) to a boundary edge of the patch."""
    cnt = defaultdict(int)
    for _, pts in tiles:
        for e in edges_of(pts):
            cnt[e] += 1
    bd = [e for e in cnt if (e[1], e[0]) not in cnt]
    out = []
    for (a, b) in bd:
        za, zb = Pt(a), Pt(b)
        # new tile must traverse the edge as b -> a (ccw on the other side)
        dvec = tuple(za - zb)
        d = UI.index(dvec)
        for kind, shape in SHAPES:
            pts = tile_vertices(zb, d, shape)
            if area(pts) <= 0:
                continue
            out.append((kind, pts))
    return out


def boundary_signature(tiles):
    cnt = defaultdict(int)
    for _, pts in tiles:
        for e in edges_of(pts):
            cnt[e] += 1
    bd = [e for e in cnt if (e[1], e[0]) not in cnt]
    # simply connected check: Euler characteristic of the patch
    V = {v for _, pts in tiles for v in map(key, pts)}
    E = {frozenset(e) for _, pts in tiles for e in edges_of(pts)}
    if len(V) - len(E) + len(tiles) != 1:
        return None
    o = min(V)
    sh = lambda v: tuple(x - y for x, y in zip(v, o))
    return frozenset((sh(a), sh(b)) for a, b in bd)


def enumerate_patches(kmax):
    start = [("t", tile_vertices(ORIGIN, 0, TRI)), ("r", tile_vertices(ORIGIN, 0, RHO_A))]
    level = {}
    for t in start:
        level[frozenset([canon_tile(*t)])] = [t]
    allp = dict(level)
    for k in range(2, kmax + 1):
        new = {}
        for kk, tiles in level.items():
            for opt in attach_options(tiles):
                if overlaps(opt[1], tiles):
                    continue
                nt = tiles + [opt]
                # translation-normalize the canonical key
                V = [p for _, pts in nt for p in pts]
                o = Pt(min(V))
                ck = frozenset(canon_tile(kind, [p - o for p in pts]) for kind, pts in nt)
                if ck not in new:
                    new[ck] = nt
        level = new; allp.update(new)
        print(f"k={k}: {len(new)} patches", flush=True)
    return allp


if __name__ == "__main__":
    kmax = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    allp = enumerate_patches(kmax)
    groups = defaultdict(list)
    for ck, tiles in allp.items():
        sig = boundary_signature(tiles)
        if sig is not None:
            groups[sig].append(tiles)
    moves = {s: g for s, g in groups.items() if len(g) > 1}
    print("regions with >1 tiling:", len(moves))
    import pickle
    summ = []
    for s, g in moves.items():
        comp = sorted({(sum(1 for t in tl if t[0] == 't'), sum(1 for t in tl if t[0] == 'r')) for tl in g})
        summ.append((len(s), len(g), comp))
    from collections import Counter
    for (nb, ng, comp), c in sorted(Counter((a, b, tuple(cc)) for a, b, cc in summ).items()):
        print(f"  boundary edges {nb}, #tilings {ng}, compositions (tri, rho) {comp}: {c} regions (all orientations)")
    pickle.dump({s: g for s, g in moves.items()}, open(f"moves_k{kmax}.pkl", "wb"))
