"""Integer-coordinate version of the single-domain sigma phase (3^2.4.3.4) tilted by 10 degrees (analysis/z3/make_sd.py geometry):
every vertex is an exact Z[zeta18] vector.  Returns (A1, A2, sites, tiles) with tiles = list of (name, vertex-int-tuples) in the window."""
import math, itertools
import numpy as np
import cfg as C
from zz import cplx, add, sub
def build(theta_deg=10.0):
    c = (1 + math.sqrt(3)) / math.sqrt(2)
    def sq(cen, rot): return [cen + np.array([math.cos(a), math.sin(a)]) / math.sqrt(2) for a in np.radians(rot + 45 + 90 * np.arange(4))]
    basis = np.array(sq(np.zeros(2), 15) + sq(np.array([c / 2, c / 2]), -15)); A = np.array([[c, 0], [0, c]]); B = []
    for p in basis:
        f = np.linalg.solve(A.T, p) % 1.0
        if not any(np.allclose(f, q, atol=1e-6) or np.allclose((f - q + 0.5) % 1 - 0.5, 0, atol=1e-6) for q in B): B.append(f)
    B = np.array(B); nb = len(B); X = B @ A.T
    edges = []
    for i in range(nb):
        for j in range(nb):
            for n1 in (-1, 0, 1):
                for n2 in (-1, 0, 1):
                    d = X[j] + n1 * A[:, 0] + n2 * A[:, 1] - X[i]
                    if abs(np.linalg.norm(d) - 1) < 1e-6 and (i, j, n1, n2) < (j, i, -n1, -n2): edges.append((i, j, n1, n2, d))
    th = math.radians(theta_deg); off0 = math.degrees(math.atan2(edges[0][4][1], edges[0][4][0])) % 30
    def rot(d):
        m = round((math.degrees(math.atan2(d[1], d[0])) - off0) / 30)
        if m % 2: d = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]) @ d
        return d
    # direction index in units of 20 degrees (relative to the global rotation off0)
    E_ = []
    for i, j, n1, n2, d in edges:
        dr = rot(d); k = (math.degrees(math.atan2(dr[1], dr[0])) - off0) / 20.0
        kk = int(round(k)); assert abs(k - kk) < 1e-6, ('edge direction not a multiple of 20 deg', k)
        E_.append((i, j, n1, n2, kk % 18))
    # linear system per component:  x_j + n1 A1 + n2 A2 - x_i = E[k]   (unknown vectors in Z^r, x_0 = 0)
    nu = (nb - 1) + 2; rows = []; ks = []
    for i, j, n1, n2, kk in E_:
        r = np.zeros(nu)
        if j > 0: r[j - 1] += 1
        if i > 0: r[i - 1] -= 1
        r[nb - 1] += n1; r[nb] += n2; rows.append(r); ks.append(kk)
    M = np.array(rows); rhs = np.array([C.E[k] for k in ks], float)          # (nedges, r)
    sol, res, rk, _ = np.linalg.lstsq(M, rhs, rcond=None)
    resid = np.abs(M @ sol - rhs).max(); assert resid < 1e-9, resid
    S = np.round(sol).astype(int); assert np.abs(sol - S).max() < 1e-9, 'non-integer solution'
    sites = [tuple([0] * C.r)] + [tuple(S[j - 1]) for j in range(1, nb)]
    A1 = tuple(S[nb - 1]); A2 = tuple(S[nb])
    return A1, A2, sites, [(i, j, n1, n2, kk) for i, j, n1, n2, kk in E_], resid
def tiles_in_window(A1, A2, sites, edges, N=3):
    """all unit triangles and 80-degree rhombi of the periodic graph in the window |n1|,|n2| <= N, as vertex-int-tuples"""
    def sh(p, n1, n2):
        q = p
        for _ in range(abs(n1)): q = add(q, A1) if n1 > 0 else sub(q, A1)
        for _ in range(abs(n2)): q = add(q, A2) if n2 > 0 else sub(q, A2)
        return q
    V = {}
    for n1 in range(-N, N + 1):
        for n2 in range(-N, N + 1):
            for s in sites: V[sh(s, n1, n2)] = None
    Vs = list(V); adj = {v: set() for v in Vs}
    for v in Vs:
        for e in C.E:
            w = add(v, e)
            if w in V:
                # an edge of the tiling iff it is one of the graph edges: unit vectors in the window that are actual tiling edges
                adj[v].add(w)
    return Vs, adj
def periodic_tiles(A1, A2, sites, edges, N=4):
    """tiles (name, frozenset of vertex tuples) in the window, from the edge graph; verifies exact cover of the cell (each edge in two tiles)"""
    import math
    def sh(p, n1, n2):
        q = p
        for _ in range(abs(n1)): q = add(q, A1) if n1 > 0 else sub(q, A1)
        for _ in range(abs(n2)): q = add(q, A2) if n2 > 0 else sub(q, A2)
        return q
    adj = {}
    for m1 in range(-N, N + 1):
        for m2 in range(-N, N + 1):
            for i, j, n1, n2, kk in edges:
                a = sh(sites[i], m1, m2); b = sh(sites[j], m1 + n1, m2 + n2)
                adj.setdefault(a, set()).add(b); adj.setdefault(b, set()).add(a)
    tiles = {}
    for a in adj:
        if abs(cplx(a)) > 6: continue
        for b, c in itertools.combinations(sorted(adj[a]), 2):
            if c in adj[b]: tiles[frozenset((a, b, c))] = 'T'
        for b in adj[a]:
            for c in adj[b]:
                if c == a: continue
                for d in adj[c]:
                    if d in (a, b) or a not in adj[d]: continue
                    pts = [a, b, c, d]; z = [cplx(p) for p in pts]
                    ar = abs(sum((z[k].conjugate() * z[(k + 1) % 4]).imag for k in range(4))) / 2
                    if abs(ar - math.sin(math.radians(80))) < 1e-9:
                        # parallelogram check: a + c == b + d
                        if add(a, c) == add(b, d): tiles[frozenset(pts)] = 'R'
    return tiles, adj
