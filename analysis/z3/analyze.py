"""Tile analysis of HCSS snapshots: core-contact bond network -> triangles / quads / defects,
quad corner angles, vertex types (triangles, acute, obtuse) and the 'exactly-one-cover' test."""
import sys, math
import numpy as np
from collections import Counter
from scipy.spatial import Delaunay


def tiles(fn, rc=1.12):
    z = np.load(fn); s = z["s"]; a, b, c = z["box"]; lam = float(z["lam"])
    M = np.array([[a, b], [0, c]]); P = s @ M.T; N = len(P)
    X = []; ID = []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            X.append(P + i * M[:, 0] + j * M[:, 1]); ID.append(np.arange(N))
    X = np.vstack(X); ID = np.concatenate(ID)
    tri = Delaunay(X); S = tri.simplices; nb = tri.neighbors
    L = lambda p, q: np.linalg.norm(X[p] - X[q])
    # union triangles across Delaunay edges that are NOT bonds
    parent = list(range(len(S)))
    def f(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for t in range(len(S)):
        for k in range(3):
            u = nb[t, k]
            if u < 0: continue
            p, q = S[t, (k + 1) % 3], S[t, (k + 2) % 3]
            if L(p, q) > rc:
                parent[f(t)] = f(u)
    groups = {}
    for t in range(len(S)): groups.setdefault(f(t), []).append(t)
    Minv = np.linalg.inv(M)
    def inbox(p):
        g = Minv @ p; return 0 <= g[0] < 1 and 0 <= g[1] < 1
    polys = []
    for g, ts in groups.items():
        # boundary edges of the merged polygon
        edges = Counter()
        for t in ts:
            for k in range(3):
                e = tuple(sorted((S[t, (k + 1) % 3], S[t, (k + 2) % 3])))
                edges[e] += 1
        bd = [e for e, m in edges.items() if m == 1]
        vs = sorted(set(v for e in bd for v in e))
        cen = X[vs].mean(0)
        if not inbox(cen) or len(vs) > 12: continue
        # order vertices by angle
        ang = np.arctan2(*(X[vs] - cen).T[::-1])
        vs = [vs[i] for i in np.argsort(ang)]
        polys.append(vs)
    return X, ID, N, polys, lam


def corner_angles(X, poly):
    n = len(poly); out = []
    for i in range(n):
        p, v, q = X[poly[i - 1]], X[poly[i]], X[poly[(i + 1) % n]]
        u, w = p - v, q - v
        out.append(math.degrees(math.acos(np.clip(u @ w / np.linalg.norm(u) / np.linalg.norm(w), -1, 1))))
    return out


def analyze(fn, acute_cut=None):
    X, ID, N, polys, lam = tiles(fn)
    kinds = Counter(len(p) for p in polys)
    quads = [p for p in polys if len(p) == 4]
    qang = np.array([corner_angles(X, p) for p in quads]) if quads else np.zeros((0, 4))
    sdiag = np.array([min(np.linalg.norm(X[p[0]] - X[p[2]]), np.linalg.norm(X[p[1]] - X[p[3]])) for p in quads])
    # vertex environments: (n_tri, n_acute, n_obtuse, n_other)
    env = {}
    for p in polys:
        angs = corner_angles(X, p)
        for v, an in zip(p, angs):
            e = env.setdefault(ID[v], Counter())
            if len(p) == 3: e["t"] += 1
            elif len(p) == 4: e["a" if an < 90 else "o"] += 1
            else: e["x"] += 1
    types = Counter((e["t"], e["a"], e["o"], e["x"]) for e in env.values() if sum(e.values()) > 0)
    tot = sum(types.values())
    ok = sum(v for (t, a, o, x), v in types.items() if x == 0 and a == o)
    return dict(lam=lam, polys=dict(sorted(kinds.items())), n_quads=len(quads), qang=qang, sdiag=sdiag,
                types=types, frac_ok=ok / tot, nvert=tot, Ntri=kinds.get(3, 0))


if __name__ == "__main__":
    for fn in sys.argv[1:]:
        r = analyze(fn)
        q = r["qang"]
        print(f"== {fn}  λ={r['lam']}")
        print("  polygons by #sides:", r["polys"], " N_t/N_q =", round(r["Ntri"] / max(r["n_quads"], 1), 3))
        if len(q):
            acute = q[q < 90]
            print(f"  quad acute-corner angle: mean {acute.mean():.2f}°, median {np.median(acute):.2f}°, sd {acute.std():.2f}°")
            print(f"  quad short diagonal: mean {r['sdiag'].mean():.3f}, median {np.median(r['sdiag']):.3f}, frac < λ: {(r['sdiag'] < r['lam']).mean():.3f}")
        print(f"  vertices obeying 'exactly-one' rule (n_acute = n_obtuse, no defect polygon): {r['frac_ok']:.3f}")
        for k, v in r["types"].most_common(10):
            print(f"    (tri, acute, obtuse, defect) = {k}: {v / r['nvert']:.3f}")
