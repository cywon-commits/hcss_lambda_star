"""Map an exact z18 sample (sampler.py) onto the square-triangle tiling by the family rotation (B family turned by -10 deg) and check that the image is a valid
square-triangle tiling of the cylinder P0: consistent development (mod P0), triangle area 0.433, square area 1, no overlaps (shapely, periodic images).
Usage: st12_map_sample.py sample.npz a0..a5 [ntiles_for_overlap_test]"""
import sys, math, collections, json
import numpy as np
from shapely.geometry import Polygon
from shapely.strtree import STRtree
z18 = np.exp(2j * np.pi / 18); z12 = np.exp(2j * np.pi / 12)
def check(f, a, nov=4000):
    a = np.array(a); P = a @ z18 ** np.arange(6); ph = P / abs(P); S = np.load(f); V = S['V'].astype(int); names = S['names']
    cx = lambda v: v @ z18 ** np.arange(6)
    def fold(v): k = math.floor((cx(v) * ph.conjugate()).real / abs(P)); return tuple(int(x) for x in (v - k * a))
    tiles = []
    for t in range(len(names)):
        nv = 3 if names[t] == 'T' else 4; v = V[t][:nv]; k = math.floor((sum(cx(x) for x in v) / nv * ph.conjugate()).real / abs(P)); tiles.append([tuple(int(x) for x in (u - k * a)) for u in v])
    def dirj(u, w): e = cx(np.array(w)) - cx(np.array(u)); return int(round(math.degrees(math.atan2(e.imag, e.real)) / 20)) % 18
    cl = sorted({dirj(t[i], t[(i + 1) % len(t)]) % 3 for t in tiles for i in range(len(t))})
    if len(cl) != 2: return dict(file=f, a=a.tolist(), families=len(cl), mapped=False)
    fam = (cl[0], cl[1]) if (cl[1] - cl[0]) % 3 == 1 else (cl[1], cl[0])
    to12 = lambda j: 2 * (((j - fam[0]) % 18) // 3) if j % 3 == fam[0] else 2 * (((j - 1 - fam[0]) % 18) // 3) + 1
    steps = [i if x > 0 else i + 9 for i, x in enumerate(a) for _ in range(abs(x))]; P0 = sum(z12 ** to12(j) for j in steps)
    adj = collections.defaultdict(list)
    for t in tiles:
        n = len(t)
        for i in range(n):
            u, w = fold(np.array(t[i])), fold(np.array(t[(i + 1) % n])); e = z12 ** to12(dirj(t[i], t[(i + 1) % n])); adj[u].append((w, e)); adj[w].append((u, -e))
    pos = {}; s0 = next(iter(adj)); pos[s0] = 0j; q = [s0]
    while q:
        u = q.pop()
        for w, e in adj[u]:
            if w not in pos: pos[w] = pos[u] + e; q.append(w)
    bad = 0
    for u in pos:
        for w, e in adj[u]:
            d = pos[w] - pos[u] - e; kk = (d * P0.conjugate()).real / abs(P0) ** 2
            if abs(d - round(kk) * P0) > 1e-8: bad += 1
    ph0 = P0 / abs(P0); polys = []
    for t in tiles:
        p = [pos[fold(np.array(v))] for v in t]; pc = [p[0]]
        for x in p[1:]: pc.append(x - round(((x - p[0]) * P0.conjugate()).real / abs(P0) ** 2) * P0)
        c = sum(pc) / len(pc); kc = math.floor((c * ph0.conjugate()).real / abs(P0)); polys.append(Polygon([((x - kc * P0).real, (x - kc * P0).imag) for x in pc]))
    allp = [Polygon([(x + sh * P0.real, y + sh * P0.imag) for x, y in pg.exterior.coords]) for sh in (-1, 0, 1) for pg in polys]; tree = STRtree(allp); over = 0
    for i, pg in enumerate(polys[:nov]):
        for j in tree.query(pg):
            if j != i + len(polys) and allp[j].intersection(pg).area > 1e-7: over += 1
    areas = collections.Counter(round(p.area, 3) for p in polys)
    return dict(file=f, a=a.tolist(), families=2, mapped=True, P0=[P0.real, P0.imag], inconsistent_edges=bad, areas=dict(areas), overlaps=over, tested=min(nov, len(polys)))
if __name__ == '__main__':
    print(json.dumps(check(sys.argv[1], [int(x) for x in sys.argv[2:8]], int(sys.argv[8]) if len(sys.argv) > 8 else 4000)))
