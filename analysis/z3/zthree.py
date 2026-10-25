"""Z3 family analysis of triangle–rhombus(≈80°) tilings from HCSS snapshots.
For each clean vertex (only triangles and quads around it): cyclic corner sequence (t/a/o),
family winding w = #a - #o (each acute corner: family +1, obtuse: -1, triangle: 0),
number of distinct families visited, and arrangement class of (0,2,2) vertices (aoao vs aaoo).
Rhombus 'pair label' = {f, f+1} from its edge directions (direction/20° mod 3, after removing the global offset)."""
import sys, math
import numpy as np
from collections import Counter, defaultdict
from analyze import tiles, corner_angles


def run(fn, verbose=True):
    X, ID, N, polys, lam = tiles(fn)
    # global 20° offset from bond directions
    E = set()
    for p in polys:
        if len(p) in (3, 4):
            for i in range(len(p)):
                E.add(tuple(sorted((p[i], p[(i + 1) % len(p)]))))
    d = np.array([math.degrees(math.atan2(*(X[b] - X[a])[::-1])) % 180 for a, b in E])
    off = (np.degrees(np.angle(np.mean(np.exp(1j * 2 * np.pi * (d % 20) / 20)))) / 360 * 20) % 20
    fam = lambda a, b: int(np.floor(((math.degrees(math.atan2(*(X[b] - X[a])[::-1])) - off + 10) % 60) / 20))
    # vertex -> list of (angle of corner bisector, kind)
    around = defaultdict(list)
    quad_pair = {}
    for qi, p in enumerate(polys):
        angs = corner_angles(X, p)
        n = len(p)
        for i, v in enumerate(p):
            c = X[p].mean(0) - X[v]
            kind = "t" if n == 3 else ("a" if angs[i] < 90 else "o") if n == 4 else "x"
            around[ID[v]].append((math.atan2(c[1], c[0]), kind))
        if n == 4:
            fs = sorted({fam(p[0], p[1]), fam(p[1], p[2])})
            quad_pair[qi] = tuple(fs)
    stats = Counter(); arr = Counter(); wind = Counter()
    for v, L in around.items():
        kinds = [k for _, k in sorted(L)]
        if "x" in kinds or len(kinds) < 3:
            continue
        a, o, t = kinds.count("a"), kinds.count("o"), kinds.count("t")
        if abs(60 * t + 80 * a + 100 * o - 360) > 1e-9:
            continue
        # families visited along the cyclic sequence
        f = 0; seen = {0}
        for k in kinds:
            f += {"t": 0, "a": 1, "o": -1}[k]; seen.add(f % 3)
        w = a - o
        stats[(t, a, o)] += 1; wind[w] += 1
        if (t, a, o) == (0, 2, 2):
            s = "".join(kinds)
            arr["aaoo" if ("aa" in s + s[0]) else "aoao"] += 1
        stats[("nfam", len(seen))] += 1
    pairs = Counter(quad_pair.values())
    if verbose:
        tot = sum(v for k, v in stats.items() if k[0] != "nfam")
        print(f"== {fn}  offset {off:.1f}°  clean flat vertices {tot}")
        print("   family winding (a−o):", dict(sorted(wind.items())), "  → vortex +1 (E) / −1 (D)")
        print("   vertices visiting 2 families:", stats[('nfam', 2)], " 3 families:", stats[('nfam', 3)], " 1:", stats[('nfam', 1)])
        print("   (0,2,2) arrangements:", dict(arr))
        print("   rhombus pair labels:", {"".join(map(str,k)): n for k, n in sorted(pairs.items())})
    return stats, arr, wind, pairs


if __name__ == "__main__":
    for fn in sys.argv[1:]:
        run(fn)
