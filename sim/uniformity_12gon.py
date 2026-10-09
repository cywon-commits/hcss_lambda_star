"""Step 1 validation: does move 0 + move 1 (no refill) sample the 5827 fillings of an isolated 12-gon uniformly?
Isolated 12-gon (12 fixed boundary vertices + 13 mobile interior vertices) in a large periodic box.
Control: move 1 only (should be stuck in the connected component of the start filling).
Usage: python3 uniformity_12gon.py [R chains] [snapshots per chain] [moves: 01|1|0]
Statistic: R * V_between / V_within with V_between = sum_j (cbar_j - mu)^2, V_within = mean_r sum_j (c_rj - mu)^2.
It is ~1 for an unbiased sampler (chains independent) and >> 1 if the stationary distribution is not uniform."""
import sys, json, math, numpy as np
import hcss_mc as mc, moves4 as m4

LAM = mc.LAM_STAR; L = LAM * (1 + 1e-7)
F = mc.dodecagon_fillings(); nF = len(F)
FK = {tuple(sorted((round(x, 3) + 0.0, round(y, 3) + 0.0) for x, y in f * L)): j for j, f in enumerate(F)}
PAT = m4.pattern_arrays(L)
R12 = L / (2 * math.sin(math.radians(15)))
VB = np.array([[R12 * math.cos(math.radians(15 + 30 * k)), R12 * math.sin(math.radians(15 + 30 * k))] for k in range(12)])
TOL = (0.03, 0.03, 0.05, math.radians(3))


def run(seed, nsnap, moves, start=0, every=40, box_side=40.0):
    O = np.array([box_side / 2, box_side / 2])
    pts = np.vstack([O + VB, O + F[start] * L]); box = np.array([box_side, 0.0, box_side])
    s = (pts / box_side) % 1.0; img = np.zeros((len(s), 2), np.int64)
    n = mc.total_count(s, *box, LAM); n0 = n
    cnt = np.zeros(nF, np.int64); rng = np.random.default_rng(seed); acc0 = acc1 = 0
    for k in range(nsnap):
        if '0' in moves:
            n, _, _, a = m4.move0_attempts(s, img, box, LAM, 1e3, every, n, int(rng.integers(1, 2**31 - 1)), *PAT, 1e-3); acc0 += a
        if '1' in moves:
            n, _, _, a = mc.flip_moves(s, img, box, LAM, 1e3, every, n, int(rng.integers(1, 2**31 - 1)), *TOL); acc1 += a
        rel = (s[12:] * box_side - O) if False else None
        X = s[:, :] * box_side - O
        X = X[(np.linalg.norm(X, axis=1) < R12 - 0.05)]
        key = tuple(sorted((round(x, 3) + 0.0, round(y, 3) + 0.0) for x, y in X))
        cnt[FK[key]] += 1
    assert n == n0
    return cnt, acc0, acc1


if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    nsnap = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
    moves = sys.argv[3] if len(sys.argv) > 3 else '01'
    C = []; out = dict(moves=moves, R=R, nsnap=nsnap)
    for r in range(R):
        c, a0, a1 = run(1000 + r, nsnap, moves, start=int(np.random.default_rng(r).integers(nF)) if moves != '1' else 0)
        C.append(c); print('chain', r, 'visited', int((c > 0).sum()), 'of', nF, 'acc0', a0, 'acc1', a1, flush=True)
    C = np.array(C, float); mu = C.sum(1, keepdims=True) / nF
    chi_each = ((C - mu) ** 2 / mu).sum(1)
    cbar = C.mean(0); Vb = ((cbar - mu.mean()) ** 2).sum(); Vw = (((C - mu) ** 2).sum(1)).mean()
    out.update(visited_per_chain=[int((c > 0).sum()) for c in C], chi2_per_chain=chi_each.tolist(), dof=nF - 1,
               ratio=float(R * Vb / Vw), mean_count=float(mu.mean()),
               union_visited=int((C.sum(0) > 0).sum()), min_count=float(C.sum(0).min()), max_count=float(C.sum(0).max()))
    print(json.dumps(out, indent=1))
    json.dump(out, open('../runs/uniformity_%s.json' % moves, 'w'), indent=1)
