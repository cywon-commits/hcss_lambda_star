"""Tests for move 0 (moves4.py). python -m pytest -q test_move0.py"""
import math, numpy as np
import hcss_mc as mc, moves4 as m4

LAM = mc.LAM_STAR
L = LAM * (1 + 1e-7)
TOL = (0.08, 0.20, 0.30, math.radians(15))
PAT = m4.pattern_arrays(L)


def move0_hexagon_state(box_side=40.0, noise=0.0, seed=0):
    bd, A, B = m4._template()
    O = np.array([box_side / 2, box_side / 2])
    pts = np.vstack([O + L * bd, O + L * A])           # interior particle last
    if noise > 0:
        pts = pts + np.random.default_rng(seed).normal(0, noise, pts.shape)
    box = np.array([box_side, 0.0, box_side])
    return (pts / box_side) % 1.0, box, O + L * A, O + L * B


def test_patterns_24_and_mirror_closed():
    rel, d = m4.pattern_arrays(1.0); rel2, d2 = m4.pattern_arrays(1.0, True)
    r3 = lambda x: tuple((np.round(x, 3) + 0.0).ravel())
    key = lambda r, dd: (tuple(sorted(r3(v) for v in r)), r3(dd))
    S1 = {key(r, dd) for r, dd in zip(rel, d)}; S2 = {key(r, dd) for r, dd in zip(rel2, d2)}
    assert len(S1) == 24 and S2 == S1


def test_isolated_move0_half_occupancy():
    s, box, XA, XB = move0_hexagon_state()
    img = np.zeros((len(s), 2), np.int64); n = mc.total_count(s, *box, LAM)
    occ = []
    for k in range(3000):
        n, _, _, _ = m4.move0_attempts(s, img, box, LAM, 1e3, 9, n, 1 + k, *PAT, 1e-3)
        x = s[-1] * box[0]; occ.append(int(np.linalg.norm(x - XA) < 0.1 * L))
        assert min(np.linalg.norm(x - XA), np.linalg.norm(x - XB)) < 1e-6
    assert abs(np.mean(occ) - 0.5) < 0.05 and n == mc.total_count(s, *box, LAM)


def test_move0_conserves_pairs_in_random_tiling():
    t = __import__("tiling_mc").Tiling("dodeca", 400, seed=2)
    img = t.img
    n0 = t.n_pairs; tot = 0; acc = 0
    for k in range(200):
        t.n_pairs, seen, prop, a = m4.move0_attempts(t.s, img, t.box, t.lam, 1e3, 200, t.n_pairs, 10 + k, *PAT, 1e-3)
        tot += seen; acc += a
        t.n_pairs, _, _, _ = mc.flip_moves(t.s, img, t.box, t.lam, 1e3, 200, t.n_pairs, 500 + k, 0.03, 0.03, 0.05, math.radians(3))
    assert acc > 0 and t.n_pairs == n0 == mc.total_count(t.s, *t.box, t.lam)
