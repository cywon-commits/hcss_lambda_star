"""Phase-0 verification of the generalised reference code.   cd reference80 && python3 -m pytest -q test_phase0.py
(the three large lambda* regression cylinders are run by run_regression.sh)"""
import sys, os, random, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.setrecursionlimit(100000)
import numpy as np, pytest
import cfg as C
HERE = os.path.dirname(os.path.abspath(__file__))
def _sel(name):
    C.select(name)
    import front; return front
def test_tiles_z18():
    C.select('z18')
    from zz import add
    for name, offs, ang in C.TILES:
        s = (0,) * 6
        for k in offs: s = add(s, C.E[k])
        assert s == (0,) * 6
        angs = []
        for i in range(len(offs)):
            turn = (offs[i] - offs[i - 1]) % 18; turn = turn - 18 if turn > 9 else turn; angs.append((9 - turn) * 20)
        assert angs[0] == {'T': 60, 'R80': 80, 'R100': 100}[name] and sum(angs) == 180 * (len(offs) - 2)
        if name != 'T': assert angs in ([80, 100, 80, 100], [100, 80, 100, 80])
    assert C.E[6] == (-1, 0, 0, 1, 0, 0) and C.E[7] == (0, -1, 0, 0, 1, 0) and C.E[8] == (0, 0, -1, 0, 0, 1)
def test_region_counts_lam12():
    front = _sel('lam12')
    S = [0, 1, -3, 1, 0]; infl = lambda poly: [(d + s) % 12 for d in poly for s in S]
    checks = [([0, 4, 8], 1), ([0, 1, 6, 7], 1), ([0, 1, 2, 6, 7, 8], 2), (infl([0, 4, 8]), 20), (infl([0, 1, 6, 7]), 23), (list(range(12)), 5827),
              ([7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5] + list(range(1, 12)), 38120461), ([10, 2] + list(range(1, 12)), 6865)]
    for d, e in checks: assert front.fill_count(front.canon_closed(d))[0] == e
def test_18gon_and_independent():
    front = _sel('z18'); import indep_count as I
    d = list(range(18)); c, vw, vec = front.fill_info(front.canon_closed(d))
    assert c == 6 and vec == (6, 6, 6) + (2,) * 9 and I.count(d) == 6
    assert front.lifted_cls(front.canon_closed(d))[0].tolist() == list(vec)
def test_12gon_independent():
    front = _sel('lam12'); import indep_count as I
    d = list(range(12)); assert front.fill_count(front.canon_closed(d))[0] == I.count(d) == 5827
def test_patches_conservation_and_small_regions():
    front = _sel('z18'); import patches as Pa, indep_count as I
    rng = random.Random(11); ncomp = 0
    for k in range(2, 11):
        for t in range(12):
            r = Pa.random_patch(k, rng)
            if r is None: continue
            tiles, U = r; d = Pa.boundary_loop(tiles, U); cd = front.canon_closed(d)
            c, vw, vec = front.fill_info(cd)
            assert sum(vec) == k                                  # class vector accounts for every tile
            if k <= 5: assert c == 1                             # corollary 2: no local rearrangement with <= 5 tiles
            lc = front.lifted_cls(cd); assert lc[2] and tuple(lc[0]) == tuple(vec) and lc[1] < 1e-6   # lifted-area linear system reproduces the class counts
            if k <= 7 and t < 3: assert I.count(d) == c; ncomp += 1
    assert not front.VIOLATIONS and ncomp > 0
def test_sigma_phase_followed_through_transfer_matrix():
    _sel('z18'); import follow_tiling as F, sigma_phase as SP
    A1, A2, *_ = SP.build(10.0)
    for P in (A2, A1):
        r = F.run(tuple(int(x) for x in P), steps=200)
        assert r['ok'] and r['visited_in_TM'] == 200
def test_regression_small_lam12_cylinders():
    front = _sel('lam12'); import cyl, cyl_ref
    from zz import cplx
    for a, n in (((2, 2, 0, -1), 664), ((2, 3, 0, -1), 4453)):
        ref = np.load(os.path.join(HERE, '..', 'reference', 'data', 'ref_%d_%d_%d_%d.npz' % a))
        cyl.PHAT[0] = cplx(a) / abs(cplx(a)); d0 = cyl.find_d0(a); order, tr = cyl_ref.build(a, d0)
        assert len(order) == int(ref['n']) == n and len(tr) == len(ref['fr'])
        assert [t[0] for t in tr] == list(ref['fr']) and [t[1] for t in tr] == list(ref['to']) and [t[2] for t in tr] == list(ref['c']) and [t[3] for t in tr] == list(ref['w'])
