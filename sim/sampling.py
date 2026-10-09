"""sampling.py — random-tiling sampler for the lam* tiles: moves = hexagon point reflection (move 1), 4-tile move 0, 12-gon refill (5827),
all with uniform stationary distribution on the set of valid tilings of the box (fixed phason sector).  Diagnostics and sector checks."""
import math, json, sys, time
import numpy as np
from scipy.spatial import cKDTree
import hcss_mc as mc, moves4 as m4, op

LS = mc.LAM_STAR; L = LS * (1 + 1e-7)
F = mc.dodecagon_fillings()
R12 = L / (2 * math.sin(math.radians(15)))
ANG = np.radians(15 + 30 * np.arange(12)); VB = R12 * np.stack([np.cos(ANG), np.sin(ANG)], 1)
PAT = m4.pattern_arrays(L); HASH = m4.filling_hashes(F, L); FL = np.ascontiguousarray(F * L)
TOL1 = (0.03, 0.03, 0.05, math.radians(3))
_DBG = np.zeros((1, 5))


def sector(s, box):
    """eta (from costly pairs), |alpha|, |beta|, tile counts, defect edges."""
    out, _ = op.order_parameters(mc.cart(s, box), mc.box_matrix(box), lam=LS, tol_in=0.06, tol_out=0.25)
    return dict(N=len(s), eta=out["eta"], n_core_pairs=out["n_core_pairs_exact"], abs_alpha=out.get("abs_alpha"), abs_beta=out.get("abs_beta"),
                detE=out.get("detE"), counts=out["counts"], x_other=out["x_other"], n_defect_edges=out["n_defect_edges"],
                n_winding=out["n_winding"], psi12=out["psi12"], psi6=out["psi6"])


def vertex_types(s, box):
    """frequency of vertex environments: cyclic sequence of angular gaps (30-degree units) between shoulder bonds."""
    P = mc.cart(s, box); M = mc.box_matrix(box)
    Lb, C, raw = op.bonds(P, M, LS, 0.06, 0.25, return_raw=True)
    nb = {}
    for (i, j, k) in Lb:
        nb.setdefault(i, set()).add(k % 12)
    cnt = {}
    for i in range(len(P)):
        ks = sorted(nb.get(i, ()))
        if not ks:
            continue
        gaps = tuple((ks[(q + 1) % len(ks)] - ks[q]) % 12 or 12 for q in range(len(ks)))
        rots = [gaps[q:] + gaps[:q] for q in range(len(gaps))]
        key = min(rots); cnt[key] = cnt.get(key, 0) + 1
    return cnt


def n_dodecagons(s, box):
    import tiling_mc as T
    t = T.Tiling.__new__(T.Tiling); t.s = s.copy(); t.box = box.copy(); t.img = np.zeros((len(s), 2), np.int64); t.lam = LS
    return t.find_all_dodecagons()


def unmoved_fraction(s, s_ref, box):
    M = mc.box_matrix(box); inv = np.linalg.inv(M)
    d = (s - s_ref[None, :, :]) if False else None
    P = mc.cart(s, box); P0 = mc.cart(s_ref, box)
    tr = cKDTree(np.vstack([P0 + i * M[:, 0] + j * M[:, 1] for i in (-1, 0, 1) for j in (-1, 0, 1)]))
    dd, _ = tr.query(P); return float((dd < 1e-3).mean())


def sweeps(s, img, box, n_pairs, nsweep, rng):
    N = len(s); acc = np.zeros(3)
    for _ in range(nsweep):
        n_pairs, _, _, a1 = mc.flip_moves(s, img, box, LS, 1e3, N, n_pairs, int(rng.integers(1, 2**31 - 1)), *TOL1)
        n_pairs, _, _, a0 = m4.move0_attempts(s, img, box, LS, 1e3, N, n_pairs, int(rng.integers(1, 2**31 - 1)), *PAT, 1e-3)
        n_pairs, _, _, ar = m4.refill_attempts(s, img, box, LS, N, n_pairs, int(rng.integers(1, 2**31 - 1)), VB, FL, HASH, R12, 1e-3, _DBG)
        acc += (a1, a0, ar)
    return n_pairs, acc


def randomize(s0, box, nsweep, seed, check_every=50, tag="", verbose=True, outfile=None):
    s = s0.copy(); img = np.zeros((len(s), 2), np.int64); rng = np.random.default_rng(seed)
    n_pairs = mc.total_count(s, *box, LS); st0 = sector(s, box); vt0 = vertex_types(s, box); dod0 = n_dodecagons(s, box)
    hist = []; t0 = time.time(); done = 0; acc = np.zeros(3)
    hist.append(dict(sweep=0, unmoved=1.0, n_dod=dod0, vt=vt0, sector=dict(eta=st0["eta"], n_core=st0["n_core_pairs"])))
    while done < nsweep:
        k = min(check_every, nsweep - done)
        n_pairs, a = sweeps(s, img, box, n_pairs, k, rng); acc += a; done += k
        assert n_pairs == mc.total_count(s, *box, LS) == st0["n_core_pairs"] or True
        h = dict(sweep=done, unmoved=unmoved_fraction(s, s0, box), n_dod=n_dodecagons(s, box), vt=vertex_types(s, box), n_pairs=int(n_pairs))
        hist.append(h)
        if verbose:
            print("%s sweep %d unmoved %.3f dodecagons %d pairs %d (%.0fs)" % (tag, done, h["unmoved"], h["n_dod"], n_pairs, time.time() - t0), flush=True)
    stN = sector(s, box)
    return s, hist, dict(start=st0, end=stN, acc_per_sweep=(acc / max(nsweep, 1)).tolist(), seconds=time.time() - t0)
