"""moves4.py — the "move 0" local 4-tile rearrangement (data/local_moves_4tile.json, move 0) as an MC move.
Region: unit-edge hexagon that is NOT centrally symmetric, 2 triangles + 2 rhombi; the two fillings differ by the position of ONE
interior vertex (P_A <-> P_B, |P_A-P_B| = core distance).  Move 1 of the catalogue (centrally symmetric hexagon) is
hcss_mc.flip_moves (point reflection).
Pattern p = 2*k + role: rotation k = 0..11 of the template by 30 deg*k, role 0/1 = the moving particle sits at P_A / P_B.
PAT_REL[p] = the six boundary vertices relative to the moving particle (units of the tile edge), PAT_D[p] = displacement to the other position.
Proposal (as flip_moves): pick particle i uniformly, pick one of its n_old matching patterns uniformly, move; accept
min(1, n_old/n_new) if the new state has no core overlap and dU = 0 (beta large); reverse pattern (k, 1-role) must be found at the new position."""
import json, math, os
import numpy as np
from numba import njit
import hcss_mc as mc
from hcss_mc import build_cells, local_count, _neigh, _move_cell, MAXNB, total_count

_HERE = os.path.dirname(os.path.abspath(__file__))


def _template():
    d = json.load(open(os.path.join(_HERE, "local_moves_4tile.json")))["moves"][0]["tilings"]
    pts = [{tuple(np.round(v, 6)): np.array(v, float) for t in til for v in t["vertices_xy"]} for til in d]
    common = set(pts[0]) & set(pts[1])
    A = [pts[0][k] for k in pts[0] if k not in common][0]; B = [pts[1][k] for k in pts[1] if k not in common][0]
    assert len(common) == 6 and len(pts[0]) == 7 and len(pts[1]) == 7
    bd = np.array([pts[0][k] for k in sorted(common)])
    # snap the 9-digit JSON values to exact Z[zeta12] values: x,y of the vertices are k*cos30 + l*0.5 combinations
    return bd, A, B


def patterns(include_mirror=False):
    bd, A, B = _template()
    pats = []
    def rot(k, mirror):
        th = math.radians(30 * k); R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        if mirror:
            R = R @ np.diag([1.0, -1.0])
        return R
    out = []
    for mirror in ((False, True) if include_mirror else (False,)):
        for k in range(12):
            R = rot(k, mirror)
            b = bd @ R.T; a = R @ A; bb = R @ B
            out.append((b - a, bb - a))     # role 0: moving particle at A
            out.append((b - bb, a - bb))    # role 1: at B
    return out


def pattern_arrays(lam, include_mirror=False):
    P = patterns(include_mirror)
    rel = np.array([p[0] for p in P]) * lam; d = np.array([p[1] for p in P]) * lam
    return rel, d


def _canon(P):
    """hashable description of a placed region: boundary set (round) -> used to test whether mirror patterns are new"""
    return None


@njit(cache=True)
def _match_patterns(i, x0, x1, s, a, b, c, lam, nx, ny, count, members, pat_rel, pat_d, tol, matched):
    ids = np.empty(MAXNB, np.int64); rel = np.empty((MAXNB, 2))
    m = _neigh(i, x0, x1, s, a, b, c, 1.7 * lam, nx, ny, count, members, ids, rel)
    nm = 0
    for p in range(pat_rel.shape[0]):
        ok = True
        for v in range(6):
            found = False
            for k in range(m):
                dx = rel[k, 0] - pat_rel[p, v, 0]; dy = rel[k, 1] - pat_rel[p, v, 1]
                if dx * dx + dy * dy < tol * tol:
                    found = True; break
            if not found:
                ok = False; break
        if ok:
            matched[nm] = p; nm += 1
    return nm


@njit(cache=True)
def move0_attempts(s, img, box, lam, beta, n_attempt, n_pairs, seed_state, pat_rel, pat_d, tol):
    """returns (n_pairs, seen, proposed, accepted)"""
    np.random.seed(seed_state)
    a, b, c = box[0], box[1], box[2]
    N = s.shape[0]
    nx, ny, count, members, cell_of, slot_of = build_cells(s, a, b, c, lam)
    if nx < 5 or ny < 5:
        return n_pairs, 0, 0, 0
    npat = pat_rel.shape[0]
    m1 = np.empty(npat, np.int64); m2 = np.empty(npat, np.int64)
    seen = 0; prop = 0; acc = 0
    for t in range(n_attempt):
        i = np.random.randint(N)
        n_old = _match_patterns(i, s[i, 0], s[i, 1], s, a, b, c, lam, nx, ny, count, members, pat_rel, pat_d, tol, m1)
        if n_old == 0:
            continue
        seen += 1
        p = m1[np.random.randint(n_old)]
        dx = pat_d[p, 0]; dy = pat_d[p, 1]
        dsy = dy / c; dsx = (dx - b * dsy) / a
        rx = s[i, 0] + dsx; ry = s[i, 1] + dsy
        fx = math.floor(rx); fy = math.floor(ry)
        nsx = rx - fx; nsy = ry - fy
        nnew = local_count(i, nsx, nsy, s, a, b, c, lam, nx, ny, count, members)
        if nnew < 0:
            continue
        prop += 1
        n_new = _match_patterns(i, nsx, nsy, s, a, b, c, lam, nx, ny, count, members, pat_rel, pat_d, tol, m2)
        rev = False
        for z in range(n_new):
            if m2[z] == (p ^ 1):
                rev = True
        if not rev:
            continue
        nold = local_count(i, s[i, 0], s[i, 1], s, a, b, c, lam, nx, ny, count, members)
        dU = nnew - nold
        arg = math.log(n_old / n_new) - beta * dU
        if arg >= 0 or np.random.random() < math.exp(arg):
            s[i, 0] = nsx; s[i, 1] = nsy
            img[i, 0] += int(fx); img[i, 1] += int(fy)
            newc = (int(nsx * nx) % nx) + nx * (int(nsy * ny) % ny)
            _move_cell(i, newc, count, members, cell_of, slot_of)
            n_pairs += dU
            acc += 1
    return n_pairs, seen, prop, acc


# ---------------------------------------------------------------------------- 12-gon refill with state-independent proposal
# Proposal: pick particle i uniformly and k in 0..11 uniformly; candidate centre c = x_i - VB_k.  A given 12-gon centre is hit by
# exactly its 12 boundary vertices, so P(centre) = 1/N whatever the state.  If the 12 boundary vertices are present and the 13
# interior particles form one of the 5827 fillings, a filling j is chosen uniformly (forward and reverse probabilities are equal,
# acceptance 1 provided the new state has the same number of shoulder pairs and no overlap).
@njit(cache=True)
def _hash_pts(P, n):
    key = np.empty(n, np.int64)
    for q in range(n):
        key[q] = int(round(P[q, 0] * 1000.0)) * 1000003 + int(round(P[q, 1] * 1000.0))
    key.sort()
    h = np.int64(1469598103)
    for q in range(n):
        h = (h * np.int64(1099511628211) + key[q]) & np.int64(0x7fffffffffffffff)
    return h


def filling_hashes(F, L):
    return np.sort(np.array([_hash_pts(np.ascontiguousarray(f * L), 13) for f in F], np.int64))


@njit(cache=True)
def refill_attempts(s, img, box, lam, n_attempt, n_pairs, seed_state, VB, FL, hashes, R12, tol, dbg):
    """FL = fillings (nF,13,2) already multiplied by the tile edge L; VB = 12 boundary vertex vectors (relative to the centre)."""
    np.random.seed(seed_state)
    a, b, c = box[0], box[1], box[2]
    N = s.shape[0]
    nx, ny, count, members, cell_of, slot_of = build_cells(s, a, b, c, lam)
    if nx < 5 or ny < 5:
        return n_pairs, 0, 0, 0
    nF = FL.shape[0]
    ids = np.empty(MAXNB, np.int64); rel = np.empty((MAXNB, 2)); inner = np.empty(16, np.int64); ipos = np.empty((13, 2))
    seen = 0; prop = 0; acc = 0
    for t in range(n_attempt):
        i = np.random.randint(N); k = np.random.randint(12)
        # centre in fractional coordinates
        dx = -VB[k, 0]; dy = -VB[k, 1]
        dsy = dy / c; dsx = (dx - b * dsy) / a
        x0 = s[i, 0] + dsx; x1 = s[i, 1] + dsy
        x0 -= math.floor(x0); x1 -= math.floor(x1)
        m = _neigh(-1, x0, x1, s, a, b, c, R12 + 0.1, nx, ny, count, members, ids, rel)
        ok = True
        for q in range(12):
            f = False
            for z in range(m):
                ddx = rel[z, 0] - VB[q, 0]; ddy = rel[z, 1] - VB[q, 1]
                if ddx * ddx + ddy * ddy < tol * tol:
                    f = True; break
            if not f:
                ok = False; break
        if not ok:
            continue
        ni = 0
        for z in range(m):
            if rel[z, 0] ** 2 + rel[z, 1] ** 2 < (R12 - 0.05) ** 2:
                if ni < 13:
                    inner[ni] = ids[z]; ipos[ni, 0] = rel[z, 0]; ipos[ni, 1] = rel[z, 1]
                ni += 1
        if ni != 13:
            continue
        h = _hash_pts(ipos, 13)
        lo = 0; hi = nF - 1; found = False
        while lo <= hi:
            mid = (lo + hi) // 2
            if hashes[mid] == h:
                found = True; break
            elif hashes[mid] < h:
                lo = mid + 1
            else:
                hi = mid - 1
        if not found:
            continue
        seen += 1
        j = np.random.randint(nF)
        # new interior positions (assign to the 13 ids in order); fractional centre -> cartesian
        old = np.empty((13, 2))
        for q in range(13):
            old[q, 0] = s[inner[q], 0]; old[q, 1] = s[inner[q], 1]
        for q in range(13):
            ddx = FL[j, q, 0]; ddy = FL[j, q, 1]
            dsy2 = ddy / c; dsx2 = (ddx - b * dsy2) / a
            nx_ = x0 + dsx2; ny_ = x1 + dsy2
            nx_ -= math.floor(nx_); ny_ -= math.floor(ny_)
            s[inner[q], 0] = nx_; s[inner[q], 1] = ny_
        nn = total_count(s, a, b, c, lam)
        if prop < dbg.shape[0]:
            dbg[prop, 0] = x0; dbg[prop, 1] = x1; dbg[prop, 2] = j; dbg[prop, 3] = nn; dbg[prop, 4] = n_pairs
        prop += 1
        if nn == n_pairs:
            acc += 1
            nx, ny, count, members, cell_of, slot_of = build_cells(s, a, b, c, lam)
        else:
            for q in range(13):
                s[inner[q], 0] = old[q, 0]; s[inner[q], 1] = old[q, 1]
    return n_pairs, seen, prop, acc
