"""Phase-0 check 4: a known periodic tiling (sigma phase tilted by 10 deg, exact Z[zeta18] coordinates) is a path in the cylinder
transfer matrix built for P = one of its period vectors.  Following the tiling tile by tile through the same split_shift machinery used by
build() must always find exactly one admissible transition whose tile belongs to the tiling, and every visited front must be a state of
the transfer matrix generated from the standard start front (find_d0)."""
import sys, math
import numpy as np
import cfg as C
from zz import cplx, add, sub, star
import cyl, cyl_ref, sigma_phase as SP
from cyl import canon_front, find_d0, PHAT, front_split
from collections import deque
def reduce_lat(A1, A2, vs):
    """canonical representative of a vertex set modulo the 2D lattice Z A1 + Z A2 (exact: integer shifts)"""
    M = np.array([[cplx(A1).real, cplx(A2).real], [cplx(A1).imag, cplx(A2).imag]])
    c = sum(cplx(v) for v in vs) / len(vs); f = np.linalg.solve(M, np.array([c.real, c.imag]))
    n1, n2 = math.floor(f[0] + 1e-9), math.floor(f[1] + 1e-9)
    sh = tuple(-n1 * a - n2 * b for a, b in zip(A1, A2))
    return frozenset(add(v, sh) for v in vs)
def reduce_mod(P, vs):
    """canonical representative of a vertex set modulo translations by multiples of P"""
    pc = cplx(P); ph = pc / abs(pc)
    a = np.mean([(cplx(v) * ph.conjugate()).real for v in vs]); k = math.floor(a / abs(pc))
    sh = tuple(-k * x for x in P)
    return frozenset(add(v, sh) for v in vs)
def run(P, steps=400, verbose=True):
    A1, A2, sites, edges, res = SP.build(10.0)
    tiles, adj = SP.periodic_tiles(A1, A2, sites, edges)
    TS = {reduce_lat(A1, A2, t) for t in tiles}
    PHAT[0] = cplx(P) / abs(cplx(P)); negP = tuple(-x for x in P)
    # shortest tiling-edge path from v0 to v0 + P
    v0 = sites[0]; tgt = add(v0, P); prev = {v0: None}; q = deque([v0])
    while q and tgt not in prev:
        u = q.popleft()
        for w in sorted(adj.get(u, ())):
            if w not in prev: prev[w] = u; q.append(w)
    assert tgt in prev, 'no edge path'
    path = []; u = tgt
    while prev[u] is not None: path.append(u); u = prev[u]
    path.append(v0); path.reverse()
    dirs = []
    for a, b in zip(path[:-1], path[1:]):
        dirs.append([k for k in range(C.n) if add(a, C.E[k]) == b][0])
    r = front_split(list(dirs), P, negP)
    assert r is not None and not r[1], 'path is not a valid front'
    s0 = r[0]
    # reference transfer matrix from the standard start front
    d0 = find_d0(tuple(P)); order, tr = cyl_ref.build(tuple(P), d0); states = set(order)
    # position of the canonical start vertex
    rots = [tuple(dirs[i:] + dirs[:i]) for i in range(len(dirs))]; i0 = rots.index(s0)
    pstart = v0
    for t in range(i0): pstart = add(pstart, C.E[dirs[t]])
    state = s0; visited = []; n, h = C.n, C.half; inTM = 0; seen_pos = {}
    for step in range(steps):
        d = list(state); din, dout = d[-1], d[0]
        turn = (dout - din) % n; turn = turn - n if turn > h else turn
        hits = []
        for t, td, ang in C.TILES:
            if ang > h - turn: continue
            tdir = [(dout + x) % n for x in td]; pts = [pstart]
            for x in tdir[:-1]: pts.append(add(pts[-1], C.E[x]))
            if reduce_lat(A1, A2, pts) not in TS: continue
            rep = [(x + h) % n for x in reversed(tdir[1:])]
            rr = cyl_ref.split_shift(rep + d[1:], tuple(P), negP)
            if rr is None: continue
            hits.append((t, rr, pts))
        if len(hits) != 1: return dict(ok=False, step=step, nhits=len(hits), msg='expected exactly one admissible tiling tile')
        t, rr, pts = hits[0]; f, holes, off = rr
        if holes: return dict(ok=False, step=step, msg='hole encountered (not followed)')
        inTM += (state in states); visited.append(state)
        pstart = add(pstart, off); state = f
    return dict(ok=True, steps=steps, front_len=len(s0), n_states_TM=len(order), visited_in_TM=inTM, distinct_visited=len(set(visited)), P_abs=abs(cplx(P)))
if __name__ == '__main__':
    sys.setrecursionlimit(100000)
    A1, A2, *_ = SP.build(10.0)
    for name, P in (('A1', A1), ('A2', A2), ('A1+A2', add(A1, A2)), ('A1-A2', sub(A1, A2))):
        P = tuple(int(x) for x in P); print(name, P, run(P))
