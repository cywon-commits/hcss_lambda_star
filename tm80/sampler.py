"""Exact (Boltzmann, uniform at fixed vertex number) sampler of z18 tilings of a cylinder, for one branching SCC of the transfer graph.
Walk: P(front i -> j via tile t and uniformly chosen hole fillings) = c-weighted  val * l_j / l_i  (val = exp(lu*w), rho = 1; l = left Perron vector),
exactly the Markov measure of the infinite uniform tiling (microcanonical in the vertex number, phi = 0).  Tiles are written with absolute
integer coordinates in Z^6.  Usage:  python3 tm80/sampler.py a0..a5 steps seed out.npz"""
import sys, os, math, random, time
os.environ.setdefault('HCSS_CFG', 'z18')
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); sys.path.insert(0, os.path.join(ROOT, 'reference80'))
import numpy as np
import cfg as C
from zz import add, cplx, star
import front
from front import _fill, cancel, pos, crosses, area, tile_class, fill_info
import cyl, cyl_ref, tilt
from cyl import lift, PHAT, find_d0, canon_front
from cyl_ref import cancel_shift
sys.setrecursionlimit(100000)
Z0 = tuple(0 for _ in range(6))
def pieces(nd, q):
    """closed_pieces with absolute start positions: list of (canonical loop, start position) or None"""
    n = C.n; nd = list(nd); ch = True
    while ch and len(nd) >= 2:
        ch = False
        for i in range(len(nd)):
            j = (i + 1) % len(nd)
            if (nd[i] - nd[j]) % n == C.half:
                if j == 0: q = add(q, C.E[nd[0]])
                nd = [x for k, x in enumerate(nd) if k not in (i, j)]; ch = True; break
    if not nd: return []
    P = [q]
    for x in nd: P.append(add(P[-1], C.E[x]))
    seen = {}
    for i, p in enumerate(P[:-1]):
        if p in seen:
            i0 = seen[p]; a = pieces(nd[i0:i], P[i0]); b = pieces(nd[:i0] + nd[i:], P[0])
            return None if a is None or b is None else a + b
        seen[p] = i
    Cc = [cplx(p) for p in P]
    if crosses([(Cc[i], Cc[i + 1]) for i in range(len(nd))]) or area(nd) < 1e-9: return None
    ii = min(range(len(P) - 1), key=lambda k: (round(Cc[k].imag, 9), round(Cc[k].real, 9)))
    return [(nd[ii:] + nd[:ii], P[ii])]
def split_pos(d, q0, Pv, negP):
    """cyl_ref.split_shift with absolute positions: (front, [(hole loop, start pos)], off)"""
    d, st = cancel_shift(d); q0 = add(q0, st)
    P = lift(d); idx = {}
    for i, p in enumerate(P[:-1]): idx.setdefault(p, []).append(i)
    for i, p in enumerate(P[:-1]):
        for q, typ in ((p, 0), (add(p, Pv), 1)):
            for j in idx.get(q, []):
                if j <= i: continue
                if typ == 0: hole = d[i:j]; hq = add(q0, P[i]); rest = d[:i] + d[j:]; rq = q0; rst = Z0
                else: hole = d[:i] + d[j:]; hq = q0; rest = d[i:j]; rq = add(q0, P[i]); rst = P[i]
                hp = pieces(hole, hq)
                if hp is None: return None
                r = split_pos(rest, rq, Pv, negP)
                if r is None: return None
                return (r[0], r[1] + hp, add(st, add(rst, r[2])))
        if add(p, negP) in idx: return None
    Cc = [cplx(p) for p in P]; pc = cplx(Pv); n = len(d); segs = []
    for sh in (-pc, 0, pc): segs += [(Cc[k] + sh, Cc[k + 1] + sh) for k in range(n)]
    if crosses(segs): return None
    Q = P[:-1]; ys = [round((cplx(p) * PHAT[0].conjugate()).imag, 9) for p in Q]; m = min(ys)
    f, i = min((tuple(d[i:] + d[:i]), i) for i in range(len(d)) if ys[i] == m)
    return (f, [], add(st, Q[i]))
def sample_fill(loop, p0, rng, out):
    """uniformly random filling of the closed loop (start position p0); tiles appended to out as (name, class, vertex list)"""
    if not loop: return
    n, h = C.n, C.half; loop = list(loop)
    din, dout = loop[-1], loop[0]; turn = (dout - din) % n; turn = turn - n if turn > h else turn
    opts = []
    for t, td, ang in C.TILES:
        if ang > h - turn: continue
        tdir = [(dout + x) % n for x in td]; pts = [p0]
        for x in tdir[:-1]: pts.append(add(pts[-1], C.E[x]))
        rep = [(x + h) % n for x in reversed(tdir[1:])]
        ps = pieces(rep + loop[1:], p0)
        if ps is None: continue
        c = 1
        for lp, q in ps:
            c *= _fill(tuple(lp))[0]
            if not c: break
        if c: opts.append((c, t, pts, ps))
    tot = sum(o[0] for o in opts); u = rng.random() * tot
    for c, t, pts, ps in opts:
        u -= c
        if u < 0: break
    out.append((t, tile_class(t, dout), pts))
    for lp, q in ps: sample_fill(lp, q, rng, out)
def prepare(a, solve_largest=True):
    a = tuple(a); PHAT[0] = cplx(a) / abs(cplx(a)); d0 = find_d0(a); order, tr = cyl_ref.build(a, d0)
    fr, to, c, w, ar, off, na, cls = zip(*tr)
    off = np.array(off); a_ = np.array([(cplx(o) * np.conj(PHAT[0])).real for o in off]); Pc = cplx(a)
    qs = [np.array([star(tuple(o), m) for o in off]) - (a_ / abs(Pc)) * star(a, m) for m in C.PERP]
    D = dict(fr=np.array(fr), to=np.array(to), c=np.array(c, float), w=np.array(w, float), ar=np.array(ar), n=len(order), P=Pc,
             Pss=[star(a, m) for m in C.PERP], qs=qs, cls=np.array(cls, float))
    comps, lab = tilt.sccs(D); br = sorted([x for x in comps if x[1]], key=lambda x: -len(x[0]))
    best = None
    for nodes, _ in br[:6]:
        R = tilt.restrict(D, nodes); r = tilt.solve(R, want_vectors=True)
        if r is not None and (best is None or r['s'] > best[1]['s']): best = (nodes, r, R)
    return order, tr, best
def run(a, steps, seed, burn=300):
    rng = random.Random(seed); order, tr, (nodes, sol, R) = prepare(a)
    Pv = tuple(a); negP = tuple(-x for x in Pv); idx = -np.ones(len(order), int); idx[nodes] = np.arange(len(nodes))
    l = sol['l']; lu = sol['lu']; n, h = C.n, C.half
    k = int(nodes[rng.randrange(len(nodes))]); state = order[k]; ps = Z0; tiles = []
    for step in range(steps + burn):
        d = list(state); din, dout = d[-1], d[0]; turn = (dout - din) % n; turn = turn - n if turn > h else turn
        opts = []
        for t, td, ang in C.TILES:
            if ang > h - turn: continue
            tdir = [(dout + x) % n for x in td]; pts = [ps]
            for x in tdir[:-1]: pts.append(add(pts[-1], C.E[x]))
            rep = [(x + h) % n for x in reversed(tdir[1:])]
            r = split_pos(rep + d[1:], ps, Pv, negP)
            if r is None: continue
            f, holes, off = r; cc = 1; w = C.VW[t]
            for hl, hq in holes:
                ci, wi, _ = fill_info(tuple(hl)); cc *= ci; w += wi
            if cc == 0: continue
            j = _state_index(order, f)
            if idx[j] < 0: continue
            wt = cc * math.exp(lu * int(round(2 * w))) * l[idx[j]]
            opts.append((wt, t, pts, holes, f, off, dout))
        tot = sum(o[0] for o in opts); u = rng.random() * tot
        for o in opts:
            u -= o[0]
            if u < 0: break
        wt, t, pts, holes, f, off, dout = o
        out = [(t, tile_class(t, dout), pts)]
        for hl, hq in holes: sample_fill(hl, hq, rng, out)
        if step >= burn: tiles += out
        state = f; ps = add(ps, off)
    return tiles, sol
_IDX = {}
def _state_index(order, f):
    key = id(order)
    if key not in _IDX: _IDX[key] = {s: i for i, s in enumerate(order)}
    return _IDX[key][f]
if __name__ == '__main__':
    a = [int(x) for x in sys.argv[1:7]]; steps = int(sys.argv[7]); seed = int(sys.argv[8]); outp = sys.argv[9]
    t0 = time.time(); tiles, sol = run(a, steps, seed)
    names = np.array([t[0] for t in tiles]); cl = np.array([t[1] for t in tiles]); V = np.array([[list(v) for v in t[2]] + [[0] * 6] * (4 - len(t[2])) for t in tiles])
    np.savez_compressed(outp, a=a, names=names, cls=cl, V=V, s=sol['s'], lu=sol['lu'])
    print('tiles', len(tiles), 's=%.5f' % sol['s'], '%.0fs' % (time.time() - t0))
