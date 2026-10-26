"""Transfer matrix of the z18 two-family cylinder P' = (a0..a5) and of its square-triangle image P0 (cfg st12), same code path (cyl_ref.build, tilt).
  HCSS_CFG=z18  python3 st12_cyl.py a0 a1 a2 a3 a4 a5 out.json
  HCSS_CFG=st12 python3 st12_cyl.py a0 a1 a2 a3 a4 a5 out.json        (a = the z18 period; it is mapped to Z[zeta12] inside)
Output: states, transitions, branching SCCs (size, s, rho(0), nodes as cyclic words mapped to z12 directions)."""
import sys, json, itertools, time
import numpy as np
import cfg as C
from zz import add, cplx, star
import cyl, cyl_ref, tilt
from cyl import front_split, PHAT, canon_front
sys.setrecursionlimit(100000)
def z18_steps(a): return [i if x > 0 else i + 9 for i, x in enumerate(a) for _ in range(abs(x))]
def to_z12(j, fam):
    """z18 direction j of class A = fam[0] -> even z12 direction 2m (j = 3m + A); class B = A + 1 (mod 3) -> odd direction 2m + 1 with (j - 1) = 3m + A
    (B lines are the A lines turned by +20 deg in z18 and by +30 deg in z12, i.e. the B family is rotated by -10 deg when going to z12)"""
    if j % 3 == fam[0]: return 2 * (((j - fam[0]) % 18) // 3)
    return 2 * (((j - 1 - fam[0]) % 18) // 3) + 1
def to_z18(k, fam):  # inverse (used for the necklace comparison)
    m, odd = divmod(k, 2); return (3 * m + fam[0] + odd) % 18
def main():
    a = [int(x) for x in sys.argv[1:7]]; out = sys.argv[7]; steps = z18_steps(a); cl = sorted({j % 3 for j in steps})
    assert len(cl) == 2, 'not a two-family period'
    fam = (cl[0], cl[1]) if (cl[1] - cl[0]) % 3 == 1 else (cl[1], cl[0])          # B = A + 1 (mod 3)
    if C.name == 'z18': P = tuple(a); dirs = steps
    else:
        dirs = [to_z12(j, fam) for j in steps]; P = tuple(0 for _ in range(C.r))
        for k in dirs: P = add(P, C.E[k])
    PHAT[0] = cplx(P) / abs(cplx(P)); negP = tuple(-x for x in P); d0 = None; t0 = time.time()
    if len(sys.argv) > 8 and C.name != 'z18':               # start from a front of the z18 dominant SCC, mapped to z12 (the standard start can be a dead end in the other geometry)
        w18 = json.load(open(sys.argv[8]))['words'][0][0]; d0 = [to_z12(j, fam) for j in w18]
        r = front_split(list(d0), P, negP); assert r is not None and not r[1], 'mapped front is not a valid square-triangle front'
    else:
        for perm in itertools.permutations(dirs):
            r = front_split(list(perm), P, negP)
            if r is not None and not r[1]: d0 = list(perm); break
    assert d0 is not None
    order, tr = cyl_ref.build(P, d0)
    fr, to, c, w, ar, off, na, cls = zip(*tr)
    off = np.array(off); a_ = np.array([(cplx(o) * np.conj(PHAT[0])).real for o in off]); Pc = cplx(P)
    qs = [np.array([star(tuple(o), m) for o in off]) - (a_ / abs(Pc)) * star(P, m) for m in C.PERP]
    D = dict(fr=np.array(fr), to=np.array(to), c=np.array(c, float), w=np.array(w, float), ar=np.array(ar), n=len(order), P=Pc, Pss=[star(P, m) for m in C.PERP], qs=qs,
             cls=np.array(cls, float) if C.name == 'z18' else None, na=np.array(na, float))
    comps, lab = tilt.sccs(D); res = []
    for nodes, br in comps:
        if not br: continue
        R = tilt.restrict(D, nodes); sol = tilt.solve(R, want_vectors=True)
        if sol is None: continue
        M = R['c'][:, None]; import scipy.sparse as sp
        A = sp.csr_matrix((R['c'], (R['to'], R['fr'])), shape=(R['n'], R['n'])); rho0 = tilt.eig(A, R['n'])[0]
        conv = (lambda k: k) if C.name == 'z18' else (lambda k: to_z18(k, fam))          # necklaces are canonicalised AFTER mapping to z18 directions
        words = []
        for i in nodes:
            w0 = [conv(k) for k in order[i]]; words.append(min(tuple(w0[j:] + w0[:j]) for j in range(len(w0))))
        words.sort()
        import scipy.sparse as sp2
        val = sol['val']; l_, r_ = sol['l'], sol['r']; nat = R['na'] if 'na' in R else None
        avg = lambda X: l_ @ (sp2.csr_matrix((val * X, (R['to'], R['fr'])), shape=(R['n'], R['n'])) @ r_)
        nT = 2 * avg(R['na']) / avg(R['w']) if nat is not None else None; nV = 1.0      # triangles per vertex (w = 2 x vertex weight)
        res.append(dict(nT_per_vertex=None if nT is None else float(nT), size=int(len(nodes)), ntrans=int(len(R['fr'])), s=float(sol['s']), lu=float(sol['lu']), rho0=float(rho0), nv=float(sol['nv']), words=words))
    res.sort(key=lambda r: -r['s'])
    summary = dict(cfg=C.name, a=a, fam=fam, P=list(P), W=abs(Pc), states=len(order), transitions=len(tr), branching=[{k: v for k, v in r.items() if k != 'words'} for r in res], sec=time.time() - t0)
    json.dump(dict(summary=summary, words=[r['words'] for r in res]), open(out, 'w')); print(json.dumps(summary))
if __name__ == '__main__': main()
