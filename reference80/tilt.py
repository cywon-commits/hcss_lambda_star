"""Entropy of the cylinder transfer matrix as a function of the perpendicular-space tilt fields (generalised to any number of perp spaces).
Weights  c * exp(lu*w + sum_m Re(conj(phi_m) q_m)),  lu fixed by rho = 1.   Works for the lambda* files (one perp space: key q) and the
z18 files (q5, q7, cls).  solve(D, phis) with phis = list of complex (one per perp space)."""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla, sys, math
SIG5 = {'T': -math.sqrt(3) / 4, 'R': math.sin(math.radians(40))}      # signed perpendicular tile areas (z18)
SIG7 = {'T': math.sqrt(3) / 4, 'R': -math.sin(math.radians(20))}
def load(f):
    z = np.load(f); D = dict(fr=z['fr'], to=z['to'], c=z['c'].astype(float), w=z['w'].astype(float), ar=z['ar'], n=int(z['n']), P=complex(z['P']))
    if 'q' in z.files: D.update(qs=[z['q']], Pss=[complex(z['Ps'])], cls=None)
    else: D.update(qs=[z['q5'], z['q7']], Pss=[complex(z['Ps5']), complex(z['Ps7'])], cls=z['cls'].astype(float))
    return D
def eig(M, n):
    """Perron eigenvalue and left/right eigenvectors.  A shift M + delta*I removes the periodicity of (nearly) deterministic transfer matrices
    (cycles) which otherwise makes ARPACK stagnate; the eigenvectors are unchanged."""
    if n < 3000:
        A = M.toarray(); ev, R = np.linalg.eig(A); i = np.argmax(ev.real); evl, L = np.linalg.eig(A.T); j = np.argmax(evl.real)
        return ev[i].real, np.abs(R[:, i].real), np.abs(L[:, j].real)
    x = np.ones(n)
    for _ in range(30):                                   # rough scale of the spectral radius by power iteration
        y = M @ x; nr = np.linalg.norm(y); x = y / nr
    delta = 0.5 * nr
    S = M + delta * sp.identity(n, format='csr')
    v, R = sla.eigs(S, k=1, which='LR', tol=1e-12, maxiter=20000); vl, L = sla.eigs(S.T.tocsr(), k=1, which='LR', tol=1e-12, maxiter=20000)
    return v[0].real - delta, np.abs(R[:, 0].real), np.abs(L[:, 0].real)
def has_cycle(D):
    """True if the transition graph contains a directed cycle (otherwise the cylinder cannot be tiled indefinitely)"""
    if 'cyc' not in D:
        import scipy.sparse.csgraph as cg
        G = sp.csr_matrix((np.ones(len(D['fr'])), (D['fr'], D['to'])), shape=(D['n'], D['n']))
        nc, lab = cg.connected_components(G, directed=True, connection='strong')
        cnt = np.bincount(lab, minlength=nc); selfl = np.zeros(nc, bool); m = D['fr'] == D['to']; selfl[lab[D['fr'][m]]] = True
        D['cyc'] = bool(((cnt > 1) | selfl).any())
    return D['cyc']
def _matrix(D, lu, ex):
    val = D['c'] * np.exp(lu * D['w'] + ex)
    return val, sp.csr_matrix((val, (D['to'], D['fr'])), shape=(D['n'], D['n']))
def _rho_only(D, lu, ex):
    val, M = _matrix(D, lu, ex)
    if D['n'] < 400:
        return float(np.max(np.linalg.eigvals(M.toarray()).real))
    return eig(M, D['n'])[0]
def solve(D, phis=None, lu=None, want_vectors=False):
    nm = len(D['qs']); phis = [0j] * nm if phis is None else phis
    if not has_cycle(D): return None                          # no infinite tiling of this cylinder
    ex = sum((np.conj(phi) * q).real for phi, q in zip(phis, D['qs']))
    def newton(lu0):
        lu = lu0
        for _ in range(40):
            val, M = _matrix(D, lu, ex)
            if not np.isfinite(val).all() or not val.any(): return None
            try: rho, r, l = eig(M, D['n'])
            except Exception: return None
            if not np.isfinite(rho) or rho <= 0: return None
            if abs(np.log(rho)) < 1e-12: return lu
            def avg(X): return l @ (sp.csr_matrix((val * X, (D['to'], D['fr'])), shape=(D['n'], D['n'])) @ r)
            step = np.log(rho) / (avg(D['w']) / (rho * (l @ r)))
            if not np.isfinite(step) or abs(step) > 3: return None
            lu -= step
        return None
    lu = newton(-0.3 if lu is None else lu)
    if lu is None:                                              # fallback: bracket and bisect log rho(lu) = 0 (rho increases with lu)
        lo, hi = -3.0, 3.0
        while _rho_only(D, lo, ex) >= 1: lo -= 2.0               # rho increases with lu (weights exp(lu w), w > 0)
        while _rho_only(D, hi, ex) <= 1: hi += 2.0
        for _ in range(45):
            mid = 0.5 * (lo + hi)
            if _rho_only(D, mid, ex) < 1: lo = mid
            else: hi = mid
            if hi - lo < 1e-10: break
        lu = newton(0.5 * (lo + hi)) or 0.5 * (lo + hi)
    val, M = _matrix(D, lu, ex); rho, r, l = eig(M, D['n'])
    def avg(X): return l @ (sp.csr_matrix((val * X, (D['to'], D['fr'])), shape=(D['n'], D['n'])) @ r)
    Fu = avg(D['w'])
    per = lambda X: 2 * avg(X) / Fu                      # per-vertex average
    qv = [per(q.real) + 1j * per(q.imag) for q in D['qs']]; yv = per(D['ar']) / abs(D['P'])
    f = -2 * lu; s = f - sum((np.conj(phi) * x).real for phi, x in zip(phis, qv))
    Ph = D['P'] / abs(D['P']); out = dict(s=s, nv=1 / per(D['ar']), lu=lu, alphas=[], betas=[])
    for Ps, x in zip(D['Pss'], qv):
        g = Ps / abs(D['P']); h = x / yv
        out['alphas'].append((g - 1j * h) / 2 / Ph); out['betas'].append((g + 1j * h) / 2 / np.conj(Ph))
    if D.get('cls') is not None:
        cl = np.array([per(D['cls'][:, k]) for k in range(D['cls'].shape[1])])      # tiles of each class per vertex
        nt, nr = cl[:3].sum(), cl[3:].sum(); at = math.sqrt(3) / 4; ar = math.sin(math.radians(80))
        out.update(cls=cl, Nt=nt, Nr=nr, Nt_over_Nr=nt / nr,
                   area5=(SIG5['T'] * nt + SIG5['R'] * nr) / (at * nt + ar * nr), area7=(SIG7['T'] * nt + SIG7['R'] * nr) / (at * nt + ar * nr))
    if want_vectors: out.update(rho=rho, r=r, l=l, val=val)
    return out
def sccs(D):
    """non-trivial strongly connected components of the transition graph: list of (nodes, branching?)"""
    import scipy.sparse.csgraph as cg
    n = D['n']; G = sp.csr_matrix((np.ones(len(D['fr'])), (D['fr'], D['to'])), shape=(n, n))
    nc, lab = cg.connected_components(G, directed=True, connection='strong')
    ins = lab[D['fr']] == lab[D['to']]; cnt = np.bincount(lab, minlength=nc)
    ein = np.bincount(lab[D['fr']][ins], weights=D['c'][ins], minlength=nc)
    out = []
    for k in np.nonzero(ein > 0)[0]:
        out.append((np.nonzero(lab == k)[0], bool(ein[k] > cnt[k])))
    return out, lab
def restrict(D, nodes):
    idx = -np.ones(D['n'], int); idx[nodes] = np.arange(len(nodes)); m = (idx[D['fr']] >= 0) & (idx[D['to']] >= 0)
    E = dict(fr=idx[D['fr'][m]], to=idx[D['to'][m]], c=D['c'][m], w=D['w'][m], ar=D['ar'][m], n=len(nodes), P=D['P'], Pss=D['Pss'],
             qs=[q[m] for q in D['qs']], cls=None if D.get('cls') is None else D['cls'][m])
    return E
def solve_sccs(D, phis=None, maxscc=12):
    """entropy of every branching non-trivial SCC (a pure cycle has s = 0 and is reported as frozen).  Returns dict(frozen=#cycles, results=[...] sorted by s)"""
    comps, lab = sccs(D); res = []; frozen = 0
    br = [c for c in comps if c[1]]; frozen = sum(1 for c in comps if not c[1])
    br.sort(key=lambda c: -len(c[0]))
    for nodes, _ in br[:maxscc]:
        r = solve(restrict(D, nodes), phis)
        if r is not None: r['size'] = len(nodes); res.append(r)
    res.sort(key=lambda r: -r['s'])
    return dict(frozen=frozen, n_branching=len(br), results=res)
if __name__ == '__main__':
    D = load(sys.argv[1]); r0 = solve(D)
    if r0 is None: print('no infinite tiling (acyclic transfer graph)'); sys.exit(0)
    print('phi=0: s=%.6f  nv=%.5f' % (r0['s'], r0['nv']))
    for m, (a, b) in enumerate(zip(r0['alphas'], r0['betas'])): print('  perp space %d: alpha=%s beta=%s det E=%.5f' % (m, np.round(a, 5), np.round(b, 5), abs(a) ** 2 - abs(b) ** 2))
    if 'Nt_over_Nr' in r0: print('  N_t/N_r=%.4f  area densities sigma5 %.4f sigma7 %.4f' % (r0['Nt_over_Nr'], r0['area5'], r0['area7']))
