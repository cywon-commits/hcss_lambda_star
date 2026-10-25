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
    if n < 3000:
        A = M.toarray(); ev, R = np.linalg.eig(A); i = np.argmax(ev.real); evl, L = np.linalg.eig(A.T); j = np.argmax(evl.real)
        return ev[i].real, np.abs(R[:, i].real), np.abs(L[:, j].real)
    v, R = sla.eigs(M, k=1, which='LR', tol=1e-12); vl, L = sla.eigs(M.T.tocsr(), k=1, which='LR', tol=1e-12)
    return v[0].real, np.abs(R[:, 0].real), np.abs(L[:, 0].real)
def solve(D, phis=None, lu=-0.3, want_vectors=False):
    nm = len(D['qs']); phis = [0j] * nm if phis is None else phis
    ex = sum((np.conj(phi) * q).real for phi, q in zip(phis, D['qs']))
    for _ in range(40):
        val = D['c'] * np.exp(lu * D['w'] + ex); M = sp.csr_matrix((val, (D['to'], D['fr'])), shape=(D['n'], D['n']))
        rho, r, l = eig(M, D['n'])
        def avg(X): return l @ (sp.csr_matrix((val * X, (D['to'], D['fr'])), shape=(D['n'], D['n'])) @ r)
        Fu = avg(D['w'])
        if abs(np.log(rho)) < 1e-12: break
        lu -= np.log(rho) / (Fu / (rho * (l @ r)))
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
if __name__ == '__main__':
    D = load(sys.argv[1]); r0 = solve(D)
    print('phi=0: s=%.6f  nv=%.5f' % (r0['s'], r0['nv']))
    for m, (a, b) in enumerate(zip(r0['alphas'], r0['betas'])): print('  perp space %d: alpha=%s beta=%s det E=%.5f' % (m, np.round(a, 5), np.round(b, 5), abs(a) ** 2 - abs(b) ** 2))
    if 'Nt_over_Nr' in r0: print('  N_t/N_r=%.4f  area densities sigma5 %.4f sigma7 %.4f' % (r0['Nt_over_Nr'], r0['area5'], r0['area7']))
