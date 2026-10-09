"""TASK3 Step 4: Delta_vib, D_tB, verdict and phase window from the Frenkel-Ladd results.
Reads runs/fe_long/*.json (preferred, 200k NPT sweeps) and runs/fe/*.json.  Errors in the JSON are 1 sigma; tables print 2 sigma.
Usage: python3 analysis/task3_analysis.py"""
import glob, json, math, os, re, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
S_RANDOM12, S_RANDOM312 = 0.594, 0.591
S_DOD_NEW, S_DOD_OLD = math.log(5827)/19, math.log(4421)/19
def load():
    D = {}
    for sub in ('fe', 'fe_long', 'fe_rr'):   # later directories override earlier ones when the new run is usable
        for f in sorted(glob.glob(os.path.join(ROOT, 'runs', sub, '*.json'))):
            n = os.path.basename(f)[:-5]
            m = re.match(r'(B|A|dodeca|r12|r312)_(?:N(\d+)_)?(?:s(\d+)_)?P([\d.]+)_T([\d.]+)$', n)
            if not m: continue
            d = json.load(open(f)); kind, N, s, P, T = m.groups()
            d['_src'] = sub
            if n in D and not (math.isfinite(d['beta_g']) and d['components']['dA1_overlap_free_fraction'] >= 0.8): continue
            D[n] = dict(kind=kind, Ntag=int(N) if N else None, seed=int(s) if s else None, P=float(P), T=float(T), bg=d['beta_g'], err=d['beta_g_err'],
                                         drift=d['npt']['drift_v_last_minus_first_half_of_sampling'], free=d['components']['dA1_overlap_free_fraction'], v=d['npt']['v'], N=d['N'], src=sub, npt_h=d['npt']['h'])
    return D
def good(r): return math.isfinite(r['bg']) and r['free'] >= 0.8
def pick(D, kind, P, T, Ntag=None):
    return sorted([r for r in D.values() if r['kind'] == kind and abs(r['P']-P) < 1e-9 and abs(r['T']-T) < 1e-9 and (Ntag is None or r['Ntag'] == Ntag)], key=lambda r: r['seed'] or 0)
def mean_err(rs):
    rs = [r for r in rs if good(r)]
    if not rs: return None
    x = np.array([r['bg'] for r in rs]); e = np.array([r['err'] for r in rs]); n = len(x)
    return dict(n=n, mean=x.mean(), err=math.sqrt((e**2).sum())/n, std=x.std(ddof=1) if n > 1 else float('nan'), fl_err_typ=float(e.mean()), v=np.mean([r['v'] for r in rs]),
                drift=max(abs(r['drift']) for r in rs), free=min(r['free'] for r in rs))
def main():
    D = load(); pts = sorted({(r['P'], r['T']) for r in D.values()}, key=lambda a: (a[1], a[0]))
    out = {}; lines = []
    P_ = lambda *a: (print(*a), lines.append(' '.join(str(x) for x in a)))
    P_('%-14s | %-8s %-8s %-8s %-8s %-8s %-8s | D_tB(r12) D_tB(r312) D_tB(dod new) D_tB(dod old) | Dvib12 Dvib312' % ('(P,T)', 'B931', 'B1600', 'dod931', 'dod1539', 'r12', 'r312'))
    for P, T in pts:
        R = {k: mean_err(pick(D, k, P, T, nt)) for k, nt in (('B931', None), ('B1600', None), ('dod931', None), ('dod1539', None), ('r12', None), ('r312', None))} if False else None
        g = {}
        g['B931'] = mean_err(pick(D, 'B', P, T, 931)); g['B1600'] = mean_err(pick(D, 'B', P, T, 1600)); g['A931'] = mean_err(pick(D, 'A', P, T, 931)); g['A1600'] = mean_err(pick(D, 'A', P, T, 1600))
        g['dod931'] = mean_err(pick(D, 'dodeca', P, T, 931)); g['dod1539'] = mean_err(pick(D, 'dodeca', P, T, 1539))
        g['r12'] = mean_err(pick(D, 'r12', P, T)); g['r312'] = mean_err(pick(D, 'r312', P, T)); out[(P, T)] = g
        f = lambda k: '%.4f' % g[k]['mean'] if g.get(k) else '   -   '
        def dd(a, b, sa=0.0):
            if g.get(a) and g.get(b): return g[a]['mean'] - sa - g[b]['mean'], math.hypot(g[a]['err'], g[b]['err'])
            return None
        tB12 = dd('r12', 'B1600', S_RANDOM12); tB312 = dd('r312', 'B931', S_RANDOM312); tBd = dd('dod1539', 'B1600', S_DOD_NEW); tBdo = dd('dod1539', 'B1600', S_DOD_OLD)
        v12 = dd('r12', 'dod1539'); v312 = dd('r312', 'dod931')
        s = lambda t: '%+.3f+-%.3f' % (t[0], 2*t[1]) if t else '    -    '
        P_('(%.3f,%.2f)    | %s %s %s %s %s %s | %s %s %s %s | %s %s' % (P, T, f('B931'), f('B1600'), f('dod931'), f('dod1539'), f('r12'), f('r312'), s(tB12), s(tB312), s(tBd), s(tBdo), s(v12), s(v312)))
    return D, out, lines
if __name__ == '__main__' and not (len(sys.argv) > 1 and sys.argv[1] == 'full'):
    main()


def details(D):
    """per-point ensemble statistics (n, mean, sample std, typical FL 1-sigma error) for the random tilings and dodeca references"""
    print('\nensemble statistics (beta*g per particle; std = sample standard deviation over samples, fl = typical single-run FL 1-sigma error)')
    pts = sorted({(r['P'], r['T']) for r in D.values()}, key=lambda a: (a[1], a[0]))
    rows = []
    for P, T in pts:
        for k, nt in (('r12', None), ('r312', None), ('dodeca', 931), ('dodeca', 1539), ('B', 931), ('B', 1600), ('A', 931), ('A', 1600)):
            g = mean_err(pick(D, k, P, T, nt))
            if g: rows.append((P, T, k + ('' if nt is None else str(nt)), g['n'], g['mean'], g['err'], g['std'], g['fl_err_typ']))
    for r in rows: print('(%.3f,%.2f) %-9s n=%d mean %.4f  err(1s) %.4f  std %.4f  fl %.4f %s' % (*r, '<-- spread > 2*FL' if r[6] == r[6] and r[6] > 2*r[7] else ''))
    return rows


def full_table(D, path):
    L = ['| run | source | N | beta*g | 2sigma | overlap-free | NPT volume drift | flag |', '|---|---|---|---|---|---|---|---|']
    for n, r in sorted(D.items(), key=lambda kv: (kv[1]['T'], kv[1]['P'], kv[0])):
        fl = []
        if abs(r['drift']) > 0.005: fl.append('drift>0.005')
        if r['free'] < 0.8: fl.append('free<0.8')
        if not math.isfinite(r['bg']): fl.append('inf')
        L.append('| %s | %s | %d | %s | %.4f | %.3f | %+.4f | %s |' % (n, r['src'], r['N'], '%.4f' % r['bg'] if math.isfinite(r['bg']) else 'inf', 2*r['err'], r['free'], r['drift'], ','.join(fl)))
    open(path, 'w').write('\n'.join(L) + '\n'); return L


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'full':
    D = load(); details(D); full_table(D, os.path.join(ROOT, 'runs', 'task3_fl_table.md'))
