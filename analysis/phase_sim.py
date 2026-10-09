"""TASK3 Step 4b: re-determine sigma_B, sigma_A from the new reference/verification points and redraw the phase window.
(New file: analysis/phase.py itself is left unchanged as required; it is the model whose constants are updated here.)
Usage: python3 analysis/phase_sim.py   -> figures/phase_diagram_sim.png, runs/phase_sim.json"""
import math, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import task3_analysis as T3
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT = T3.ROOT; S3 = math.sqrt(3); Pst = S3 - 1; LS = 2*math.cos(math.radians(15)); cB = 2.0; cA = 2/S3
D = T3.load()
def ens(kind, P, T, Ntag=None): return T3.mean_err(T3.pick(D, kind, P, T, Ntag))
def d_tx(struct, Pnt, T, Ntag, sconf=T3.S_RANDOM12):
    a = ens('r12', Pnt, T); b = ens(struct, Pnt, T, Ntag)
    if not a or not b: return None
    return dict(P=Pnt, T=T, D=a['mean'] - sconf - b['mean'], err=math.hypot(a['err'], b['err']), dv=a['v'] - b['v'])
def linfit(pts, P0):
    """D(P) = D0 + m (P-P0); weighted LSQ; returns root and its 1-sigma"""
    P = np.array([p['P'] for p in pts]); y = np.array([p['D'] for p in pts]); e = np.array([p['err'] for p in pts])
    X = np.c_[np.ones_like(P), P - P0]; W = 1/e
    cf, *_ = np.linalg.lstsq(X*W[:, None], y*W, rcond=None); cov = np.linalg.inv((X*W[:, None]).T @ (X*W[:, None]))
    D0, m = cf; root = P0 - D0/m
    gr = np.array([-1/m, D0/m**2]); err = math.sqrt(gr @ cov @ gr)
    chi2 = float((((X@cf - y)/e)**2).sum()); return root, err, m, math.sqrt(cov[1, 1]), chi2, len(P)
res = {}; lines = []
def pr(*a):
    s = ' '.join(str(x) for x in a); print(s); lines.append(s)
for T in (0.06, 0.08):
    Pts = sorted({r['P'] for r in D.values() if abs(r['T']-T) < 1e-9})
    tB = [d for d in (d_tx('B', p, T, 1600) for p in Pts) if d]; tA = [d for d in (d_tx('A', p, T, 1600) for p in Pts) if d]
    pr('T=%.2f  D_tB(P): %s' % (T, ' '.join('%.3f:%+.3f(+-%.3f)' % (d['P'], d['D'], 2*d['err']) for d in tB)))
    pr('       D_tA(P): %s' % ' '.join('%.3f:%+.3f(+-%.3f)' % (d['P'], d['D'], 2*d['err']) for d in tA))
    rB = linfit([d for d in tB if d['P'] >= 0.735], 0.76); rA = linfit(tA, 0.72) if len(tA) >= 2 else None
    beta = 1/T; dvB = np.mean([d['dv'] for d in tB if d['P'] >= 0.735]); 
    pr('       B side: P_tB = %.4f +- %.4f (2s %.4f)  slope %.2f +- %.2f  (thermodynamic beta*(v_t-v_B) = %.2f)  chi2 %.2f/%d pts' % (rB[0], rB[1], 2*rB[1], rB[2], rB[3], beta*dvB, rB[4], rB[5]))
    sB = (rB[0] - Pst)/(cB*T); esB = rB[1]/(cB*T)
    res[T] = dict(P_tB=rB[0], eP_tB=rB[1], sigma_B=sB, esigma_B=esB, slopeB=rB[2])
    pr('       sigma_B = (P_tB - P*)/(2T) = %.4f +- %.4f (1s)' % (sB, esB))
    if rA:
        pr('       A side: P_tA = %.4f +- %.4f  slope %.2f (beta*(v_t-v_A) from NPT: %.2f)' % (rA[0], rA[1], rA[2], beta*np.mean([d['dv'] for d in tA])))
        sA = (Pst - rA[0])/(cA*T); res[T].update(P_tA=rA[0], eP_tA=rA[1], sigma_A=sA, esigma_A=rA[1]/(cA*T))
        pr('       sigma_A = (P* - P_tA)/(cA T) = %.4f +- %.4f' % (sA, rA[1]/(cA*T)))
    else:
        # one A point only: use the thermodynamic slope beta*(v_t - v_A)
        d = tA[0]; m = beta*d['dv']; root = d['P'] - d['D']/m; sA = (Pst - root)/(cA*T)
        res[T].update(P_tA=root, eP_tA=d['err']/abs(m), sigma_A=sA, esigma_A=d['err']/abs(m)/(cA*T), A_extrapolated=True)
        pr('       A side (single point, thermodynamic slope %.2f): P_tA = %.4f ; sigma_A = %.4f' % (m, root, sA))
sBm = np.mean([res[T]['sigma_B'] for T in res]); sAm = np.mean([res[T]['sigma_A'] for T in res])
pr('sigma_B: T=0.06 %.4f, T=0.08 %.4f (mean %.4f)   sigma_A: %.4f, %.4f (mean %.4f)   [phase.py with s_conf=0.594: sigma_B=0.2733, sigma_A=0.9189]' % (res[0.06]['sigma_B'], res[0.08]['sigma_B'], sBm, res[0.06]['sigma_A'], res[0.08]['sigma_A'], sAm))
json.dump({str(k): v for k, v in res.items()} | dict(sigma_B_mean=sBm, sigma_A_mean=sAm), open(os.path.join(ROOT, 'runs', 'phase_sim.json'), 'w'), indent=1)
# ------------------ figure
sB0, sA0 = 0.1211 + (0.594 - math.log(4421)/19), 0.7666 + (0.594 - math.log(4421)/19)
fig, ax = plt.subplots(figsize=(7.2, 5.6)); Tg = np.linspace(0, 0.1, 100)
ax.fill_betweenx(Tg, Pst - cA*sAm*Tg, Pst + cB*sBm*Tg, color='#cfe3f5', label='random-tiling window (sigma_B=%.3f, sigma_A=%.3f, this work)' % (sBm, sAm))
ax.plot(Pst - cA*sA0*Tg, Tg, 'k:', lw=1); ax.plot(Pst + cB*sB0*Tg, Tg, 'k:', lw=1, label='prediction (s_conf=0.594, calibrated constants)')
ax.plot(Pst + cB*(0.1211)*Tg, Tg, '--', c='gray', lw=1, label='old window (s_conf = ln4421/19)')
for T in (0.06, 0.08):
    ax.plot([res[T]['P_tB']], [T], 's', c='navy', ms=7, label='measured P_tB (D_tB = 0)' if T == 0.06 else None)
    ax.plot([res[T]['P_tA']], [T], 's', c='darkgreen', ms=7, mfc='none', label='measured P_tA (D_tA = 0)' if T == 0.06 else None)
shown = set()
for (P, T), g in sorted(T3.main()[1].items()) if False else []:
    pass
for T in (0.06, 0.08):
    for p in sorted({r['P'] for r in D.values() if abs(r['T']-T) < 1e-9}):
        d = d_tx('B', p, T, 1600)
        if not d: continue
        dA = d_tx('A', p, T, 1600); sig = 2*d['err']
        if d['D'] < -sig and (dA is None or dA['D'] < -2*dA['err']): c, lab = '#1f77b4', 'tiling wins'
        elif d['D'] > sig: c, lab = '#d62728', 'B wins'
        elif dA is not None and dA['D'] > 2*dA['err']: c, lab = '#2ca02c', 'A wins'
        else: c, lab = '#cccccc', 'undecided (2 sigma)'
        ax.plot(p, T, 'o', ms=9, mfc=c, mec='k', label=lab if lab not in shown else None); shown.add(lab)
for p in sorted({r['P'] for r in D.values() if abs(r['T']-0.06) < 1e-9}):
    if not d_tx('B', p, 0.06, 1600):
        dA = d_tx('A', p, 0.06, 1600)
        if dA: ax.plot(p, 0.06, 'o', ms=9, mfc='white', mec='#1f77b4', label='tiling beats A (B not computed)')
ax.plot(Pst, 0, 'k*', ms=11); ax.set_xlim(0.62, 0.82); ax.set_ylim(0, 0.1); ax.set_xlabel('P'); ax.set_ylabel('T'); ax.set_title('lambda = 1.93: random tiling vs A, B (Frenkel-Ladd)')
ax.legend(fontsize=7, loc='lower right'); plt.tight_layout(); plt.savefig(os.path.join(ROOT, 'figures', 'phase_diagram_sim.png'), dpi=130)
open(os.path.join(ROOT, 'runs', 'phase_sim.txt'), 'w').write('\n'.join(lines) + '\n')
