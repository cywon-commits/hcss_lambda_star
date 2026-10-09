"""TASK1b: refit the cylinder tilt scans in symmetry-adapted variables.
   s = s0 - (1/(2 nv)) [ Kbar (|g|^2+|h|^2)/2 + (dK/8) Re(Ph^2 (g+i h)^2) ] + lambda . l(h)  [+ delta term inside lambda]
   alpha=(g-ih)/(2Ph), beta=(g+ih)/(2 conj Ph), g=P*/|P|, h free complex (tilt).  |a|^2-|b|^2 = -Im(g conj h) is linear in h:
   delta enters only the linear coefficient:  lambda_delta = -(delta/(2 nv)) sigma,  sigma = conj(g) e^{i(phi_m+arg P)}  (real when the mirror is respected).
   Mirror of the alpha plane: 0deg: alpha -> conj(alpha) (axis 0); 15deg: alpha -> zeta^4 conj(alpha) (axis 60deg); 6.2/5.1deg: none.
Usage: python3 task1b_fit.py [scan_dir=analysis/scan_data]"""
import sys, numpy as np
D = sys.argv[1] if len(sys.argv) > 1 else 'analysis/scan_data'
z = np.exp(1j*np.pi/6)
CYL = [('2_2_0_-1', 0.0), ('2_3_0_-1', None), ('2_3_1_-1', 60.0), ('3_3_0_-1', None), ('3_4_0_-2', 0.0), ('3_4_1_-1', 60.0), ('4_4_0_-2', 0.0)]
def prep(n, rng):
    a = np.array([int(x) for x in n.split('_')]); P = a@np.array([z**k for k in range(4)]); Ps = a@np.array([z**(5*k) for k in range(4)])
    R = np.load(f'{D}/scan_{n}_{rng}.npy'); s = R[:, 2]; al = R[:, 3]+1j*R[:, 4]; be = R[:, 5]+1j*R[:, 6]; nv = R[:, 7].mean()
    Ph = P/abs(P); g = Ps/abs(P); h = (np.conj(Ph)*be-Ph*al)/1j
    assert np.allclose(Ph*al+np.conj(Ph)*be, g, atol=1e-9)
    return dict(P=P, W=abs(P), Ph=Ph, g=g, h=h, s=s, nv=nv, al=al, be=be)
def fit(d, phim, constrained=True):
    h, Ph, g, nv = d['h'], d['Ph'], d['g'], d['nv']; f = -1/(2*nv)
    cols = [np.ones_like(h.real), f*abs(h)**2/2, f*np.real(Ph**2*(g+1j*h)**2)/8]
    if constrained and phim is not None:
        cols.append(np.imag(h*np.exp(-1j*(np.radians(phim)+np.angle(Ph)))))
    else: cols += [h.real, h.imag]
    X = np.array(cols).T; cf, *_ = np.linalg.lstsq(X, d['s'], rcond=None)
    res = X@cf-d['s']; dof = len(res)-X.shape[1]; cov = np.linalg.inv(X.T@X)*np.sum(res**2)/dof
    return cf, np.sqrt(np.diag(cov)), np.sqrt(np.mean(res**2)), cov
out = {}
print('== step 1: (Kbar, dK, linear lambda) per cylinder, mirror-constrained linear term ==')
print('%-9s %5s %6s | %6s %6s %8s %7s | range: dKbar ddK dlam | Kbar_old dK_old | rms free-lin rms' % ('cyl', '|P|', 'rng', 'Kbar', 'dK', 'lambda', 'sigma_c'))
for n, phim in CYL:
    r = {}
    for rng in ('0.3', '0.2'):
        d = prep(n, rng); cf, se, rms, cov = fit(d, phim)
        cfu, seu, rmsu, _ = fit(d, phim, constrained=False)
        r[rng] = (cf, se, rms, cfu, rmsu, d)
    d = r['0.3'][5]; cf, se, rms, cfu, rmsu = r['0.3'][0], r['0.3'][1], r['0.3'][2], r['0.3'][3], r['0.3'][4]
    dr = r['0.3'][0]-r['0.2'][0]
    # earlier (K_a,K_r,K_i) -> (Kbar, dK) for comparison
    R = np.load(f'{D}/scan_{n}_0.3.npy'); s = R[:, 2]; al = R[:, 3]+1j*R[:, 4]; be = R[:, 5]+1j*R[:, 6]; nv = R[:, 7].mean()
    X = np.c_[np.ones_like(s), -abs(al)**2/(2*nv), -be.real**2/(2*nv), -be.imag**2/(2*nv)]; Ka, Kr, Ki = np.linalg.lstsq(X, s, rcond=None)[0][1:]
    Kold, dKold = (Ka+(Kr+Ki)/2)/2, Kr-Ki
    lam = cf[3] if phim is not None else cf[3:5]
    sig = np.conj(d['g'])*np.exp(1j*(np.radians(phim)+np.angle(d['Ph']))) if phim is not None else np.nan
    out[n] = dict(W=d['W'], arg=np.degrees(np.angle(d['P'])), g=d['g'], nv=d['nv'], Kbar=cf[1], dK=cf[2], lam=lam, sig=sig, phim=phim,
                  e_Kbar=np.hypot(se[1], dr[1]), e_dK=np.hypot(se[2], dr[2]), e_lam=np.hypot(se[3], dr[3]) if phim is not None else np.nan,
                  K02=(r['0.2'][0][1], r['0.2'][0][2]), Kold=Kold, dKold=dKold, rms=rms, rmsu=rmsu, lamu=cfu[3:], Kbar_u=cfu[1], dK_u=cfu[2])
    o = out[n]
    print('%-9s %5.3f %6.3f | %6.3f %+6.3f %+8.4f %7.4f | %+.3f %+.3f %+.4f | %6.3f %+6.3f | %.1e %.1e' % (n, o['W'], abs(o['g']), o['Kbar'], o['dK'], lam if phim is not None else np.nan,
          o['e_lam'], dr[1], dr[2], dr[3] if phim is not None else np.nan, Kold, dKold, rms, rmsu))
print('   (Kbar, dK) errors = quad(cov, 0.3-0.2 difference):')
for n, o in out.items(): print('   %-9s Kbar=%.3f+-%.3f  dK=%+.3f+-%.3f  free-linear fit: Kbar=%.3f dK=%+.3f  sigma(real?)=%s' % (n, o['Kbar'], o['e_Kbar'], o['dK'], o['e_dK'], o['Kbar_u'], o['dK_u'], 'n/a' if np.isnan(o['sig']) else '%.4f%+.1ei' % (o['sig'].real, o['sig'].imag)))
print('   unconstrained linear vector (Re h, Im h) angle vs mirror-allowed direction:')
for n, o in out.items():
    if o['phim'] is None: print('   %-9s no mirror: lambda=(%.4f, %.4f)' % (n, *o['lamu'])); continue
    Ph = np.exp(1j*np.angle(out[n]['g'])*0)  # placeholder to keep names simple
    d = prep(n, '0.3'); ang = np.degrees(np.angle(complex(*o['lamu']))); allowed = np.degrees(np.angle(1j*np.exp(1j*(np.radians(o['phim'])+np.angle(d['Ph'])))))
    print('   %-9s lambda=(%.4f, %.4f)  angle=%.1f deg, expected +-(%.1f) deg, other-direction fraction=%.3f' % (n, *o['lamu'], ang, allowed % 180, abs(np.sin(np.radians(ang-allowed)))))
# ---------------- step 2: delta ----------------
print('\n== step 2: delta from the mirror-allowed linear coefficient lambda = -delta*sigma/(2 nv) + c/W^2 ==')
def solve_delta(names, sets, power=2):
    """sets: list of direction labels per cylinder; unknowns: delta, c[label]"""
    labels = sorted(set(sets)); rows = []; y = []; w = []
    for n, lb in zip(names, sets):
        o = out[n]; row = [-o['sig'].real/(2*o['nv'])]+[o['W']**-power*(lb == L) for L in labels]
        if power == 24: row = [-o['sig'].real/(2*o['nv'])]+[o['W']**-2*(lb == L) for L in labels]+[o['W']**-4*(lb == L) for L in labels]
        rows.append(row); y.append(o['lam']); w.append(1/o['e_lam'])
    X = np.array(rows); y = np.array(y); w = np.array(w)
    cf, *_ = np.linalg.lstsq(X*w[:, None], y*w, rcond=None); res = (X@cf-y)
    try: cov = np.linalg.inv((X*w[:, None]).T@(X*w[:, None]))
    except np.linalg.LinAlgError: cov = np.full((X.shape[1],)*2, np.nan)
    dof = len(y)-X.shape[1]; chi2 = np.sum((res*w)**2)
    return cf, np.sqrt(np.diag(cov)), res, chi2, dof, labels
zero = ['2_2_0_-1', '3_4_0_-2', '4_4_0_-2']; fif = ['2_3_1_-1', '3_4_1_-1']
for n in zero+fif: o = out[n]; print('   %-9s |P|=%.3f sigma=%+.4f lambda=%+.4f+-%.4f  (delta part per unit delta: %+.4f)' % (n, o['W'], o['sig'].real, o['lam'], o['e_lam'], -o['sig'].real/(2*o['nv'])))
RES = {}
for tag, names, sets, power in [('0deg only, c/W^2', zero, ['0']*3, 2), ('0deg only, c/W (power 1)', zero, ['0']*3, 1),
                                ('0deg only, c/W^2 + c4/W^4 (exact, 3 pts)', zero, ['0']*3, 24), ('joint 0deg+15deg, c0/W^2, c15/W^2', zero+fif, ['0']*3+['15']*2, 2)]:
    cf, se, res, chi2, dof, labels = solve_delta(names, sets, power)
    RES[tag] = (cf[0], se[0])
    print('   %-44s delta=%+.3f +- %.3f  c=%s  resid=%s chi2/dof=%s' % (tag, cf[0], se[0], np.round(cf[1:], 3), np.round(res, 4), '%.2f' % (chi2/dof) if dof > 0 else 'n/a'))
# ---------------- series extrapolation of Kbar, dK ----------------
print('\n== series extrapolation Kbar(W)=Kbar_inf+b/W^2 ==')
def ext(names, key, ekey, order):
    W = np.array([out[n]['W'] for n in names]); K = np.array([out[n][key] for n in names]); e = np.array([out[n][ekey] for n in names])
    X = np.c_[np.ones_like(W), W**-2] if order == 1 else np.c_[np.ones_like(W), W**-2, W**-4]
    wgt = 1/e; cf = np.linalg.lstsq(X*wgt[:, None], K*wgt, rcond=None)[0]; return cf, X@cf-K
EXT = {}
for sname, names in (('0deg (3.732, 6.464, 7.464)', zero), ('15deg (5.278, 7.210)', fif), ('large three (6.464, 7.210, 7.464)', ['3_4_0_-2', '3_4_1_-1', '4_4_0_-2'])):
    for key, ekey in (('Kbar', 'e_Kbar'), ('dK', 'e_dK')):
        c1, r1 = ext(names, key, ekey, 1); line = '%-34s %-5s inf(1/W^2)=%+.3f b=%+.2f resid=%s' % (sname, key, c1[0], c1[1], np.round(r1, 3))
        if len(names) >= 3 and 'large' not in sname:
            c2, _ = ext(names, key, ekey, 2); line += '  | +1/W^4: %+.3f' % c2[0]; EXT[(sname, key)] = (c1[0], c2[0])
        else: EXT[(sname, key)] = (c1[0], np.nan)
        print('  ', line)
np.save('analysis/scan_data/task1b_summary.npy', {k: {kk: vv for kk, vv in v.items() if kk not in ('lamu',)} for k, v in out.items()}, allow_pickle=True)
# ---------------- model-light bound: finite-size part f(W) = lambda - lambda_delta must be non-increasing in W (per direction) ----------------
print('\n== delta bounds assuming only: f(W)=lambda+delta*sigma/(2nv) is non-increasing in W within a direction series ==')
def mono_bounds(names):
    lo, hi = -np.inf, np.inf
    for a, b in zip(names[:-1], names[1:]):          # f(a) >= f(b)  (W_a < W_b);  f = lam - u*delta, u=-sigma/(2nv)
        ua, ub = -out[a]['sig'].real/(2*out[a]['nv']), -out[b]['sig'].real/(2*out[b]['nv'])
        # lam_a - ua d >= lam_b - ub d  ->  (ub-ua) d >= lam_b - lam_a
        c = ub-ua; rhs = out[b]['lam']-out[a]['lam']
        if c > 0: lo = max(lo, rhs/c)
        elif c < 0: hi = min(hi, rhs/c)
    return lo, hi
b0 = mono_bounds(zero); b15 = mono_bounds(fif)
lo, hi = max(b0[0], b15[0]), min(b0[1], b15[1])
print('   0deg series      : %+.3f <= delta <= %+.3f' % b0)
print('   15deg series     : %+.3f <= delta <= %+.3f' % b15)
print('   common delta     : %+.3f <= delta <= %+.3f' % (lo, hi))
# 0deg with f also required >= 0 at the largest W (f -> 0+ from above)
u7 = -out['4_4_0_-2']['sig'].real/(2*out['4_4_0_-2']['nv']); print('   0deg + f(7.464)>=0 : delta >= %+.3f' % (-out['4_4_0_-2']['lam']/(-u7)))
nv = 1+1/np.sqrt(3); xp = 4/(np.sqrt(3)*nv**2); print('\n== phase window: shift of a fixed threshold  d ln mu = -delta/(2 nv |x\'|) = %.4f*(-delta),  dP/T = -dlnmu/0.683 (TASK conversion) ==' % (1/(2*nv*xp)))
for dl in (0.06, 0.07, 0.10, 0.16, 0.20):
    sh = -dl/(2*nv*xp); print('   |delta|=%.2f : |d ln mu|=%.4f  |dP/T|=%.4f  (fraction of the tiling window 1.608T: %.1f%%)' % (dl, abs(sh), abs(sh)/0.683, 100*abs(sh)/0.683/1.608))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
mk = {0.0: 'o', 60.0: 's', None: 'x'}
for n, o in out.items():
    m = mk[o['phim']]; lab = {'o': '0deg', 's': '15deg', 'x': 'tilted (no mirror)'}[m]
    ax[0].errorbar(o['W'], o['Kbar'], o['e_Kbar'], fmt=m, c='C0' if m == 'o' else 'C1' if m == 's' else 'gray', label=lab)
    ax[1].errorbar(o['W'], o['dK'], o['e_dK'], fmt=m, c='C0' if m == 'o' else 'C1' if m == 's' else 'gray')
    if m != 'x': ax[2].errorbar(o['W'], o['lam'], o['e_lam'], fmt=m, c='C0' if m == 'o' else 'C1', mfc='none' if o['sig'].real < 0 else None)
ax[0].set_ylabel('Kbar'); ax[1].set_ylabel('dK = K_r - K_i'); ax[2].set_ylabel('mirror-allowed linear coeff. lambda'); ax[1].axhline(0, c='k', lw=.5); ax[2].axhline(0, c='k', lw=.5)
ax[2].set_title('open marker: sigma<0 (g sign flipped)', fontsize=9)
h, l = ax[0].get_legend_handles_labels(); u = dict(zip(l, h)); ax[0].legend(u.values(), u.keys(), fontsize=8)
for a in ax: a.set_xlabel('|P|'); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig('figures/kbar_dk_delta.png', dpi=130)
