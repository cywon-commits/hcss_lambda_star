"""Phase B step 4-5: K(W) extrapolation per orientation series and isotropy verdict.
Reads runs/scan_<name>_<range>.npy (tiltscan output) and runs/chi_<name>.log; writes figures/k_extrapolation.png and prints tables."""
import numpy as np, re, json, itertools, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
CYL = {'2_2_0_-1': 0, '2_3_0_-1': 0, '2_3_1_-1': 0, '3_3_0_-1': 0, '3_4_0_-2': 0, '3_4_1_-1': 0, '4_4_0_-2': 0}
z = np.exp(1j*np.pi/6)
def kfit(f):
    R = np.load(f); s = R[:, 2]; al = R[:, 3]+1j*R[:, 4]; be = R[:, 5]+1j*R[:, 6]; nv = R[:, 7].mean()
    X = np.c_[np.ones_like(s), -abs(al)**2/(2*nv), -be.real**2/(2*nv), -be.imag**2/(2*nv)]
    cf = np.linalg.lstsq(X, s, rcond=None)[0]; return cf[1:], np.sqrt(np.mean((X@cf-s)**2)), np.abs(al).max()
info = {}
for n in CYL:
    a = np.array([int(x) for x in n.split('_')]); P = a@np.array([z**k for k in range(4)])
    Ps = a@np.array([z**(5*k) for k in range(4)])
    K3, r3, am3 = kfit(f'runs/scan_{n}_0.3.npy'); K2, r2, am2 = kfit(f'runs/scan_{n}_0.2.npy')
    chi = float(re.search(r'chi=dx/dlnmu=([-\d.e]+)', open(f'runs/chi_{n}.log').read()).group(1))
    s = float(re.search(r's=([\d.]+)', open(f'runs/chi_{n}.log').read()).group(1))
    info[n] = dict(W=abs(P), arg=np.degrees(np.angle(P)), g=abs(Ps)/abs(P), K3=K3, K2=K2, rms3=r3, rms2=r2, chi=chi, s=s)
names = ['K_alpha', 'K_r', 'K_i']
print('%-9s %6s %6s %6s | K(0.3): %6s %6s %6s | K(0.2): %6s %6s %6s | spread  chi' % ('cyl', '|P|', 'arg', '|g|', *names, *names))
for n, d in info.items():
    print('%-9s %6.3f %6.1f %6.3f |         %6.3f %6.3f %6.3f |         %6.3f %6.3f %6.3f | %5.3f %8.5f' % (n, d['W'], d['arg'], d['g'], *d['K3'], *d['K2'], np.ptp(d['K3']), d['chi']))
series = {'0deg (3.732, 6.464, 7.464)': ['2_2_0_-1', '3_4_0_-2', '4_4_0_-2'], '15deg (5.278, 7.210)': ['2_3_1_-1', '3_4_1_-1']}
res = {}
def fit(W, K, order):
    X = np.c_[np.ones_like(W)] if order == 0 else np.c_[np.ones_like(W), W**-2] if order == 1 else np.c_[np.ones_like(W), W**-2, W**-4]
    cf = np.linalg.lstsq(X, K, rcond=None)[0]; return cf, X@cf-K
print()
for sname, ns in series.items():
    W = np.array([info[n]['W'] for n in ns]); res[sname] = {}
    for ci, ch in enumerate(names):
        K3 = np.array([info[n]['K3'][ci] for n in ns]); K2 = np.array([info[n]['K2'][ci] for n in ns])
        a, ra = fit(W, K3, 1); a2, _ = fit(W, K2, 1)
        b = fit(W, K3, 2)[0] if len(W) >= 3 else None
        err_rng = abs(a[0]-a2[0]); err_form = abs(a[0]-b[0]) if b is not None else np.nan
        err_res = np.sqrt(np.sum(ra**2)/max(len(W)-2, 1)) if len(W) > 2 else 0.0
        res[sname][ch] = dict(Kinf=a[0], b=a[1], Kinf_b4=None if b is None else b[0], err_rng=err_rng, err_form=err_form, err_res=err_res, resid=ra)
        print('%-26s %-8s K_inf(1/W^2)=%6.3f b=%7.3f | 0.2-range K_inf=%6.3f | +1/W^4: %s | fit resid %s' % (sname, ch, a[0], a[1], a2[0], 'n/a (2 pts)' if b is None else '%.3f' % b[0], np.round(ra, 3)))
# verdict: compare K_inf between the two series (and between channels), sigma = quadrature of available error terms
print()
for ch in names:
    A, B = [res[s][ch] for s in series]
    sig = lambda r: np.sqrt(r['err_rng']**2+(0 if np.isnan(r['err_form']) else r['err_form'])**2+r['err_res']**2)
    sA, sB = sig(A), sig(B); dK = A['Kinf']-B['Kinf']; comb = np.hypot(sA, sB)
    print('%-8s Kinf 0deg=%.3f+-%.3f  15deg=%.3f+-%.3f(range only; 2 pts)  diff=%.3f  2sigma=%.3f -> %s' % (ch, A['Kinf'], sA, B['Kinf'], sB, dK, 2*comb, 'consistent' if abs(dK) < 2*comb else 'inconsistent'))
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
col = {'K_alpha': 'C0', 'K_r': 'C1', 'K_i': 'C2'}
for k, (sname, ns) in enumerate(list(series.items())):
    pass
for ci, ch in enumerate(names):
    for sname, mk in zip(series, ('o', 's')):
        ns = series[sname]; W = np.array([info[n]['W'] for n in ns]); K = np.array([info[n]['K3'][ci] for n in ns])
        ax[ci].plot(W**-2, K, mk, ms=7, label=sname.split(' ')[0])
        r = res[sname][ch]; x = np.linspace(0, 0.08, 20); ax[ci].plot(x, r['Kinf']+r['b']*x, '--', lw=1)
    for n in ('2_3_0_-1', '3_3_0_-1'):
        ax[ci].plot(info[n]['W']**-2, info[n]['K3'][ci], 'x', c='gray', label='other (6.2deg, 5.1deg)' if n == '2_3_0_-1' else None)
    ax[ci].set_title(ch); ax[ci].set_xlabel('1/|P|^2'); ax[ci].grid(alpha=.3)
ax[0].set_ylabel('K'); ax[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig('figures/k_extrapolation.png', dpi=130)
