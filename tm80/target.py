"""Strain-targeted cylinders: find primitive periods P' (W in [Wlo,Whi], arg in [0,20)) whose g_m = sigma_m(P')/|P'| equals alpha_m Ph' + beta_m conj(Ph')
for the (alpha,beta) of a known branching phase (landscape_ab.json), up to the 18 rotated images.  Usage: python3 tm80/target.py Wlo Whi tol [rmin rmax] > cands.json"""
import sys, json, math
import numpy as np
from math import gcd
z = np.exp(2j * np.pi / 18)
def phases(rmin, rmax, files=('tm80/data/landscape_ab.json',)):
    seen = {}
    for f in files:
        for x in json.load(open(f)):
            if rmin <= x['Nt_Nr'] <= rmax: seen.setdefault(round(x['Nt_Nr'], 3), x)
    return seen
def enum(Wlo, Whi, R=9, smax=22):
    r = np.arange(-R, R + 1); zp = z ** np.arange(6); zp5 = z ** (5 * np.arange(6)); zp7 = z ** (7 * np.arange(6)); out = []
    A = np.array(np.meshgrid(*([r] * 6), indexing='ij')).reshape(6, -1).T
    for i in range(0, len(A), 3_000_000):
        a = A[i:i + 3_000_000]; P = a @ zp; ap = np.abs(P); ang = np.degrees(np.angle(P)) % 360
        m = (ap >= Wlo) & (ap <= Whi) & (ang < 20 - 1e-9) & (np.abs(a).sum(1) <= smax); a = a[m]
        out.append(np.c_[a, ])
    a = np.vstack(out); keep = [all(gcd(*[int(abs(v)) for v in row[:3]], *[int(abs(v)) for v in row[3:]]) == 1 for _ in [0]) for row in a]
    return a[np.array(keep)]
def match(a, ph, tol):
    P = a @ z ** np.arange(6); Ph = P / np.abs(P); res = []
    g = {m: (a @ z ** (m * np.arange(6))) / np.abs(P) for m in (5, 7)}
    for key, x in ph.items():
        al = {5: complex(*x['al5']), 7: complex(*x['al7'])}; be = {5: complex(*x['be5']), 7: complex(*x['be7'])}
        best = np.full(len(a), 9.0)
        for k in range(18):
            e = np.zeros(len(a))
            for m in (5, 7):
                t = al[m] * z ** (k * (m - 1)) * Ph + be[m] * z ** (k * (m + 1)) * np.conj(Ph)
                e = np.maximum(e, np.abs(g[m] - t))
            best = np.minimum(best, e)
        for j in np.nonzero(best < tol)[0]: res.append((tuple(int(v) for v in a[j]), key, float(best[j]), float(abs(P[j]))))
    return res
if __name__ == '__main__':
    Wlo, Whi, tol = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]); rmin = float(sys.argv[4]) if len(sys.argv) > 4 else 1.55; rmax = float(sys.argv[5]) if len(sys.argv) > 5 else 2.6
    ph = phases(rmin, rmax); print(len(ph), 'phases', file=sys.stderr); a = enum(Wlo, Whi); print(len(a), 'periods', file=sys.stderr)
    r = match(a, ph, tol); print(len(r), 'matches', file=sys.stderr); json.dump(r, sys.stdout)
