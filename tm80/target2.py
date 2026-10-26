"""Larger-W strain targeting (chunked enumeration).  Usage: python3 tm80/target2.py Wlo Whi tol R rlo rhi > cands.json
phases = distinct (N_t/N_r rounded to 3 digits) branching cylinders from landscape_ab.json and targeted_cyl.jsonl with rlo <= N_t/N_r <= rhi."""
import sys, json, math
import numpy as np
from math import gcd
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from target import match
z = np.exp(2j * np.pi / 18)
def phases(rlo, rhi):
    seen = {}
    for f, ld in (('tm80/data/landscape_ab.json', json.load), ('tm80/data/targeted_cyl.jsonl', lambda fh: [json.loads(l) for l in fh])):
        for x in ld(open(f)):
            if x.get('status', 'ok') == 'ok' and 'al5' in x and rlo <= x['Nt_Nr'] <= rhi: seen.setdefault(round(x['Nt_Nr'], 3), x)
    return seen
def enum(Wlo, Whi, R, smax):
    r = np.arange(-R, R + 1); zp = z ** np.arange(6); rest = np.array(np.meshgrid(*([r] * 4), indexing='ij')).reshape(4, -1).T; out = []
    for a0 in r:
        for a1 in r:
            a = np.c_[np.full(len(rest), a0), np.full(len(rest), a1), rest]; P = a @ zp; ap = np.abs(P); ang = np.degrees(np.angle(P)) % 360
            m = (ap >= Wlo) & (ap <= Whi) & (ang < 20 - 1e-9) & (np.abs(a).sum(1) <= smax)
            if m.any(): out.append(a[m])
    a = np.vstack(out); g = np.gcd.reduce(np.abs(a), axis=1); return a[g == 1]
if __name__ == '__main__':
    Wlo, Whi, tol, R = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]); rlo, rhi = float(sys.argv[5]), float(sys.argv[6])
    ph = phases(rlo, rhi); print(len(ph), 'phases', sorted(ph), file=sys.stderr); a = enum(Wlo, Whi, R, 3 * R + 6); print(len(a), 'periods', file=sys.stderr)
    r = match(a, ph, tol); print(len(r), 'matches', file=sys.stderr); json.dump(r, sys.stdout)
