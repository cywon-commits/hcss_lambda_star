# transfer matrix on a cylinder: helpers (lift, canonical front, front split) for cfg-generalised tilings
import sys, itertools
import numpy as np
import cfg as C
from zz import add, star, cplx
from front import *
def lift(d, st=None):
    P = [tuple(0 for _ in range(C.r)) if st is None else st]
    for x in d: P.append(add(P[-1], C.E[x]))
    return P
PHAT = [1 + 0j]
def canon_front(d):
    P = lift(d)[:-1]; ys = [round((cplx(p) * PHAT[0].conjugate()).imag, 9) for p in P]; m = min(ys)
    cands = [tuple(d[i:] + d[:i]) for i in range(len(d)) if ys[i] == m]
    return min(cands)
def front_split(d, Pv, negP):
    """returns (front, [hole loops]) or None"""
    d = cancel(d)
    P = lift(d); n = len(d)
    idx = {}
    for i, p in enumerate(P[:-1]): idx.setdefault(p, []).append(i)
    for i, p in enumerate(P[:-1]):
        for q, typ in ((p, 0), (add(p, Pv), 1)):
            for j in idx.get(q, []):
                if j <= i: continue
                if typ == 0: hole = d[i:j]; rest = d[:i] + d[j:]
                else: hole = d[:i] + d[j:]; rest = d[i:j]
                hp = closed_pieces(hole)
                if hp is None: return None
                r = front_split(rest, Pv, negP)
                if r is None: return None
                return (r[0], r[1] + hp)
        if add(p, negP) in idx: return None
    Cc = [cplx(p) for p in P]; pc = cplx(Pv); segs = []
    for sh in (-pc, 0, pc): segs += [(Cc[k] + sh, Cc[k + 1] + sh) for k in range(n)]
    if crosses(segs): return None
    return (canon_front(d), [])
def path_for(a):
    st = []
    for i, x in enumerate(a): st += [i if x > 0 else i + C.half] * abs(x)
    return st
def find_d0(a, Pv=None):
    Pv = tuple(a) if Pv is None else Pv
    PHAT[0] = cplx(Pv) / abs(cplx(Pv))
    for perm in itertools.permutations(path_for(a)):
        r = front_split(list(perm), Pv, tuple(-x for x in Pv))
        if r is not None and not r[1]: return list(perm)
    raise RuntimeError('no start front')
