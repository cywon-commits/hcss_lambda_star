# module arithmetic for Z[zeta_n]; all constants come from cfg (call cfg.select(...) before use)
import numpy as np
import cfg as C
def mulz(a): return C.mulz(a)
def add(a, b): return tuple(x + y for x, y in zip(a, b))
def sub(a, b): return tuple(x - y for x, y in zip(a, b))
def neg(a): return tuple(-x for x in a)
def rot(a, k):
    for _ in range(k % C.n): a = mulz(a)
    return a
def cplx(a): return sum(a[i] * C.ZPOW[i] for i in range(C.r))
def star(a, m=None):
    """Galois conjugate zeta -> zeta^m (default: the first perpendicular map of the configuration)"""
    m = C.PERP[0] if m is None else m
    return sum(a[i] * C.zc ** (m * i) for i in range(C.r))
def mul(a, b):
    r = tuple(0 for _ in a)
    for i in range(C.r):
        t = a
        for _ in range(i): t = mulz(t)
        r = add(r, tuple(b[i] * x for x in t))
    return r
