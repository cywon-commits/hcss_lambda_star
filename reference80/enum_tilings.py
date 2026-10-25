"""Explicit tilings from the front method with absolute lattice positions (to cross-check fill_count against the independent counter)."""
import sys, math
import cfg as C
from zz import cplx, add
import front
from front import pos, cancel, crosses, area
def enum(d):
    n, h = C.n, C.half
    def pieces(nd, q):
        nd = list(nd); ch = True
        while ch and len(nd) >= 2:
            ch = False
            for i in range(len(nd)):
                j = (i + 1) % len(nd)
                if (nd[i] - nd[j]) % n == h:
                    if j == 0: q = add(q, C.E[nd[0]])
                    nd = [x for k, x in enumerate(nd) if k not in (i, j)]; ch = True; break
        if not nd: return []
        P = [q]
        for x in nd: P.append(add(P[-1], C.E[x]))
        seen = {}
        for i, p in enumerate(P[:-1]):
            if p in seen:
                i0 = seen[p]
                a = pieces(nd[i0:i], P[i0]); b = pieces(nd[:i0] + nd[i:], P[0])
                return None if a is None or b is None else a + b
            seen[p] = i
        Cc = [cplx(p) for p in P]
        if crosses([(Cc[i], Cc[i + 1]) for i in range(len(nd))]) or area(nd) < 1e-9: return None
        ii = min(range(len(P) - 1), key=lambda k: (round(Cc[k].imag, 9), round(Cc[k].real, 9)))
        return [(nd[ii:] + nd[:ii], P[ii])]
    def tilings(loop, p0):
        """all tilings (lists of (name, frozenset(vertices))) of the region with boundary loop starting at p0"""
        if not loop: return [[]]
        din, dout = loop[-1], loop[0]
        turn = (dout - din) % n; turn = turn - n if turn > h else turn
        out = []
        for t, td, ang in C.TILES:
            if ang > h - turn: continue
            tdir = [(dout + x) % n for x in td]
            pts = [p0]
            for x in tdir[:-1]: pts.append(add(pts[-1], C.E[x]))
            rep = [(x + h) % n for x in reversed(tdir[1:])]
            ps = pieces(rep + loop[1:], p0)
            if ps is None: continue
            combos = [[(t, frozenset(pts))]]
            for (lp, q) in ps:
                sub = tilings(lp, q)
                combos = [c + s for c in combos for s in sub]
                if not combos: break
            out += combos
        return out
    P = pos(d)[:-1]; Cc = [cplx(p) for p in P]
    ii = min(range(len(P)), key=lambda k: (round(Cc[k].imag, 9), round(Cc[k].real, 9)))
    return tilings(d[ii:] + d[:ii], P[ii])
