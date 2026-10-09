import pickle, numpy as np, sys
from zz import *
from collections import Counter, defaultdict
T,Rh,lam=pickle.load(open('fills.pkl','rb'))
ZERO=(0,0,0,0)
def ident(t,V):
    V=set(V)
    for x0 in V:
        for k in range(12):
            if t=='A' and add(x0,E[k]) in V and add(add(x0,E[k]),E[(k+4)%12]) in V: return ('A',x0,k)
            if t=='R' and add(x0,E[k]) in V and add(x0,E[(k+1)%12]) in V and add(add(x0,E[k]),E[(k+1)%12]) in V: return ('R',x0,k)
    raise
def tverts(t):
    ty,x0,k=t
    if ty=='A': return (x0,add(x0,E[k]),add(add(x0,E[k]),E[(k+4)%12]))
    return (x0,add(x0,E[k]),add(add(x0,E[k]),E[(k+1)%12]),add(x0,E[(k+1)%12]))
def make_sigma(ti,ri):
    ref={'A':[ident(t,V) for t,V in T[ti]], 'R':[ident(t,V) for t,V in Rh[ri]]}
    def sigma(tiles):
        out=[]
        for ty,x0,k in tiles:
            X=mul(lam,x0)
            for sty,y0,j in ref[ty]:
                out.append((sty,add(X,rot(y0,k)),(j+k)%12))
        return out
    return sigma
def canon(t): return (t[0],frozenset(tverts(t)))
STAR=[('R',ZERO,k) for k in range(12)]
inv_T=[6,12]; inv_R=[4,15,20]
