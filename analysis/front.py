# exact front-growth counting for triangle (A) + 30-degree rhombus (R) tilings, edges = unit vectors zeta^k
import numpy as np
from functools import lru_cache
from zz import E, add, sub, cplx
TILES=(('A',(0,4,8),2),('Ra',(0,1,6,7),1),('Ro',(0,5,6,11),5))
VW={'A':0.5,'Ra':1.0,'Ro':1.0}          # vertices per tile (angle sum / 2pi)
def cancel(d):
    d=list(d); ch=True
    while ch and len(d)>=2:
        ch=False
        for i in range(len(d)):
            j=(i+1)%len(d)
            if (d[i]-d[j])%12==6:
                d=[x for k,x in enumerate(d) if k not in (i,j)]; ch=True; break
    return d
def pos(d):
    P=[(0,0,0,0)]
    for x in d: P.append(add(P[-1],E[x]))
    return P
def crosses(segs):
    cr=lambda a,b:(a.conjugate()*b).imag
    n=len(segs); eps=1e-9
    for i in range(n):
        a,b=segs[i]
        for j in range(i+1,n):
            c,dd=segs[j]
            if min(abs(a-c),abs(a-dd),abs(b-c),abs(b-dd))>2.01: continue
            r,s=b-a,dd-c; den=cr(r,s)
            if abs(den)<1e-12:
                if abs(cr(c-a,r))<1e-9:
                    t0=((c-a)*r.conjugate()).real; t1=((dd-a)*r.conjugate()).real
                    if min(1,max(t0,t1))-max(0,min(t0,t1))>eps: return True
                continue
            t=cr(c-a,s)/den; u=cr(c-a,r)/den
            if eps<t<1-eps and -eps<u<1+eps: return True
            if eps<u<1-eps and -eps<t<1+eps: return True
    return False
def area(d):
    P=[cplx(p) for p in pos(d)]
    return sum((P[i].conjugate()*P[i+1]).imag for i in range(len(d)))/2
def canon_closed(d):
    P=pos(d)[:-1]; C=[cplx(p) for p in P]
    i=min(range(len(P)),key=lambda k:(round(C[k].imag,9),round(C[k].real,9)))
    return tuple(d[i:]+d[:i])
# ---------- finite regions (closed loops, CCW) ----------
@lru_cache(maxsize=None)
def fill_count(d):
    """d canonical closed CCW loop starting at lowest-left vertex. returns (#fillings, vertex-weight sum of tiles)"""
    if not d: return (1,0.0)
    d=list(d); din,dout=d[-1],d[0]
    turn=(dout-din)%12; turn=turn-12 if turn>6 else turn
    interior=6-turn; tot=0; vw=None
    for t,td,ang in TILES:
        if ang>interior: continue
        tdir=[(dout+x)%12 for x in td]
        rep=[(x+6)%12 for x in reversed(tdir[1:])]
        nd=cancel(rep+d[1:])
        r=closed_pieces(nd)
        if r is None: continue
        c=1; w=VW[t]
        for piece in r:
            cc,ww=fill_count(piece); c*=cc; w+=ww
            if c==0: break
        if c: tot+=c; vw=w
    return (tot,vw if vw is not None else 0.0)
def closed_pieces(nd):
    nd=cancel(nd)
    if not nd: return []
    P=pos(nd); seen={}
    for i,p in enumerate(P[:-1]):
        if p in seen:
            i0=seen[p]
            a=closed_pieces(nd[i0:i]); b=closed_pieces(nd[:i0]+nd[i:])
            return None if a is None or b is None else a+b
        seen[p]=i
    C=[cplx(p) for p in P]
    if crosses([(C[i],C[i+1]) for i in range(len(nd))]): return None
    if area(nd)<1e-9: return None
    return [canon_closed(nd)]
