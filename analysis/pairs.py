from front import *
import numpy as np, itertools
def cancel2(d):
    d=list(d); st=(0,0,0,0); ch=True
    while ch and len(d)>=2:
        ch=False
        for i in range(len(d)):
            j=(i+1)%len(d)
            if (d[i]-d[j])%12==6:
                if j==0: st=add(st,E[d[0]])
                d=[x for k,x in enumerate(d) if k not in (i,j)]; ch=True; break
    return d,st
def split(st,dd):
    Q=pos(dd); seen={}
    for k,p in enumerate(Q[:-1]):
        if p in seen:
            k0=seen[p]; return split(add(st,Q[k0]),dd[k0:k])+split(st,dd[:k0]+dd[k:])
        seen[p]=k
    return [(st,dd)]
def fills(st,d):
    d,sh=cancel2(d); st=add(st,sh)
    if not d: return [[]]
    pcs=split(st,d)
    if len(pcs)>1:
        out=[[]]
        for s,dd in pcs:
            f=fills(s,dd); out=[a+b for a in out for b in f]
        return out
    P=[add(st,p) for p in pos(d)]; C=[cplx(p) for p in P]
    if crosses([(C[i],C[i+1]) for i in range(len(d))]) or area(d)<1e-9: return []
    i=min(range(len(d)),key=lambda k:(round(C[k].imag,9),round(C[k].real,9)))
    d=d[i:]+d[:i]; v=P[i]; din,dout=d[-1],d[0]
    turn=(dout-din)%12; turn=turn-12 if turn>6 else turn
    out=[]
    for t,td,ang in TILES:
        if ang>6-turn: continue
        tdir=[(dout+x)%12 for x in td]; vs=[v]
        for x in tdir[:-1]: vs.append(add(vs[-1],E[x]))
        rep=[(x+6)%12 for x in reversed(tdir[1:])]
        for f in fills(v,rep+d[1:]): out.append([(t,tuple(vs))]+f)
    return out
F=fills((0,0,0,0),list(range(12)))
print('dodecagon fillings',len(F))
hist={}
for f in F:
    V=list({v for t,vs in f for v in vs}); C=np.array([cplx(v) for v in V])
    edges=set(); diag=set()
    for t,vs in f:
        for k in range(len(vs)): edges.add(frozenset((vs[k],vs[(k+1)%len(vs)])))
        if t!='A': 
            # short diagonal joins the two obtuse corners
            for a,b in itertools.combinations(vs,2):
                if abs(abs(cplx(a)-cplx(b))-0.5176)<1e-3: diag.add(frozenset((a,b)))
    D=np.abs(C[:,None]-C[None,:])
    for i,j in zip(*np.where((D>1e-6)&(D<1-1e-6))):
        if i<j:
            pr=frozenset((V[i],V[j]))
            key=round(D[i,j],4),('diag' if pr in diag else 'OTHER')
            hist[key]=hist.get(key,0)+1
print('sub-unit vertex pairs (tile units):',hist)
