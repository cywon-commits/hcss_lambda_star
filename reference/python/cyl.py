# transfer matrix on cylinder
from front import *
from zz import star
import numpy as np, sys, time
from collections import defaultdict
def lift(d,st=(0,0,0,0)):
    P=[st]
    for x in d: P.append(add(P[-1],E[x]))
    return P
PHAT=[1+0j]
def canon_front(d):
    P=lift(d)[:-1]; ys=[round((cplx(p)*PHAT[0].conjugate()).imag,9) for p in P]; m=min(ys)
    cands=[tuple(d[i:]+d[:i]) for i in range(len(d)) if ys[i]==m]
    return min(cands)
def front_split(d,Pv,negP):
    """returns (front, [hole loops]) or None"""
    d=cancel(d)
    P=lift(d); n=len(d)
    idx={}
    for i,p in enumerate(P[:-1]): idx.setdefault(p,[]).append(i)
    for i,p in enumerate(P[:-1]):
        for q,typ in ((p,0),(add(p,Pv),1)):
            for j in idx.get(q,[]):
                if j<=i: continue
                if typ==0:  # closed loop d[i:j] = hole
                    hole=d[i:j]; rest=d[:i]+d[j:]
                else:       # d[i:j] sums to Pv -> new front ; rest = hole
                    hole=d[:i]+d[j:]; rest=d[i:j]
                hp=closed_pieces(hole)
                if hp is None: return None
                r=front_split(rest,Pv,negP)
                if r is None: return None
                return (r[0],r[1]+hp)
        if add(p,negP) in idx: return None
    # crossing test with periodic images
    C=[cplx(p) for p in P]; pc=cplx(Pv)
    segs=[]
    for sh in (-pc,0,pc): segs+=[(C[k]+sh,C[k+1]+sh) for k in range(n)]
    if crosses(segs): return None
    return (canon_front(d),[])
def build(Pv,d0,maxstates=2_000_000,verbose=True):
    negP=tuple(-x for x in Pv)
    s0=canon_front(d0); states={s0:0}; order=[s0]; trans=[]   # (from,to,count,wexp2)
    k=0; t0=time.time()
    while k<len(order):
        d=list(order[k]); din,dout=d[-1],d[0]
        turn=(dout-din)%12; turn=turn-12 if turn>6 else turn; interior=6-turn
        for t,td,ang in TILES:
            if ang>interior: continue
            tdir=[(dout+x)%12 for x in td]
            rep=[(x+6)%12 for x in reversed(tdir[1:])]
            r=front_split(rep+d[1:],Pv,negP)
            if r is None: continue
            f,holes=r; c=1; w=VW[t]
            for h in holes:
                cc,ww=fill_count(h); c*=cc; w+=ww
            if c==0: continue
            if f not in states: states[f]=len(order); order.append(f)
            nA=(1 if t=='A' else 0)+sum(int(round((area(list(h))-fill_count(h)[1]/2)/((np.sqrt(3)-1)/4))) for h in holes)
            trans.append((k,states[f],c,int(round(2*w)),nA))
        k+=1
        if len(order)>maxstates: raise RuntimeError('too many states')
        if verbose and k%20000==0: print('  processed',k,'states',len(order),'%.0fs'%(time.time()-t0),flush=True)
    return order,trans
def entropy(order,trans,mu=1.0):
    import scipy.sparse as sp, scipy.sparse.linalg as sla
    n=len(order); fr,to,c,w,na=map(np.array,zip(*trans))
    def rho(u):
        M=sp.csr_matrix((c*u**w*mu**na,(to,fr)),shape=(n,n)).toarray() if n<2500 else sp.csr_matrix((c*u**w*mu**na,(to,fr)),shape=(n,n))
        if n<2500: return max(abs(np.linalg.eigvals(M)))
        return abs(sla.eigs(M,k=1,which='LM',return_eigenvectors=False,tol=1e-12)[0])
    lo,hi=0.05,1.0
    for _ in range(55):
        mid=(lo+hi)/2
        if rho(mid)>1: hi=mid
        else: lo=mid
    return -2*np.log((lo+hi)/2)
def analyse(order,trans):
    s=entropy(order,trans); d=1e-4
    xA=(entropy(order,trans,np.exp(d))-entropy(order,trans,np.exp(-d)))/(2*d)
    return s, xA, s - xA*0  # s per vertex, triangles per vertex

def path_for(a):
    st=[]
    for i,x in enumerate(a): st+= [i if x>0 else i+6]*abs(x)
    return st
if __name__=='__main__':
    import itertools
    a=tuple(int(x) for x in sys.argv[1:5]); Pv=a
    PHAT[0]=cplx(Pv)/abs(cplx(Pv))
    base=path_for(a)
    # choose an ordering of the steps that is a valid front
    d0=None
    for perm in itertools.permutations(base):
        r=front_split(list(perm),Pv,tuple(-x for x in Pv))
        if r is not None and not r[1]: d0=list(perm); break
    t0=time.time(); order,trans=build(Pv,d0)
    s,xA,_=analyse(order,trans)
    print('P=%s |P|=%.3f |P*|=%.3f states %d trans %d (%.0fs)  s/vertex=%.5f  triangles/vertex=%.4f (12-fold: %.4f)'%(a,abs(cplx(Pv)),abs(star(Pv)),len(order),len(trans),time.time()-t0,s,xA,np.sqrt(3)-1),flush=True)
