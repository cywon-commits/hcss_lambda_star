from subst import *
import numpy as np, sys
def rotmat(k):
    M=np.zeros((4,4),dtype=np.int64)
    for i in range(4): M[:,i]=rot(tuple(int(i==j) for j in range(4)),k)
    return M
ROT=[rotmat(k) for k in range(12)]
LAM=np.zeros((4,4),dtype=np.int64)
for i in range(4): LAM[:,i]=mul(lam,tuple(int(i==j) for j in range(4)))
TY={'A':0,'R':1}
def vsigma(ti,ri):
    ref={0:[ident(t,V) for t,V in T[ti]],1:[ident(t,V) for t,V in Rh[ri]]}
    # precompute children offsets per (type,k): arrays
    ch={}
    for ty in (0,1):
        ys=np.array([c[1] for c in ref[ty]]); js=np.array([c[2] for c in ref[ty]]); ts=np.array([TY[c[0]] for c in ref[ty]])
        for k in range(12):
            ch[(ty,k)]=(ts,(ROT[k]@ys.T).T,(js+k)%12)
    def step(ty,X,K):
        nt,nX,nK=[],[],[]
        LX=(LAM@X.T).T
        for t in (0,1):
            for k in range(12):
                m=(ty==t)&(K==k)
                if not m.any(): continue
                ts,offs,ks=ch[(t,k)]
                nX.append((LX[m][:,None,:]+offs[None,:,:]).reshape(-1,4))
                nt.append(np.tile(ts,m.sum())); nK.append(np.tile(ks,m.sum()))
        return np.concatenate(nt),np.concatenate(nX),np.concatenate(nK)
    return step
def vertices(ty,X,K):
    out=[X]
    E4=np.array(E)
    out.append(X+E4[K]); 
    a=ty==0
    out.append((X+E4[K]+E4[(K+4)%12])[a])
    b=~a
    out.append((X+E4[K]+E4[(K+1)%12])[b]); out.append((X+E4[(K+1)%12])[b])
    return np.unique(np.concatenate(out),axis=0)
Mx=np.array([zc**k for k in range(4)]); Ms=np.array([zc**(5*k) for k in range(4)])
