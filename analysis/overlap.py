# Solomyak overlap algorithm for sigma(ti,ri); exact Z[zeta12] arithmetic
from vec import *
import numpy as np, sys, random
from collections import defaultdict
ti,ri=int(sys.argv[1]),int(sys.argv[2]); NXI=int(sys.argv[3]) if len(sys.argv)>3 else 300
ref={0:[ident(t,V) for t,V in T[ti]],1:[ident(t,V) for t,V in Rh[ri]]}
CL=[(t,k) for t in (0,1) for k in range(12)]
def tv(c,x0):
    t,k=divmod(c,12); return tverts(('A' if t==0 else 'R',x0,k))
CH={}
for c in range(24):
    t,k=divmod(c,12)
    CH[c]=[(TY[st]*12+(j+k)%12, rot(y0,k)) for st,y0,j in ref[t]]
def poly(c,x0): return np.array([cplx(v) for v in tv(c,x0)])
def overlap(P,Q,eps=1e-9):
    for A in (P,Q):
        n=len(A)
        for i in range(n):
            e=A[(i+1)%n]-A[i]; nrm=complex(e.imag,-e.real)
            pa=[(v*nrm.conjugate()).real for v in P]; pb=[(v*nrm.conjugate()).real for v in Q]
            if max(pa)<=min(pb)+eps or max(pb)<=min(pa)+eps: return False
    return True
ZERO=(0,0,0,0)
def children(node):
    i,j,g=node; out=[]
    Lg=mul(lam,g)
    A=[(a,sa,poly(a,sa)) for a,sa in CH[i]]
    B=[(b,add(Lg,sb),poly(b,add(Lg,sb))) for b,sb in CH[j]]
    for a,sa,Pa in A:
        for b,sb,Pb in B:
            if np.abs(Pa.mean()-Pb.mean())<2.0 and overlap(Pa,Pb):
                out.append((a,b,sub(sb,sa)))
    return out
def is_coinc(node):
    i,j,g=node; return set(tv(i,ZERO))==set(tv(j,g))
# --- initial overlaps from the fixed point patch: T vs T - xi
st=vsigma(ti,ri)
ty=np.ones(12,dtype=int); X=np.zeros((12,4),dtype=np.int64); K=np.arange(12)
for _ in range(3): ty,X,K=st(ty,X,K)
cls=ty*12+K; xc=X@Mx
cen=[(cls[n],tuple(X[n])) for n in range(len(cls))]
pc=np.array([poly(c,x).mean() for c,x in cen])
grid=defaultdict(list)
for n,p in enumerate(pc): grid[(int(np.floor(p.real)),int(np.floor(p.imag)))].append(n)
core=[n for n in range(len(cen)) if abs(pc[n])<40]
random.seed(0); init=set()
for _ in range(NXI):
    a,b=random.sample(core,2); xi=sub(cen[b][1],cen[a][1])   # shift so tile b sits on tile a
    xic=cplx(xi)
    for n in random.sample(core,200):
        c,x=cen[n]; P=poly(c,x)
        q=pc[n]+xic
        for dx in(-2,-1,0,1,2):
            for dy in(-2,-1,0,1,2):
                for m in grid.get((int(np.floor(q.real))+dx,int(np.floor(q.imag))+dy),[]):
                    c2,x2=cen[m]; y=sub(x2,xi); Q=poly(c2,y)
                    if abs(P.mean()-Q.mean())<2.0 and overlap(P,Q): init.add((c,c2,sub(y,x)))
print('initial overlaps',len(init))
# closure
nodes=set(init); fr=list(init); edges={}
while fr:
    nf=[]
    for nd in fr:
        ch=children(nd); edges[nd]=ch
        for c in ch:
            if c not in nodes: nodes.add(c); nf.append(c)
    fr=nf
print('closed overlap graph: nodes',len(nodes))
alive=set(nodes)
while True:
    dead=[n for n in alive if not any(c in alive for c in edges[n])]
    if not dead: break
    alive-=set(dead)
print('pruned dead (polygon-artefact) overlaps:',len(nodes)-len(alive))
nodes=alive
coin={n for n in nodes if is_coinc(n)}
nc=[n for n in nodes if n not in coin]; idx={n:i for i,n in enumerate(nc)}
# reachability to coincidence
good=set(coin); changed=True
while changed:
    changed=False
    for n in nc:
        if n not in good and any(c in good for c in edges[n]): good.add(n); changed=True
print('non-coincident nodes',len(nc),' all lead to coincidence:',all(n in good for n in nc))
A=np.zeros((len(nc),len(nc)))
for n in nc:
    for c in edges[n]:
        if c in idx and c in nodes: A[idx[n],idx[c]]+=1
ev=np.linalg.eigvals(A); rho=max(abs(ev))
lamv=2+np.sqrt(3)
print('rho(non-coinc)=%.6f   lambda^2=%.4f   log rho/log lambda=%.5f   log5/loglam=%.5f'%(rho,lamv**2,np.log(rho)/np.log(lamv),np.log(5)/np.log(lamv)))
import pickle; pickle.dump((nc,edges,coin),open(f'ovl_{ti}_{ri}.pkl','wb'))
