# Rigorous overlap-coincidence test with FRACTAL (self-similar) tile supports.
# Potential overlaps = superset of true overlaps (polygons dilated by delta >= Hausdorff dev. of fractal edge).
import sys, itertools, numpy as np, time
from vec import *
from collections import defaultdict
ti,ri=int(sys.argv[1]),int(sys.argv[2])
lamv=2+np.sqrt(3)
# --- 1. rigorous bound on deviation of the limit edge curve from its chord
S=[0,1,-3,1,0]; z=np.exp(1j*np.pi/6)
p=np.concatenate([[0],np.cumsum(z**np.array(S))])/lamv
delta1=np.abs(p.imag).max(); DELTA=delta1*lamv/(lamv-1)
print('delta1=%.6f  delta bound=%.6f'%(delta1,DELTA))
# --- 2. substitution data (anchor offsets) and window bound
ref={0:[ident(t,V) for t,V in T[ti]],1:[ident(t,V) for t,V in Rh[ri]]}
CH={}
for c in range(24):
    t,k=divmod(c,12); CH[c]=[(TY[st]*12+(j+k)%12, rot(y0,k)) for st,y0,j in ref[t]]
Dmax=max(abs(star(y)) for c in CH for _,y in CH[c])
B=Dmax*lamv/(lamv-1); RINT=4*B
print('max|digit*|=%.4f  anchor-window radius bound B=%.4f  |gamma*| bound=%.4f'%(Dmax,B,RINT))
# --- 3. tile geometry
def tv(c,x0):
    t,k=divmod(c,12); return tverts(('A' if t==0 else 'R',x0,k))
ZERO=(0,0,0,0)
POLY0={c:np.array([cplx(v) for v in tv(c,ZERO)]) for c in range(24)}
def axes(P):
    e=np.roll(P,-1)-P; n=e*1j; return n/np.abs(n)
AX={c:axes(POLY0[c]) for c in range(24)}
def potential(i,j,gc):
    P=POLY0[i]; Q=POLY0[j]+gc
    for n in np.concatenate([AX[i],AX[j]]):
        pa=(P*np.conj(n)).real; pb=(Q*np.conj(n)).real
        gap=max(pb.min()-pa.max(), pa.min()-pb.max())
        if gap>2*DELTA+1e-9: return False
    return True
def coinc(i,j,g): return set(tv(i,ZERO))==set(tv(j,g))
# --- 4. candidate gamma: |gamma|<=RPHYS, |gamma*|<=RINT
RPHYS=2*1.932+2*DELTA+0.05
N=int(np.ceil(RINT+RPHYS))+1
cand=[]
rng=range(-N,N+1)
Mx4=np.array([z**k for k in range(4)]); Ms4=np.array([z**(5*k) for k in range(4)])
grid=np.array(list(itertools.product(rng,repeat=4)))
gp=grid@Mx4; gs=grid@Ms4
sel=(np.abs(gp)<=RPHYS)&(np.abs(gs)<=RINT)
G=[tuple(int(v) for v in g) for g in grid[sel]]
print('candidate translations',len(G),' (box N=%d)'%N)
t0=time.time()
init=set()
for g in G:
    gc=cplx(g)
    for i in range(24):
        for j in range(24):
            if potential(i,j,gc): init.add((i,j,g))
print('initial potential overlaps',len(init),'%.0fs'%(time.time()-t0),flush=True)
# --- 5. closure under substitution
def children(node):
    i,j,g=node; Lg=mul(lam,g); out=[]
    for a,sa in CH[i]:
        for b,sb in CH[j]:
            gg=sub(add(Lg,sb),sa)
            if abs(cplx(gg))>RPHYS: continue
            if potential(a,b,cplx(gg)): out.append((a,b,gg))
    return out
nodes=set(init); fr=list(init); E_={}
while fr:
    nf=[]
    for nd in fr:
        ch=children(nd); E_[nd]=ch
        for c in ch:
            if c not in nodes: nodes.add(c); nf.append(c)
    fr=nf
maxint=max(abs(star(n[2])) for n in nodes)
print('closed potential-overlap graph: nodes',len(nodes),' max|gamma*| in graph %.3f (bound %.3f)'%(maxint,RINT),'%.0fs'%(time.time()-t0),flush=True)
coin={n for n in nodes if coinc(*n)}
good=set(coin); ch=True
while ch:
    ch=False
    for n in nodes:
        if n not in good and any(c in good for c in E_[n]): good.add(n); ch=True
NR=[n for n in nodes if n not in good]; idx={n:k for k,n in enumerate(NR)}
print('coincidences',len(coin),' nodes reaching coincidence',len(good),' never reaching',len(NR))
import scipy.sparse as sp, scipy.sparse.linalg as sla
rows,cols=[],[]
for n in NR:
    for c in E_[n]:
        if c in idx: rows.append(idx[n]); cols.append(idx[c])
A=sp.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(NR),len(NR)))
if len(NR)<4000: rho=max(abs(np.linalg.eigvals(A.toarray())))
else: rho=abs(sla.eigs(A,k=1,which='LM',return_eigenvectors=False)[0])
print('rho(never-coincident subgraph) = %.6f   lambda^2 = %.6f   ->  %s'%(rho,lamv**2,'PURE POINT (overlap coincidence)' if rho<lamv**2-1e-6 else 'NOT established'))
print('log rho / log lambda = %.5f'%(np.log(rho)/np.log(lamv)))
