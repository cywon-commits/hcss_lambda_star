from subst import *
import numpy as np, sys, sympy as sp
from collections import defaultdict, Counter
ti,ri=int(sys.argv[1]),int(sys.argv[2])
ref={'A':[ident(t,V) for t,V in T[ti]],'R':[ident(t,V) for t,V in Rh[ri]]}
def sig(tiles):
    out=[];par=[]
    for n,(ty,x0,k) in enumerate(tiles):
        X=mul(lam,x0)
        for sty,y0,j in ref[ty]:
            out.append((sty,add(X,rot(y0,k)),(j+k)%12)); par.append(n)
    return out,par
def sides(t):
    V=tverts(t); n=len(V); out=[]
    for i in range(n):
        a,b=V[i],V[(i+1)%n]
        sc='A' if t[0]=='A' else ('R0' if i%2==0 else 'R1')
        out.append((frozenset((a,b)),sc))
    return out
def analyse(tiles):
    corn=defaultdict(list); edg=defaultdict(list); inc=defaultdict(list)
    for n,t in enumerate(tiles):
        V=tverts(t); m=len(V); Vc=[cplx(v) for v in V]
        for i in range(m):
            a=Vc[(i-1)%m]-Vc[i]; b=Vc[(i+1)%m]-Vc[i]
            ang=int(round(np.angle(a/b)/(np.pi/6)))%12; d=int(round(np.angle(b)/(np.pi/6)))%12
            corn[V[i]].append((d,ang,t[0])); inc[V[i]].append(n)
        for e,sc in sides(t): edg[e].append((n,sc))
    vt={}
    for v,cs in corn.items():
        if sum(c[1] for c in cs)==12:
            s=tuple(f"{c[2]}{c[1]}" for c in sorted(cs)); vt[v]=min(s[i:]+s[:i] for i in range(len(s)))
    et={e:'|'.join(sorted(sc for _,sc in L)) for e,L in edg.items() if len(L)==2}
    return vt,et,edg,inc
P=STAR
for _ in range(2): P,_=sig(P)
P2,par=sig(P)
vt,et,edg,inc=analyse(P); vt2,et2,edg2,inc2=analyse(P2)
VV=defaultdict(Counter); Sm=defaultdict(Counter); Um=defaultdict(Counter); Pm=defaultdict(Counter); Qm=defaultdict(Counter); VE=defaultdict(Counter)
lamv={mul(lam,v):v for v in vt}
vset=[set(tverts(t)) for t in P]
def shared_edge(a,b):
    sh=vset[a]&vset[b]
    if len(sh)==2:
        e=frozenset(sh)
        if e in et: return e
    return None
def common_vertex(ps):
    c=set.intersection(*[vset[p] for p in ps])
    return list(c)[0] if len(c)==1 else None
def edge_ok(e): return all(v in vt for v in e)
for w,tw in vt2.items():
    if w in lamv: VV[('v',lamv[w])][tw]+=1; continue
    ps=list(set(par[n] for n in inc2[w]))
    if len(ps)==1: Sm[('t',ps[0])][tw]+=1; continue
    if len(ps)==2:
        e=shared_edge(*ps)
        if e is not None:
            if edge_ok(e): Um[('e',e)][tw]+=1
            continue
    v=common_vertex(ps)
    if v is not None and v in vt: VV[('v',v)][tw]+=1
for e2,L in edg2.items():
    if len(L)!=2: continue
    pa,pb=par[L[0][0]],par[L[1][0]]
    if pa==pb: Pm[('t',pa)][et2[e2]]+=1; continue
    e=shared_edge(pa,pb)
    if e is not None:
        if edge_ok(e): Qm[('e',e)][et2[e2]]+=1
        continue
    v=common_vertex([pa,pb])
    if v is not None and v in vt: VE[('v',v)][et2[e2]]+=1
# keep only vertices / edges whose full neighbourhood lies well inside patch
cv=cplx
Rmax=max(abs(cplx(v)) for v in vt)*0.6
def inner(k):
    x=k[1]
    if k[0]=='t': return abs(cplx(tverts(P[x])[0]))<Rmax
    if k[0]=='e': return all(abs(cplx(v))<Rmax for v in x)
    return abs(cplx(x))<Rmax
def collapse(D,keyf,allkeys):
    out={}
    for k in allkeys:
        if not inner(k): continue
        out.setdefault(keyf(k),set()).add(frozenset(D.get(k,Counter()).items()))
    for cl,s in out.items():
        if len(s)!=1: print('INCONSISTENT',cl,[dict(x) for x in s])
    return {cl:dict(next(iter(s))) for cl,s in out.items()}
tiles_k=[('t',n) for n in range(len(P))]
edges_k=[('e',e) for e in et if edge_ok(e)]
verts_k=[('v',v) for v in vt]
S=collapse(Sm,lambda k:P[k[1]][0],tiles_k); Pt=collapse(Pm,lambda k:P[k[1]][0],tiles_k)
U=collapse(Um,lambda k:et[k[1]],edges_k); Q=collapse(Qm,lambda k:et[k[1]],edges_k)
VVc=collapse(VV,lambda k:vt[k[1]],verts_k); VEc=collapse(VE,lambda k:vt[k[1]],verts_k)
print('S',S);print('U',U);print('Q',Q);print('Pt',Pt)
print('vertex-attributed vertices',{ '-'.join(a):{'-'.join(b):c for b,c in d.items()} for a,d in VVc.items()})
print('vertex-attributed edges',{ '-'.join(a):d for a,d in VEc.items()})
VT=sorted(set(vt2.values())|set(vt.values())); ET=sorted(set(et.values())|set(et2.values()))
print('vertex types',len(VT),'edge classes',ET)
l2=sp.Integer(7)+4*sp.sqrt(3); t={'A':2/sp.sqrt(3),'R':sp.Integer(1)}
nE,nV=len(ET),len(VT); N=nE+nV
Mx_=sp.zeros(N,N); b=sp.zeros(N,1)
for e,d in Q.items():
    for e2,c in d.items(): Mx_[ET.index(e2),ET.index(e)]+=c
for v,d in VEc.items():
    for e2,c in d.items(): Mx_[ET.index(e2),nE+VT.index(v)]+=c
for e,d in U.items():
    for v2,c in d.items(): Mx_[nE+VT.index(v2),ET.index(e)]+=c
for v,d in VVc.items():
    for v2,c in d.items(): Mx_[nE+VT.index(v2),nE+VT.index(v)]+=c
for ty,d in Pt.items():
    for e2,c in d.items(): b[ET.index(e2)]+=c*t[ty]
for ty,d in S.items():
    for v2,c in d.items(): b[nE+VT.index(v2)]+=c*t[ty]
print('eigenvalues of coupled edge/vertex matrix:',sp.Matrix(Mx_).eigenvals())
sol=(l2*sp.eye(N)-Mx_).LUsolve(b)
sol=[sp.radsimp(sp.simplify(x)) for x in sol]
eps=sol[:nE]; nu=sol[nE:]
print('edges/area =',sp.simplify(sum(eps)),' (expect 2+sqrt3)')
print('vertices/area =',sp.radsimp(sp.simplify(sum(nu))),' (expect 1+sqrt3/3)')
tot=sp.radsimp(sp.simplify(sum(nu)))
emp=Counter(v for k,v in vt2.items() if abs(cplx(k))<0.5*max(abs(cplx(x)) for x in vt2))
ne=sum(emp.values())
for v,x in sorted(zip(VT,nu),key=lambda z:-float(z[1])):
    f=sp.radsimp(sp.simplify(x/tot))
    print('%-34s freq %-22s = %.6f  (patch %.4f)   window area %s'%('-'.join(v),f,float(f),emp[v]/ne,sp.radsimp(sp.simplify(3*x))))
