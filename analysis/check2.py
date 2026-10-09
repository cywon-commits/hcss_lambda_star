from zz import *
import numpy as np, random, sys
DEPTH=int(sys.argv[1])
S=[0,1,-3,1,0]
def infl(poly): return [(d+s)%12 for d in poly for s in S]
def pts(dirs,start=(0,0,0,0)):
    P=[start]
    for d in dirs: P.append(add(P[-1],E[d%12]))
    return P[:-1]
def pip(q,poly,strict=False,tol=1e-9):
    # winding / on-boundary
    n=len(poly)
    for i in range(n):
        a,b=poly[i],poly[(i+1)%n]
        cr=((b-a).conjugate()*(q-a)).imag
        if abs(cr)<tol*abs(b-a) and -tol<=((q-a)*(b-a).conjugate()).real<=abs(b-a)**2+tol:
            return not strict
    ins=False
    for i in range(n):
        a,b=poly[i],poly[(i+1)%n]
        if (a.imag>q.imag)!=(b.imag>q.imag):
            x=a.real+(q.imag-a.imag)*(b.real-a.real)/(b.imag-a.imag)
            if x>q.real: ins=not ins
    return ins
def solve(base):
    R=pts(infl(base)); Rc=[cplx(p) for p in R]
    area=abs(sum((Rc[i].conjugate()*Rc[(i+1)%len(Rc)]).imag for i in range(len(Rc)))/2)
    # lattice points inside
    L=set(R); fr=list(R)
    for _ in range(DEPTH):
        nf=[]
        for p in fr:
            for e in E:
                q=add(p,e)
                if q not in L and pip(cplx(q),Rc): L.add(q); nf.append(q)
        fr=nf
    cands={}
    for v in L:
        for d in range(12):
            for t,td in (('A',[0,4,8]),('R',[0,1,6,7]),('R',[0,5,6,11])):
                V=pts([d+x for x in td],v); Vc=[cplx(x) for x in V]
                if not all(x in L for x in V): continue
                cen=sum(Vc)/len(Vc)
                chk=Vc+[(Vc[i]+Vc[(i+1)%len(Vc)])/2 for i in range(len(Vc))]+[cen]
                if not all(pip(q,Rc) for q in chk): continue
                if any(pip(r,Vc,strict=True) and not any(abs(r-x)<1e-9 for x in Vc) for r in Rc): continue
                cands[frozenset(V)]=(t,Vc,tuple(V))
    cl=list(cands.values())
    random.seed(1)
    samples=[]
    for t,Vc,_ in cl:
        cen=sum(Vc)/len(Vc)
        for w in Vc:
            samples.append(cen+0.37*(w-cen)+complex(random.uniform(-1e-3,1e-3),random.uniform(-1e-3,1e-3)))
    samples=[s for s in samples if pip(s,Rc,strict=True)]
    cover=[sum(1<<j for j,s in enumerate(samples) if pip(s,Vc,strict=True)) for t,Vc,_ in cl]
    cols={}
    for i,c in enumerate(cover):
        j=0
        while c:
            if c&1: cols.setdefault(j,[]).append(i)
            c>>=1; j+=1
    sols=[]
    def X(uncov,chosen):
        if not uncov:
            sols.append(list(chosen)); return
        best=None
        u=uncov; j=0
        while u:
            if u&1:
                opts=[i for i in cols.get(j,[]) if cover[i]&uncov==cover[i]]
                if best is None or len(opts)<len(best):
                    best=opts
                    if len(best)<=1: break
            u>>=1; j+=1
        for i in best:
            chosen.append(i); X(uncov&~cover[i],chosen); chosen.pop()
    X((1<<len(samples))-1,[])
    tA=np.sqrt(3)/4; good=[s for s in sols if abs(sum(tA if cl[i][0]=='A' else .5 for i in s)-area)<1e-9]
    print(len(L),len(cl),len(samples),len(sols),len(good)); return good,cl
import pickle; pickle.dump((solve([0,4,8]),solve([0,1,6,7])),open('chk%d.pkl'%DEPTH,'wb'))
