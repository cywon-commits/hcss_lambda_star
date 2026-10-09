import pickle, numpy as np
from collections import defaultdict
from shapely.geometry import Polygon, LineString
from zz import cplx, add, E
R,SEG,DaI,DbI,TI=pickle.load(open('vmodes.pkl','rb'))
X=SEG['B1'][0]; Xc=cplx(X)
SEGL={k:LineString([(cplx(a).real,cplx(a).imag),(cplx(b).real,cplx(b).imag)]) for k,(a,b) in SEG.items()}
def ordered(vs):
    C=[cplx(v) for v in vs]; c=sum(C)/len(C); C.sort(key=lambda z:np.angle(z-c)); return [(z.real,z.imag) for z in C]
def segs_of(t):
    P=Polygon(ordered(t[1])).buffer(-1e-7)
    return tuple(k for k in ('B1','B2','B3') if P.intersection(SEGL[k]).length>1e-7)
th=np.angle(cplx(SEG['B1'][1])-Xc)
def mirror(z): return Xc+np.exp(2j*th)*np.conj(z-Xc)
def key(S): return frozenset((t[0],frozenset((round(cplx(v).real,5)+0,round(cplx(v).imag,5)+0) for v in vs)) for t,vs in S)
def mkey(S): return frozenset((t[0],frozenset((round(mirror(cplx(v)).real,5)+0,round(mirror(cplx(v)).imag,5)+0) for v in vs)) for t,vs in S)
info={}
for S,n in R.items():
    groups=defaultdict(list); multi=False
    for t in S:
        sg=segs_of(t)
        if len(sg)>1: multi=True
        groups[sg].append(t)
    bond=(not multi) and all(len(g)==2 and all(x[0]=='R' for x in g) for g in groups.values())
    Xvertex=any(X in t[1] for t in S) if S else True
    info[S]=dict(n=n,bond=bond,segs=sorted(set(k for t in S for k in segs_of(t))),ntiles=len(S),
                 types=''.join(sorted(t[0] for t in S)),Xv=Xvertex)
tot=sum(R.values()); nb=sum(v['n'] for S,v in info.items() if v['bond'] or not S)
print('total %d | decoupled+bond-decomposable %d (expected 51857353) | vertex modes %d (%.1f%%)'%(tot,nb,tot-nb,100*(tot-nb)/tot))
# mirror orbits
K={key(S):S for S in R}; seen=set(); orbits=[]
for S in R:
    k=key(S)
    if k in seen: continue
    mk=mkey(S); seen|={k,mk}; orbits.append([S]+([K[mk]] if mk!=k and mk in K else []))
print('distinct crossing sets %d -> mirror orbits %d'%(len(R),len(orbits)))
rows=[]
for orb in orbits:
    S=orb[0]; v=info[S]
    kind='decoupled' if not S else ('bond' if v['bond'] else 'VERTEX')
    rows.append((kind,len(orb),v['n'],v['ntiles'],v['types'],'+'.join(v['segs']),v['Xv'],sum(info[s]['n'] for s in orb)))
rows.sort(key=lambda r:(r[0]!='decoupled',r[0]!='bond',-r[7]))
print('%-9s %-5s %-12s %-6s %-8s %-10s %-9s'%('kind','orbit','count each','tiles','types','segments','X vertex'))
for r in rows: print('%-9s %-5d %-12d %-6d %-8s %-10s %-9s'%(r[0],r[1],r[2],r[3],r[4],r[5],r[6]))
pickle.dump((rows,orbits,info),open('vclass.pkl','wb'))
