# Classify every filling of the DDT cluster by its crossing set S (tiles whose interior meets the skeleton star)
import sys, numpy as np, pickle, time; sys.setrecursionlimit(100000)
from collections import Counter
from front import pos, crosses, area, TILES
from zz import add, E, cplx
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union
from shapely.geometry.polygon import orient
exec(open('pairs.py').read().split("def fills")[0].split("from front import *")[1])   # cancel2, split
O=(0,0,0,0); X=E[0]; Y=add(E[0],E[4])
def poly_int(start,dirs):
    P=[start]
    for d in dirs[:-1]: P.append(add(P[-1],E[d%12]))
    return P
def dodec(a,b):
    d=[k for k in range(12) if add(a,E[k])==b][0]; return poly_int(a,[(d+k)%12 for k in range(12)])
DaI=dodec(X,O); DbI=dodec(Y,X); TI=[O,X,Y]
shp=lambda P: Polygon([(cplx(p).real,cplx(p).imag) for p in P])
DcI=dodec(O,Y)
REG=unary_union([shp(DaI),shp(DbI),shp(DcI),shp(TI)])
def shared(P,Q,v): return [p for p in P if p in Q and p!=v][0]
SEG={'DaDb':(X,shared(DaI,DbI,X)),'DbDc':(Y,shared(DbI,DcI,Y)),'DcDa':(O,shared(DcI,DaI,O)),'TDa':(X,O),'TDb':(X,Y),'TDc':(Y,O)}
SEGL={k:LineString([(cplx(a).real,cplx(a).imag),(cplx(b).real,cplx(b).imag)]) for k,(a,b) in SEG.items()}
# region boundary as int dirs with absolute start
g=orient(REG,1.0); cs=list(g.exterior.coords)[:-1]
allpts={tuple(np.round([cplx(p).real,cplx(p).imag],6)):p for p in DaI+DbI+DcI+TI}
start=allpts[tuple(np.round(cs[0],6))]; dirs=[]
for k in range(len(cs)):
    a=complex(*cs[k]); b=complex(*cs[(k+1)%len(cs)]); L=abs(b-a); dd=int(round(np.angle(b-a)/(np.pi/6)))%12
    dirs+=[dd]*int(round(L))
def crossing(vs):
    P=shp(vs).buffer(-1e-7)
    return any(P.intersection(SEGL[k]).length>1e-7 for k in SEGL)
memo={}
def fill(st,d):
    d,sh=cancel2(d); st=add(st,sh)
    if not d: return Counter({frozenset():1})
    pcs=split(st,d)
    if len(pcs)>1:
        out=Counter({frozenset():1})
        for s,dd in pcs:
            f=fill(s,dd); new=Counter()
            for a,na in out.items():
                for b,nb in f.items(): new[a|b]+=na*nb
            out=new
            if not out: break
        return out
    P=[add(st,p) for p in pos(d)]; C=[cplx(p) for p in P]
    if crosses([(C[i],C[i+1]) for i in range(len(d))]) or area(d)<1e-9: return Counter()
    i=min(range(len(d)),key=lambda k:(round(C[k].imag,9),round(C[k].real,9)))
    d=d[i:]+d[:i]; v=P[i]; key=(v,tuple(d))
    if key in memo: return memo[key]
    din,dout=d[-1],d[0]; turn=(dout-din)%12; turn=turn-12 if turn>6 else turn
    out=Counter()
    for t,td,ang in TILES:
        if ang>6-turn: continue
        tdir=[(dout+x)%12 for x in td]; vs=[v]
        for x in tdir[:-1]: vs.append(add(vs[-1],E[x]))
        lab=frozenset([(t[0],frozenset(vs))]) if crossing(vs) else frozenset()
        rep=[(x+6)%12 for x in reversed(tdir[1:])]
        for a,n in fill(v,rep+d[1:]).items(): out[a|lab]+=n
    memo[key]=out; return out
t0=time.time(); R=fill(start,dirs)
print('total fillings',sum(R.values()),' (DDDT)   distinct crossing sets',len(R),' %.0fs'%(time.time()-t0))
pickle.dump((dict(R),SEG,DaI,DbI,DcI,TI),open('vmodes3.pkl','wb'))
