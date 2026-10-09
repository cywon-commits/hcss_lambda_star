# Optimised crossing-set enumeration for DDT / DDDT clusters (same output as vmodes.py / vmodes3.py).
#   python3 vmodes_fast.py DDT    -> vmodes_fast_DDT.pkl   (must give 61174241 fillings, 29 crossing sets)
#   python3 vmodes_fast.py DDDT   -> vmodes_fast_DDDT.pkl  (writes the same tuple layout as vmodes3.pkl)
# Changes vs vmodes*.py: crossing tiles are interned to int ids and labels are int bitmasks;
# sub-regions that the skeleton does not pass through are counted with front.fill_count.
import sys, numpy as np, pickle, time; sys.setrecursionlimit(100000)
from collections import Counter
from front import pos, crosses, area, TILES, closed_pieces, fill_count
from zz import add, E, cplx
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union
from shapely.geometry.polygon import orient
exec(open('pairs.py').read().split("def fills")[0].split("from front import *")[1])   # cancel2, split
which=sys.argv[1] if len(sys.argv)>1 else 'DDDT'
O=(0,0,0,0); X=E[0]; Y=add(E[0],E[4])
def poly_int(start,dirs):
    P=[start]
    for d in dirs[:-1]: P.append(add(P[-1],E[d%12]))
    return P
def dodec(a,b):
    d=[k for k in range(12) if add(a,E[k])==b][0]; return poly_int(a,[(d+k)%12 for k in range(12)])
shp=lambda P: Polygon([(cplx(p).real,cplx(p).imag) for p in P])
def shared(P,Q,v): return [p for p in P if p in Q and p!=v][0]
DaI=dodec(X,O); DbI=dodec(Y,X); TI=[O,X,Y]
if which=='DDT':
    cells=[DaI,DbI,TI]
    SEG={'B1':(X,shared(DaI,DbI,X)),'B2':(X,O),'B3':(X,Y)}
else:
    DcI=dodec(O,Y); cells=[DaI,DbI,DcI,TI]
    SEG={'DaDb':(X,shared(DaI,DbI,X)),'DbDc':(Y,shared(DbI,DcI,Y)),'DcDa':(O,shared(DcI,DaI,O)),'TDa':(X,O),'TDb':(X,Y),'TDc':(Y,O)}
REG=unary_union([shp(c) for c in cells])
SEGL={k:LineString([(cplx(a).real,cplx(a).imag),(cplx(b).real,cplx(b).imag)]) for k,(a,b) in SEG.items()}
SKEL=unary_union(list(SEGL.values()))
g=orient(REG,1.0); cs=list(g.exterior.coords)[:-1]
allpts={tuple(np.round([cplx(p).real,cplx(p).imag],6)):p for c in cells for p in c}
start=allpts[tuple(np.round(cs[0],6))]; dirs=[]
for k in range(len(cs)):
    a=complex(*cs[k]); b=complex(*cs[(k+1)%len(cs)]); L=abs(b-a); dd=int(round(np.angle(b-a)/(np.pi/6)))%12
    dirs+=[dd]*int(round(L))
# --- crossing tiles interned to bitmask ids
TID={}; TLAB=[]
def crossing_id(t0,vs):
    key=(t0,frozenset(vs)); r=TID.get(key)
    if r is None:
        P=shp(vs).buffer(-1e-7)
        r=(1<<len(TLAB)) if any(P.intersection(SEGL[k]).length>1e-7 for k in SEGL) else 0
        TID[key]=r
        if r: TLAB.append(key)
    return r
# --- skeleton-free regions: plain count
FREE={}
def far_count(st,d):
    """number of fillings if the closed loop (st,d) misses the skeleton interior, else None"""
    key=(st,tuple(d)); r=FREE.get(key,-1)
    if r!=-1: return r
    P=Polygon([(cplx(add(st,p)).real,cplx(add(st,p)).imag) for p in pos(d)[:-1]])
    if P.is_valid and P.buffer(-1e-7).intersection(SKEL).length>1e-7: r=None
    else:
        # closed_pieces works from the origin: translation invariant
        pcs=closed_pieces(list(d))
        if pcs is None: r=0
        else:
            r=1
            for p in pcs: r*=fill_count(p)[0]
    FREE[key]=r; return r
memo={}; nfar=[0]
def fill(st,d):
    d,sh=cancel2(d); st=add(st,sh)
    if not d: return {0:1}
    pcs=split(st,d)
    if len(pcs)>1:
        out={0:1}
        for s,dd in pcs:
            f=fill(s,dd); new={}
            for a,na in out.items():
                for b,nb in f.items(): new[a|b]=new.get(a|b,0)+na*nb
            out=new
            if not out: break
        return out
    P=[add(st,p) for p in pos(d)]; C=[cplx(p) for p in P]
    if crosses([(C[i],C[i+1]) for i in range(len(d))]) or area(d)<1e-9: return {}
    i=min(range(len(d)),key=lambda k:(round(C[k].imag,9),round(C[k].real,9)))
    d=d[i:]+d[:i]; v=P[i]; key=(v,tuple(d))
    r=memo.get(key)
    if r is not None: return r
    fc=far_count(v,d)
    if fc is not None:
        nfar[0]+=1; memo[key]={0:fc} if fc else {}; return memo[key]
    din,dout=d[-1],d[0]; turn=(dout-din)%12; turn=turn-12 if turn>6 else turn
    out={}
    for t,td,ang in TILES:
        if ang>6-turn: continue
        tdir=[(dout+x)%12 for x in td]; vs=[v]
        for x in tdir[:-1]: vs.append(add(vs[-1],E[x]))
        lab=crossing_id(t[0],vs)
        rep=[(x+6)%12 for x in reversed(tdir[1:])]
        for a,n in fill(v,rep+d[1:]).items(): out[a|lab]=out.get(a|lab,0)+n
    memo[key]=out; return out
t0=time.time(); Rm=fill(start,dirs)
R={}
for m,n in Rm.items():
    S=frozenset(TLAB[i] for i in range(len(TLAB)) if m>>i&1); R[S]=R.get(S,0)+n
print(which,'total fillings',sum(R.values()),' distinct crossing sets',len(R),' memo',len(memo),' far-regions',nfar[0],' %.0fs'%(time.time()-t0),flush=True)
out=(dict(R),SEG)+tuple(cells)
pickle.dump(out,open('vmodes_fast_%s.pkl'%which,'wb'))
