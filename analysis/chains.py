import sys, numpy as np; sys.setrecursionlimit(100000)
from front import *
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
from shapely.geometry.polygon import orient
from collections import Counter
def poly_int(start,dirs):
    P=[start]
    for d in dirs[:-1]: P.append(add(P[-1],E[d%12]))
    return P
def shp(P): return Polygon([(cplx(p).real,cplx(p).imag) for p in P])
D1=shp(poly_int((0,0,0,0),list(range(12))))
D2=shp(poly_int((0,0,0,0),[(6+k)%12 for k in range(12)]) if False else poly_int(E[0],[(6+k)%12 for k in range(12)]))
REG=unary_union([D1,D2]); print('pair region area %.4f (2 x %.4f)'%(REG.area,D1.area))
seg=LineString([(0,0),(1,0)])
def tile(t,v,d):
    td=dict((x[0],x[1]) for x in TILES)[t]; return (t,tuple(poly_int(v,[(d+x)%12 for x in td])))
def tshp(T): return shp(T[1])
def overl(a,b): return a.intersection(b).area>1e-7
chains=[]
def extend(chain):
    T=chain[-1]; P=T[1]; n=len(P); C=[cplx(p) for p in P]
    if (E[0] in P): chains.append(list(chain)); return   # reached (1,0) as a vertex
    # exit point: boundary crossing of segment with largest x
    best=None
    for i in range(n):
        a,b=C[i],C[(i+1)%n]
        if abs((b-a).imag)<1e-12: continue
        s=-a.imag/(b-a).imag
        if -1e-9<=s<=1+1e-9:
            x=(a+s*(b-a)).real
            if best is None or x>best[0]: best=(x,i,s)
    x,i,s=best
    if x>=1-1e-9: return            # tile pokes beyond (1,0) without having it as vertex -> invalid
    if s<1e-9 or s>1-1e-9: return  # exits through a vertex on the open segment: forbidden
    a,b=P[i],P[(i+1)%n]
    dba=[k for k in range(12) if add(b,E[k])==a][0]
    for t,td,ang in TILES:
        N=tile(t,b,dba); S=tshp(N)
        if not S.within(REG.buffer(1e-7)): continue
        if any(overl(S,tshp(c)) for c in chain): continue
        if not overl(S,seg.buffer(1e-6)): continue
        extend(chain+[N])
for t,v,d in [('A',(0,0,0,0),11)]+[('Ro',(0,0,0,0),d) for d in (8,9,10,11)]:
    T=tile(t,v,d)
    if tshp(T).within(REG.buffer(1e-7)): extend([T])
print('candidate crossing chains',len(chains))
def ring_dirs(g):
    g=orient(g,1.0); cs=list(g.exterior.coords)[:-1]; out=[]
    for k in range(len(cs)):
        a=complex(*cs[k]); b=complex(*cs[(k+1)%len(cs)]); L=abs(b-a); dd=int(round(np.angle(b-a)/(np.pi/6)))%12
        out+= [dd]*int(round(L))
    return out
total=0; stats=[]
for ch in chains:
    rem=REG.difference(unary_union([tshp(c) for c in ch])).buffer(-1e-9).buffer(1e-9)
    geoms=list(rem.geoms) if rem.geom_type=='MultiPolygon' else [rem]
    cnt=1
    for g in geoms:
        if g.area<1e-6: continue
        dirs=ring_dirs(g)
        pcs=closed_pieces(dirs)
        if pcs is None: cnt=0; break
        for p in pcs: cnt*=fill_count(p)[0]
    total+=cnt; stats.append((len(ch),''.join(c[0][0] for c in ch),cnt))
print('sum over chains =',total,' (expected coupled count 4166532)')
nz=[s for s in stats if s[2]>0]
print('coupling modes with nonzero weight:',len(nz))
print('by chain length:',Counter(s[0] for s in nz))
for L,ty,c in sorted(nz,key=lambda s:-s[2]): print('   len',L,ty,c)

for ch in chains: print('mode tiles:',[(c[0],[tuple(np.round([cplx(p).real,cplx(p).imag],3)) for p in c[1]]) for c in ch])
import sympy; print('2083266 =',sympy.factorint(2083266))
# bite counts: each dodecagon minus its part of the crossing rhombi
for ch in chains:
    for Dk,nm in ((D1,'upper'),(D2,'lower')):
        rem=Dk.difference(unary_union([tshp(c) for c in ch]))
        dirs=ring_dirs(rem); pcs=closed_pieces(dirs); c=1
        for p in pcs: c*=fill_count(p)[0]
        print('   ',nm,'dodecagon with bite: fillings',c,' area %.4f'%rem.area)
    break
