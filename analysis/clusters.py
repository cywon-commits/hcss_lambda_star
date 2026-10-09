import sys, numpy as np, time; sys.setrecursionlimit(100000)
exec(open('chains.py').read().split("seg=LineString")[0])
def ring_dirs(g):
    g=orient(g,1.0); cs=list(g.exterior.coords)[:-1]; out=[]
    for k in range(len(cs)):
        a=complex(*cs[k]); b=complex(*cs[(k+1)%len(cs)]); L=abs(b-a); dd=int(round(np.angle(b-a)/(np.pi/6)))%12
        out+= [dd]*int(round(L))
    return out
def count_region(polys):
    R=unary_union(polys).buffer(1e-9).buffer(-1e-9)
    dirs=ring_dirs(R); pcs=closed_pieces(dirs); c=1
    for p in pcs: c*=fill_count(p)[0]
    return c,R.area
def dodec_on_edge(a,b):
    """regular dodecagon of side 1 on the left of the directed edge a->b (a,b int4)"""
    d=[k for k in range(12) if add(a,E[k])==b][0]
    return shp(poly_int(a,[(d+k)%12 for k in range(12)]))
O=(0,0,0,0); X=E[0]; Y=add(E[0],E[4])           # unit triangle O,X,Y (CCW)
Tri=shp([O,X,Y])
Da=dodec_on_edge(X,O); Db=dodec_on_edge(Y,X); Dc=dodec_on_edge(O,Y)   # outward dodecagons
t0=time.time()
N1=fill_count(canon_closed(list(range(12))))[0]
for nm,pl in (('DDD',[Da,Db,Dc]),('DDDT',[Da,Db,Dc,Tri])):
    c,a=count_region(pl); print('%-5s area %7.3f  fillings %d   (%.0fs)'%(nm,a,c,time.time()-t0),flush=True)
