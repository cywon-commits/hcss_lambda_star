import sys, numpy as np, itertools; sys.setrecursionlimit(100000)
exec(open('chains.py').read().split("total=0")[0])        # gives chains (2 DD modes for edge O->X), REG etc.
from zz import rot
exec(open('clusters.py').read().split("t0=time.time()")[0].split("exec(")[0])  # nothing
def ring_dirs(g):
    g=orient(g,1.0); cs=list(g.exterior.coords)[:-1]; out=[]
    for k in range(len(cs)):
        a=complex(*cs[k]); b=complex(*cs[(k+1)%len(cs)]); L=abs(b-a); dd=int(round(np.angle(b-a)/(np.pi/6)))%12
        out+= [dd]*int(round(L))
    return out
MODES=[[c[1] for c in ch] for ch in chains]            # vertex lists for edge O->X ; cells: D1 above (left), D2 below
def place(mode,a,b):
    d=[k for k in range(12) if add(a,E[k])==b][0]
    return [shp([add(a,rot(v,d)) for v in T]) for T in mode]
def dodec_on_edge(a,b):
    d=[k for k in range(12) if add(a,E[k])==b][0]
    return shp(poly_int(a,[(d+k)%12 for k in range(12)]))
def cnt(g):
    if g.is_empty or g.area<1e-6: return 1
    gs=list(g.geoms) if g.geom_type=='MultiPolygon' else [g]; c=1
    for x in gs:
        if x.area<1e-6: continue
        if len(x.interiors): return None
        pcs=closed_pieces(ring_dirs(x))
        if pcs is None: return 0
        for p in pcs: c*=fill_count(p)[0]
    return c
O=(0,0,0,0); X=E[0]; Y=add(E[0],E[4]); Tri=shp([O,X,Y])
Da=dodec_on_edge(X,O); Db=dodec_on_edge(Y,X)
cells=[Da,Db,Tri]; REGION=unary_union(cells)
bonds=[(X,Y) if False else None]
# bonds as directed edges with both modes (each mode = list of polygons)
def bond_modes(a,b): return [place(m,a,b) for m in MODES]+[place(m,b,a) for m in MODES]
Bdd=bond_modes(Y,X) if False else None
# shared edge between Da and Db: Da is on left of X->O ; Db on left of Y->X ; they share the edge from X going away from triangle
sh=[k for k in range(12) if Da.intersection(Db).length>0.9]
inter=Da.intersection(Db); print('Da∩Db length',round(inter.length,3))
cs=list(inter.coords) if inter.geom_type=='LineString' else None
pa=X; dpts=[np.round(complex(*p),6) for p in cs]
# find int endpoints: X and X+e_k
other=[add(X,E[k]) for k in range(12) if any(abs(cplx(add(X,E[k]))-p)<1e-6 for p in dpts)][0]
bond_list={'DaDb':(X,other),'DaT':(X,O),'DbT':(Y,X)}
def options(a,b):
    opts=[('0',[])]
    seen=[]
    for m in MODES:
        for (p,q) in ((a,b),(b,a)):
            P=place(m,p,q); U=unary_union(P)
            if U.within(REGION.buffer(1e-7)) and not any(U.symmetric_difference(s).area<1e-7 for s in seen):
                seen.append(U); opts.append(('m%d'%len(seen),P))
    return opts
OPT={k:options(*v) for k,v in bond_list.items()}
print({k:len(v) for k,v in OPT.items()},'(states per bond incl. decoupled)')
total=0
for combo in itertools.product(*[OPT[k] for k in bond_list]):
    tiles=[t for _,P in combo for t in P]
    if any(tiles[i].intersection(tiles[j]).area>1e-7 for i in range(len(tiles)) for j in range(i+1,len(tiles))): continue
    U=unary_union(tiles) if tiles else None
    c=1
    for cell in cells:
        rem=cell.difference(U) if U is not None else cell
        r=cnt(rem.buffer(-1e-9).buffer(1e-9) if not rem.is_empty else rem)
        if not r: c=0; break
        c*=r
    if c: print('  ',[s for s,_ in combo],c)
    total+=c
print('bond-state sum for DDT =',total,'  exact DDT = 61174241')
