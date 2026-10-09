# Step 3: classify the DDDT crossing sets, independence test, C3v orbits / irreps.
#   python3 vclass3x.py [vmodes_fast_DDDT.pkl]        (needs vmodes_fast_DDT.pkl too)
# Definitions
#   primitive modes = single bond modes (2 R tiles across ONE skeleton segment) and the DDT vertex modes
#                     (22 crossing sets of DDT = tiles crossing all 3 segments at a junction, X vacated),
#                     placed at each of the three junctions X, Y, O by the lattice rotation R.
#   independent set = union of pairwise tile-disjoint primitives; its weight is the product over the four
#                     cells of the fillings of (cell minus the mode tiles)  (bondmodel.py method).
#   (d) = crossing sets that are NOT a union of disjoint primitives (several junctions move together).
import sys, pickle, itertools, numpy as np
from collections import defaultdict, Counter
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union
from shapely.geometry.polygon import orient
from zz import cplx, add, sub, E, rot, mulz
from front import closed_pieces, fill_count

f3=sys.argv[1] if len(sys.argv)>1 else 'vmodes_fast_DDDT.pkl'
R3,SEG3,DaI,DbI,DcI,TI=pickle.load(open(f3,'rb'))
R2,SEG2,_,_,_=pickle.load(open('vmodes_fast_DDT.pkl','rb'))
O=(0,0,0,0); X=E[0]; Y=add(E[0],E[4])
tot3=sum(R3.values())
# ---------- lattice-exact symmetry maps ----------
def conj(a): return tuple(sum(a[i]*E[(12-i)%12][j] for i in range(4)) for j in range(4))
def Rrot(z): return add(rot(z,4),X)                 # O->X->Y->O
def Mir(z): return add(X,rot(conj(sub(z,X)),10))    # fixes X, swaps Y<->O
assert [Rrot(O),Rrot(X),Rrot(Y)]==[X,Y,O] and Mir(Y)==O and Mir(O)==Y and Mir(X)==X
GROUP={'e':lambda z:z,'C3':Rrot,'C3^2':lambda z:Rrot(Rrot(z)),'s_X':Mir,'s_Y':lambda z:Rrot(Mir(Rrot(Rrot(z)))),'s_O':lambda z:Rrot(Rrot(Mir(Rrot(z))))}
def tmap(f,S): return frozenset((t,frozenset(f(v) for v in vs)) for t,vs in S)
# ---------- geometry ----------
shp=lambda P: Polygon([(cplx(p).real,cplx(p).imag) for p in P])
def ordered(vs):
    C=[cplx(v) for v in vs]; c=sum(C)/len(C); C.sort(key=lambda z:np.angle(z-c)); return [(z.real,z.imag) for z in C]
tpoly=lambda t: Polygon(ordered(t[1]))
CELLS=[shp(DaI),shp(DbI),shp(DcI),shp(TI)]
for f in GROUP.values():   # cluster is C3v symmetric
    img=unary_union([Polygon([(cplx(f(p)).real,cplx(f(p)).imag) for p in P]) for P in (DaI,DbI,DcI,TI)])
    assert img.symmetric_difference(unary_union(CELLS)).area<1e-9
SEGL={k:LineString([(cplx(a).real,cplx(a).imag),(cplx(b).real,cplx(b).imag)]) for k,(a,b) in SEG3.items()}
JUNC={'X':('DaDb','TDa','TDb'),'Y':('DbDc','TDb','TDc'),'O':('DcDa','TDc','TDa')}
def segs_of(t):
    P=tpoly(t).buffer(-1e-7); return frozenset(k for k in SEGL if P.intersection(SEGL[k]).length>1e-7)
def ring_dirs(g):
    g=orient(g,1.0); cs0=list(g.exterior.coords)[:-1]; cs=[]; out=[]
    for p in cs0:   # drop float-noise slivers from the buffer round trip
        if not cs or abs(complex(*p)-complex(*cs[-1]))>1e-6: cs.append(p)
    if len(cs)>1 and abs(complex(*cs[0])-complex(*cs[-1]))<=1e-6: cs.pop()
    for k in range(len(cs)):
        a=complex(*cs[k]); b=complex(*cs[(k+1)%len(cs)]); L=abs(b-a); dd=int(round(np.angle(b-a)/(np.pi/6)))%12
        assert abs(L-round(L))<1e-6 and L>0.5, 'non-integer edge L=%r from %r to %r'%(L,a,b)
        out+=[dd]*int(round(L))
    return out
def cnt(g):
    if g.is_empty or g.area<1e-6: return 1
    c=1
    for x in (list(g.geoms) if g.geom_type=='MultiPolygon' else [g]):
        if x.area<1e-6: continue
        if len(x.interiors): raise RuntimeError('hole in remainder region')
        pcs=closed_pieces(ring_dirs(x))
        if pcs is None: return 0
        for p in pcs: c*=fill_count(p)[0]
    return c
def weight(S):
    """product of fillings of the four cells minus the tiles of S"""
    U=unary_union([tpoly(t) for t in S]) if S else None
    c=1
    for cell in CELLS:
        rem=cell if U is None else cell.difference(U)
        rem=rem.buffer(-1e-9).buffer(1e-9) if not rem.is_empty else rem
        c*=cnt(rem)
        if c==0: return 0
    return c
# ---------- primitive library from DDT, placed by R at X, Y, O ----------
def seg_groups(S,segf):
    g=defaultdict(list)
    for t in S: g[segf(t)].append(t)
    return g
X2=SEG2['B1'][0]
SEGL2={k:LineString([(cplx(a).real,cplx(a).imag),(cplx(b).real,cplx(b).imag)]) for k,(a,b) in SEG2.items()}
def segs2(t):
    P=tpoly(t).buffer(-1e-7); return frozenset(k for k in SEGL2 if P.intersection(SEGL2[k]).length>1e-7)
bond2=[];vert2=[]
for S in R2:
    if not S: continue
    g=seg_groups(S,segs2); multi=any(len(k)>1 for k in g)
    if not multi and len(g)==1 and len(S)==2 and all(t[0]=='R' for t in S): bond2.append(S)
    elif multi or any(len(v)!=2 or any(t[0]!='R' for t in v) for v in g.values()) : vert2.append(S)
nb=sum(1 for S in R2 if S and not any(len(k)>1 for k in seg_groups(S,segs2)) and all(len(v)==2 and all(t[0]=='R' for t in v) for v in seg_groups(S,segs2).values()))
print('DDT: single-bond primitives %d, vertex primitives %d  (bond-decomposable nonempty sets %d)'%(len(bond2),len(vert2),nb))
assert len(vert2)==22 and len(bond2)==4, (len(vert2),len(bond2))
Rn=[lambda z:z,Rrot,lambda z:Rrot(Rrot(z))]
PRIM={}   # frozenset -> (kind, junction, id)
for j,(jn,f) in enumerate(zip('XYO',Rn)):
    for i,S in enumerate(bond2): PRIM.setdefault(tmap(f,S),('bond',jn,i))
    for i,S in enumerate(vert2): PRIM[tmap(f,S)]=('vertex',jn,i)
bondP=[S for S,v in PRIM.items() if v[0]=='bond']; vertP=[S for S,v in PRIM.items() if v[0]=='vertex']
print('DDDT primitives: %d single-bond modes (deduplicated), %d vertex modes (3 x 22)'%(len(bondP),len(vertP)))
assert len(vertP)==66
for S in PRIM: assert all(len(t[1])>=3 for t in S)
# ---------- independent combinations ----------
prims=list(PRIM); polys=[unary_union([tpoly(t) for t in S]) for S in prims]
ovl=[[ (i!=j and polys[i].intersection(polys[j]).area>1e-7) or bool(prims[i]&prims[j] and i!=j) for j in range(len(prims))] for i in range(len(prims))]
ICL=defaultdict(list)   # union set -> list of decompositions (tuple of primitive indices)
def dfs(chosen,start):
    if chosen: ICL[frozenset().union(*[prims[i] for i in chosen])].append(tuple(chosen))
    for i in range(start,len(prims)):
        if all(not ovl[i][j] for j in chosen): dfs(chosen+[i],i+1)
dfs([],0)
ICL[frozenset()].append(())
print('independent combinations: %d decompositions -> %d distinct unions'%(sum(len(v) for v in ICL.values()),len(ICL)))
# ---------- classification of every DDDT crossing set ----------
def kinds(dec): return Counter(PRIM[prims[i]][0] for i in dec)
cat=Counter(); catn=Counter(); rowinfo={}
indep_sum=0; mism=0; ambiguous=0; notin=[]
for S,n in R3.items():
    if S in ICL:
        decs=ICL[S]; kv=set(kinds(d)['vertex'] for d in decs)
        if len(kv)>1: ambiguous+=1
        nv=min(kv)
        c='a: unbound' if not S else ('b: bond modes only' if nv==0 else ('c: one junction vertex mode (+bonds)' if nv==1 else 'c2: several vertex modes, independent'))
        indep_sum+=n
        w=weight(S)
        if w!=n: mism+=1; print('  weight mismatch',w,n)
    else:
        c='d: cooperative (not a union of primitives)'; notin.append(S)
    cat[c]+=1; catn[c]+=n; rowinfo[S]=c
print('\nDDDT total fillings %d, crossing sets %d, ICL sets not realised by exact enumeration: %d'%(tot3,len(R3),len([S for S in ICL if S not in R3])))
print('product-formula weight == exact count for all independent sets:',mism==0,' | sets with ambiguous decomposition:',ambiguous)
print('%-46s %8s %16s %8s'%('class','#sets','fillings','share'))
for c in sorted(cat): print('%-46s %8d %16d %7.3f%%'%(c,cat[c],catn[c],100*catn[c]/tot3))
print('independent-combination sum %d  vs exact %d : difference %d (%.4f%%)'%(indep_sum,tot3,tot3-indep_sum,100*(tot3-indep_sum)/tot3))
print('unbound check 5827^3 =',5827**3,' ->',R3.get(frozenset()))
# ---------- C3v orbits ----------
key=lambda S:S
orbits=[]; seen=set()
for S in sorted(R3,key=lambda S:(len(S),-R3[S])):
    if S in seen: continue
    imgs={g:tmap(f,S) for g,f in GROUP.items()}
    for T in imgs.values(): assert T in R3, 'image of crossing set not in enumeration'
    orb=list(dict.fromkeys(imgs.values())); seen|=set(orb)
    fixed={g:int(imgs[g]==S) for g in GROUP}
    orbits.append((S,orb,imgs))
def chars(orb):
    ch={}
    for g,f in GROUP.items(): ch[g]=sum(1 for T in orb if tmap(f,T)==T)
    return ch
rows=[]
for S,orb,imgs in orbits:
    ch=chars(orb); e=ch['e']; c3=ch['C3']; s=(ch['s_X']+ch['s_Y']+ch['s_O'])/3
    A1=(e+2*c3+3*s)/6; A2=(e+2*c3-3*s)/6; Ee=(2*e-2*c3)/6
    stab={1:'C1',2:'Cs',3:'C3',6:'C3v'}[6//len(orb)]
    n=R3[S]; rows.append((rowinfo[S][0:2].strip(':'),len(orb),stab,len(S),''.join(sorted(t[0] for t in S)),n,'+'.join(sorted(set(k for t in S for k in segs_of(t)))),(A1,A2,Ee),S,orb))
pickle.dump((rows,rowinfo,dict(R3)),open('vclass3x.pkl','wb'))
irr=Counter()
print('\nC3v orbits: %d'%len(orbits))
print('%-4s %-5s %-4s %-5s %-9s %-12s %-24s %s'%('cls','orb','stab','tiles','types','count each','segments','A1,A2,E'))
for r in sorted(rows,key=lambda r:(r[0],-r[5])): print('%-4s %-5d %-4s %-5d %-9s %-12d %-24s %s'%(r[0],r[1],r[2],r[3],r[4],r[5],r[6],tuple(int(round(x)) for x in r[7])))
for r in rows:
    for nm,x in zip(('A1','A2','E'),r[7]): irr[nm]+=int(round(x))
print('total irreps in the permutation rep of crossing sets:',dict(irr),' (check: A1+A2+2E =',irr['A1']+irr['A2']+2*irr['E'],'=',len(R3),')')
