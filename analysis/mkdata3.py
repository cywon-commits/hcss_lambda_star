import numpy as np, json, pickle, math, itertools
from collections import Counter
exec(open('pairs.py').read().split("F=fills")[0])
from zz import cplx, rot, add, sub
F=fills((0,0,0,0),list(range(12)))
B=pos(list(range(12)))[:-1]; Bc=np.array([cplx(p) for p in B]); c=Bc.mean()
ang=sorted((np.degrees(np.angle(Bc-c))%360).round(6)); print('boundary vertex angles', ang[:3],'... (expect 15,45,75)')
inter=[]
for f in F:
    V={v for t,vs in f for v in vs}-set(B)
    assert len(V)==13
    inter.append(sorted([(cplx(v)-c).real,(cplx(v)-c).imag] for v in V))
inter=np.array(inter); print('interior array',inter.shape)
# compare with the old 4421 set
old=np.load('/home/claude/hardcore_square_shoulder/dodecagon_fillings.npz')['interior']
key=lambda P: tuple(sorted((round(x,3)+0.0,round(y,3)+0.0) for x,y in P))
newk={key(p) for p in inter}; oldk={key(p) for p in old}
print('old 4421 subset of new 5827:',oldk<=newk,' missing in old:',len(newk-oldk))
np.savez('/home/claude/task3/data/dodecagon_fillings_5827.npz',interior=inter,
  note=np.array('All 5827 edge-to-edge fillings of the unit-edge regular 12-gon (vertices at 15+30k deg) by unit triangles and 30-deg rhombi; 13 interior vertices relative to the centre. Exact enumeration (hcss_lambda_star front.py), validated: inflated triangle 20, inflated rhombus 23. Supersedes the 4421 set (which is a strict subset).'))
# ---- local move catalogue: pairs of fillings differing by exactly 4 tiles
TS=[frozenset((t[0],frozenset(vs)) for t,vs in f) for f in F]
def canon(A,Bt):
    best=None
    for k in range(12):
        tr=lambda S:[(t,tuple(sorted(rot(v,k) for v in vs))) for t,vs in S]
        a=tr(A); b=tr(Bt)
        allv=[v for _,vs in a+b for v in vs]; m=min(allv)
        sh=lambda S: tuple(sorted((t,tuple(sorted(sub(v,m) for v in vs))) for t,vs in S))
        rep=tuple(sorted([sh(a),sh(b)]))
        if best is None or rep<best: best=rep
    return best
cat=Counter(); n=len(TS)
idx={}
for i in range(n):
    for j in range(i+1,n):
        d1=TS[i]-TS[j]
        if len(d1)==4:
            idx.setdefault(canon(d1,TS[j]-TS[i]),0); idx[canon(d1,TS[j]-TS[i])]+=1
print('distinct 4-tile local moves (up to rotation+translation):',len(idx),' total pairs',sum(idx.values()))
out=[]
for rep,cnt in sorted(idx.items(),key=lambda x:-x[1]):
    def area(S): return sum(math.sqrt(3)/4 if t=='A' else 0.5 for t,_ in S)
    shape=sorted({v for t,vs in rep[0] for v in vs})
    out.append(dict(count_in_dodecagon_pairs=cnt,
        tilings=[[{'type':t,'vertices_int4':[list(v) for v in vs],'vertices_xy':[[round(cplx(v).real,9),round(cplx(v).imag,9)] for v in vs]} for t,vs in S] for S in rep],
        n_vertices=len(shape)))
    print('  move: %d occurrences, tiles %s / %s'%(cnt,''.join(sorted(t for t,_ in rep[0])),''.join(sorted(t for t,_ in rep[1]))))
json.dump(dict(note='Local moves = pairs of regions tiled two ways that differ by exactly 4 tiles (2A+2R). Extracted from all 5827 dodecagon fillings (pairs differing by 4 tiles: 36,360). Coordinates: Z[zeta12] int4 (basis 1,z,z^2,z^3), unit edge; apply any of the 12 rotations and any lattice translation. Moves 0..: canonical representatives.',moves=out),
          open('/home/claude/task3/data/local_moves_4tile.json','w'),indent=1)

# ergodicity check inside the dodecagon: graph with only the centrosymmetric move vs both moves
reps=sorted(idx.items(),key=lambda x:-x[1]); rep_noncs,rep_cs=reps[0][0],reps[1][0]
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
rows={0:[],1:[]}
for i in range(n):
    for j in range(i+1,n):
        d1=TS[i]-TS[j]
        if len(d1)==4:
            r=canon(d1,TS[j]-TS[i]); k=0 if r==rep_noncs else 1
            rows[k].append((i,j))
for name,E_ in (('centrosymmetric hexagon only (current engine)',rows[1]),('non-centrosymmetric only',rows[0]),('both moves',rows[0]+rows[1])):
    a,b=zip(*E_); M=sp.coo_matrix((np.ones(len(a)),(a,b)),shape=(n,n))
    nc,lab=connected_components(M,directed=False); sizes=np.bincount(lab)
    print('%-48s components %4d   largest %4d   isolated %4d'%(name,nc,sizes.max(),(sizes==1).sum()))
