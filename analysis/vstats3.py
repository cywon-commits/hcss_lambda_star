# extra statistics for the DDDT classification (reads vclass3x.pkl)
import pickle, sys
from collections import Counter, defaultdict
sys.argv=['x']; 
exec(open('vclass3x.py').read().split("# ---------- classification")[0])   # primitives, ICL, geometry
rows,rowinfo,R=pickle.load(open('vclass3x.pkl','rb'))
juncv={'X':X,'Y':Y,'O':O}
def involved(S):
    cr=set().union(*[segs_of(t) for t in S]) if S else set()
    return [j for j,ss in JUNC.items() if all(s in cr for s in ss)], cr
def vacated(S): return [j for j,v in juncv.items() if not any(v in t[1] for t in S)]
# (1) single-junction vertex sets: crossed segs inside one junction's triple, all three crossed, not a bond union
print('--- single-junction check (crossing sets whose crossed segments lie inside one junction triple)')
tot_single=0
for j,ss in JUNC.items():
    sel=[S for S in R if S and set().union(*[segs_of(t) for t in S])<=set(ss) and len(set().union(*[segs_of(t) for t in S]))==3]
    vert=[S for S in sel if any(PRIM.get(S,('',))[0]=='vertex' for _ in [0])]
    nonvert=[S for S in sel if S not in PRIM]
    print(j,'sets crossing all 3 of its segments and nothing else:',len(sel),'| of them DDT-vertex primitives placed here:',len(vert),'| other:',len(nonvert),
          '| in ICL:',sum(S in ICL for S in nonvert))
# (2) the 22 DDT vertex modes: chirality / X vacated / three segs, placed in DDDT
print('--- 66 vertex primitives: all cross 3 segs, vacate their junction, chiral?')
chir=0
for S,(k,j,i) in PRIM.items():
    if k!='vertex': continue
    jj=juncv[j]; cr=set().union(*[segs_of(t) for t in S])
    assert cr==set(JUNC[j]) and jj not in set().union(*[t[1] for t in S]), (j,i)
    m=tmap(GROUP['s_'+j],S); chir+= (m!=S)
print('  OK; mirror-asymmetric (chiral) vertex primitives:',chir,'of 66')
# (3) class (d): involved junctions, vacated junctions, segments crossed
d=[S for S in R if rowinfo[S].startswith('d')]
t=Counter(); tv=Counter(); ts=Counter(); tn=defaultdict(int)
for S in d:
    inv,cr=involved(S); va=vacated(S)
    t[len(inv)]+=1; tn[len(inv)]+=R[S]; tv[len(va)]+=1; ts[len(cr)]+=1
print('--- class (d): %d sets'%len(d))
print('  #junctions with all 3 segments crossed -> sets, fillings:',{k:(t[k],tn[k]) for k in sorted(t)})
print('  #vacated junctions (no tile has the triangle vertex as a corner) -> sets:',dict(sorted(tv.items())))
print('  #skeleton segments crossed -> sets:',dict(sorted(ts.items())))
print('  tile-count range: %d..%d'%(min(len(S) for S in d),max(len(S) for S in d)))
# (4) classes c/c2 by vertex junction, and the realised-count stats
print('--- ICL unions with weight 0 / not realised:',len([S for S in ICL if S not in R]))
print('--- class counts per C3v orbit size:',Counter((r[0],r[1]) for r in rows))
print('--- class d by C3v stabiliser:',Counter(r[2] for r in rows if r[0]=='d'),' fillings by stab:',{s:sum(r[5]*r[1] for r in rows if r[0]=='d' and r[2]==s) for s in set(r[2] for r in rows if r[0]=='d')})
