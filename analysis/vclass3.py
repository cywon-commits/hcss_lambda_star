import pickle, numpy as np
from collections import defaultdict, Counter
from shapely.geometry import Polygon, LineString
from zz import cplx
R,SEG,DaI,DbI,DcI,TI=pickle.load(open('vmodes3.pkl','rb'))
SEGL={k:LineString([(cplx(a).real,cplx(a).imag),(cplx(b).real,cplx(b).imag)]) for k,(a,b) in SEG.items()}
JUNC={'X':('DaDb','TDa','TDb'),'Y':('DbDc','TDb','TDc'),'O':('DcDa','TDc','TDa')}
def ordered(vs):
    C=[cplx(v) for v in vs]; c=sum(C)/len(C); C.sort(key=lambda z:np.angle(z-c)); return [(z.real,z.imag) for z in C]
def segs_of(t):
    P=Polygon(ordered(t[1])).buffer(-1e-7); return frozenset(k for k in SEGL if P.intersection(SEGL[k]).length>1e-7)
tot=sum(R.values()); cat=Counter(); njm=Counter()
for S,n in R.items():
    if not S: cat['decoupled']+=n; continue
    # connected crossing clusters: tiles linked if they share a crossed segment
    tiles=[(t,segs_of(t)) for t in S]
    multi=any(len(sg)>1 for _,sg in tiles)
    crossed=set().union(*[sg for _,sg in tiles])
    juncs=[j for j,ss in JUNC.items() if all(s in crossed for s in ss) and any(len(sg)>1 for _,sg in tiles if sg&set(ss))]
    if not multi: cat['bond modes only']+=n
    else: cat['%d junction mode(s)'%len(juncs) if juncs else 'other multi-segment']+=n
    njm[len(juncs)]+=1
print('DDDT total fillings',tot,' distinct crossing sets',len(R))
for k,v in sorted(cat.items(),key=lambda x:-x[1]): print('  %-26s %14d  %.2f%%'%(k,v,100*v/tot))
print('crossing sets by number of active junctions:',dict(njm))
