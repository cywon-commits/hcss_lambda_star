import numpy as np, pickle
exec(open('pairs.py').read().split("F=fills")[0])
F=fills((0,0,0,0),list(range(12)))
TS=[frozenset((t[0],frozenset(vs)) for t,vs in f) for f in F]   # tiles as (type-letter, vertex set)
idx={s:i for i,s in enumerate(TS)}
# the two tilings of each 30-degree hexagon (edges a,a+1,a+2,a+6,a+7,a+8) placed at v
hexfill={}
for a in range(12):
  for typ,(p,q,r) in (('H1',(0,1,2)),('H2',(0,1,5))):
    H=fills((0,0,0,0),[(a+x)%12 for x in (p,q,r,p+6,q+6,r+6)])
    H=[[(t[0],vs) for t,vs in h] for h in H]
    assert len(H)==2,len(H)
    hexfill[(typ,a)]=H
nbr=[set() for _ in TS]
for i,f in enumerate(F):
    verts={v for t,vs in f for v in vs}
    tiles=TS[i]
    for v in verts:
        for a in hexfill:
            sh=[frozenset((t,frozenset(add(v,x) for x in vs)) for t,vs in h) for h in hexfill[a]]
            for k in (0,1):
                if sh[k]<=tiles:
                    j=idx[(tiles-sh[k])|sh[1-k]]; nbr[i].add(j)
deg=np.array([len(n) for n in nbr])
print('states',len(TS),' degree min/mean/max',deg.min(),round(deg.mean(),3),deg.max())
# connectivity & bipartiteness
col=-np.ones(len(TS),int); col[0]=0; st=[0]; bip=True
while st:
    i=st.pop()
    for j in nbr[i]:
        if col[j]<0: col[j]=1-col[i]; st.append(j)
        elif col[j]==col[i]: bip=False
print('connected',(col>=0).all(),' bipartite',bip)
pickle.dump((nbr,deg,col),open('flipgraph.pkl','wb'))
