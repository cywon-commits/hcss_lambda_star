from subst import *
def vertex_configs(tiles):
    # corner angles at each vertex: list of (vertex, angle-in-30deg, start dir)
    corn=defaultdict(list)
    for t in tiles:
        V=tverts(t); n=len(V)
        Vc=[cplx(v) for v in V]
        for i in range(n):
            a=Vc[(i-1)%n]-Vc[i]; b=Vc[(i+1)%n]-Vc[i]
            ang=int(round(np.angle(a/b)/(np.pi/6)))%12   # CCW angle from b to a
            d=int(round(np.angle(b)/(np.pi/6)))%12
            corn[V[i]].append((d,ang,t[0]))
    full={}
    for v,cs in corn.items():
        if sum(c[1] for c in cs)==12:
            cs=sorted(cs); s=tuple(f"{c[2]}{c[1]}" for c in cs)
            # canonical up to rotation (cyclic shift)
            full[v]=min(s[i:]+s[:i] for i in range(len(s)))
    return full
import sys
DEP=int(sys.argv[1]) if len(sys.argv)>1 else 3
res={}
for ti in inv_T:
    for ri in inv_R:
        sg=make_sigma(ti,ri)
        types=Counter()
        for seed in ([('A',ZERO,0)],[('R',ZERO,0)]):
            P=seed
            for _ in range(DEP): P=sg(P)
            types.update(vertex_configs(P).values())
        star=('R1',)*12
        print(ti,ri,'#vertex types',len(types),'12-star legal:',star in types)
        res[(ti,ri)]=types
pickle.dump(res,open('vtypes.pkl','wb'))
