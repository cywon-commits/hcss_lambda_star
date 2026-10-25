import sys, math, numpy as np
from collections import Counter
from analyze import tiles
def famstats(fn):
    X,ID,N,polys,lam=tiles(fn)
    E=set()
    for p in polys:
        if len(p) in (3,4):
            for i in range(len(p)): E.add(tuple(sorted((p[i],p[(i+1)%len(p)]))))
    d=np.array([math.degrees(math.atan2(*(X[b]-X[a])[::-1]))%180 for a,b in E])
    off=(np.degrees(np.angle(np.mean(np.exp(1j*2*np.pi*(d%20)/20))))/360*20)%20
    fam=lambda a,b:int(np.floor(((math.degrees(math.atan2(*(X[b]-X[a])[::-1]))-off+10)%60)/20))
    tf=Counter(fam(p[0],p[1]) for p in polys if len(p)==3)
    lab=Counter(tuple(sorted({fam(p[0],p[1]),fam(p[1],p[2])})) for p in polys if len(p)==4)
    lab={k:v for k,v in lab.items() if len(k)==2}
    w=np.exp(2j*np.pi*np.arange(3)/3)
    pt=np.array([tf.get(i,0) for i in range(3)],float); pt/=pt.sum()
    pr=np.array([lab.get(k,0) for k in ((0,1),(1,2),(0,2))],float); pr/=pr.sum()   # labelled by missing family 2,0,1
    return pt, abs((pt*w).sum()), pr, abs((pr*w).sum())
if __name__=="__main__":
    for fn in sys.argv[1:]:
        pt,Mt,pr,Mr=famstats(fn); print(f"{fn:22s} triangle families {np.round(pt,2)} M_tri={Mt:.2f} | rhombus labels {np.round(pr,2)} M_rh={Mr:.2f}")
