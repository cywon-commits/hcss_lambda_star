from vec import *
import numpy as np, time
ti,ri=int(__import__('sys').argv[1]),int(__import__('sys').argv[2])
st=vsigma(ti,ri)
ty=np.ones(12,dtype=int); X=np.zeros((12,4),dtype=np.int64); K=np.arange(12)
for _ in range(4): ty,X,K=st(ty,X,K)
cid=ty*12+K; freq=np.bincount(cid,minlength=24).astype(float); freq/=freq.sum()
ref={0:[ident(t,V) for t,V in T[ti]],1:[ident(t,V) for t,V in Rh[ri]]}
li=1/(2+np.sqrt(3))
par,chi,off=[],[],[]
for t in (0,1):
    for k in range(12):
        for sty,y0,j in ref[t]:
            par.append(t*12+k); chi.append(TY[sty]*12+(j+k)%12); off.append(star(rot(y0,k)))
par,chi,off=map(np.array,(par,chi,off))
# backward transition: into class c choose map m with prob ∝ freq[par[m]]
into=[np.where(chi==c)[0] for c in range(24)]
prob=[freq[par[m]]/freq[par[m]].sum() for m in into]
# tile-corner offsets (internal) per class
E4=np.array(E)
def corners(c):
    t,k=divmod(c,12)
    if t==0: vs=[E4[k],E4[k]+E4[(k+4)%12]]
    else: vs=[E4[k],E4[k]+E4[(k+1)%12],E4[(k+1)%12]]
    return [0j]+[star(tuple(v)) for v in vs]
COR=[np.array(corners(c)) for c in range(24)]
rng=np.random.default_rng(0)
hs=[0.016,0.008,0.004,0.002,0.001]
occ={h:set() for h in hs}
NW=400000; depth=16; t0=time.time(); tot=0
for batch in range(int(__import__('sys').argv[3])):
    c=rng.choice(24,size=NW,p=freq)
    seq=np.empty((depth,NW),dtype=np.int64)
    for d in range(depth):
        m=np.empty(NW,dtype=np.int64)
        for cc in range(24):
            sel=np.where(c==cc)[0]
            if len(sel)==0: continue
            m[sel]=rng.choice(into[cc],size=len(sel),p=prob[cc])
        seq[depth-1-d]=m; c=par[m]
    z=np.zeros(NW,complex)
    for d in range(depth): z=li*z+off[seq[d]]
    cl=chi[seq[-1]]
    pts=np.concatenate([z[cl==cc][:,None]+COR[cc][None,:] for cc in range(24)],axis=None) if False else None
    allp=[]
    for cc in range(24):
        s=cl==cc
        if s.any(): allp.append((z[s][:,None]+COR[cc][None,:]).ravel())
    allp=np.concatenate(allp); tot+=len(allp)
    for h in hs:
        key=np.floor(allp.real/h).astype(np.int64)*1000003+np.floor(allp.imag/h).astype(np.int64)
        occ[h].update(np.unique(key).tolist())
print('points',tot,'time',time.time()-t0)
vw=3+np.sqrt(3); res=[]
for h in hs:
    S=occ[h]; ks=np.array(list(S))
    nb=0
    for k in ks:
        if not all((k+di*1000003+dj) in S for di in(-1,0,1) for dj in(-1,0,1)): nb+=1
    A=len(S)*h*h; res.append((h,nb,A-vw))
    print('h=%.4f occ.area %.4f excess %.4f boundary pix %d pts/pix %.1f'%(h,A,A-vw,nb,tot/len(S)))
r=np.array(res)
for i in range(len(r)-1):
    print('h %.4f->%.4f  D(boundary)=%.3f  D(excess)=%.3f'%(r[i,0],r[i+1,0],np.log(r[i+1,1]/r[i,1])/np.log(2), 2-np.log(r[i,2]/r[i+1,2])/np.log(2)))
print('log5/loglam =',np.log(5)/np.log(2+np.sqrt(3)))
