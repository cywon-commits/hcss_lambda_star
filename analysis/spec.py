import numpy as np, scipy.sparse as sp, sys, time
from tmgen import build, entropy
from zz import cplx
def mats(order,tr,u,mu=1.0):
    n=len(order); fr=np.array([x[0] for x in tr]); to=np.array([x[1] for x in tr]); c=np.array([x[2] for x in tr],float)
    w=np.array([x[3] for x in tr]); tri=np.array([1 if x[4] in ('A','T') else 0 for x in tr])
    return sp.csr_matrix((c*u**w*mu**tri,(to,fr)),shape=(n,n)).toarray(), (fr,to)
def ratios(ev, tag):
    ev=ev[np.abs(ev)>1e-8*np.abs(ev).max()]
    up=ev[ev.imag>1e-6*np.abs(ev).max()]           # upper half plane, away from the real axis
    if len(up)<30: return None
    D=np.abs(up[:,None]-up[None,:]); np.fill_diagonal(D,np.inf)
    idx=np.argsort(D,axis=1)[:,:2]
    z=(up[idx[:,0]]-up)/(up[idx[:,1]]-up)
    return len(up), np.mean(np.abs(z)), np.mean(np.cos(np.angle(z)))
rng=np.random.default_rng(1)
cases=[('sqtri',(4,4,0,-2),[0,0,0,0,1,1,1,1,9,9]),('sqtri',(3,4,1,-1),[0,0,0,1,1,1,1,2,9]),('sqtri',(5,5,0,-2),None),
       ('ours',(2,2,0,-1),[0,0,1,1,9]),('ours',(2,3,0,-1),[0,0,1,1,1,9])]
import itertools, cyl
for model,Pv,d0 in cases:
    if d0 is None:
        import tmgen; tmgen.configure(model); cyl.PHAT[0]=cplx(Pv)/abs(cplx(Pv))
        for perm in itertools.permutations(cyl.path_for(Pv)):
            r=cyl.front_split(list(perm),Pv,tuple(-x for x in Pv))
            if r is not None and not r[1]: d0=list(perm); break
    t0=time.time()
    try: o,t=build(model,Pv,d0,maxstates=6000)
    except RuntimeError: print(model,Pv,'too many states'); continue
    if len(o)>6000: continue
    s,u=entropy(o,t); M,(fr,to)=mats(o,t,u)
    ev=np.linalg.eigvals(M); r=ratios(ev,model)
    # controls: same sparsity, random positive weights
    R=M.copy(); R[R>0]=rng.uniform(0.5,1.5,size=(R>0).sum()); rr=ratios(np.linalg.eigvals(R),'rnd')
    # commutation of T(u,mu1), T(u,mu2)
    A,_=mats(o,t,u,0.7); B,_=mats(o,t,u,1.4); com=np.linalg.norm(A@B-B@A)/(np.linalg.norm(A)*np.linalg.norm(B))
    print('%-6s |P|=%.3f states %5d s=%.4f | <|r|>=%.3f <cos>=%+.3f (N=%s) | random-weight ctrl <|r|>=%.3f <cos>=%+.3f | [T(mu),T(mu\')] rel=%.3f  (%.0fs)'%(
        model,abs(cplx(Pv)),len(o),s,r[1],r[2],r[0],rr[1],rr[2],com,time.time()-t0),flush=True)
print('reference: 2D Poisson <|r|>=0.667 <cos>=0 ; Ginibre <|r|>=0.738 <cos>=-0.240')
