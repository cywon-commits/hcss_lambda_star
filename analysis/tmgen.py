# generic cylinder growth transfer matrix for edge-to-edge tilings with unit edges in the 12 directions
import numpy as np, sys
import front, cyl
from zz import add, E, cplx
MODELS={
 'ours':  [('A',(0,4,8),2,0.5),('Ra',(0,1,6,7),1,1.0),('Ro',(0,5,6,11),5,1.0)],
 'sqtri': [('S',(0,3,6,9),3,1.0),('T',(0,4,8),2,0.5)],
 'lozenge':[('L60',(0,2,6,8),2,1.0),('L120',(0,4,6,10),4,1.0)],
}
def configure(model):
    tl=MODELS[model]
    front.TILES=tuple((n,o,a) for n,o,a,_ in tl); front.VW={n:v for n,_,_,v in tl}
    front.fill_count.cache_clear()
    cyl.TILES=front.TILES; cyl.VW=front.VW
def build(model,Pv,d0,maxstates=400000):
    configure(model)
    cyl.PHAT[0]=cplx(Pv)/abs(cplx(Pv))
    negP=tuple(-x for x in Pv)
    s0=cyl.canon_front(d0); states={s0:0}; order=[s0]; tr=[]; k=0
    while k<len(order):
        d=list(order[k]); din,dout=d[-1],d[0]
        turn=(dout-din)%12; turn=turn-12 if turn>6 else turn
        for t,td,ang in front.TILES:
            if ang>6-turn: continue
            tdir=[(dout+x)%12 for x in td]; rep=[(x+6)%12 for x in reversed(tdir[1:])]
            r=cyl.front_split(rep+d[1:],Pv,negP)
            if r is None: continue
            f,holes=r; c=1; w=front.VW[t]; nt=1 if t in ('A','T') else 0
            for h in holes:
                cc,ww=front.fill_count(h); c*=cc; w+=ww
            if c==0: continue
            if f not in states: states[f]=len(order); order.append(f)
            tr.append((k,states[f],c,int(round(2*w)),t))
        k+=1
        if len(order)>maxstates: raise RuntimeError('too many states')
    return order,tr
def entropy(order,tr):
    import scipy.sparse as sp, scipy.sparse.linalg as sla
    n=len(order); fr,to,c,w=[np.array([x[i] for x in tr]) for i in range(4)]
    def rho(u):
        M=sp.csr_matrix((c*u**w,(to,fr)),shape=(n,n))
        if n<3000: return max(abs(np.linalg.eigvals(M.toarray())))
        return abs(sla.eigs(M,k=1,which='LM',return_eigenvectors=False,tol=1e-12)[0])
    lo,hi=0.02,1.0
    for _ in range(50):
        m=(lo+hi)/2
        if rho(m)>1: hi=m
        else: lo=m
    return -2*np.log((lo+hi)/2),(lo+hi)/2
if __name__=='__main__':
    for model,Pv,d0 in [('ours',(2,2,0,-1),[0,0,1,1,9]),
                        ('sqtri',(2,2,0,-1),[0,0,1,1,9]),('sqtri',(2,3,1,-1),[0,0,1,1,1,2,9]),('sqtri',(3,4,0,-2),[0,0,0,1,1,1,1,9,9]),
                        ('lozenge',(3,0,0,0),[2,10]*3),('lozenge',(4,0,0,0),[2,10]*4),('lozenge',(6,0,0,0),[2,10]*6),('lozenge',(8,0,0,0),[2,10]*8)]:
        try:
            o,t=build(model,Pv,d0); s,u=entropy(o,t)
            print('%-8s P=%-14s |P|=%.3f states %6d  s/vertex=%.5f'%(model,Pv,abs(cplx(Pv)),len(o),s),flush=True)
        except Exception as e: print(model,Pv,'ERR',e)
