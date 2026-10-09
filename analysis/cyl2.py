# cylinder TM that also records the lattice displacement of the front (for phason tilt fields)
import sys, itertools, time, numpy as np
from front import *
from zz import star, zc
import cyl
from cyl import lift, PHAT
def cancel_shift(d):
    d=list(d); st=(0,0,0,0); ch=True
    while ch and len(d)>=2:
        ch=False
        for i in range(len(d)):
            j=(i+1)%len(d)
            if (d[i]-d[j])%12==6:
                if j==0: st=add(st,E[d[0]])
                d=[x for k,x in enumerate(d) if k not in (i,j)]; ch=True; break
    return d,st
def split_shift(d,Pv,negP):
    d,st=cancel_shift(d)
    P=lift(d); idx={}
    for i,p in enumerate(P[:-1]): idx.setdefault(p,[]).append(i)
    for i,p in enumerate(P[:-1]):
        for q,typ in ((p,0),(add(p,Pv),1)):
            for j in idx.get(q,[]):
                if j<=i: continue
                if typ==0: hole=d[i:j]; rest=d[:i]+d[j:]; rst=(0,0,0,0)
                else:      hole=d[:i]+d[j:]; rest=d[i:j]; rst=P[i]
                hp=closed_pieces(hole)
                if hp is None: return None
                r=split_shift(rest,Pv,negP)
                if r is None: return None
                return (r[0],r[1]+hp,add(st,add(rst,r[2])))
        if add(p,negP) in idx: return None
    C=[cplx(p) for p in P]; pc=cplx(Pv); n=len(d); segs=[]
    for sh in (-pc,0,pc): segs+=[(C[k]+sh,C[k+1]+sh) for k in range(n)]
    if crosses(segs): return None
    # canonical rotation and its offset
    Q=P[:-1]; ys=[round((cplx(p)*PHAT[0].conjugate()).imag,9) for p in Q]; m=min(ys)
    cands=[(tuple(d[i:]+d[:i]),i) for i in range(len(d)) if ys[i]==m]
    f,i=min(cands)
    return (f,[],add(st,Q[i]))
AREA={'A':np.sqrt(3)/4,'Ra':0.5,'Ro':0.5}
def build(Pv,d0):
    negP=tuple(-x for x in Pv)
    s0=cyl.canon_front(d0); states={s0:0}; order=[s0]; tr=[]
    k=0
    while k<len(order):
        d=list(order[k]); din,dout=d[-1],d[0]
        turn=(dout-din)%12; turn=turn-12 if turn>6 else turn
        for t,td,ang in TILES:
            if ang>6-turn: continue
            tdir=[(dout+x)%12 for x in td]; rep=[(x+6)%12 for x in reversed(tdir[1:])]
            r=split_shift(rep+d[1:],Pv,negP)
            if r is None: continue
            f,holes,off=r; c=1; w=VW[t]; ar=AREA[t]
            for h in holes:
                cc,ww=fill_count(h); c*=cc; w+=ww; ar+=area(list(h))
            if c==0: continue
            if f not in states: states[f]=len(order); order.append(f)
            tr.append((k,states[f],c,int(round(2*w)),ar,off))
        k+=1
    return order,tr
if __name__=='__main__':
    a=tuple(int(x) for x in sys.argv[1:5]); Pv=a; PHAT[0]=cplx(Pv)/abs(cplx(Pv)); cyl.PHAT[0]=PHAT[0]
    base=cyl.path_for(a); d0=None
    for perm in itertools.permutations(base):
        r=cyl.front_split(list(perm),Pv,tuple(-x for x in Pv))
        if r is not None and not r[1]: d0=list(perm); break
    t0=time.time(); order,tr=build(Pv,d0)
    fr,to,c,w,ar,off=zip(*tr)
    off=np.array(off); dphys=off@np.array([zc**k for k in range(4)]); dperp=np.array([star(tuple(o)) for o in off])
    Pc=cplx(Pv); Ps=star(Pv); a_=(dphys*np.conj(Pc/abs(Pc))).real
    q=dperp-(a_/abs(Pc))*Ps                      # lateral-ambiguity-free perp displacement
    np.savez('tm2_%d_%d_%d_%d.npz'%a,fr=fr,to=to,c=c,w=w,ar=ar,q=q,n=len(order),P=Pc,Ps=Ps)
    print('saved',a,len(order),'%.0fs'%(time.time()-t0),flush=True)
