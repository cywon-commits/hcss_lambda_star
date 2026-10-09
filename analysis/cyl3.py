# cylinder TM recording, per transition, the rhombus contact-bond nematic sum X2 = sum exp(2i theta_b)
import sys, itertools, time, numpy as np
from front import *
from zz import star, zc
import cyl
from cyl2 import split_shift, AREA
exec(open('pairs.py').read().split("F=fills")[0].split("from front import *")[1])   # defines fills(start,dirs) -> absolute tile lists
def bond2(t,d):
    th=np.radians(30*d+105) if t=='Ra' else np.radians(30*d+75)
    return np.exp(2j*th)
def hole_bonds(h):
    out=[]
    for f in fills((0,0,0,0),list(h)):
        x=0j; ntri=0
        for t,vs in f:
            if t[0]=='A': ntri+=1; continue
            C=[cplx(v) for v in vs]
            for a,b in itertools.combinations(C,2):
                if abs(abs(a-b)-0.517638)<1e-4: x+=np.exp(2j*np.angle(b-a))
        out.append((x,ntri))
    return out
def build(Pv,d0):
    negP=tuple(-x for x in Pv)
    s0=cyl.canon_front(d0); states={s0:0}; order=[s0]; tr=[]; k=0; hcache={}
    while k<len(order):
        d=list(order[k]); din,dout=d[-1],d[0]
        turn=(dout-din)%12; turn=turn-12 if turn>6 else turn
        for t,td,ang in TILES:
            if ang>6-turn: continue
            tdir=[(dout+x)%12 for x in td]; rep=[(x+6)%12 for x in reversed(tdir[1:])]
            r=split_shift(rep+d[1:],Pv,negP)
            if r is None: continue
            f,holes,off=r
            base_x=bond2(t,dout) if t!='A' else 0j
            w=VW[t]; ar=AREA[t]
            opts=[(base_x,0)]
            for h in holes:
                if h not in hcache: hcache[h]=hole_bonds(h)
                cc,ww=fill_count(h); w+=ww; ar+=area(list(h))
                opts=[(x+y,0) for x,_ in opts for y,_ in hcache[h]]
            if not opts: continue
            if f not in states: states[f]=len(order); order.append(f)
            for x,_ in opts: tr.append((k,states[f],1,int(round(2*w)),ar,off,x))
        k+=1
    return order,tr
if __name__=='__main__':
    a=tuple(int(x) for x in sys.argv[1:5]); Pv=a; cyl.PHAT[0]=cplx(Pv)/abs(cplx(Pv))
    import cyl2; cyl2.PHAT[0]=cyl.PHAT[0]
    base=cyl.path_for(a); d0=None
    for perm in itertools.permutations(base):
        r=cyl.front_split(list(perm),Pv,tuple(-x for x in Pv))
        if r is not None and not r[1]: d0=list(perm); break
    t0=time.time(); order,tr=build(Pv,d0)
    fr,to,c,w,ar,off,x2=zip(*tr)
    off=np.array(off); dphys=off@np.array([zc**k for k in range(4)]); dperp=np.array([star(tuple(o)) for o in off])
    Pc=cplx(Pv); Ps=star(Pv); lat=(dphys*np.conj(Pc/abs(Pc))).real; q=dperp-(lat/abs(Pc))*Ps
    np.savez('tm3_%d_%d_%d_%d.npz'%a,fr=fr,to=to,c=c,w=w,ar=ar,q=q,x2=np.array(x2),n=len(order),P=Pc,Ps=Ps)
    print('saved',a,len(order),'transitions',len(tr),'%.0fs'%(time.time()-t0),flush=True)
