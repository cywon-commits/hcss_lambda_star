import pickle
from zz import *
(trs,trc),(rhs,rhc)=pickle.load(open('chk3.pkl','rb'))
lam=add(add(E[0],E[0]),add(add(E[1],E[1]),E[9]))
def F(sols,cl): return [frozenset((cl[i][0],frozenset(cl[i][2])) for i in s) for s in sols]
T=F(trs,trc); Rh=F(rhs,rhc)
def img(Fl,f): return frozenset((t,frozenset(f(v) for v in V)) for t,V in Fl)
c3=lambda x: add(lam,rot(x,4))
c2=lambda x: sub(mul(lam,add(E[0],E[1])),x)
for name,L,f,o in (('tri C3',T,c3,3),('rho C2',Rh,c2,2)):
    idx={x:i for i,x in enumerate(L)}
    assert all(img(x,f) in idx for x in L)
    inv=[i for i,x in enumerate(L) if img(x,f)==x]
    seen=set();orb=[]
    for i,x in enumerate(L):
        if i in seen: continue
        o_=[];g=x
        for _ in range(o): o_.append(idx[g]); g=img(g,f)
        seen|=set(o_); orb.append(sorted(set(o_)))
    print(name,'invariant',inv,'orbit sizes',sorted(len(a) for a in orb))
pickle.dump((T,Rh,lam),open('fills.pkl','wb'))
