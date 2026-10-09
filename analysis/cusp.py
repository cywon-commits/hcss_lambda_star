import sys, itertools, numpy as np, pickle
sys.argv=['x']+sys.argv[1:]
import cyl
from cyl import *
a=tuple(int(x) for x in sys.argv[1:5]); Pv=a; cyl.PHAT[0]=cplx(Pv)/abs(cplx(Pv))
base=path_for(a); d0=None
for perm in itertools.permutations(base):
    r=front_split(list(perm),Pv,tuple(-x for x in Pv))
    if r is not None and not r[1]: d0=list(perm); break
order,trans=build(Pv,d0,verbose=False)
res=[]
for lm in np.linspace(-0.6,0.6,13):
    f=entropy(order,trans,np.exp(lm)); h=1e-3
    x=(entropy(order,trans,np.exp(lm+h))-entropy(order,trans,np.exp(lm-h)))/(2*h)
    res.append((lm,f,x,f-x*lm))   # Legendre: s(x)=f-x*ln mu
    print('ln mu=%+.2f  x=%.4f  s(x)=%.5f'%(lm,x,f-x*lm),flush=True)
pickle.dump(res,open('cusp_%d%d%d%d.pkl'%tuple(a),'wb'))
