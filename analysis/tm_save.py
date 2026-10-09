import sys, itertools, pickle, numpy as np
import cyl
from cyl import *
a=tuple(int(x) for x in sys.argv[1:5]); Pv=a; cyl.PHAT[0]=cplx(Pv)/abs(cplx(Pv))
base=path_for(a); d0=None
for perm in itertools.permutations(base):
    r=front_split(list(perm),Pv,tuple(-x for x in Pv))
    if r is not None and not r[1]: d0=list(perm); break
order,trans=build(Pv,d0,verbose=False)
fr,to,c,w,na=map(np.array,zip(*trans))
np.savez('tm_%d_%d_%d_%d.npz'%a,fr=fr,to=to,c=c,w=w,na=na,n=len(order))
print('saved',a,len(order),flush=True)
