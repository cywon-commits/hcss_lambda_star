import numpy as np, sys
from nematic import load, solve
f=sys.argv[1]; D=load(f); r0=solve(D); lu=r0['lu']
print(f,'n=%d'%D['n'])
for h in [0.0,0.25,0.5,0.75,1.0,1.5]:
    rows=[]
    for pr in (-0.2,0,0.2):
        for pi in (-0.2,0,0.2):
            r=solve(D,phi=pr+1j*pi,h=h,lu=lu); lu=r['lu']; rows.append((r['s'],r['alpha'].real,r['alpha'].imag,r['nv'],r['psi']))
    s,x,y,nv,psi=map(np.array,zip(*rows))
    X=np.c_[np.ones_like(s),x,y,x*x,y*y,x*y]; cf=np.linalg.lstsq(X,s,rcond=None)[0]
    H=np.array([[2*cf[3],cf[5]],[cf[5],2*cf[4]]])          # Hessian of s per vertex w.r.t. (Re a, Im a)
    ev=np.linalg.eigvalsh(-H)*2*nv.mean()                    # -> stiffness-like numbers (K units, constrained cylinder mode)
    c=rows[4]                                               # phi=0
    print('  h=%.2f  stiffness eigenvalues of free strain mode: %s   alpha(phi=0)=%s  Psi2(phi=0)=%s'%(h,np.round(ev,3),np.round(c[1]+1j*c[2],4),np.round(c[4],3)),flush=True)
