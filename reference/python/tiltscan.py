import numpy as np, sys
from tilt import load, solve
D=load(sys.argv[1]); rng=float(sys.argv[2]); n=int(sys.argv[3])
rows=[]; lu=-0.3
for pr in np.linspace(-rng,rng,n):
    for pi in np.linspace(-rng,rng,n):
        r=solve(D,pr+1j*pi,lu); lu=r['lu']
        rows.append((pr,pi,r['s'],r['alpha'].real,r['alpha'].imag,r['beta'].real,r['beta'].imag,r['nv']))
rows=np.array(rows); np.save(sys.argv[1].replace('.npz','_scan.npy'),rows)
print('alpha range Re [%.3f,%.3f] Im [%.3f,%.3f]'%(rows[:,3].min(),rows[:,3].max(),rows[:,4].min(),rows[:,4].max()))
# fit s = c + a1 Re(al) + a2 Im(al) + q11 Re^2 + q22 Im^2 + q12 Re Im
x,y,s,nv=rows[:,3],rows[:,4],rows[:,2],rows[:,7].mean()
X=np.c_[np.ones_like(x),x,y,x*x,y*y,x*y]; cf,res,_,_=np.linalg.lstsq(X,s,rcond=None)
print('fit coeffs c=%.6f a=(%.4f,%.4f) Q=(%.3f,%.3f,%.3f)  rms resid %.2e'%(*cf,np.sqrt(np.mean((X@cf-s)**2))))
np.save(sys.argv[1].replace('.npz','_fit.npy'),cf)
