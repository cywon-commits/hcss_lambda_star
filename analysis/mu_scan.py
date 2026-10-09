"""TASK1b: composition response x(ln mu) of a cylinder transfer matrix (needs fr,to,c,w,na,n in the npz).
Locked plateau between thresholds L- < ln mu < L+ ; in the bulk (harmonic + cusp):
   L+ = (K_b/(2 n_v) + s1)/|x'| ,  L- = -(K_a/(2 n_v) - s1)/|x'| ,  x' = -4/(sqrt3 n_v^2) = -0.928 ,  s1 = -s0 (1-1/sqrt3)/n_v
=> width  L+ - L- = Kbar/(n_v |x'|)          (check against Kbar = 1.76  ->  1.203)
   delta = (K_a - K_b)/2 = n_v (2 s1 - |x'| (L+ + L-))
Usage: python3 mu_scan.py file.npz [Lmax=1.6] [n=33]"""
import sys, numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as sla
def load(f):
    z=np.load(f); return z['fr'],z['to'],z['c'].astype(float),z['w'],z['na'],int(z['n'])
def eig(M,n):
    if n<3000:
        A=M.toarray(); ev,R=np.linalg.eig(A); i=np.argmax(abs(ev)); evl,L=np.linalg.eig(A.T); j=np.argmax(abs(evl))
        return ev[i].real,np.abs(R[:,i].real),np.abs(L[:,j].real)
    v,R=sla.eigs(M,k=1,which='LR',tol=1e-13); vl,L=sla.eigs(M.T.tocsr(),k=1,which='LR',tol=1e-13)
    return v[0].real,np.abs(R[:,0].real),np.abs(L[:,0].real)
def solve(data,lmu,lu=-0.30):
    fr,to,c,w,na,n=data
    for it in range(30):
        val=c*np.exp(lu*w+lmu*na)
        M=sp.csr_matrix((val,(to,fr)),shape=(n,n))
        rho,r,l=eig(M,n)
        Mw=sp.csr_matrix((val*w,(to,fr)),shape=(n,n)); Mn=sp.csr_matrix((val*na,(to,fr)),shape=(n,n))
        lr=l@r; Fu=(l@(Mw@r))/(rho*lr); Fm=(l@(Mn@r))/(rho*lr)
        F=np.log(rho)
        if abs(F)<1e-13: break
        lu-=F/Fu
    return -2*lu, 2*Fm/Fu, lu
f=sys.argv[1]; Lmax=float(sys.argv[2]) if len(sys.argv)>2 else 1.6; n=int(sys.argv[3]) if len(sys.argv)>3 else 33
D=load(f); L=np.linspace(-Lmax,Lmax,n); xs=[]; ss=[]; lu=-0.3
s0,x0,lu=solve(D,0.0,lu)
for l in L:
    s,x,lu=solve(D,l,lu); xs.append(x); ss.append(s)
xs=np.array(xs); d1=np.gradient(xs,L); d2=np.gradient(d1,L)
nv=1+1/np.sqrt(3); xp=4/(np.sqrt(3)*nv**2); s1=-s0*(1-1/np.sqrt(3))/nv
iL=np.argmax(np.where(L<0,np.abs(d2),-1)); iR=np.argmax(np.where(L>0,np.abs(d2),-1))
Lm,Lp=L[iL],L[iR]
print('%s  s0=%.5f x0=%.5f'%(f,s0,x0))
for l,x,a in zip(L,xs,d1): print('   lnmu=%+.3f  x=%.5f  dx/dlnmu=%.5f'%(l,x,a))
print('curvature-peak thresholds L-=%.3f L+=%.3f  width=%.3f (Kbar=1.76 predicts %.3f)  delta=%.3f'%(Lm,Lp,Lp-Lm,1.76/(nv*xp),nv*(2*s1-xp*(Lp+Lm))))
np.save(f.replace('.npz','_mu.npy'),np.c_[L,xs,np.array(ss)])
