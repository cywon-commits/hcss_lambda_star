import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla, sys
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
for f in sys.argv[1:]:
    D=load(f); s0,x0,lu=solve(D,0.0)
    h=0.05; xs=[solve(D,k*h,lu)[1] for k in (-2,-1,1,2)]
    chi=(-xs[3]+8*xs[2]-8*xs[1]+xs[0])/(12*h)
    print('%s  n=%d  s=%.6f  x=%.6f  chi=dx/dlnmu=%.5f'%(f,D[5],s0,x0,chi),flush=True)
