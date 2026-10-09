import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla, sys
def load(f):
    z=np.load(f); return dict(fr=z['fr'],to=z['to'],c=z['c'].astype(float),w=z['w'].astype(float),ar=z['ar'],q=z['q'],n=int(z['n']),P=complex(z['P']),Ps=complex(z['Ps']))
def eig(M,n):
    if n<3000:
        A=M.toarray(); ev,R=np.linalg.eig(A); i=np.argmax(ev.real); evl,L=np.linalg.eig(A.T); j=np.argmax(evl.real)
        return ev[i].real,np.abs(R[:,i].real),np.abs(L[:,j].real)
    v,R=sla.eigs(M,k=1,which='LR',tol=1e-12); vl,L=sla.eigs(M.T.tocsr(),k=1,which='LR',tol=1e-12)
    return v[0].real,np.abs(R[:,0].real),np.abs(L[:,0].real)
def solve(D,phi,lu=-0.3):
    ex=(np.conj(phi)*D['q']).real
    for _ in range(40):
        val=D['c']*np.exp(lu*D['w']+ex); M=sp.csr_matrix((val,(D['to'],D['fr'])),shape=(D['n'],D['n']))
        rho,r,l=eig(M,D['n'])
        def avg(X): return l@(sp.csr_matrix((val*X,(D['to'],D['fr'])),shape=(D['n'],D['n']))@r)
        Fu=avg(D['w'])
        if abs(np.log(rho))<1e-12: break
        lu-=np.log(rho)/(Fu/(rho*(l@r)))
    per=lambda X: 2*avg(X)/Fu          # per-vertex average
    qv=per(D['q'].real)+1j*per(D['q'].imag); yv=per(D['ar'])/abs(D['P'])
    f=-2*lu; s=f-(np.conj(phi)*qv).real
    Ph=D['P']/abs(D['P']); g=D['Ps']/abs(D['P']); h=qv/yv
    alpha=(g-1j*h)/2/Ph; beta=(g+1j*h)/2/np.conj(Ph)
    return dict(s=s,alpha=alpha,beta=beta,nv=1/(per(D['ar'])),lu=lu)
if __name__=='__main__':
    D=load(sys.argv[1]); r0=solve(D,0)
    print('phi=0: s=%.6f  alpha=%s  beta=%s  n_v=%.5f  det E=%.5f'%(r0['s'],np.round(r0['alpha'],5),np.round(r0['beta'],5),r0['nv'],abs(r0['alpha'])**2-abs(r0['beta'])**2))
