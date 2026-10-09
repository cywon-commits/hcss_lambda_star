import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla, sys
from tilt import eig
def load(f):
    z=np.load(f); return dict(fr=z['fr'],to=z['to'],c=z['c'].astype(float),w=z['w'].astype(float),ar=z['ar'],q=z['q'],x2=z['x2'],n=int(z['n']),P=complex(z['P']),Ps=complex(z['Ps']))
def solve(D,phi=0j,h=0j,lu=-0.3):
    ex=(np.conj(phi)*D['q']).real+(np.conj(h)*D['x2']).real
    for _ in range(40):
        val=D['c']*np.exp(lu*D['w']+ex); M=sp.csr_matrix((val,(D['to'],D['fr'])),shape=(D['n'],D['n']))
        rho,r,l=eig(M,D['n'])
        avg=lambda X: l@(sp.csr_matrix((val*X,(D['to'],D['fr'])),shape=(D['n'],D['n']))@r)
        Fu=avg(D['w'])
        if abs(np.log(rho))<1e-12: break
        lu-=np.log(rho)/(Fu/(rho*(l@r)))
    per=lambda X: 2*avg(X)/Fu
    psi=per(D['x2'].real)+1j*per(D['x2'].imag)
    qv=per(D['q'].real)+1j*per(D['q'].imag); yv=per(D['ar'])/abs(D['P'])
    Ph=D['P']/abs(D['P']); g=D['Ps']/abs(D['P']); hh=qv/yv
    al=(g-1j*hh)/2/Ph; be=(g+1j*hh)/2/np.conj(Ph)
    return dict(psi=psi,alpha=al,beta=be,s=-2*lu-(np.conj(phi)*qv).real-(np.conj(h)*psi).real,lu=lu,nv=1/per(D['ar']))
if __name__=='__main__':
    f=sys.argv[1]; D=load(f); r0=solve(D)
    print(f,' n=%d  Psi2(h=0)=%s  alpha=%s beta=%s'%(D['n'],np.round(r0['psi'],5),np.round(r0['alpha'],4),np.round(r0['beta'],4)))
    # nematic susceptibility, two field directions (theta_h = 0 and 45 deg)
    dh=0.05
    for nm,u in (('h real (theta_h=0)',1+0j),('h imag (theta_h=45deg)',1j)):
        rp=solve(D,h=dh*u,lu=r0['lu']); rm=solve(D,h=-dh*u,lu=r0['lu'])
        dpsi=(rp['psi']-rm['psi'])/(2*dh)
        print('   %-24s dPsi2/dh = %s  -> chi2 = %.4f'%(nm,np.round(dpsi,5),(dpsi*np.conj(u)).real))
    # strain-nematic coupling kappa: tilt scan at h=0
    rows=[]
    for pr in np.linspace(-0.3,0.3,5):
        for pi in np.linspace(-0.3,0.3,5):
            r=solve(D,phi=pr+1j*pi,lu=r0['lu']); rows.append((r['psi'],r['alpha'],r['beta']))
    psi,al,be=map(np.array,zip(*rows))
    X=np.c_[np.ones_like(psi),np.conj(al)*be,al**2*np.conj(be),al,np.conj(al),be,np.conj(be)]
    cf=np.linalg.lstsq(X,psi,rcond=None)[0]; res=np.abs(X@cf-psi).max()
    print('   fit (with cylinder-anisotropy linear terms): kappa=%s  kappa3=%s  |linear coeffs|=%s  max resid %.1e'%(np.round(cf[1],3),np.round(cf[2],2),np.round(np.abs(cf[3:]),3),res))
    np.save(f.replace('.npz','_nem.npy'),dict(cf=cf,r0=r0),allow_pickle=True)
