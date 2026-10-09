import numpy as np, sys
for f in sys.argv[1:]:
    R=np.load(f); s=R[:,2]; al=R[:,3]+1j*R[:,4]; be=R[:,5]+1j*R[:,6]; nv=R[:,7].mean()
    X=np.c_[np.ones_like(s),-np.abs(al)**2/(2*nv),-be.real**2/(2*nv),-be.imag**2/(2*nv)]
    cf,_,rk,_=np.linalg.lstsq(X,s,rcond=None); rms=np.sqrt(np.mean((X@cf-s)**2))
    s0,Ka,Kr,Ki=cf
    print('%-28s s0=%.5f  K_alpha=%.3f  K_r(Re beta)=%.3f  K_i(Im beta)=%.3f   rank %d  rms %.1e'%(f,s0,Ka,Kr,Ki,rk,rms))
