from vec import *
import sys
ti,ri,n=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])
st=vsigma(ti,ri)
ty=np.ones(12,dtype=int); X=np.zeros((12,4),dtype=np.int64); K=np.arange(12)
for _ in range(n): ty,X,K=st(ty,X,K)
V=vertices(ty,X,K); x=V@Mx; xs=V@Ms
r=0.4*np.abs(x).max(); sel=np.abs(x)<r; xi=xs[sel]; N=sel.sum()
print((ti,ri),'tiles',len(ty),'N',N,'dens',N/(np.pi*r*r))
for h in [0.04,0.02,0.01,0.005]:
    ix=np.floor(xi.real/h).astype(int); iy=np.floor(xi.imag/h).astype(int)
    keys,cnt=np.unique(np.stack([ix,iy],1),axis=0,return_counts=True)
    S=set(map(tuple,keys))
    inter=np.array([all((a+da,b+db) in S for da in(-1,0,1) for db in(-1,0,1)) for a,b in keys])
    mc=np.mean(cnt[inter])
    print('  h=%.3f occ.area %.4f | interior-count vol est %.4f | interior cnt mean %.1f sd %.2f'%(h,len(keys)*h*h,N*h*h/mc,mc,cnt[inter].std()))
np.savez('pts_%d_%d.npz'%(ti,ri),xi=xi)
