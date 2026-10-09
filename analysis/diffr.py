import numpy as np
z=np.exp(1j*np.pi/6); lam=2+np.sqrt(3)
# --- (a) physical supertile edge curve: self-avoidance + box dimension
dirs=np.array([0]); S=np.array([0,1,-3,1,0])
for n in range(8): dirs=(dirs[:,None]+S[None,:]).ravel()
p=np.concatenate([[0],np.cumsum(z**dirs)])/lam**8
key=np.round(p*lam**8*1e6).astype(np.int64)
print('level 8: steps',len(dirs),' distinct vertices',len(np.unique(np.round(p*lam**8,6))),'(self-avoiding if = steps+1)')
hs=np.array([1/2**k for k in range(4,10)]); nb=[]
for h in hs: nb.append(len(set(zip(np.floor(p.real/h).astype(int),np.floor(p.imag/h).astype(int)))))
sl=[np.log(nb[i+1]/nb[i])/np.log(2) for i in range(len(nb)-1)]
print('edge curve box counts',nb,' local slopes',np.round(sl,3),' log5/loglam=%.4f'%(np.log(5)/np.log(lam)))
# --- (b) Bragg intensities |W^(q)|^2 vs |k_perp|
def radial(xi,L=2.0,G=2048):
    h=2*L/G
    H,_,_=np.histogram2d(xi.real,xi.imag,bins=G,range=[[-L,L],[-L,L]])
    F=np.fft.fftshift(np.fft.fft2(H))/len(xi)
    q=np.fft.fftshift(np.fft.fftfreq(G,d=h))*2*np.pi
    QX,QY=np.meshgrid(q,q,indexing='ij'); Q=np.hypot(QX,QY)
    corr=(np.sinc(QX*h/2/np.pi)*np.sinc(QY*h/2/np.pi))**2     # undo pixel binning
    I=np.abs(F)**2/corr
    edges=np.logspace(np.log10(5),np.log10(0.35*np.pi/h),40)
    c=0.5*(edges[1:]+edges[:-1]); m=[I[(Q>=edges[i])&(Q<edges[i+1])].mean() for i in range(len(c))]
    return c,np.array(m)
res={}
for nm in ('6_15','6_4'):
    xi=np.load(f'pts_{nm}.npz')['xi']; res[nm]=radial(xi)
# control: polygonal (regular dodecagon) window, low-discrepancy grid points, same area 3+sqrt3
A=3+np.sqrt(3); R=np.sqrt(A/3)          # dodecagon area = 3 R^2
g=np.linspace(-R,R,1300); X,Y=np.meshgrid(g,g); P=(X+1j*Y).ravel()
ang=np.angle(P); ok=np.abs(P)*np.cos((ang%(np.pi/6))-np.pi/12)<=R*np.cos(np.pi/12)
res['dodecagon']=radial(P[ok])
for nm,(c,m) in res.items():
    sel=(c>15)&(c<250)
    s=np.polyfit(np.log(c[sel]),np.log(m[sel]),1)[0]
    print('%-10s slope of <I>(k_perp) on 15<k<250: %.3f'%(nm,s))
print('predicted: 6_15 %.3f   6_4 %.3f   polygon -3'%(-(4-np.log(5)/np.log(lam)),-(4-1.653)))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(6,4.5))
for nm,(c,m) in res.items(): ax.loglog(c,m/m[0],'o-',ms=3,label=nm)
ax.set_xlabel('|k_perp|'); ax.set_ylabel('<I> (shell average, normalised)'); ax.legend(); plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/bragg_vs_kperp.png',dpi=120)
