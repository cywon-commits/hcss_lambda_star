import pickle, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from zz import cplx
exec(open('vclass.py').read().split("tot=sum")[0])
rows,orbits,_=pickle.load(open('vclass.pkl','rb'))
sel=[o for o in orbits if o[0]]   # skip decoupled
sel.sort(key=lambda o:(not info[o[0]]['bond'],-info[o[0]]['n']))
fig,axs=plt.subplots(2,7,figsize=(21,6.6))
outline=[DaI,DbI,TI]
for ax,orb in zip(axs.ravel(),sel):
    S=orb[0]; v=info[S]
    for P in outline:
        C=[cplx(p) for p in P]+[cplx(P[0])]; ax.plot([z.real for z in C],[z.imag for z in C],c='0.75',lw=0.8)
    for t in S:
        Z=ordered(t[1]); ax.fill([z[0] for z in Z],[z[1] for z in Z],fc='#f2b134' if t[0]=='A' else '#4a7fb5',ec='k',lw=0.8,alpha=0.9)
    for k,(a,b) in SEG.items(): ax.plot([cplx(a).real,cplx(b).real],[cplx(a).imag,cplx(b).imag],'r-',lw=1.6)
    ax.plot(Xc.real,Xc.imag,'ko' if v['Xv'] else 'kx',ms=6)
    ax.set_xlim(Xc.real-1.7,Xc.real+1.7); ax.set_ylim(Xc.imag-1.5,Xc.imag+1.9); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(('bond' if v['bond'] else 'vertex')+'  %s\n%d per member (x2 mirror)'%(v['types'],v['n']),fontsize=9)
plt.suptitle('DDT cluster: all coupling modes (red = skeleton edges B1,B2,B3 meeting at junction X; o = X is a vertex, x = X vacated)',fontsize=11)
plt.tight_layout(); plt.savefig('/mnt/user-data/outputs/vertex_modes.png',dpi=110)
