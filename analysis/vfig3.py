# Step 4: one representative per C3v orbit for class (c) and (d) of the DDDT crossing-set classification
import sys, pickle, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.argv=['x']
exec(open('vclass3x.py').read().split("# ---------- classification")[0])
rows,rowinfo,R=pickle.load(open('vclass3x.pkl','rb'))
juncv={'X':X,'Y':Y,'O':O}
def draw(ax,r):
    cls,osz,stab,nt,types,n,segs,irr,S,orb=r
    for P in (DaI,DbI,DcI,TI):
        C=[cplx(p) for p in P]+[cplx(P[0])]; ax.plot([z.real for z in C],[z.imag for z in C],c='0.8',lw=0.6)
    for t in S:
        Z=ordered(t[1]); ax.fill([z[0] for z in Z],[z[1] for z in Z],fc='#f2b134' if t[0]=='A' else '#4a7fb5',ec='k',lw=0.5,alpha=0.9)
    for k,(a,b) in SEG3.items(): ax.plot([cplx(a).real,cplx(b).real],[cplx(a).imag,cplx(b).imag],'r-',lw=1.2)
    for nm,v in juncv.items():
        z=cplx(v); has=any(v in t[1] for t in S); ax.plot(z.real,z.imag,'ko' if has else 'kx',ms=4)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('%s  %d tiles  %s\n%.3g x%d'%(cls,nt,stab,n,osz),fontsize=6.5)
def sheet(sel,fn,title,ncol=8):
    nr=(len(sel)+ncol-1)//ncol; fig,axs=plt.subplots(nr,ncol,figsize=(2.6*ncol,2.5*nr+0.6),squeeze=False)
    for ax in axs.ravel(): ax.axis('off')
    for ax,r in zip(axs.ravel(),sel): draw(ax,r)
    plt.suptitle(title,fontsize=11); plt.tight_layout(rect=(0,0,1,0.98)); plt.savefig(fn,dpi=100); plt.close()
c=sorted([r for r in rows if r[0]=='c'],key=lambda r:-r[5]); d=sorted([r for r in rows if r[0]=='d'],key=lambda r:-r[5])
sheet(c,'../figures/dddt_vertex_c.png','DDDT class (c): one-junction vertex mode (+bond modes), one member per C3v orbit (red = skeleton, o = triangle vertex occupied, x = vacated); title: tiles, stabiliser, fillings per member x orbit size')
h=(len(d)+1)//2
sheet(d[:h],'../figures/dddt_coop_d1.png','DDDT class (d): cooperative modes, orbits 1-%d of %d (sorted by fillings per member)'%(h,len(d)))
sheet(d[h:],'../figures/dddt_coop_d2.png','DDDT class (d): cooperative modes, orbits %d-%d of %d'%(h+1,len(d),len(d)))
print(len(c),len(d))
