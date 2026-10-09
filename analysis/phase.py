import numpy as np, math, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
LS=2*math.cos(math.radians(15)); S3=math.sqrt(3); Pst=S3-1
s_old=math.log(4421)/19; s_new=0.594; dS=s_new-s_old
# calibrated (results5, kappa=1.7) entropy constants, which contain s_old
sB0,sA0=0.1211,0.7666; sB,sA=sB0+dS,sA0+dS
print('s_old=%.4f  correction %+.4f  ->  sigma_B=%.4f sigma_A=%.4f'%(s_old,dS,sB,sA))
# exact geometry at lambda*
et=(3-S3)/2; vt=(3+S3)/2; vA=S3/2*LS**2; vB=math.sqrt(LS**2-0.25)
print('at lambda*: v_t-v_B=%.5f (1/2)  v_A-v_t=%.5f (sqrt3/2)  P*=%.5f'%(vt-vB,vA-vt,(1-et)/(vt-vB)))
cB=1/(vt-vB); cA=1/(vA-vt)
Kc=1.75; nv=1+1/S3; s1=-0.594*(1-1/S3)/nv; cusp=Kc/(2*nv); fld=0.170*LS**2
lock_hi=(cusp-s1)/fld; lock_lo=-(cusp+s1)/fld
print('window/T: [%.3f, %+.3f]   (old s_conf: [%.3f, %+.3f])'%(-cA*sA,cB*sB,-cA*sA0,cB*sB0))
print('lock window/T: [%.3f, %+.3f]'%(lock_lo,lock_hi))
# lambda dependence (theory_window.py, kappa=1.7 for lambda<lambda*), 3.12.12 geometry as in the calibration
E_T=12/19; KAP=1.7
def vA_(l): return S3/2*l*l
def vB_(l): return math.sqrt(l*l-0.25)
def vt_(l): L=max(l,LS); return L*L*(14*S3+24)/76-KAP*max(0,LS-l)
def win(l,T,sA,sB): return (E_T-T*sA)/(vA_(l)-vt_(l)), (1-E_T+T*sB)/(vt_(l)-vB_(l))
# check old measured points at lambda=1.93 (results4), D_tB, D_tA shift by -dS
pts=[(0.660,0.06,-0.732,0.302),(0.700,0.06,-0.400,-0.263),(0.735,0.06,-0.111,-0.772),(0.760,0.06,0.091,-1.126),
     (0.700,0.08,-0.323,-0.396),(0.735,0.08,-0.093,-0.761),(0.770,0.08,0.121,-1.142)]
print('re-judged results4 points at lambda=1.93 (D_tB, D_tA after +s_conf correction):')
for P,T,dB,dA in pts:
    nb,na=dB-dS,dA-dS; win_=('tiling' if nb<0 and na<0 else ('B' if nb>0 and nb>na else 'A'))
    print('   P=%.3f T=%.2f  old (%+.3f,%+.3f) -> new (%+.3f,%+.3f)  winner %s'%(P,T,dB,dA,nb,na,win_))
# ---------------- figure
fig,ax=plt.subplots(1,2,figsize=(13,5.2))
T=np.linspace(0,0.135,200); Tm=0.12
a=ax[0]
a.fill_betweenx(T,Pst-cA*sA*T,Pst+cB*sB*T,where=T<=Tm,color='#cfe3f5',label='random tiling (12-fold QC)')
a.fill_betweenx(T,Pst-cA*sA*T,np.maximum(Pst+lock_lo*T,Pst-cA*sA*T),where=T<=Tm,color='#f7dfb8',label='tiling, composition unlocked (triangle-rich, beta strain)')
a.plot(Pst-cA*sA0*T,T,'k:',lw=1); a.plot(Pst+cB*sB0*T,T,'k:',lw=1,label='old window (s_conf = ln4421/19)')
a.plot(Pst+lock_lo*T,T,'--',c='#c77c00',lw=1); a.plot(Pst+lock_hi*T,T,'--',c='#c77c00',lw=1,label='composition lock limits')
a.axhline(Tm,c='gray',lw=0.8); a.text(0.63,Tm+0.002,'tiling melting (defect unbinding) T~0.12',fontsize=8,color='gray')
a.text(0.632,0.03,'A crystal',fontsize=11); a.text(0.787,0.07,'B crystal',fontsize=11)
for P,Tt,dB,dA in pts:
    nb,na=dB-dS,dA-dS; c='#1f77b4' if (nb<0 and na<0) else ('#d62728' if nb>na else '#2ca02c')
    a.plot(P,Tt,'o',ms=7,mfc=c,mec='k')
a.plot(Pst,0,'k*',ms=12); a.text(Pst+0.002,0.003,'P* = sqrt3-1',fontsize=8)
a.set_xlim(0.62,0.82); a.set_ylim(0,0.135); a.set_xlabel('P  (epsilon/sigma^2)'); a.set_ylabel('T  (epsilon/k_B)')
a.set_title('(a) lambda = lambda* : P-T plane'); a.legend(fontsize=7,loc='upper right')
b=ax[1]; L=np.linspace(1.80,1.99,400)
lo,hi=np.array([win(l,0.06,sA,sB) for l in L]).T; ok=hi>lo
b.fill_between(L,lo,hi,where=ok,color='#9ecae1',label='random tiling, T=0.06 (s_conf=0.594)')
lo8,hi8=np.array([win(l,0.08,sA,sB) for l in L]).T; ok8=hi8>lo8
b.plot(L[ok8],lo8[ok8],'-',c='#08519c',lw=1.2,label='same, T=0.08'); b.plot(L[ok8],hi8[ok8],'-',c='#08519c',lw=1.2)
lo0,hi0=np.array([win(l,0.06,sA0,sB0) for l in L]).T; ok0=hi0>lo0
b.plot(L[ok0],lo0[ok0],'k:',lw=1.2,label='old, T=0.06 (s_conf=ln4421/19)'); b.plot(L[ok0],hi0[ok0],'k:',lw=1.2)
b.axvspan(1.80,1.85,color='0.92'); b.text(1.802,0.62,'kappa-model\nextrapolation',fontsize=7,color='gray')
b.axvline(LS,c='gray',lw=0.8); b.text(LS+0.002,0.88,'lambda*',fontsize=8)
b.text(1.86,0.86,'B crystal',fontsize=11); b.text(1.945,0.64,'A crystal',fontsize=11)
b.set_xlabel('lambda (shoulder width / core)'); b.set_ylabel('P  (epsilon/sigma^2)'); b.set_ylim(0.6,0.9); b.set_xlim(1.80,1.99)
b.set_title('(b) lambda-P plane'); b.legend(fontsize=7,loc='upper right')
plt.tight_layout(); plt.savefig('/mnt/user-data/outputs/phase_diagram.png',dpi=130)
# lambda-extent of the band at each T (new vs old)
for Tt in (0.04,0.06,0.08,0.10):
    def ext(sa,sb):
        ok=[l for l in np.linspace(1.70,2.10,4001) if win(l,Tt,sa,sb)[1]>win(l,Tt,sa,sb)[0]]
        return (min(ok),max(ok)) if ok else None
    e1=ext(sA,sB); e0=ext(sA0,sB0)
    print('T=%.2f  tiling band in lambda: new %s   old %s'%(Tt,np.round(e1,3),np.round(e0,3)))
