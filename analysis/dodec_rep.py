import numpy as np, itertools, sys
from collections import Counter
exec(open('pairs.py').read().split("F=fills")[0])
F=fills((0,0,0,0),list(range(12)))
from zz import cplx
V=[cplx(p) for p in pos(list(range(12)))[:-1]]; c=np.mean(V); z=np.exp(1j*np.pi/6)
tv=np.angle(V[0]-c); td=np.angle((V[0]+V[1])/2-c)
G=[]   # (label, class, map)
for k in range(12): G.append((f'C^{k}',('rot',min(k,12-k)),lambda x,k=k:c+z**k*(x-c)))
for k in range(6): G.append((f'sv{k}',('sv',),lambda x,k=k:c+np.exp(2j*(tv+k*np.pi/6))*np.conj(x-c)))
for k in range(6): G.append((f'sd{k}',('sd',),lambda x,k=k:c+np.exp(2j*(td+k*np.pi/6))*np.conj(x-c)))
key=lambda w:(int(round(w.real*1e5)),int(round(w.imag*1e5)))
def enc(f,g=lambda x:x): return frozenset((t[0],frozenset(key(g(cplx(v))) for v in vs)) for t,vs in [(t,vs) for t,vs in f])
S=[enc(f) for f in F]; idx={s:i for i,s in enumerate(S)}
perm=[]
for lab,cl,g in G:
    im=[idx[enc(f,g)] for f in F]; perm.append(im)
perm=np.array(perm)
chi=(perm==np.arange(len(F))[None,:]).sum(1)
print('fillings',len(F))
for (lab,cl,g),x in zip(G,chi): print(f'  {lab:6s} fixes {x}')
# stabilizers
stab=Counter()
for i in range(len(F)):
    H=tuple(G[j][0] for j in range(24) if perm[j,i]==i); stab[H]+=1
print('stabilizer types:')
for H,n in sorted(stab.items(),key=lambda t:-len(t[0])): print('  |H|=%d'%len(H),n,'states', H if len(H)<=6 else '...')
# D12 irreps
def irrep_chars():
    ch={}
    rot=[g[1] for g in G]
    def row(f): return np.array([f(g) for g in G])
    k=lambda g:int(g[0].split('^')[1]) if g[0].startswith('C^') else None
    ch['A1']=row(lambda g:1)
    ch['A2']=row(lambda g:1 if g[1][0]=='rot' else -1)
    ch['B1']=row(lambda g:(-1)**k(g) if g[1][0]=='rot' else (1 if g[1][0]=='sv' else -1))
    ch['B2']=row(lambda g:(-1)**k(g) if g[1][0]=='rot' else (-1 if g[1][0]=='sv' else 1))
    for j in range(1,6): ch[f'E{j}']=row(lambda g,j=j:2*np.cos(2*np.pi*j*k(g)/12) if g[1][0]=='rot' else 0)
    return ch
CH=irrep_chars()
print('D12 decomposition of the 5827-dim permutation rep:')
tot=0
for nm,x in CH.items():
    m=(chi*x).sum()/24; tot+=m*x[0]; print(f'  {nm}: {m:.4f}')
print('  dimension check',tot)
# restriction to site group 6mm of 3.12.12 (rotations by 60deg + edge-midpoint mirrors) and to C12 (chiral)
for name,sel in (('C6v (3.12.12 site, sd mirrors)',[i for i,g in enumerate(G) if (g[0].startswith('C^') and int(g[0][2:])%2==0) or g[1][0]=='sd']),
                 ('C12 (rotations only)',[i for i,g in enumerate(G) if g[0].startswith('C^')])):
    print(name,'order',len(sel),' orbits (A1 multiplicity) =',chi[sel].sum()/len(sel))
np.save('dodec_perm.npy',perm)
