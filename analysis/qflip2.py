import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla, pickle
from collections import Counter
A,deg,col,CH=pickle.load(open('qflip.pkl','rb')); perm=np.load('dodec_perm.npy'); N=len(deg)
labels=[f'C^{k}' for k in range(12)]+[f'sv{k}' for k in range(6)]+[f'sd{k}' for k in range(6)]
def stab(i): return tuple(labels[g] for g in range(24) if perm[g,i]==i)
def orbit(i): return sorted(set(perm[:,i]))
for nm,sel in (('max degree',np.where(deg==deg.max())[0]),('min degree',np.where(deg==deg.min())[0])):
    print(nm,deg[sel[0]],'#states',len(sel),'stabilizers',Counter(len(stab(i)) for i in sel),'example',stab(sel[0]))
exec(open('qflip.py').read().split("def irrep_of")[1].join(["def irrep_of",""]) if False else "")
def irrep_of(v):
    w={}
    for nm,ch in CH.items():
        pv=sum(ch[g]*v[np.argsort(perm[g])] for g in range(24))*ch[0]/24; w[nm]=np.dot(pv,v)
    return max(w,key=lambda x:w[x])
for vt in (-3.0,2.0,3.0):
    H=-A+vt*sp.diags(deg); ev,V=sla.eigsh(H,k=16,which='SA',tol=1e-12); o=np.argsort(ev); ev,V=ev[o],V[:,o]
    g0=np.abs(V[:,0]); top=np.argsort(-g0)[:3]
    print(f'v/t={vt}: lowest levels (E-E0) and irreps:')
    print('   ',[(round(e-ev[0],4),irrep_of(V[:,i])) for i,e in enumerate(ev[:16])])
    print('    GS weight on top state(s):',[(int(i),round(g0[i]**2,3),int(deg[i]),len(stab(i))) for i in top])
    # weight of GS on degree classes
    w=Counter(); 
    for i in range(N): w[int(deg[i])]+=g0[i]**2
    print('    GS weight by flip-number:',{k:round(x,3) for k,x in sorted(w.items()) if x>0.01})
