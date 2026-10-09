import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla, pickle
A,deg,col,CH=pickle.load(open('qflip.pkl','rb')); perm=np.load('dodec_perm.npy')
def irrep_of(v):
    w={nm:np.dot(sum(ch[g]*v[np.argsort(perm[g])] for g in range(24))*ch[0]/24,v) for nm,ch in CH.items()}
    return max(w,key=lambda x:w[x])
for vt in (-1.0,0.0,1.0,2.0):
    H=A+vt*sp.diags(deg); ev,V=sla.eigsh(H,k=6,which='SA',tol=1e-12); o=np.argsort(ev); ev,V=ev[o],V[:,o]
    print('t<0 (sign-flipped tunneling) v/|t|=%4.1f  E0=%.4f  levels:'%(vt,ev[0]),[(round(e-ev[0],4),irrep_of(V[:,i])) for i,e in enumerate(ev)])
