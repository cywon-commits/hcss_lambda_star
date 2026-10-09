import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla, pickle
diff=np.load('diffmat.npy'); perm=np.load('dodec_perm.npy')   # perm[g,i] = index of g(T_i)
N=diff.shape[0]
A=sp.csr_matrix((diff==4).astype(float)); deg=np.asarray(A.sum(1)).ravel()
print('flip graph: degree min/mean/max',deg.min(),deg.mean().round(3),deg.max())
# connectivity, bipartiteness
from scipy.sparse.csgraph import connected_components
nc,lab=connected_components(A); print('components',nc)
col=-np.ones(N,int); col[0]=0; st=[0]; bip=True
Ad=A.tolil().rows
while st:
    i=st.pop()
    for j in Ad[i]:
        if col[j]<0: col[j]=1-col[i]; st.append(j)
        elif col[j]==col[i]: bip=False
print('bipartite',bip)
# symmetry check of the graph
for g in range(24):
    P=sp.csr_matrix((np.ones(N),(perm[g],np.arange(N))),shape=(N,N))
    assert abs(P@A@P.T-A).sum()==0
if bip:
    flips=[int(np.all(col[perm[g]]==col)) - int(np.all(col[perm[g]]!=col)) for g in range(24)]
    print('parity character of group elements (+1 keeps sublattice, -1 swaps):',flips)
# irreps of D12 in the element order used in dodec_rep.py: C^0..C^11, sv0..5, sd0..5
k=np.r_[np.arange(12),np.zeros(12,int)]; isrot=np.r_[np.ones(12,bool),np.zeros(12,bool)]; issv=np.r_[np.zeros(12,bool),np.ones(6,bool),np.zeros(6,bool)]
CH={'A1':np.ones(24),'A2':np.where(isrot,1,-1),
    'B1':np.where(isrot,(-1.0)**k,np.where(issv,1,-1)),'B2':np.where(isrot,(-1.0)**k,np.where(issv,-1,1))}
for j in range(1,6): CH[f'E{j}']=np.where(isrot,2*np.cos(2*np.pi*j*k/12),0)
def irrep_of(v):
    w={}
    for nm,ch in CH.items():
        d=ch[0]; pv=sum(ch[g]*v[np.argsort(perm[g])] for g in range(24))*d/24   # (U_g v)(i)=v(g^-1 i)
        w[nm]=np.dot(pv,v)
    return max(w,key=lambda x:w[x]),w
pickle.dump((A,deg,col,CH),open('qflip.pkl','wb'))
res=[]
for vt in [-3,-2,-1,-0.5,0,0.5,0.8,0.9,1.0,1.1,1.2,1.5,2.0]:
    H=-A+vt*sp.diags(deg)
    ev,V=sla.eigsh(H,k=6,which='SA',tol=1e-12)
    o=np.argsort(ev); ev=ev[o]; V=V[:,o]
    g0=V[:,0]*np.sign(V[:,0].sum())
    ir=[irrep_of(V[:,i])[0] for i in range(4)]
    rk=np.ones(N)/np.sqrt(N)
    pr=1/np.sum(g0**4)
    print('v/t=%5.2f  E0=%9.4f  gap=%.4f  irreps(0..3)=%s  <n_flip>=%.3f  |<RK|psi>|^2=%.4f  participation=%.0f  minamp>0:%s'%(
        vt,ev[0],ev[1]-ev[0],ir,(g0**2*deg).sum(),np.dot(rk,g0)**2,pr,(g0>-1e-12).all()),flush=True)
    res.append((vt,ev,g0))
pickle.dump(res,open('qflip_res.pkl','wb'))
