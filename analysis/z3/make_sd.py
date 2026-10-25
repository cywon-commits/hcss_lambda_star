"""Single-domain 6-fold start: snub-square (sigma) tiling tilted by the family-rotation mechanism (theta=10°),
scaled to rho*=0.98, written in hcss_mc (s, box=(a,b,c)) format."""
import numpy as np, math, sys, os
from collections import deque
c=(1+math.sqrt(3))/math.sqrt(2)
def sq(cen,rot): return [cen+np.array([math.cos(a),math.sin(a)])/math.sqrt(2) for a in np.radians(rot+45+90*np.arange(4))]
basis=np.array(sq(np.zeros(2),15)+sq(np.array([c/2,c/2]),-15))
# dedupe basis modulo lattice
A=np.array([[c,0],[0,c]])
B=[]
for p in basis:
    f=np.linalg.solve(A.T,p)%1.0
    if not any(np.allclose(f,q,atol=1e-6) or np.allclose((f-q+0.5)%1-0.5,0,atol=1e-6) for q in B): B.append(f)
B=np.array(B); nb=len(B); print("basis sites",nb)
X=B@A.T
# edges with image shifts
edges=[]
for i in range(nb):
    for j in range(nb):
        for n1 in (-1,0,1):
            for n2 in (-1,0,1):
                d=X[j]+n1*A[:,0]+n2*A[:,1]-X[i]
                if abs(np.linalg.norm(d)-1)<1e-6 and (i,j,n1,n2)<(j,i,-n1,-n2): edges.append((i,j,n1,n2,d))
print("edges per cell",len(edges))
th=math.radians(float(sys.argv[1]) if len(sys.argv)>1 else 10.0)
off0=math.degrees(math.atan2(edges[0][4][1],edges[0][4][0]))%30
def rot(d):
    m=round((math.degrees(math.atan2(d[1],d[0]))-off0)/30)
    if m%2: d=np.array([[math.cos(th),-math.sin(th)],[math.sin(th),math.cos(th)]])@d
    return d
E=[(i,j,n1,n2,rot(d)) for i,j,n1,n2,d in edges]
# unknowns: x_1..x_{nb-1} (x_0=0), A1', A2' (2 each) ; equations x_j + n1 A1 + n2 A2 - x_i = d'
nu=2*(nb-1)+4; rows=[]; rhs=[]
idx=lambda i:None if i==0 else 2*(i-1)
for i,j,n1,n2,d in E:
    for k in range(2):
        r=np.zeros(nu)
        if idx(j) is not None: r[idx(j)+k]+=1
        if idx(i) is not None: r[idx(i)+k]-=1
        r[2*(nb-1)+k]+=n1; r[2*(nb-1)+2+k]+=n2
        rows.append(r); rhs.append(d[k])
sol,res,rk,_=np.linalg.lstsq(np.array(rows),np.array(rhs),rcond=None)
print("closure residual",np.abs(np.array(rows)@sol-rhs).max())
Xs=np.vstack([np.zeros(2),sol[:2*(nb-1)].reshape(-1,2)]); A1=sol[-4:-2]; A2=sol[-2:]
# supercell
n=int(sys.argv[2]) if len(sys.argv)>2 else 17
M=np.column_stack([n*A1,n*A2]); P=np.array([Xs[k]+i*A1+j*A2 for i in range(n) for j in range(n) for k in range(nb)])
rho=len(P)/abs(np.linalg.det(M)); scale=math.sqrt(rho/0.98); P*=scale; M*=scale
# rotate so first vector along x, upper-triangular
ang=math.atan2(M[1,0],M[0,0]); R=np.array([[math.cos(-ang),-math.sin(-ang)],[math.sin(-ang),math.cos(-ang)]])
M=R@M; P=P@R.T
if M[1,1]<0: M[:,1]*=-1
a,b,cc=M[0,0],M[0,1],M[1,1]; b-=round(b/a)*a
s=np.stack([(P[:,0]-b*P[:,1]/cc)/a, P[:,1]/cc],1)%1.0
print("N",len(P),"scale (edge length)",round(scale,4),"box",round(a,3),round(b,3),round(cc,3))
sys.path.insert(0,os.environ.get("HCSS_MC_DIR","../../../hardcore_square_shoulder")); import hcss_mc as mc  # set HCSS_MC_DIR to the hcss_mc.py folder
print("shoulder pairs per particle", mc.total_count(s,a,b,cc,1.27)/len(P))
np.savez("sd_theta10.npz",s=s,box=np.array([a,b,cc]),lam=1.27,T=0.24)
