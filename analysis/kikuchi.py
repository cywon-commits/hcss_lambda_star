"""Cluster-variation (Kikuchi) estimates of the 3.12.12-sector entropy from exact region counts.
Per 3.12.12 cell (19 particles): 1 dodecagon D, 2 triangles, 3 D-D edges, 6 D-T edges, 2 DDDT (triangle-centred) clusters."""
from math import log
N_D=5827; N_DT=6865; N_DD=38120461; N_DDDT=749563960730
bond =3*log(N_DD)+6*log(N_DT)-11*log(N_D)          # maximal clusters DD, DT ; intersections D (a=-11), T (a=-2, ln1=0)
vert =2*log(N_DDDT)-3*log(N_DD)+log(N_D)            # maximal clusters DDDT ; intersections DD (a=-1), D (a=+1)
for name,v in (('skeleton (independent dodecagons)',log(N_D)),('bond level (DD, DT)',bond),('vertex level (DDDT)',vert)):
    print('%-36s ln Z/cell = %7.3f   s = %.4f per particle'%(name,v,v/19))
print('exact 3.12.12-sector entropy (transfer matrix): 0.591')
