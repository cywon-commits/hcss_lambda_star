"""Convert the C++ builder's raw output directory into the npz format read by tilt.py / chi.py.
Directory layout (all little-endian, one value per transition unless noted):
  meta.json : {"P_int4":[a0,a1,a2,a3], "n_states":N, "n_transitions":M}
  fr.i4 to.i4   int32   source / target state index (BFS discovery order, start state = 0)
  c.u8          uint64  multiplicity (product of hole fillings)
  w.i4          int32   2 x (vertices added)   [triangle 0.5, rhombus 1, plus holes]
  na.i4         int32   triangles added (tile + holes)
  ar.f8         float64 area added (tile + holes)
  off.i4        int32   4 x M : lattice offset of the new canonical start vertex (Z[zeta12] basis 1,z,z^2,z^3)
Usage: python3 bin2npz.py <dir> <out.npz>"""
import sys, json, numpy as np
d,out=sys.argv[1],sys.argv[2]; meta=json.load(open(f'{d}/meta.json'))
rd=lambda f,t:np.fromfile(f'{d}/{f}',dtype=t)
fr,to,c,w,na,ar=rd('fr.i4','<i4'),rd('to.i4','<i4'),rd('c.u8','<u8'),rd('w.i4','<i4'),rd('na.i4','<i4'),rd('ar.f8','<f8')
off=rd('off.i4','<i4').reshape(-1,4).astype(np.int64)
z=np.exp(1j*np.pi/6); Mx=np.array([z**k for k in range(4)]); Ms=np.array([z**(5*k) for k in range(4)])
a=np.array(meta['P_int4']); Pc=a@Mx; Ps=a@Ms
dphys=off@Mx; dperp=off@Ms; lat=(dphys*np.conj(Pc/abs(Pc))).real
q=dperp-(lat/abs(Pc))*Ps
np.savez_compressed(out,fr=fr.astype(np.int64),to=to.astype(np.int64),c=c.astype(np.int64),w=w.astype(np.int64),
                    ar=ar,q=q,na=na.astype(np.int64),n=meta['n_states'],P=Pc,Ps=Ps)
print('wrote',out,meta)
