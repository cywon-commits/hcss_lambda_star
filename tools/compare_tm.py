"""Compare a transfer-matrix file (npz) against the Python reference.
   Exact equality required for n, fr, to, c, w, na ; ar and q to 1e-9.
   Usage: python3 compare_tm.py reference.npz candidate.npz"""
import sys, numpy as np
A=np.load(sys.argv[1]); B=np.load(sys.argv[2]); ok=True
ok&=int(A['n'])==int(B['n']); print('n_states', int(A['n']), int(B['n']))
for k in ('fr','to','c','w','na'):
    e=A[k].shape==B[k].shape and np.array_equal(A[k],B[k]); ok&=e; print(f'{k:3s} exact:',e)
for k in ('ar','q'):
    e=A[k].shape==B[k].shape and np.allclose(A[k],B[k],atol=1e-9,rtol=0); ok&=e; print(f'{k:3s} 1e-9 :',e)
print('PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
