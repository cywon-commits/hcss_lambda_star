"""Compare a tm80 npz with a reference npz (reference80/cyl_ref.py or reference/data).  Exact: n, fr, to, c, w, na (and cls, off for z18); 1e-9: ar, q (q5, q7).
Usage: python3 compare80.py reference.npz candidate.npz"""
import sys, numpy as np
A = np.load(sys.argv[1]); B = np.load(sys.argv[2]); ok = True
ok &= int(A['n']) == int(B['n']); print('n_states', int(A['n']), int(B['n']))
exact = ['fr', 'to', 'c', 'w', 'na'] + [k for k in ('cls', 'off') if k in A.files and k in B.files]
close = [k for k in ('ar', 'q', 'q5', 'q7') if k in A.files]
for k in exact:
    e = A[k].shape == B[k].shape and np.array_equal(A[k], B[k]); ok &= e; print(f'{k:4s} exact:', e)
for k in close:
    e = A[k].shape == B[k].shape and np.allclose(A[k], B[k], atol=1e-9, rtol=0); ok &= e; print(f'{k:4s} 1e-9 :', e)
print('PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
