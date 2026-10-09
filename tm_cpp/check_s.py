"""Phase-A check 3: s at zero tilt from the C++ npz vs expected.json (1e-9)."""
import sys, json, numpy as np
sys.path.insert(0, 'reference/python')
from tilt import load, solve
exp = json.load(open('reference/expected.json'))['cylinders']
ok = True
for key, v in exp.items():
    n = '_'.join(map(str, v['P_int4']))
    r = solve(load(f'runs/cpp_{n}.npz'), 0)
    tol = 1e-9 if len(str(v['s_phi0'])) > 10 else 5e-7   # last reference entry is stored with 6 digits
    good = abs(r['s'] - v['s_phi0']) < tol; ok &= good
    print(key, 's=%.12f expected=%.12f diff=%.1e %s' % (r['s'], v['s_phi0'], r['s'] - v['s_phi0'], 'OK' if good else 'FAIL'))
print('ALL OK' if ok else 'FAIL'); sys.exit(not ok)
