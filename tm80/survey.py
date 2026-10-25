"""Survey of cylinders: for many periods P (arg in [0,20) deg, by symmetry) build the z18 transfer graph with tm80 and report whether it has
branching strongly connected components (entropy > 0).  Usage: python3 tm80/survey.py out.csv Wmin Wmax maxsteps [nproc]"""
import sys, os, itertools, math, subprocess, json, tempfile, shutil
import numpy as np, scipy.sparse as sp, scipy.sparse.csgraph as cg
from multiprocessing import Pool
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
z = np.exp(2j * np.pi / 18)
def candidates(Wmin, Wmax, smax, rmax=None):
    R = 6; r = np.arange(-R, R + 1); out = []
    A = np.array(np.meshgrid(*([r] * 6), indexing='ij')).reshape(6, -1).T.astype(np.int16)
    zp = z ** np.arange(6)
    for i in range(0, len(A), 2_000_000):
        a = A[i:i + 2_000_000].astype(np.int32); s = np.abs(a).sum(1); P = a @ zp; ap = np.abs(P)
        m = (s >= 3) & (s <= smax) & (ap >= Wmin) & (ap <= Wmax); a = a[m]; P = P[m]
        ang = np.degrees(np.angle(P)) % 360; m2 = (ang >= -1e-9) & (ang < 20 - 1e-9); a = a[m2]
        from math import gcd
        for row in a:
            g = 0
            for x in row: g = gcd(g, int(abs(x)))
            if g == 1:                                                  # primitive vectors only
                t = tuple(int(x) for x in row)
                if rmax is not None:
                    Pc = sum(t[i] * z ** i for i in range(6)); s5 = abs(sum(t[i] * z ** (5 * i) for i in range(6))); s7 = abs(sum(t[i] * z ** (7 * i) for i in range(6)))
                    if max(s5, s7) / abs(Pc) > rmax: continue
                out.append(t)
    return out
def one(a):
    d = tempfile.mkdtemp(prefix='sv_', dir=os.path.join(ROOT, 'runs'))
    try:
        r = subprocess.run([os.path.join(ROOT, 'tm80', 'tm80'), 'build', 'z18'] + [str(x) for x in a] + [d], capture_output=True, text=True, timeout=900, env=dict(os.environ, OMP_NUM_THREADS='1'))
        if r.returncode != 0: return (a, 'fail', 0, 0, 0, 0, 0)
        n = json.load(open(d + '/meta.json'))['n_states']
        fr = np.fromfile(d + '/fr.i4', '<i4'); to = np.fromfile(d + '/to.i4', '<i4'); c = np.fromfile(d + '/c.u8', '<u8').astype(float)
        G = sp.csr_matrix((np.ones(len(fr)), (fr, to)), shape=(n, n)); nc, lab = cg.connected_components(G, directed=True, connection='strong')
        ins = lab[fr] == lab[to]; cnt = np.bincount(lab, minlength=nc); ein = np.bincount(lab[fr][ins], weights=c[ins], minlength=nc)
        nontriv = ein > 0; branch = nontriv & (ein > cnt)
        return (a, 'ok', n, len(fr), int(nontriv.sum()), int(branch.sum()), int(cnt[branch].max()) if branch.any() else 0)
    finally: shutil.rmtree(d, ignore_errors=True)
if __name__ == '__main__':
    out, Wmin, Wmax, smax = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]); npr = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    rmax = float(sys.argv[6]) if len(sys.argv) > 6 else None
    C = candidates(Wmin, Wmax, smax, rmax); print(len(C), 'candidates', flush=True)
    with Pool(npr) as p, open(out, 'w') as f:
        f.write('a,status,states,transitions,nontrivial_scc,branching_scc,largest_branching_size,W,s5_over_W,s7_over_W\n')
        for res in p.imap_unordered(one, C, chunksize=1):
            a = res[0]; P = sum(a[i] * z ** i for i in range(6)); s5 = abs(sum(a[i] * z ** (5 * i) for i in range(6))); s7 = abs(sum(a[i] * z ** (7 * i) for i in range(6)))
            f.write('"%s",%s,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f\n' % (' '.join(map(str, a)), res[1], *res[2:], abs(P), s5 / abs(P), s7 / abs(P))); f.flush()
