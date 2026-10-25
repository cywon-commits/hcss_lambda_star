"""Analyse one z18 cylinder: build with tm80, SCC structure, entropy of the branching SCCs (tilt.solve_sccs), composition, area densities, strain.
analyze(a) -> dict.   Used by target.py / phases.py.   Set OMP_NUM_THREADS=1 when running many in parallel."""
import sys, os, json, shutil, tempfile, subprocess, math
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); sys.path.insert(0, os.path.join(ROOT, 'reference80')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
z = np.exp(2j * np.pi / 18)
def analyze(a, keep=None, max_states=3_000_000, timeout=3600):
    import tilt, bin2npz_mod
    a = tuple(int(x) for x in a); d = tempfile.mkdtemp(prefix='an_', dir=os.path.join(ROOT, 'runs'))
    try:
        r = subprocess.run([os.path.join(ROOT, 'tm80', 'tm80'), 'build', 'z18'] + [str(x) for x in a] + [d], capture_output=True, text=True, timeout=timeout, env=dict(os.environ, OMP_NUM_THREADS='1'))
        if r.returncode: return dict(a=a, status='build-fail')
        meta = json.load(open(d + '/meta.json'))
        if meta['n_states'] > max_states: return dict(a=a, status='too-large', states=meta['n_states'])
        f = d + '.npz'; bin2npz_mod.convert(d, f); D = tilt.load(f); P = complex(D['P'])
        R = tilt.solve_sccs(D, maxscc=40)
        out = dict(a=a, status='ok', W=abs(P), arg=math.degrees(math.atan2(P.imag, P.real)), states=D['n'], transitions=len(D['fr']), frozen=R['frozen'], n_branching=R['n_branching'],
                   s5W=abs(sum(a[i] * z ** (5 * i) for i in range(6))) / abs(P), s7W=abs(sum(a[i] * z ** (7 * i) for i in range(6))) / abs(P))
        if R['results']:
            b = R['results'][0]
            out.update(s=float(b['s']), size=int(b['size']), Nt_Nr=float(b['Nt_over_Nr']), area5=float(b['area5']), area7=float(b['area7']), nv=float(b['nv']),
                       cls=[float(x) for x in b['cls']], al5=[b['alphas'][0].real, b['alphas'][0].imag], be5=[b['betas'][0].real, b['betas'][0].imag],
                       al7=[b['alphas'][1].real, b['alphas'][1].imag], be7=[b['betas'][1].real, b['betas'][1].imag], all_s=[float(x['s']) for x in R['results']], all_size=[int(x['size']) for x in R['results']])
        if keep: shutil.copytree(d, keep, dirs_exist_ok=True); shutil.copy(f, keep + '.npz')
        if os.path.exists(f): os.remove(f)
        return out
    finally: shutil.rmtree(d, ignore_errors=True)
if __name__ == '__main__':
    print(json.dumps(analyze([int(x) for x in sys.argv[1:7]]), indent=1))
