"""Entropy at phi = 0 (unconstrained maximum of each cylinder) for the cylinders listed in survey CSVs (branching ones only).
Usage: python3 tm80/landscape.py out.csv survey1.csv [survey2.csv ...]"""
import sys, os, csv, json, shutil, tempfile, subprocess
import numpy as np
from multiprocessing import Pool
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); sys.path.insert(0, os.path.join(ROOT, 'reference80')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
def one(a):
    import tilt, bin2npz_mod
    d = tempfile.mkdtemp(prefix='ls_', dir=os.path.join(ROOT, 'runs'))
    try:
        r = subprocess.run([os.path.join(ROOT, 'tm80', 'tm80'), 'build', 'z18'] + [str(x) for x in a] + [d], capture_output=True, text=True, timeout=1800, env=dict(os.environ, OMP_NUM_THREADS='1'))
        if r.returncode: return a, None
        f = d + '.npz'; bin2npz_mod.convert(d, f)
        D = tilt.load(f); R = tilt.solve_sccs(D)
        if not R['results']: os.remove(f); return a, None
        res = R['results'][0]
        out = dict(a=a, n=D['n'], size=res['size'], n_branching=R['n_branching'], frozen=R['frozen'], s=res['s'], nv=res['nv'], Nt_Nr=res['Nt_over_Nr'], area5=res['area5'], area7=res['area7'],
                   cls=[float(x) for x in res['cls']], al5=complex(res['alphas'][0]), be5=complex(res['betas'][0]), al7=complex(res['alphas'][1]), be7=complex(res['betas'][1]), P=complex(D['P']),
                   all_s=[float(r['s']) for r in R['results']])
        os.remove(f); return a, out
    except Exception as e:
        return a, dict(err=str(e))
    finally: shutil.rmtree(d, ignore_errors=True)
if __name__ == '__main__':
    out = sys.argv[1]; C = []
    for fn in sys.argv[2:]:
        for r in csv.DictReader(open(fn)):
            if r['status'] == 'ok' and int(r['branching_scc']) > 0: C.append(tuple(int(x) for x in r['a'].split()))
    C = sorted(set(C)); print(len(C), 'cylinders', flush=True); rows = []
    with Pool(4) as p:
        for a, o in p.imap_unordered(one, C, chunksize=1):
            if o is None or 'err' in o: print('skip', a, o); continue
            rows.append(o)
    json.dump([{k: ([v.real, v.imag] if isinstance(v, complex) else (list(v) if isinstance(v, tuple) else v)) for k, v in r.items()} for r in rows], open(out, 'w'))
    print('done', len(rows))
