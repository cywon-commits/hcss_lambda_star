"""Run analyze_cyl.analyze over a targeted candidate list.  Usage: run_targets.py cands.json out.jsonl nproc per_phase Wmax"""
import sys, os, json, collections
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
import analyze_cyl
def job(t):
    a, key, err = t; r = analyze_cyl.analyze(a); r['phase'] = key; r['err'] = err; return r
if __name__ == '__main__':
    c, out, npr, per, Wmax = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
    by = collections.defaultdict(list)
    for a, k, e, W in json.load(open(c)):
        if W <= Wmax: by[k].append((e, tuple(a)))
    jobs = [(a, k, e) for k, v in by.items() for e, a in sorted(v)[:per]]
    done = set()
    if os.path.exists(out): done = {tuple(json.loads(l)['a']) for l in open(out)}
    jobs = [j for j in jobs if j[0] not in done]; print(len(jobs), 'jobs', flush=True)
    with Pool(npr) as p, open(out, 'a') as f:
        for r in p.imap_unordered(job, jobs, chunksize=1): f.write(json.dumps(r) + '\n'); f.flush()
