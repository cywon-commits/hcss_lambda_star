"""For each sampled tiling: (a) number of edge families used (edge direction k mod 3), (b) number of rhombus Z3 labels used, (c) number of vortices
((1,0,3) and (2,3,0) vertices).  Usage: count_families.py out.json sample1.npz:a0,..,a5 ..."""
import sys, json, math
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from analyze_sample import stats
z = np.exp(2j * np.pi / 18)
def count(f, a):
    S = np.load(f); V = S['V'].astype(int); names = S['names']; cls = S['cls']; fam = np.zeros(3, int); dirs = np.zeros(18, int)
    for t in range(len(names)):
        nv = 3 if names[t] == 'T' else 4; c = V[t][:nv] @ z ** np.arange(6)
        for i in range(nv):
            e = c[(i + 1) % nv] - c[i]; k = int(round(math.degrees(math.atan2(e.imag, e.real)) / 20)) % 18; dirs[k] += 1; fam[k % 3] += 1
    rh = [(int(cls[t]) - 3 + 2) % 3 for t in range(len(names)) if names[t] != 'T']; lab = np.bincount(rh, minlength=3)
    st = stats(f, a)
    return dict(file=f, a=a, tiles=len(names), families_used=int((fam > 0).sum()), edge_counts=fam.tolist(), labels_used=int((lab > 0).sum()), label_counts=lab.tolist(),
                vortices=st['vort_pos'] + st['vort_neg'], vertices=st['vertices'])
if __name__ == '__main__':
    out = []
    for arg in sys.argv[2:]:
        f, a = arg.split(':'); out.append(count(f, [int(x) for x in a.split(',')])); r = out[-1]
        print(r['file'], 'families', r['families_used'], r['edge_counts'], 'labels', r['labels_used'], r['label_counts'], 'vortices', r['vortices'], flush=True)
    json.dump(out, open(sys.argv[1], 'w'), indent=1)
