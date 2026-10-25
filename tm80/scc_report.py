"""Strongly connected components of a transfer-matrix npz: size, internal edges, whether the component has branching (entropy > 0).
Usage: python3 tm80/scc_report.py file.npz"""
import sys, numpy as np, scipy.sparse as sp, scipy.sparse.csgraph as cg
z = np.load(sys.argv[1]); n = int(z['n']); fr, to = z['fr'], z['to']; c = z['c']
G = sp.csr_matrix((np.ones(len(fr)), (fr, to)), shape=(n, n)); nc, lab = cg.connected_components(G, directed=True, connection='strong')
cnt = np.bincount(lab, minlength=nc); inside = lab[fr] == lab[to]
ein = np.bincount(lab[fr][inside], minlength=nc); cin = np.bincount(lab[fr][inside], weights=c[inside], minlength=nc)
print('states', n, 'transitions', len(fr), 'SCCs', nc, ' c max', c.max())
rows = [(int(cnt[k]), int(ein[k]), float(cin[k])) for k in range(nc) if ein[k] > 0]
rows.sort(reverse=True)
print('non-trivial SCCs (size, internal edges, sum of multiplicities):', len(rows)); print(rows[:12])
print('SCCs with branching (internal edges > size):', sum(1 for r in rows if r[1] > r[0] or r[2] > r[0]))
