"""Load the committed crossing-set JSON files.
   from load_crossing_sets import load; R, meta = load('DDDT')     # R: {frozenset({(kind, frozenset(vertices))}): n}, same layout as vmodes3.pkl[0]
   python3 load_crossing_sets.py            # check: JSON round-trips exactly to the pickles (if present) and to the known totals"""
import gzip, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
FILES = {'DDDT': 'data/dddt_crossing_sets.json.gz', 'DDT': 'data/ddt_crossing_sets.json.gz'}
def load(which='DDDT'):
    with gzip.open(os.path.join(HERE, FILES[which]), 'rt', encoding='utf-8') as f: D = json.load(f)
    R = {}
    for e in D['sets']:
        S = frozenset((k, frozenset(tuple(v) for v in vs)) for k, vs in e['tiles'])
        assert S not in R and isinstance(e['n'], int); R[S] = e['n']
    assert len(R) == D['meta']['n_sets'] and sum(R.values()) == D['meta']['total_fillings']
    return R, D['meta']
if __name__ == '__main__':
    import pickle
    TOT = {'DDDT': (749563960730, 948), 'DDT': (61174241, 29)}
    ok = True
    for w, pk in (('DDDT', 'vmodes3.pkl'), ('DDT', 'vmodes_fast_DDT.pkl')):
        R, meta = load(w); good = (sum(R.values()), len(R)) == TOT[w]; ok &= good
        print(w, 'sets', len(R), 'total', sum(R.values()), 'matches known totals:', good)
        p = os.path.join(HERE, pk)
        if os.path.exists(p):
            P = pickle.load(open(p, 'rb'))[0]; same = (P == R); ok &= same
            print('  identical to', pk, ':', same, '| keys', set(P) == set(R), '| all counts equal:', all(P[k] == R[k] for k in P) if set(P) == set(R) else False)
        else: print('  (%s not present; skipped pickle comparison)' % pk)
    print('OK' if ok else 'MISMATCH'); sys.exit(0 if ok else 1)
