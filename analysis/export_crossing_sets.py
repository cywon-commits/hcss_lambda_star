"""Export the crossing-set dictionaries (pickle) to portable JSON.gz.
   python3 export_crossing_sets.py                      # vmodes3.pkl -> data/dddt_crossing_sets.json.gz, vmodes_fast_DDT.pkl -> data/ddt_crossing_sets.json.gz
Entry: {"tiles": [[kind, [[a0,a1,a2,a3], ...]], ...], "n": <exact integer filling count>}
  kind 'A' = triangle, 'R' = 30-degree rhombus; vertices are integer 4-vectors in Z[zeta12] (basis 1,z,z^2,z^3; z^4=z^2-1),
  vertices sorted lexicographically, tiles sorted by (kind, vertices), entries sorted by (number of tiles, tiles).
  "n" is written as a JSON integer (Python ints are arbitrary precision; values here are < 2^63) -- readers must not use floats.
File also carries the cluster definition: junction points O, X, Y, the skeleton segments and the three/four cells."""
import sys, gzip, json, pickle
def dump(pkl, out, name):
    R, SEG, *cells = pickle.load(open(pkl, 'rb'))
    ent = []
    for S, n in R.items():
        tiles = sorted([t[0], sorted(list(v) for v in t[1])] for t in S)
        ent.append({'tiles': tiles, 'n': int(n)})
    ent.sort(key=lambda e: (len(e['tiles']), json.dumps(e['tiles'])))
    meta = {'cluster': name, 'total_fillings': sum(e['n'] for e in ent), 'n_sets': len(ent),
            'segments': {k: [list(a), list(b)] for k, (a, b) in SEG.items()}, 'cells': [[list(p) for p in c] for c in cells],
            'source': pkl}
    with gzip.open(out, 'wt', encoding='utf-8', compresslevel=9) as f: json.dump({'meta': meta, 'sets': ent}, f, separators=(',', ':'))
    print(out, name, 'sets', len(ent), 'total', meta['total_fillings'])
if __name__ == '__main__':
    dump('vmodes3.pkl', 'data/dddt_crossing_sets.json.gz', 'DDDT')
    dump('vmodes_fast_DDT.pkl', 'data/ddt_crossing_sets.json.gz', 'DDT')
