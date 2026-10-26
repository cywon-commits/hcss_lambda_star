"""Region-count identity between the square-triangle tiling (cfg st12) and its family-rotated image (cfg z18, 80-degree rhombi).
Direction map  f(k) = floor(3k/2) (theta = -10 deg, B family rotated):  even z12 directions k -> 3k/2, odd k -> (3k-1)/2.
  HCSS_CFG=st12 python3 st12_regions.py gen out.json [nloops seed]    random patches (3-14 tiles) + dodecagon, their filling counts
  HCSS_CFG=z18  python3 st12_regions.py map out.json                  counts of the mapped loops in the 80-degree system and class vectors"""
import sys, json, random
import cfg as C, front
def fmap(k): return (3 * k) // 2
def gen(out, nloops=60, seed=1):
    import patches
    rng = random.Random(seed); loops = [list(range(12))]
    tries = 0
    while len(loops) < nloops and tries < 4000:
        tries += 1; k = rng.randint(3, 14); r = patches.random_patch(k, rng)
        if r is None: continue
        d = patches.boundary_loop(*r)
        if d is None: continue
        loops.append(d)
    res = [dict(loop=d, count=front.fill_count(tuple(d))[0], w=front.fill_count(tuple(d))[1]) for d in loops]
    json.dump(res, open(out, 'w')); print(len(res), 'loops; max count', max(r['count'] for r in res))
def mapped(out):
    res = json.load(open(out)); bad = 0; viol = 0; third = 0
    for r in res:
        d = [fmap(k) for k in r['loop']]; c, w, v = front.fill_info(tuple(d)); r['count18'] = c; r['w18'] = w; r['cls18'] = list(v)
        if c != r['count'] or abs(w - r['w']) > 1e-9: bad += 1
        tf = v[1] if len(v) else 0
        # third family: triangle family 1 (z18 class 1) and rhombus labels other than the single used one
        third += int(v[1] > 0)
    json.dump(res, open(out, 'w')); print(len(res), 'loops; count/weight mismatches:', bad, '; loops with family-1 triangles:', third, '; class violations:', len(front.VIOLATIONS))
if __name__ == '__main__':
    if sys.argv[1] == 'gen': gen(sys.argv[2], *(int(x) for x in sys.argv[3:5]))
    else: mapped(sys.argv[2])
