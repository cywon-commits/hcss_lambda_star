"""TASK2b: redefine DDDT cooperative modes by connected components of crossing tiles. Reads data/*.json.gz only.
   python3 task2b.py   (log: runs/task2b.log; figures: figures/dddt_coop_cs.png, figures/dddt_ext_single.png)"""
from dddt_common import *
import itertools, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
FIG = os.path.join(HERE, '..', 'figures')
_seg = {}
def tsegs(t):
    if t not in _seg: _seg[t] = segs_of(t)
    return _seg[t]
def fully(S):      # junctions at which all three skeleton segments are crossed by tiles of S
    cr = set().union(*[tsegs(t) for t in S]) if S else set()
    return [j for j in 'XYO' if set(JUNC[j]) <= cr]
def vac_whole(S):  # results02 definition: junction that is a vertex of no crossing tile of S
    return [j for j in 'XYO' if not any(JPT[j] in t[1] for t in S)]
def vac_geo(S):    # junction lying in the union of the tiles but not a vertex of any of them
    if not S: return []
    U = unary_union([tpoly(t) for t in S]).buffer(1e-7)
    return [j for j in 'XYO' if U.contains(Point(cplx(JPT[j]).real, cplx(JPT[j]).imag)) and not any(JPT[j] in t[1] for t in S)]
def components(S):
    T = list(S); par = list(range(len(T)))
    def f(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    for i, j in itertools.combinations(range(len(T)), 2):
        if T[i][1] & T[j][1]: par[f(i)] = f(j)
    g = defaultdict(list)
    for i in range(len(T)): g[f(i)].append(T[i])
    return [frozenset(v) for v in g.values()]
log = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); log.append(s)
P('DDDT total %d sets %d; independent (primitive unions) %d ; cooperative (class d) %d' % (tot3, len(R3), sum(1 for S in R3 if S in ICL), len(COOP)))
coopn = sum(R3[S] for S in COOP); P('cooperative fillings %d (%.4f%%)' % (coopn, 100*coopn/tot3))
# ---------------- 1. weight tables ----------------
for name, vf in (('vacated = vertex of no crossing tile (results02 definition)', vac_whole), ('vacated = junction covered by the tiles but not their vertex (geometric)', vac_geo)):
    tab = defaultdict(lambda: [0, 0])
    for S in COOP:
        k = (len(vf(S)), len(fully(S))); tab[k][0] += 1; tab[k][1] += R3[S]
    P('\n[1] cooperative sets by (#vacated junctions, #fully-crossed junctions): count / fillings ; %s' % name)
    P('%-9s | %s | %s' % ('vacated', ' | '.join('fully=%d %18s' % (f, '') for f in (1, 2, 3)), 'row total'))
    for v in range(4):
        cells = []; rc = rn = 0
        for f in (1, 2, 3):
            c, n = tab.get((v, f), [0, 0]); cells.append('%4d %18d' % (c, n)); rc += c; rn += n
        P('%-9d | %s | %4d %18d' % (v, ' | '.join(cells), rc, rn))
    for f in (1, 2, 3):
        c = sum(tab.get((v, f), [0, 0])[0] for v in range(4)); n = sum(tab.get((v, f), [0, 0])[1] for v in range(4)); P('   fully=%d: %d sets, %d fillings' % (f, c, n))
    for v in range(4):
        c = sum(tab.get((v, f), [0, 0])[0] for f in (1, 2, 3)); n = sum(tab.get((v, f), [0, 0])[1] for f in (1, 2, 3)); P('   vacated=%d: %d sets, %d fillings' % (v, c, n))
# ---------------- 2. connected components ----------------
CK = {}  # component -> (kind, vac, segs, comp string)
def ckind(C):
    vac = vac_geo(C); segs = frozenset().union(*[tsegs(t) for t in C]); comp = ''.join(sorted(t[0] for t in C))
    if len(vac) >= 2: k = 'K3 cooperative (>=2 vacated)'
    elif len(vac) == 1:
        k = 'K1 vertex (one junction, own 3 segments)' if segs <= set(JUNC[vac[0]]) else 'K2 extended single junction'
    else: k = 'K0 no vacated junction'
    return k, tuple(vac), tuple(sorted(segs)), comp
setcomps = {}
for S in R3:
    cs = components(S); setcomps[S] = cs
    for C in cs:
        if C not in CK: CK[C] = ckind(C)
P('\n[2] distinct connected components over all %d crossing sets: %d' % (len(R3), len(CK)))
orb = defaultdict(list)
for C in CK: orb[orbit_key(C)].append(C)
P('    C3v orbits of components: %d' % len(orb))
share = lambda pred: sum(n for S, n in R3.items() if any(pred(CK[C]) for C in setcomps[S]))
rows = []
for ok, mem in orb.items():
    C0 = mem[0]; k, vac, segs, comp = CK[C0]; stab = {1: 'C1', 2: 'Cs', 3: 'C3', 6: 'C3v'}[6//len(mem)]
    sets = [S for S in R3 if any(C in set(mem) for C in setcomps[S])]; nsum = sum(R3[S] for S in sets)
    # alone-weight: fillings of the set consisting of this component only (if realised)
    alone = R3.get(frozenset(C0), 0)
    rows.append((k, vac, segs, comp, len(C0), len(mem), stab, len(sets), nsum, alone, C0))
rows.sort(key=lambda r: (r[0], -r[8]))
P('\n    kind | vacated | crossed segments | tiles | #tiles | members | stab | #sets containing | fillings of those sets (overlapping) | share | alone')
for r in rows: P('    %-40s %-6s %-28s %-8s %2d %3d %-3s %4d %16d %7.3f%% %14d' % (r[0], ','.join(r[1]) or '-', '+'.join(r[2]), r[3], r[4], r[5], r[6], r[7], r[8], 100*r[8]/tot3, r[9]))
P('\n    by kind: #orbits, #components, #sets containing (kind), share of all fillings (sets containing at least one component of the kind)')
for kind in sorted(set(r[0] for r in rows)):
    rr = [r for r in rows if r[0] == kind]; sets = [S for S in R3 if any(CK[C][0] == kind for C in setcomps[S])]
    P('    %-40s orbits %3d components %4d sets %4d fillings %16d  %7.3f%%' % (kind, len(rr), sum(r[5] for r in rr), len(sets), sum(R3[S] for S in sets), 100*sum(R3[S] for S in sets)/tot3))
# set-level partition by the "highest" component kind present
order = ['K3 cooperative (>=2 vacated)', 'K2 extended single junction', 'K1 vertex (one junction, own 3 segments)', 'K0 no vacated junction']
cat = Counter(); catn = Counter()
for S, n in R3.items():
    ks = set(CK[C][0] for C in setcomps[S]); c = next((o for o in order if o in ks), 'none (unbound)'); cat[c] += 1; catn[c] += n
P('\n    partition of the 948 sets by highest component kind present: sets / fillings / share')
for c in order+['none (unbound)']: P('    %-40s %4d %16d %8.3f%%' % (c, cat[c], catn[c], 100*catn[c]/tot3))
# true cooperative vs old cooperative
newcoop = [S for S in R3 if any(CK[C][0].startswith('K3') for C in setcomps[S])]
P('    sets with a true cooperative component (vacated>=2): %d, fillings %d (%.4f%%) ; old class d: %d sets, %d fillings (%.4f%%)' % (len(newcoop), sum(R3[S] for S in newcoop), 100*sum(R3[S] for S in newcoop)/tot3, len(COOP), coopn, 100*coopn/tot3))
oldnot = [S for S in COOP if S not in set(newcoop)]
P('    old-cooperative sets WITHOUT a true cooperative component (reclassified): %d sets, %d fillings (%.4f%%)' % (len(oldnot), sum(R3[S] for S in oldnot), 100*sum(R3[S] for S in oldnot)/tot3))
newnotold = [S for S in newcoop if S in ICL]; P('    sets with a K3 component that the old scheme called independent: %d' % len(newnotold))
# extended single junction
ext = [r for r in rows if r[0].startswith('K2')]
P('\n    extended single-junction orbits: %d, members %d' % (len(ext), sum(r[5] for r in ext)))
for r in ext: P('      orbit: vacated %s crossed %s tiles %s (%d) stab %s members %d, sets %d, fillings %d (%.3f%%)' % (','.join(r[1]), '+'.join(r[2]), r[3], r[4], r[6], r[5], r[7], r[8], 100*r[8]/tot3))
# ---------------- 3. independence of observed components ----------------
comps = list(CK); N = len(comps)
verts = [frozenset(v for t in C for v in t[1]) for C in comps]; pl = [unary_union([tpoly(t) for t in C]) for C in comps]
ok = [[i != j and not (verts[i] & verts[j]) and pl[i].intersection(pl[j]).area < 1e-7 for j in range(N)] for i in range(N)]
P('\n[3] independence: %d distinct components; compatible (vertex-disjoint, non-overlapping) pairs: %d' % (N, sum(map(sum, ok))//2))
found = 0; missing = []; nsub = 0; bad = []
def dfs(chosen, cand):
    global found, nsub
    if chosen:
        nsub += 1; U = frozenset().union(*[comps[i] for i in chosen])
        if U in R3:
            found += 1
            # consistency: components of the realised set are exactly the chosen ones
            if set(setcomps[U]) != set(comps[i] for i in chosen): bad.append(U)
        else: missing.append(chosen[:])
    for k, i in enumerate(cand): dfs(chosen+[i], [j for j in cand[k+1:] if ok[i][j]])
dfs([], list(range(N)))
P('    pairwise-compatible component families: %d ; realised in the enumeration: %d ; not realised: %d ; decomposition mismatches: %d' % (nsub, found, len(missing), len(bad)))
excl = Counter(); holes = 0
for ch in missing:
    U = frozenset().union(*[comps[i] for i in ch])
    try: w = weight(U)
    except RuntimeError: w = 'hole'
    if w == 'hole': holes += 1; excl[(len(ch), 'hole')] += 1
    elif w == 0: excl[(len(ch), 'weight0')] += 1
    else: excl[(len(ch), 'NONZERO weight %d' % w)] += 1
P('    not-realised families by (size, reason): %s' % dict(excl))
sizes = Counter(len(c) for c in missing); P('    not-realised by family size: %s' % dict(sizes))
for ch in missing[:12]:
    P('      missing: ' + ' + '.join('[%s %s %s]' % (CK[comps[i]][0][:2], ','.join(CK[comps[i]][1]) or '-', '+'.join(CK[comps[i]][2])) for i in ch))
# ---------------- 4. figures ----------------
def draw(ax, S, title):
    for Pp in (DaI, DbI, DcI, TI):
        C = [cplx(p) for p in Pp]+[cplx(Pp[0])]; ax.plot([z.real for z in C], [z.imag for z in C], c='0.8', lw=0.6)
    for t in S:
        Z = ordered(t[1]); ax.fill([z[0] for z in Z], [z[1] for z in Z], fc='#f2b134' if t[0] == 'A' else '#4a7fb5', ec='k', lw=0.5, alpha=0.9)
    for k, (a, b) in SEG3.items(): ax.plot([cplx(a).real, cplx(b).real], [cplx(a).imag, cplx(b).imag], 'r-', lw=1.2)
    for nm, v in JPT.items():
        z = cplx(v); has = any(v in t[1] for t in S); ax.plot(z.real, z.imag, 'ko' if has else 'kx', ms=4)
    ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title, fontsize=7)
def sheet(items, fn, title, ncol=3):
    nr = (len(items)+ncol-1)//ncol; fig, axs = plt.subplots(nr, ncol, figsize=(4.2*ncol, 4.0*nr+0.6), squeeze=False)
    for ax in axs.ravel(): ax.axis('off')
    for ax, (S, t) in zip(axs.ravel(), items): draw(ax, S, t)
    plt.suptitle(title, fontsize=10); plt.tight_layout(rect=(0, 0, 1, 0.97)); plt.savefig(fn, dpi=100); plt.close()
cs = []; seen = set()
for S in sorted(COOP, key=lambda S: (-R3[S], len(S))):
    k = orbit_key(S)
    if k in seen: continue
    seen.add(k); orbS = {tmap(f, S) for f in GROUP.values()}
    if len(orbS) == 3: cs.append((S, orbS))
P('\n[4] cooperative orbits with stabiliser Cs: %d' % len(cs))
items = []
for S, orbS in cs:
    syms = [g for g, f in GROUP.items() if g.startswith('s_') and tmap(f, S) == S]
    cr = sorted(set().union(*[tsegs(t) for t in S])); ncomp = len(components(S))
    P('    Cs orbit: %d tiles %s, mirror %s, vacated(whole) %s, vacated(geo) %s, crossed %s, components %d, fillings per member %d (orbit total %d)' % (len(S), ''.join(sorted(t[0] for t in S)), ','.join(syms), ','.join(vac_whole(S)), ','.join(vac_geo(S)) or '-', '+'.join(cr), ncomp, R3[S], 3*R3[S]))
    items.append((S, '%d tiles %s  mirror %s  vacated(geo) %s\n%d components, %.4g fillings x3' % (len(S), ''.join(sorted(t[0] for t in S)), ','.join(syms), ','.join(vac_geo(S)) or '-', ncomp, R3[S])))
sheet(items, os.path.join(FIG, 'dddt_coop_cs.png'), 'DDDT cooperative modes with a mirror (stabiliser Cs): one member per orbit (red skeleton, o vertex of a tile, x not)')
sheet([(r[10], '%s %s\n%s members %d, %.3g fillings' % (','.join(r[1]), r[3], '+'.join(r[2]), r[5], r[8])) for r in ext][:12] or [(frozenset(), 'none')], os.path.join(FIG, 'dddt_ext_single.png'), 'Extended single-junction components (one vacated junction + segment beyond its own three): one per orbit')
open(os.path.join(HERE, '..', 'runs', 'task2b.log'), 'w').write('\n'.join(log)+'\n')
