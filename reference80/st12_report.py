import json, sys
rows = []
for a in sys.argv[1:]:
    n = a.replace(' ', '_').replace('-', 'm'); A = json.load(open('../runs/cmp_%s_z18.json' % n)); B = json.load(open('../runs/cmp_%s_st12.json' % n))
    sa, sb = A['summary'], B['summary']; ba, bb = sa['branching'][0], sb['branching'][0]
    wa = {tuple(w) for w in A['words'][0]}; wb = {tuple(w) for w in B['words'][0]}
    rows.append(dict(a=sa['a'], W80=sa['W'], W90=sb['W'], states=(sa['states'], sb['states']), trans=(sa['transitions'], sb['transitions']), scc=(ba['size'], bb['size']), scc_trans=(ba['ntrans'], bb['ntrans']),
                     s=(ba['s'], bb['s']), ds=abs(ba['s'] - bb['s']), nT=(ba['nT_per_vertex'], bb['nT_per_vertex']), rho0=(ba['rho0'], bb['rho0']), drho=abs(ba['rho0'] - bb['rho0']), common=len(wa & wb), only80=len(wa - wb), only90=len(wb - wa), nbr=(len(sa['branching']), len(sb['branching']))))
    print(rows[-1])
json.dump(rows, open('../runs/st12_cyl_compare.json', 'w'), indent=1)
