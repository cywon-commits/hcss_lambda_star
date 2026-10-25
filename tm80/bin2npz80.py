"""Convert the raw directory written by tm80 into the npz layout of reference80/cyl_ref.py (and tools/bin2npz.py for lam12).
Usage: python3 bin2npz80.py <dir> <out.npz>"""
import sys, json, numpy as np
d, out = sys.argv[1], sys.argv[2]; meta = json.load(open(f'{d}/meta.json')); rk = meta['rank']
rd = lambda f, t: np.fromfile(f'{d}/{f}', dtype=t)
kw = dict(fr=rd('fr.i4', '<i4').astype(np.int64), to=rd('to.i4', '<i4').astype(np.int64), c=rd('c.u8', '<u8').astype(np.int64), w=rd('w.i4', '<i4').astype(np.int64),
          na=rd('na.i4', '<i4').astype(np.int64), ar=rd('ar.f8', '<f8'), n=meta['n_states'], P=complex(*meta['P']), d0=np.array(meta['d0']),
          off=rd('off.i4', '<i4').reshape(-1, rk).astype(np.int64))
if meta['cfg'] == 'lam12':
    kw.update(q=rd('q5.c16', '<c16'), Ps=complex(*meta['Ps5']))
else:
    kw.update(q5=rd('q5.c16', '<c16'), q7=rd('q7.c16', '<c16'), Ps5=complex(*meta['Ps5']), Ps7=complex(*meta['Ps7']), cls=rd('cls.i2', '<i2').reshape(-1, 12).astype(np.int64))
np.savez_compressed(out, **kw); print('wrote', out, meta['cfg'], meta['n_states'], meta['n_transitions'], 'violations', meta.get('violations'), 'area mismatches', meta.get('area_mismatch'))
