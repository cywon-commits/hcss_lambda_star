import json, numpy as np
def convert(d, out):
    meta = json.load(open(f'{d}/meta.json')); rk = meta['rank']; rd = lambda f, t: np.fromfile(f'{d}/{f}', dtype=t)
    kw = dict(fr=rd('fr.i4', '<i4').astype(np.int64), to=rd('to.i4', '<i4').astype(np.int64), c=rd('c.u8', '<u8').astype(np.int64), w=rd('w.i4', '<i4').astype(np.int64),
              na=rd('na.i4', '<i4').astype(np.int64), ar=rd('ar.f8', '<f8'), n=meta['n_states'], P=complex(*meta['P']), d0=np.array(meta['d0']), off=rd('off.i4', '<i4').reshape(-1, rk).astype(np.int64))
    kw.update(q5=rd('q5.c16', '<c16'), q7=rd('q7.c16', '<c16'), Ps5=complex(*meta['Ps5']), Ps7=complex(*meta['Ps7']), cls=rd('cls.i2', '<i2').reshape(-1, 12).astype(np.int64))
    np.savez_compressed(out, **kw)
