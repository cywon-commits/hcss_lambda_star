"""Configuration of the tile system.  select('lam12') -> lambda* tiling (Z[zeta12], A + 30-degree rhombi);
select('z18') -> triangle + 80-degree rhombus tiling (Z[zeta18]).  All other modules read the attributes of THIS module at call time
(import cfg as C ; C.n ...), so the choice can be made once at program start:  cfg.select(os.environ.get('HCSS_CFG','lam12'))."""
import math, os
import numpy as np

_DEF = {
    'lam12': dict(n=12, r=4, red=(-1, 0, 1, 0),                      # zeta^4 = zeta^2 - 1
                  tiles=(('A', (0, 4, 8), 2), ('Ra', (0, 1, 6, 7), 1), ('Ro', (0, 5, 6, 11), 5)),
                  vw={'A': 0.5, 'Ra': 1.0, 'Ro': 1.0}, area={'A': math.sqrt(3)/4, 'Ra': 0.5, 'Ro': 0.5}, perp=(5,)),
    'z18': dict(n=18, r=6, red=(-1, 0, 0, 1, 0, 0),                  # zeta^6 = zeta^3 - 1
                tiles=(('T', (0, 6, 12), 3), ('R80', (0, 4, 9, 13), 4), ('R100', (0, 5, 9, 14), 5)),
                vw={'T': 0.5, 'R80': 1.0, 'R100': 1.0}, area={'T': math.sqrt(3)/4, 'R80': math.sin(math.radians(80)), 'R100': math.sin(math.radians(80))}, perp=(5, 7)),
}
name = None
HOOKS = []          # callables run after every select() (caches of dependent modules register their cache_clear here)


def select(nm):
    global name, n, r, red, TILES, VW, AREA, PERP, half, zc, E, ZP, ZPOW
    d = _DEF[nm]; name = nm
    n, r, red, TILES, VW, AREA, PERP = d['n'], d['r'], d['red'], d['tiles'], d['vw'], d['area'], d['perp']
    half = n // 2
    zc = np.exp(2j * np.pi / n)
    ZPOW = [zc ** k for k in range(r)]
    E = []
    a = (1,) + (0,) * (r - 1)
    for k in range(n):
        E.append(a); a = mulz(a)
    for hk in HOOKS: hk()
    assert mulz(E[n - 1]) == E[0] and all(tuple(-x for x in E[k]) == E[(k + half) % n] for k in range(n))


def mulz(a):
    top = a[r - 1]
    out = [0] + list(a[:r - 1])
    return tuple(out[i] + top * red[i] for i in range(r))


select(os.environ.get('HCSS_CFG', 'lam12'))
