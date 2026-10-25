"""Rank of lifted area 2-forms of tile orientation classes in Z[zeta18] (expect 12: 3 triangle families + 9 rhombus orientations)."""
import itertools, sympy as sp
from moves import UI
pairs = list(itertools.combinations(range(6), 2))
wedge = lambda a, b: [a[i] * b[j] - a[j] * b[i] for i, j in pairs]
forms, names = [], []
for d in range(6): forms.append(wedge(UI[d], UI[(d + 6) % 18])); names.append(f"T{d}")
for d in range(9): forms.append(wedge(UI[d], UI[(d + 4) % 18])); names.append(f"R{d}")
A = sp.Matrix(forms).T
print("rank of 15 oriented tiles:", A.rank(), "(T_d and T_{d+3} are up/down triangles of one class)")
for v in A.nullspace():
    v = v * sp.ilcm(*[x.q for x in v]); print("relation:", {n: int(c) for n, c in zip(names, v) if c != 0})
