import numpy as np, pickle
from collections import Counter
exec(open('pairs.py').read().split("F=fills")[0])
F=fills((0,0,0,0),list(range(12)))
TS=[frozenset((t[0],frozenset(vs)) for t,vs in f) for f in F]
allt={}
for s in TS:
    for t in s: allt.setdefault(t,len(allt))
M=np.zeros((len(TS),len(allt)),np.uint8)
for i,s in enumerate(TS):
    for t in s: M[i,allt[t]]=1
M=M.astype(np.int32)
common=M@M.T            # shared tiles
n=M.sum(1)
diff=n[:,None]-common   # tiles of i not in j (removed)
np.fill_diagonal(diff,999)
print('distinct tiles',len(allt))
c=Counter(diff.min(1)); print('min #tiles changed to nearest neighbour:',sorted(c.items()))
print('pairs by #tiles changed:',sorted(Counter(diff[np.triu_indices(len(TS),1)].ravel()).items())[:6])
np.save('diffmat.npy',diff.astype(np.int16))
pickle.dump((TS,allt),open('dodec_tiles.pkl','wb'))
