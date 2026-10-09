# exact region counts that ANY port of the region counter must reproduce
import sys; sys.setrecursionlimit(100000)
from front import fill_count, canon_closed
S=[0,1,-3,1,0]; infl=lambda poly:[(d+s)%12 for d in poly for s in S]
checks=[('triangle',[0,4,8],1),('rhombus',[0,1,6,7],1),('hexagon 30deg (0,1,2)',[0,1,2,6,7,8],2),
        ('inflated triangle',infl([0,4,8]),20),('inflated rhombus',infl([0,1,6,7]),23),
        ('regular dodecagon side 1',list(range(12)),5827),
        ('two edge-sharing dodecagons',[7,8,9,10,11,0,1,2,3,4,5]+list(range(1,12)),38120461),
        ('dodecagon + triangle',[10,2]+list(range(1,12)),6865)]
ok=True
for name,d,exp in checks:
    n=fill_count(canon_closed(d))[0]; ok&=(n==exp); print('%-30s %10d  expected %10d  %s'%(name,n,exp,'OK' if n==exp else 'FAIL'))
print('ALL OK' if ok else 'SOME FAILED')
