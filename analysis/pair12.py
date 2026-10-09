import sys, time; sys.setrecursionlimit(100000)
from front import *
t0=time.time()
one=fill_count(canon_closed(list(range(12))))[0]
two=[7,8,9,10,11,0,1,2,3,4,5]+list(range(1,12))
n2,_=fill_count(canon_closed(two))
print('single dodecagon',one,' edge-sharing pair',n2,' decoupled (shared edge present) =',one**2,
      ' coupled =',n2-one**2,' ratio N2/N1^2 = %.4f'%(n2/one**2),' time %.0fs'%(time.time()-t0),' memo states',fill_count.cache_info().currsize)
