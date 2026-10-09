import numpy as np, itertools
# Z[zeta12] in basis 1,z,z^2,z^3 ; z^4 = z^2 - 1
def mulz(a):  # multiply by zeta
    a0,a1,a2,a3=a; return (-a3,a0,a1+a3,a2)
E=[(1,0,0,0)]
for k in range(11): E.append(mulz(E[-1]))
def add(a,b): return tuple(x+y for x,y in zip(a,b))
def sub(a,b): return tuple(x-y for x,y in zip(a,b))
def neg(a): return tuple(-x for x in a)
def rot(a,k):
    for _ in range(k%12): a=mulz(a)
    return a
zc=np.exp(1j*np.pi/6)
def cplx(a): return sum(a[i]*zc**i for i in range(4))
def star(a): return sum(a[i]*zc**(5*i) for i in range(4))   # Galois zeta->zeta^5
def mul(a,b):
    r=(0,0,0,0)
    for i in range(4):
        t=a
        for _ in range(i): t=mulz(t)
        r=add(r,tuple(b[i]*x for x in t))
    return r
