"""Signed tile areas in physical and perpendicular spaces; zero-phason composition (12-fold check + 18-fold contradiction)."""
import math, cmath
def area(dirs, unit, m):
    z = 0j; pts = [0j]
    for d in dirs[:-1]: z += cmath.exp(1j * math.radians(unit * d * m)); pts.append(z)
    return 0.5 * sum((pts[i].conjugate() * pts[(i + 1) % len(pts)]).imag for i in range(len(pts)))
print("12-fold square-triangle: N_t/N_s =", -area((0, 3, 6, 9), 30, 5) / area((0, 4, 8), 30, 5), "(4/sqrt3 = %.6f)" % (4 / math.sqrt(3)))
T, R = (0, 6, 12), (0, 4, 9, 13)
for m in (1, 5, 7): print(f"sigma_{m}: T {area(T,20,m):+.4f}  R80 {area(R,20,m):+.4f}")
print("zero area in sigma5 needs N_t/N_r =", -area(R, 20, 5) / area(T, 20, 5), "; in sigma7 needs", -area(R, 20, 7) / area(T, 20, 7))
