"""Re-express the cylinder stiffness fits of results01 in symmetry-adapted combinations.
K_a|a|^2 + K_r(Re b)^2 + K_i(Im b)^2 = Kbar(|a|^2+|b|^2) + delta(|a|^2-|b|^2) + (dK/2) Re(b^2)
  Kbar = (K_a + Kb)/2,  delta = (K_a - Kb)/2,  dK = K_r - K_i,  Kb = (K_r+K_i)/2
|a|^2-|b|^2 = det E is a null Lagrangian: delta is invisible to local curvature/fluctuations and
enters a cylinder fit only through the linear term ~ g, hence it is ill-conditioned when |g| is small."""
rows=[  # |P|, dir, |g|, K_a, K_r, K_i   (results01 table, tilt range 0.3)
 (3.732, 0.0,0.072,1.124,2.319,1.754),(4.625, 6.2,0.169,1.659,1.357,1.924),(5.278,15.0,0.072,1.788,1.785,1.867),
 (5.620, 5.1,0.114,1.551,1.745,1.911),(6.464, 0.0,0.072,1.903,1.570,1.695),(7.210,15.0,0.019,1.540,1.953,2.040),
 (7.464, 0.0,0.072,1.757,1.736,1.753)]
print('%6s %5s %6s | %6s %7s %7s | %8s %8s'%('|P|','dir','|g|','Kbar','dK','delta','Ka+Kr','Ka+Ki'))
for W,d,g,Ka,Kr,Ki in rows:
    Kb=(Kr+Ki)/2
    print('%6.3f %5.1f %6.3f | %6.3f %+7.3f %+7.3f | %8.3f %8.3f'%(W,d,g,(Ka+Kb)/2,Kr-Ki,(Ka-Kb)/2,Ka+Kr,Ka+Ki))
