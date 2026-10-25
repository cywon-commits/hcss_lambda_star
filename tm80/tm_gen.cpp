// tm80: cylinder transfer-matrix generator, generalised from tm_cpp/tm_gen.cpp to two tile systems selected at run time:
//   lam12 : lambda* tiling (Z[zeta12], rank 4, 12 directions: A + 30-degree rhombi)       -> must reproduce tm_cpp / reference data
//   z18   : triangle + 80-degree rhombus (Z[zeta18], rank 6, 18 directions)               -> must reproduce reference80 (Python)
// Usage:  tm80 build <cfg> a0..a(r-1) outdir [d0 comma list]   |   tm80 regions <cfg>   |   tm80 d0 <cfg> a0..a(r-1)
// State numbering, transition order and all multiplicities follow reference80/cyl_ref.py exactly.
// Output (little endian, one value per transition): fr.i4 to.i4 c.u8 w.i4 na.i4 ar.f8 off.i4 x rank, cls.i2 x 12 (z18), q5.c16 [q7.c16] (complex128)
// plus meta.json.  Geometry: exact integer vectors in Z[zeta_n]; canonical start vertices use exact tie tests (x == conj x) and doubles for the order;
// crossing tests use doubles with the reference tolerances.  Compile with -ffp-contract=off.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

static const int MAXR = 6, MAXN = 18;
typedef std::array<int, MAXR> V;
typedef std::string Seq;  // direction sequence, one byte per step
typedef unsigned long long u64;
static const double PI = 3.14159265358979323846;

// ---------------- configuration ----------------
struct Tile { const char* name; int td[4]; int len; int ang; int vw2; double ar; };
static std::string CFG; static int NDIR, RK, HALF; static int RED[MAXR];
static Tile TILES[3]; static std::vector<int> PERP; static bool Z18 = false;
static V E[MAXN]; static V CONJROW[MAXR]; static double ZC_RE[MAXR], ZC_IM[MAXR];
static const V ZERO = {0, 0, 0, 0, 0, 0};
static inline V vadd(const V& a, const V& b) { V r; for (int i = 0; i < MAXR; i++) r[i] = a[i] + b[i]; return r; }
static inline V vsub(const V& a, const V& b) { V r; for (int i = 0; i < MAXR; i++) r[i] = a[i] - b[i]; return r; }
static inline V vneg(const V& a) { V r; for (int i = 0; i < MAXR; i++) r[i] = -a[i]; return r; }
static inline V mulz(const V& a) {
  V r = ZERO; int top = a[RK - 1];
  for (int i = 1; i < RK; i++) r[i] = a[i - 1];
  for (int i = 0; i < RK; i++) r[i] += top * RED[i];
  return r;
}
static void select_cfg(const std::string& c) {
  CFG = c;
  if (c == "lam12") {
    NDIR = 12; RK = 4; int red[4] = {-1, 0, 1, 0}; for (int i = 0; i < 4; i++) RED[i] = red[i];
    TILES[0] = {"A", {0, 4, 8, 0}, 3, 2, 1, std::sqrt(3.0) / 4}; TILES[1] = {"Ra", {0, 1, 6, 7}, 4, 1, 2, 0.5}; TILES[2] = {"Ro", {0, 5, 6, 11}, 4, 5, 2, 0.5};
    PERP = {5}; Z18 = false;
  } else if (c == "z18") {
    NDIR = 18; RK = 6; int red[6] = {-1, 0, 0, 1, 0, 0}; for (int i = 0; i < 6; i++) RED[i] = red[i];
    double s80 = std::sin(80.0 * PI / 180.0);
    TILES[0] = {"T", {0, 6, 12, 0}, 3, 3, 1, std::sqrt(3.0) / 4}; TILES[1] = {"R80", {0, 4, 9, 13}, 4, 4, 2, s80}; TILES[2] = {"R100", {0, 5, 9, 14}, 4, 5, 2, s80};
    PERP = {5, 7}; Z18 = true;
  } else { fprintf(stderr, "unknown cfg %s\n", c.c_str()); exit(1); }
  HALF = NDIR / 2;
  E[0] = ZERO; E[0][0] = 1;
  for (int k = 1; k < NDIR; k++) E[k] = mulz(E[k - 1]);
  for (int i = 0; i < RK; i++) { CONJROW[i] = E[(NDIR - i) % NDIR]; ZC_RE[i] = std::cos(2 * PI * i / NDIR); ZC_IM[i] = std::sin(2 * PI * i / NDIR); }
}
struct Cx { double re, im; };
static inline Cx cplx(const V& a) { Cx r{0, 0}; for (int i = 0; i < RK; i++) { r.re += a[i] * ZC_RE[i]; r.im += a[i] * ZC_IM[i]; } return r; }
static Cx starm(const V& a, int m) {   // Galois conjugate zeta -> zeta^m
  Cx r{0, 0}; for (int i = 0; i < RK; i++) { r.re += a[i] * std::cos(2 * PI * m * i / NDIR); r.im += a[i] * std::sin(2 * PI * m * i / NDIR); } return r;
}
static V conjv(const V& a) { V r = ZERO; for (int i = 0; i < RK; i++) for (int k = 0; k < RK; k++) r[k] += a[i] * CONJROW[i][k]; return r; }
static V mulv(const V& a, const V& b) {
  V r = ZERO, t = a;
  for (int i = 0; i < RK; i++) { for (int k = 0; k < RK; k++) r[k] += b[i] * t[k]; t = mulz(t); }
  return r;
}
static inline bool is_real(const V& a) { return a == conjv(a); }   // exact: x is real iff x == conj(x) in the integer basis

// ---------------- path helpers ----------------
static std::vector<V> lift(const Seq& d) {
  std::vector<V> P(d.size() + 1); P[0] = ZERO;
  for (size_t i = 0; i < d.size(); i++) P[i + 1] = vadd(P[i], E[(int)d[i]]);
  return P;
}
static Seq cancel_shift(Seq d, V& st) {
  st = ZERO; bool ch = true;
  while (ch && d.size() >= 2) {
    ch = false; int n = d.size();
    for (int i = 0; i < n; i++) {
      int j = (i + 1) % n;
      if ((((int)d[i] - (int)d[j]) % NDIR + NDIR) % NDIR == HALF) {
        if (j == 0) { st = vadd(st, E[(int)d[0]]); d.erase(n - 1, 1); d.erase(0, 1); }
        else d.erase(i, 2);
        ch = true; break;
      }
    }
  }
  return d;
}
static Seq cancel(Seq d) { V st; return cancel_shift(std::move(d), st); }

struct Seg { Cx a, b; };
static bool crosses(const std::vector<Seg>& segs) {
  const double eps = 1e-9;
  auto cr = [](Cx a, Cx b) { return a.re * b.im - a.im * b.re; };
  size_t n = segs.size();
  for (size_t i = 0; i < n; i++) {
    Cx a = segs[i].a, b = segs[i].b;
    for (size_t j = i + 1; j < n; j++) {
      Cx c = segs[j].a, dd = segs[j].b;
      double m = std::min(std::min(std::hypot(a.re - c.re, a.im - c.im), std::hypot(a.re - dd.re, a.im - dd.im)),
                          std::min(std::hypot(b.re - c.re, b.im - c.im), std::hypot(b.re - dd.re, b.im - dd.im)));
      if (m > 2.01) continue;
      Cx r{b.re - a.re, b.im - a.im}, s{dd.re - c.re, dd.im - c.im}, ca{c.re - a.re, c.im - a.im};
      double den = cr(r, s);
      if (std::fabs(den) < 1e-12) {
        if (std::fabs(cr(ca, r)) < 1e-9) {
          double t0 = ca.re * r.re + ca.im * r.im; Cx da{dd.re - a.re, dd.im - a.im};
          double t1 = da.re * r.re + da.im * r.im;
          if (std::min(1.0, std::max(t0, t1)) - std::max(0.0, std::min(t0, t1)) > eps) return true;
        }
        continue;
      }
      double t = cr(ca, s) / den, u = cr(ca, r) / den;
      if (eps < t && t < 1 - eps && -eps < u && u < 1 + eps) return true;
      if (eps < u && u < 1 - eps && -eps < t && t < 1 + eps) return true;
    }
  }
  return false;
}
static double area(const Seq& d) {
  std::vector<V> P = lift(d); double s = 0;
  for (size_t i = 0; i < d.size(); i++) { Cx p = cplx(P[i]), q = cplx(P[i + 1]); s += p.re * q.im - p.im * q.re; }
  return s / 2;
}
// canonical rotation of a closed loop: lowest (imag, real) vertex, first on ties
static Seq canon_closed(const Seq& d) {
  std::vector<V> P = lift(d); size_t n = d.size(), best = 0;
  Cx cb = cplx(P[0]);
  for (size_t k = 1; k < n; k++) {
    Cx c = cplx(P[k]);
    bool lower;
    if (c.im < cb.im - 1e-6) lower = true;                                  // clearly lower
    else if (c.im > cb.im + 1e-6) lower = false;
    else {                                                                    // imaginary parts tie exactly (checked exactly): compare real parts
      bool tie = is_real(vsub(P[k], P[best]));
      lower = tie ? (c.re < cb.re) : (c.im < cb.im);
    }
    if (lower) { best = k; cb = c; }
  }
  return d.substr(best) + d.substr(0, best);
}

// ---------------- classes (z18): triangles by family d mod 3, rhombi by pair of edge directions mod 9 ----------------
typedef std::array<int16_t, 12> Cls;
static inline int tile_class(int tile, int d) { return tile == 0 ? d % 3 : (tile == 1 ? 3 + d % 9 : 3 + (d + 5) % 9); }
static inline Cls cls_zero() { Cls c; c.fill(0); return c; }
static inline Cls cls_add(const Cls& a, const Cls& b) { Cls r; for (int i = 0; i < 12; i++) r[i] = a[i] + b[i]; return r; }
static std::atomic<long> VIOLATIONS(0);

// ---------------- finite regions ----------------
struct Fill { u64 c; int w2; Cls cls; };
static thread_local std::unordered_map<Seq, Fill> fill_cache;
static bool closed_pieces(const Seq& nd0, std::vector<Seq>& out) {
  Seq nd = cancel(nd0);
  if (nd.empty()) return true;
  std::vector<V> P = lift(nd); int n = nd.size();
  for (int i = 0; i < n; i++)
    for (int i0 = 0; i0 < i; i0++)
      if (P[i0] == P[i]) {
        if (!closed_pieces(nd.substr(i0, i - i0), out)) return false;
        return closed_pieces(nd.substr(0, i0) + nd.substr(i), out);
      }
  std::vector<Cx> C(n + 1); for (int i = 0; i <= n; i++) C[i] = cplx(P[i]);
  std::vector<Seg> segs(n); for (int i = 0; i < n; i++) segs[i] = {C[i], C[i + 1]};
  if (crosses(segs)) return false;
  if (area(nd) < 1e-9) return false;
  out.push_back(canon_closed(nd));
  return true;
}
static Fill fill_info(const Seq& d) {
  if (d.empty()) return {1, 0, cls_zero()};
  auto it = fill_cache.find(d);
  if (it != fill_cache.end()) return it->second;
  int din = d.back(), dout = d[0];
  int turn = ((dout - din) % NDIR + NDIR) % NDIR; if (turn > HALF) turn -= NDIR;
  int interior = HALF - turn; u64 tot = 0; int vw = 0; Cls vec = cls_zero(); bool have = false;
  for (int ti = 0; ti < 3; ti++) {
    const Tile& T = TILES[ti];
    if (T.ang > interior) continue;
    Seq rep; for (int k = T.len - 1; k >= 1; k--) rep.push_back((char)(((dout + T.td[k]) % NDIR + HALF) % NDIR));
    Seq nd = cancel(rep + d.substr(1)); std::vector<Seq> pcs;
    if (!closed_pieces(nd, pcs)) continue;
    u64 c = 1; int w = T.vw2; Cls v = cls_zero();
    if (Z18) v[tile_class(ti, dout)] = 1;
    for (const Seq& pc : pcs) {
      Fill f = fill_info(pc);
      if (__builtin_mul_overflow(c, f.c, &c)) { fprintf(stderr, "overflow in fill_count\n"); exit(2); }
      w += f.w2; v = cls_add(v, f.cls);
      if (c == 0) break;
    }
    if (c) { tot += c; vw = w; if (!have) { vec = v; have = true; } else if (vec != v) VIOLATIONS++; }
  }
  Fill r{tot, tot ? vw : 0, have ? vec : cls_zero()};
  fill_cache[d] = r; return r;
}

// ---------------- cylinder fronts ----------------
static V PV, NEGP, CONJP;
static void set_period(const V& Pv) { PV = Pv; NEGP = vneg(Pv); CONJP = conjv(Pv); }
// height of vertex p above the line through P (up to the factor |P|): Im(p conj P)
static inline double height(const V& p) { return cplx(mulv(p, CONJP)).im; }
// canonical rotation: min height (ties tested exactly), then lexicographically smallest sequence (ties: smallest index)
static void canon_front_idx(const Seq& d, const std::vector<V>& P, size_t& bi) {
  size_t n = d.size(); std::vector<double> h(n); size_t imin = 0;
  for (size_t i = 0; i < n; i++) { h[i] = height(P[i]); if (h[i] < h[imin]) imin = i; }
  V xm = mulv(P[imin], CONJP); bool have = false; Seq best;
  for (size_t i = 0; i < n; i++) {
    if (std::fabs(h[i] - h[imin]) > 1e-6) continue;
    if (!is_real(vsub(mulv(P[i], CONJP), xm))) continue;          // exact tie in the imaginary part
    Seq r = d.substr(i) + d.substr(0, i);
    if (!have || r < best) { best = r; bi = i; have = true; }
  }
}
static Seq canon_front(const Seq& d) { std::vector<V> P = lift(d); size_t bi; canon_front_idx(d, P, bi); return d.substr(bi) + d.substr(0, bi); }

static bool split_shift(const Seq& d0, Seq& f, std::vector<Seq>& holes, V& off) {
  V st; Seq d = cancel_shift(d0, st);
  std::vector<V> P = lift(d); int n = d.size();
  for (int i = 0; i < n; i++) {
    for (int typ = 0; typ < 2; typ++) {
      V q = typ == 0 ? P[i] : vadd(P[i], PV);
      for (int j = i + 1; j < n; j++) {
        if (!(P[j] == q)) continue;
        Seq hole, rest; V rst = ZERO;
        if (typ == 0) { hole = d.substr(i, j - i); rest = d.substr(0, i) + d.substr(j); }
        else { hole = d.substr(0, i) + d.substr(j); rest = d.substr(i, j - i); rst = P[i]; }
        std::vector<Seq> hp;
        if (!closed_pieces(hole, hp)) return false;
        Seq f2; std::vector<Seq> h2; V o2;
        if (!split_shift(rest, f2, h2, o2)) return false;
        f = f2; holes = h2; for (auto& h : hp) holes.push_back(h);
        off = vadd(st, vadd(rst, o2)); return true;
      }
    }
    V t = vadd(P[i], NEGP);
    for (int k = 0; k < n; k++) if (P[k] == t) return false;
  }
  std::vector<Cx> C(n + 1); for (int i = 0; i <= n; i++) C[i] = cplx(P[i]);
  Cx pc = cplx(PV); std::vector<Seg> segs; segs.reserve(3 * n);
  for (int s = -1; s <= 1; s++)
    for (int k = 0; k < n; k++) {
      Cx sh{s * pc.re, s * pc.im};
      segs.push_back({{C[k].re + sh.re, C[k].im + sh.im}, {C[k + 1].re + sh.re, C[k + 1].im + sh.im}});
    }
  if (crosses(segs)) return false;
  size_t bi; canon_front_idx(d, P, bi);
  f = d.substr(bi) + d.substr(0, bi); holes.clear(); off = vadd(st, P[bi]);
  return true;
}

struct Trans { Seq f; u64 c; int w2; int na; double ar; V off; Cls cls; };
static std::atomic<long> AREA_MISMATCH(0);
static void expand(const Seq& d, std::vector<Trans>& out) {
  out.clear();
  int din = d.back(), dout = d[0];
  int turn = ((dout - din) % NDIR + NDIR) % NDIR; if (turn > HALF) turn -= NDIR;
  const double sr = std::sin(80.0 * PI / 180.0);
  for (int ti = 0; ti < 3; ti++) {
    const Tile& T = TILES[ti];
    if (T.ang > HALF - turn) continue;
    Seq rep; for (int k = T.len - 1; k >= 1; k--) rep.push_back((char)(((dout + T.td[k]) % NDIR + HALF) % NDIR));
    Seq f; std::vector<Seq> holes; V off;
    if (!split_shift(rep + d.substr(1), f, holes, off)) continue;
    u64 c = 1; int w2 = T.vw2; double ar = T.ar; Cls cls = cls_zero(); if (Z18) cls[tile_class(ti, dout)] = 1;
    int na = (!Z18 && ti == 0) ? 1 : 0; double nt_h = 0; bool bad = false;
    for (const Seq& h : holes) {
      Fill fl = fill_info(h);
      if (__builtin_mul_overflow(c, fl.c, &c)) { fprintf(stderr, "overflow in c\n"); exit(2); }
      w2 += fl.w2; double a = area(h); ar += a; cls = cls_add(cls, fl.cls);
      if (!Z18) na += (int)std::nearbyint((a - fl.w2 / 4.0) / ((std::sqrt(3.0) - 1) / 4));
      else nt_h += (a - (fl.w2 / 2.0) * sr) / (std::sqrt(3.0) / 4 - sr / 2);
      if (fl.c == 0) bad = true;
    }
    if (bad || c == 0) continue;
    if (Z18) { na = cls[0] + cls[1] + cls[2]; if (std::fabs(nt_h + (ti == 0 ? 1 : 0) - na) > 1e-6) AREA_MISMATCH++; }
    out.push_back({f, c, w2, na, ar, off, cls});
  }
}

static Seq path_for(const V& a) {
  Seq st; for (int i = 0; i < RK; i++) for (int k = 0; k < std::abs(a[i]); k++) st.push_back((char)(a[i] > 0 ? i : i + HALF));
  return st;
}
static Seq find_d0(const V& a) {
  Seq base = path_for(a); size_t L = base.size(); std::vector<int> idx(L);
  for (size_t i = 0; i < L; i++) idx[i] = i;
  do {
    Seq d; for (int i : idx) d.push_back(base[i]);
    Seq f; std::vector<Seq> holes; V off;
    if (split_shift(d, f, holes, off) && holes.empty()) return d;
  } while (std::next_permutation(idx.begin(), idx.end()));
  fprintf(stderr, "no valid start front\n"); exit(1);
}
static void write_file(const std::string& path, const void* p, size_t bytes) {
  FILE* fp = fopen(path.c_str(), "wb"); if (!fp) { perror(path.c_str()); exit(1); }
  if (bytes) fwrite(p, 1, bytes, fp); fclose(fp);
}
static V parse_period(char** argv, int first) { V a = ZERO; for (int i = 0; i < RK; i++) a[i] = atoi(argv[first + i]); return a; }

static int cmd_build(int argc, char** argv) {
  if (argc < 4 + RK) { fprintf(stderr, "usage: tm80 build <cfg> a0..a%d outdir [d0,comma,list]\n", RK - 1); return 1; }
  V a = parse_period(argv, 3); std::string out = argv[3 + RK];
  set_period(a); Seq d0;
  if (argc > 4 + RK) { char* s = argv[4 + RK]; for (char* t = strtok(s, ","); t; t = strtok(nullptr, ",")) d0.push_back((char)atoi(t)); }
  else d0 = find_d0(a);
  Seq s0 = canon_front(d0);
  std::unordered_map<Seq, int> states; states.reserve(1 << 20);
  std::vector<Seq> order; order.push_back(s0); states[s0] = 0;
  std::vector<int32_t> fr, to, w, na, off; std::vector<u64> c; std::vector<double> ar; std::vector<int16_t> clsv;
  auto t0 = std::chrono::steady_clock::now();
  const size_t B = 2048; std::vector<std::vector<Trans>> buf(B); size_t k = 0;
  while (k < order.size()) {
    size_t nb = std::min(B, order.size() - k);
#pragma omp parallel for schedule(dynamic, 16)
    for (long i = 0; i < (long)nb; i++) expand(order[k + i], buf[i]);
    for (size_t i = 0; i < nb; i++)
      for (const Trans& t : buf[i]) {
        auto it = states.find(t.f); int id;
        if (it == states.end()) { id = order.size(); states.emplace(t.f, id); order.push_back(t.f); } else id = it->second;
        fr.push_back(k + i); to.push_back(id); c.push_back(t.c); w.push_back(t.w2); na.push_back(t.na); ar.push_back(t.ar);
        for (int q = 0; q < RK; q++) off.push_back(t.off[q]);
        if (Z18) for (int q = 0; q < 12; q++) clsv.push_back(t.cls[q]);
      }
    k += nb;
    if ((k / B) % 50 == 0) fprintf(stderr, "  processed %zu states %zu  %.0fs\n", k, order.size(), std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  }
  std::string cmd = "mkdir -p '" + out + "'"; if (system(cmd.c_str())) return 1;
  write_file(out + "/fr.i4", fr.data(), fr.size() * 4); write_file(out + "/to.i4", to.data(), to.size() * 4);
  write_file(out + "/c.u8", c.data(), c.size() * 8); write_file(out + "/w.i4", w.data(), w.size() * 4);
  write_file(out + "/na.i4", na.data(), na.size() * 4); write_file(out + "/ar.f8", ar.data(), ar.size() * 8);
  write_file(out + "/off.i4", off.data(), off.size() * 4);
  if (Z18) write_file(out + "/cls.i2", clsv.data(), clsv.size() * 2);
  // perpendicular displacements q_m = star_m(off) - (a/|P|) star_m(P),  a = Re(off conj(Phat))   (lateral part removed)
  Cx Pc = cplx(a); double aP = std::hypot(Pc.re, Pc.im); Cx Ph{Pc.re / aP, Pc.im / aP};
  std::string pstr;
  for (size_t mi = 0; mi < PERP.size(); mi++) {
    int m = PERP[mi]; Cx Ps = starm(a, m); std::vector<double> q(2 * fr.size());
    for (size_t t = 0; t < fr.size(); t++) {
      V o; for (int i = 0; i < RK; i++) o[i] = off[t * RK + i];
      Cx dphys = cplx(o); double lat = dphys.re * Ph.re + dphys.im * Ph.im;    // Re(dphys * conj(Phat))
      Cx dp = starm(o, m); q[2 * t] = dp.re - (lat / aP) * Ps.re; q[2 * t + 1] = dp.im - (lat / aP) * Ps.im;
    }
    write_file(out + "/q" + std::to_string(m) + ".c16", q.data(), q.size() * 8);
    char b[200]; snprintf(b, sizeof b, "%s\"Ps%d\":[%.17g,%.17g]", mi ? "," : "", m, Ps.re, Ps.im); pstr += b;
  }
  std::string d0s; for (size_t i = 0; i < d0.size(); i++) d0s += (i ? "," : "") + std::to_string((int)d0[i]);
  std::string ps; for (int i = 0; i < RK; i++) ps += (i ? "," : "") + std::to_string(a[i]);
  char head[400]; snprintf(head, sizeof head, "{\"cfg\":\"%s\",\"rank\":%d,\"n_states\":%zu,\"n_transitions\":%zu,\"P\":[%.17g,%.17g],", CFG.c_str(), RK, order.size(), fr.size(), Pc.re, Pc.im);
  std::string meta = std::string(head) + "\"P_int\":[" + ps + "],\"d0\":[" + d0s + "]," + pstr + ",\"violations\":" + std::to_string(VIOLATIONS.load()) + ",\"area_mismatch\":" + std::to_string(AREA_MISMATCH.load()) + "}\n";
  write_file(out + "/meta.json", meta.data(), meta.size());
  printf("states %zu transitions %zu  %.1fs  violations %ld area-mismatch %ld\n", order.size(), fr.size(),
         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), VIOLATIONS.load(), AREA_MISMATCH.load());
  return 0;
}

static int cmd_regions() {
  bool ok = true;
  auto test = [&](const char* name, std::vector<int> d, u64 expect, int ncls = -1) {
    Seq s; for (int x : d) s.push_back((char)x);
    Fill f = fill_info(canon_closed(s)); bool g = f.c == expect; ok &= g;
    printf("%-30s %12llu  expected %12llu  %s", name, f.c, expect, g ? "OK" : "FAIL");
    if (Z18) { printf("  cls"); for (int i = 0; i < 12; i++) printf(" %d", f.cls[i]); }
    puts("");
  };
  if (CFG == "lam12") {
    auto infl = [](std::vector<int> poly) { int S[5] = {0, 1, -3, 1, 0}; std::vector<int> r; for (int d : poly) for (int s : S) r.push_back(((d + s) % 12 + 12) % 12); return r; };
    std::vector<int> dd, ddt, twod; for (int i = 0; i < 12; i++) dd.push_back(i);
    for (int x : {7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5}) twod.push_back(x); for (int i = 1; i < 12; i++) twod.push_back(i);
    ddt = {10, 2}; for (int i = 1; i < 12; i++) ddt.push_back(i);
    test("triangle", {0, 4, 8}, 1); test("rhombus", {0, 1, 6, 7}, 1); test("hexagon 30deg (0,1,2)", {0, 1, 2, 6, 7, 8}, 2);
    test("inflated triangle", infl({0, 4, 8}), 20); test("inflated rhombus", infl({0, 1, 6, 7}), 23);
    test("regular dodecagon side 1", dd, 5827); test("two edge-sharing dodecagons", twod, 38120461); test("dodecagon + triangle", ddt, 6865);
  } else {
    std::vector<int> g18; for (int i = 0; i < 18; i++) g18.push_back(i);
    test("triangle", {0, 6, 12}, 1); test("R80", {0, 4, 9, 13}, 1); test("regular 18-gon side 1", g18, 6);
  }
  puts(ok ? "ALL OK" : "SOME FAILED"); printf("class-vector violations: %ld\n", VIOLATIONS.load());
  return ok ? 0 : 1;
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: tm80 build|regions|d0 <lam12|z18> ...\n"); return 1; }
  std::string m = argv[1]; select_cfg(argv[2]);
  if (m == "regions") return cmd_regions();
  if (m == "d0") {
    V a = parse_period(argv, 3); set_period(a); Seq d = find_d0(a);
    for (size_t i = 0; i < d.size(); i++) printf("%s%d", i ? "," : "", (int)d[i]);
    puts(""); return 0;
  }
  if (m == "build") return cmd_build(argc, argv);
  fprintf(stderr, "unknown mode\n"); return 1;
}
