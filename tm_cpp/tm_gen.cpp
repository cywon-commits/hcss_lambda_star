// C++ port of reference/python/cyl_ref.py (cylinder transfer-matrix builder for the A + 30deg-rhombus tilings).
// State numbering, transition order and all multiplicities follow the Python reference exactly.
//
//   tm_gen build  a0 a1 a2 a3 outdir [d0 as comma list]   -> raw directory (see tools/bin2npz.py)
//   tm_gen regions                                          -> region counts (cf. region_checks.py)
//   tm_gen d0     a0 a1 a2 a3                               -> start front chosen as in cyl_ref.py
//
// Geometry: vertices are exact integer 4-vectors in Z[zeta12] (basis 1,z,z^2,z^3; z^4=z^2-1).
// Canonical start vertices (lowest height / lowest-left) are chosen with exact arithmetic in Q(sqrt3);
// only the crossing tests use doubles with the reference tolerances.
// Compile with -ffp-contract=off so that the doubles follow the reference operation order.
#include <algorithm>
#include <array>
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

typedef std::array<int, 4> V4;
typedef std::string Seq;  // direction sequence, one byte (0..11) per step

static V4 E[12];
static double ZC_RE[4], ZC_IM[4];
static const double PI = 3.14159265358979323846;

static inline V4 vadd(const V4& a, const V4& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]}; }
static inline V4 vneg(const V4& a) { return {-a[0], -a[1], -a[2], -a[3]}; }
static inline V4 mulz(const V4& a) { return {-a[3], a[0], a[1] + a[3], a[2]}; }
static void init_tables() {
  E[0] = {1, 0, 0, 0};
  for (int k = 1; k < 12; k++) E[k] = mulz(E[k - 1]);
  for (int i = 0; i < 4; i++) { ZC_RE[i] = std::cos(PI * i / 6); ZC_IM[i] = std::sin(PI * i / 6); }
}
struct Cx { double re, im; };
static inline Cx cplx(const V4& a) {
  Cx r{0, 0};
  for (int i = 0; i < 4; i++) { r.re += a[i] * ZC_RE[i]; r.im += a[i] * ZC_IM[i]; }
  return r;
}

// ---- exact comparison of m + n*sqrt3 ----
static inline int sgn_mn(long m, long n) {  // sign of m + n sqrt3
  if (m == 0 && n == 0) return 0;
  if (m >= 0 && n >= 0) return 1;
  if (m <= 0 && n <= 0) return -1;
  long m2 = m * m, n2 = 3 * n * n;
  if (m > 0) return m2 > n2 ? 1 : -1;  // m>0, n<0
  return n2 > m2 ? 1 : -1;             // m<0, n>0
}
static inline int cmp_mn(long m1, long n1, long m2, long n2) { return sgn_mn(m1 - m2, n1 - n2); }
// 2*Im and 2*Re of a vertex as (m,n) meaning m + n sqrt3
static inline void im2(const V4& a, long& m, long& n) { m = a[1] + 2L * a[3]; n = a[2]; }
static inline void re2(const V4& a, long& m, long& n) { m = 2L * a[0] + a[2]; n = a[1]; }

// ---- path helpers ----
static std::vector<V4> lift(const Seq& d) {
  std::vector<V4> P(d.size() + 1);
  P[0] = {0, 0, 0, 0};
  for (size_t i = 0; i < d.size(); i++) P[i + 1] = vadd(P[i], E[(int)d[i]]);
  return P;
}
static Seq cancel(Seq d) {
  bool ch = true;
  while (ch && d.size() >= 2) {
    ch = false;
    int n = d.size();
    for (int i = 0; i < n; i++) {
      int j = (i + 1) % n;
      if ((((int)d[i] - (int)d[j]) % 12 + 12) % 12 == 6) {
        if (j == 0) { d.erase(n - 1, 1); d.erase(0, 1); }
        else d.erase(i, 2);
        ch = true;
        break;
      }
    }
  }
  return d;
}
static Seq cancel_shift(Seq d, V4& st) {
  st = {0, 0, 0, 0};
  bool ch = true;
  while (ch && d.size() >= 2) {
    ch = false;
    int n = d.size();
    for (int i = 0; i < n; i++) {
      int j = (i + 1) % n;
      if ((((int)d[i] - (int)d[j]) % 12 + 12) % 12 == 6) {
        if (j == 0) { st = vadd(st, E[(int)d[0]]); d.erase(n - 1, 1); d.erase(0, 1); }
        else d.erase(i, 2);
        ch = true;
        break;
      }
    }
  }
  return d;
}

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
      Cx r{b.re - a.re, b.im - a.im}, s{dd.re - c.re, dd.im - c.im};
      Cx ca{c.re - a.re, c.im - a.im};
      double den = cr(r, s);
      if (std::fabs(den) < 1e-12) {
        if (std::fabs(cr(ca, r)) < 1e-9) {
          double t0 = ca.re * r.re + ca.im * r.im;
          Cx da{dd.re - a.re, dd.im - a.im};
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
  std::vector<V4> P = lift(d);
  double s = 0;
  for (size_t i = 0; i < d.size(); i++) {
    Cx p = cplx(P[i]), q = cplx(P[i + 1]);
    s += p.re * q.im - p.im * q.re;
  }
  return s / 2;
}
// canonical rotation of a closed loop: lowest (imag, real) vertex, first on ties
static Seq canon_closed(const Seq& d) {
  std::vector<V4> P = lift(d);
  size_t n = d.size(), best = 0;
  long bm, bn, rm, rn, m, nn, m2, n2;
  im2(P[0], bm, bn); re2(P[0], rm, rn);
  for (size_t k = 1; k < n; k++) {
    im2(P[k], m, nn); re2(P[k], m2, n2);
    int c = cmp_mn(m, nn, bm, bn);
    if (c < 0 || (c == 0 && cmp_mn(m2, n2, rm, rn) < 0)) { best = k; bm = m; bn = nn; rm = m2; rn = n2; }
  }
  return d.substr(best) + d.substr(0, best);
}

// ---- tiles ----
struct Tile { int td[4]; int len; int ang; int vw2; double ar; };
static const Tile TILES[3] = {{{0, 4, 8, 0}, 3, 2, 1, 0.4330127018922193},
                              {{0, 1, 6, 7}, 4, 1, 2, 0.5},
                              {{0, 5, 6, 11}, 4, 5, 2, 0.5}};  // A, Ra, Ro ; vw2 = 2*vertex weight

// ---- finite regions ----
typedef unsigned long long u64;
struct Fill { u64 c; int w2; };
static thread_local std::unordered_map<Seq, Fill> fill_cache;

static bool closed_pieces(const Seq& nd0, std::vector<Seq>& out) {
  Seq nd = cancel(nd0);
  if (nd.empty()) return true;
  std::vector<V4> P = lift(nd);
  int n = nd.size();
  for (int i = 0; i < n; i++)
    for (int i0 = 0; i0 < i; i0++)
      if (P[i0] == P[i]) {
        // the first repeat in scan order: earliest i whose vertex was already seen; i0 = first occurrence
        bool a = closed_pieces(nd.substr(i0, i - i0), out);
        if (!a) return false;
        return closed_pieces(nd.substr(0, i0) + nd.substr(i), out);
      }
  std::vector<Cx> C(n + 1);
  for (int i = 0; i <= n; i++) C[i] = cplx(P[i]);
  std::vector<Seg> segs(n);
  for (int i = 0; i < n; i++) segs[i] = {C[i], C[i + 1]};
  if (crosses(segs)) return false;
  if (area(nd) < 1e-9) return false;
  out.push_back(canon_closed(nd));
  return true;
}

static Fill fill_count(const Seq& d) {
  if (d.empty()) return {1, 0};
  auto it = fill_cache.find(d);
  if (it != fill_cache.end()) return it->second;
  int din = d.back(), dout = d[0];
  int turn = ((dout - din) % 12 + 12) % 12;
  if (turn > 6) turn -= 12;
  int interior = 6 - turn;
  u64 tot = 0; int vw = 0;
  for (const Tile& T : TILES) {
    if (T.ang > interior) continue;
    Seq rep;
    for (int k = T.len - 1; k >= 1; k--) rep.push_back((char)(((dout + T.td[k]) % 12 + 6) % 12));
    Seq nd = cancel(rep + d.substr(1));
    std::vector<Seq> pcs;
    if (!closed_pieces(nd, pcs)) continue;
    u64 c = 1; int w = T.vw2;
    for (const Seq& pc : pcs) {
      Fill f = fill_count(pc);
      if (__builtin_mul_overflow(c, f.c, &c)) { fprintf(stderr, "overflow in fill_count\n"); exit(2); }
      w += f.w2;
      if (c == 0) break;
    }
    if (c) { tot += c; vw = w; }
  }
  Fill r{tot, tot ? vw : 0};
  fill_cache[d] = r;
  return r;
}

// ---- cylinder fronts ----
static V4 PV, NEGP;
static V4 CONJP;

static void set_period(const V4& Pv) {
  PV = Pv; NEGP = vneg(Pv);
  CONJP = {0, 0, 0, 0};
  for (int k = 0; k < 4; k++) {
    V4 e = E[(12 - k) % 12];
    for (int i = 0; i < 4; i++) CONJP[i] += Pv[k] * e[i];
  }
}
static V4 mulv(const V4& a, const V4& b) {
  V4 r{0, 0, 0, 0}, t = a;
  for (int i = 0; i < 4; i++) {
    for (int k = 0; k < 4; k++) r[k] += b[i] * t[k];
    t = mulz(t);
  }
  return r;
}
// height above the line through P: Im(p conj(P)) as (m,n): 2*height = m + n sqrt3 (up to the factor |P|)
static inline void height2(const V4& p, long& m, long& n) {
  V4 r = mulv(p, CONJP);
  im2(r, m, n);
}
// canonical rotation: min height, then lexicographically smallest sequence (ties: smallest index)
static void canon_front_idx(const Seq& d, const std::vector<V4>& P, size_t& bi) {
  size_t n = d.size();
  std::vector<long> hm(n), hn(n);
  long mm = 0, mn = 0; bool first = true;
  for (size_t i = 0; i < n; i++) {
    height2(P[i], hm[i], hn[i]);
    if (first || cmp_mn(hm[i], hn[i], mm, mn) < 0) { mm = hm[i]; mn = hn[i]; first = false; }
  }
  bool have = false; Seq best;
  for (size_t i = 0; i < n; i++) {
    if (cmp_mn(hm[i], hn[i], mm, mn) != 0) continue;
    Seq r = d.substr(i) + d.substr(0, i);
    if (!have || r < best) { best = r; bi = i; have = true; }
  }
}
static Seq canon_front(const Seq& d) {
  std::vector<V4> P = lift(d);
  size_t bi; canon_front_idx(d, P, bi);
  return d.substr(bi) + d.substr(0, bi);
}

// returns false if invalid. On success: front f, holes (closed canonical loops), off (offset of new canonical start)
static bool split_shift(const Seq& d0, Seq& f, std::vector<Seq>& holes, V4& off) {
  V4 st;
  Seq d = cancel_shift(d0, st);
  std::vector<V4> P = lift(d);
  int n = d.size();
  for (int i = 0; i < n; i++) {
    for (int typ = 0; typ < 2; typ++) {
      V4 q = typ == 0 ? P[i] : vadd(P[i], PV);
      for (int j = i + 1; j < n; j++) {
        if (!(P[j] == q)) continue;
        Seq hole, rest; V4 rst{0, 0, 0, 0};
        if (typ == 0) { hole = d.substr(i, j - i); rest = d.substr(0, i) + d.substr(j); }
        else { hole = d.substr(0, i) + d.substr(j); rest = d.substr(i, j - i); rst = P[i]; }
        std::vector<Seq> hp;
        if (!closed_pieces(hole, hp)) return false;
        Seq f2; std::vector<Seq> h2; V4 o2;
        if (!split_shift(rest, f2, h2, o2)) return false;
        f = f2; holes = h2;
        for (auto& h : hp) holes.push_back(h);
        off = vadd(st, vadd(rst, o2));
        return true;
      }
    }
    V4 t = vadd(P[i], NEGP);
    for (int k = 0; k < n; k++) if (P[k] == t) return false;
  }
  // crossing test against periodic images
  std::vector<Cx> C(n + 1);
  for (int i = 0; i <= n; i++) C[i] = cplx(P[i]);
  Cx pc = cplx(PV);
  std::vector<Seg> segs; segs.reserve(3 * n);
  for (int s = -1; s <= 1; s++)
    for (int k = 0; k < n; k++) {
      Cx sh{s * pc.re, s * pc.im};
      segs.push_back({{C[k].re + sh.re, C[k].im + sh.im}, {C[k + 1].re + sh.re, C[k + 1].im + sh.im}});
    }
  if (crosses(segs)) return false;
  size_t bi; canon_front_idx(d, P, bi);
  f = d.substr(bi) + d.substr(0, bi);
  holes.clear();
  off = vadd(st, P[bi]);
  return true;
}

struct Trans { Seq f; u64 c; int w2; int na; double ar; V4 off; };

static void expand(const Seq& d, std::vector<Trans>& out) {
  out.clear();
  int din = d.back(), dout = d[0];
  int turn = ((dout - din) % 12 + 12) % 12;
  if (turn > 6) turn -= 12;
  for (const Tile& T : TILES) {
    if (T.ang > 6 - turn) continue;
    Seq rep;
    for (int k = T.len - 1; k >= 1; k--) rep.push_back((char)(((dout + T.td[k]) % 12 + 6) % 12));
    Seq f; std::vector<Seq> holes; V4 off;
    if (!split_shift(rep + d.substr(1), f, holes, off)) continue;
    u64 c = 1; int w2 = T.vw2; double ar = T.ar; int na = (T.ang == 2) ? 1 : 0;
    bool bad = false;
    for (const Seq& h : holes) {
      Fill fl = fill_count(h);
      if (__builtin_mul_overflow(c, fl.c, &c)) { fprintf(stderr, "overflow in c\n"); exit(2); }
      w2 += fl.w2;
      double a = area(h);
      ar += a;
      na += (int)std::nearbyint((a - fl.w2 / 4.0) / ((std::sqrt(3.0) - 1) / 4));
      if (fl.c == 0) bad = true;
    }
    if (bad || c == 0) continue;
    out.push_back({f, c, w2, na, ar, off});
  }
}

static Seq path_for(const V4& a) {
  Seq st;
  for (int i = 0; i < 4; i++)
    for (int k = 0; k < std::abs(a[i]); k++) st.push_back((char)(a[i] > 0 ? i : i + 6));
  return st;
}
static Seq find_d0(const V4& a) {
  Seq base = path_for(a);
  size_t L = base.size();
  std::vector<int> idx(L);
  for (size_t i = 0; i < L; i++) idx[i] = i;
  do {
    Seq d; for (int i : idx) d.push_back(base[i]);
    Seq f; std::vector<Seq> holes; V4 off;
    if (split_shift(d, f, holes, off) && holes.empty()) return d;
  } while (std::next_permutation(idx.begin(), idx.end()));
  fprintf(stderr, "no valid start front\n"); exit(1);
}

static void write_file(const std::string& path, const void* p, size_t bytes) {
  FILE* fp = fopen(path.c_str(), "wb");
  if (!fp) { perror(path.c_str()); exit(1); }
  if (bytes) fwrite(p, 1, bytes, fp);
  fclose(fp);
}

static int cmd_build(int argc, char** argv) {
  if (argc < 7) { fprintf(stderr, "usage: tm_gen build a0 a1 a2 a3 outdir [d0,comma,list]\n"); return 1; }
  V4 a{atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5])};
  std::string out = argv[6];
  set_period(a);
  Seq d0;
  if (argc >= 8) { char* s = argv[7]; for (char* t = strtok(s, ","); t; t = strtok(nullptr, ",")) d0.push_back((char)atoi(t)); }
  else d0 = find_d0(a);
  Seq s0 = canon_front(d0);
  std::unordered_map<Seq, int> states; states.reserve(1 << 20);
  std::vector<Seq> order; order.push_back(s0); states[s0] = 0;
  std::vector<int32_t> fr, to, w, na, off; std::vector<u64> c; std::vector<double> ar;
  auto t0 = std::chrono::steady_clock::now();
  const size_t B = 2048;
  std::vector<std::vector<Trans>> buf(B);
  size_t k = 0;
  while (k < order.size()) {
    size_t nb = std::min(B, order.size() - k);
#pragma omp parallel for schedule(dynamic, 16)
    for (long i = 0; i < (long)nb; i++) expand(order[k + i], buf[i]);
    for (size_t i = 0; i < nb; i++) {
      for (const Trans& t : buf[i]) {
        auto it = states.find(t.f);
        int id;
        if (it == states.end()) { id = order.size(); states.emplace(t.f, id); order.push_back(t.f); }
        else id = it->second;
        fr.push_back(k + i); to.push_back(id); c.push_back(t.c); w.push_back(t.w2); na.push_back(t.na); ar.push_back(t.ar);
        for (int q = 0; q < 4; q++) off.push_back(t.off[q]);
      }
    }
    k += nb;
    if ((k / B) % 50 == 0)
      fprintf(stderr, "  processed %zu states %zu  %.0fs\n", k, order.size(),
              std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  }
  std::string sdir = out;
  std::string cmd = "mkdir -p '" + sdir + "'";
  if (system(cmd.c_str())) return 1;
  write_file(sdir + "/fr.i4", fr.data(), fr.size() * 4);
  write_file(sdir + "/to.i4", to.data(), to.size() * 4);
  write_file(sdir + "/c.u8", c.data(), c.size() * 8);
  write_file(sdir + "/w.i4", w.data(), w.size() * 4);
  write_file(sdir + "/na.i4", na.data(), na.size() * 4);
  write_file(sdir + "/ar.f8", ar.data(), ar.size() * 8);
  write_file(sdir + "/off.i4", off.data(), off.size() * 4);
  char meta[256];
  snprintf(meta, sizeof meta, "{\"P_int4\":[%d,%d,%d,%d],\"n_states\":%zu,\"n_transitions\":%zu}\n", a[0], a[1], a[2], a[3], order.size(), fr.size());
  write_file(sdir + "/meta.json", meta, strlen(meta));
  printf("states %zu transitions %zu  %.1fs\n", order.size(), fr.size(),
         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  return 0;
}

static int cmd_regions() {
  struct R { const char* name; std::vector<int> d; u64 expect; };
  auto infl = [](std::vector<int> poly) {
    int S[5] = {0, 1, -3, 1, 0}; std::vector<int> r;
    for (int d : poly) for (int s : S) r.push_back(((d + s) % 12 + 12) % 12);
    return r;
  };
  std::vector<int> dd, ddt, twod;
  for (int i = 0; i < 12; i++) dd.push_back(i);
  for (int x : {7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5}) twod.push_back(x);
  for (int i = 1; i < 12; i++) twod.push_back(i);
  ddt = {10, 2}; for (int i = 1; i < 12; i++) ddt.push_back(i);
  std::vector<R> rs = {{"triangle", {0, 4, 8}, 1}, {"rhombus", {0, 1, 6, 7}, 1},
                       {"hexagon 30deg (0,1,2)", {0, 1, 2, 6, 7, 8}, 2},
                       {"inflated triangle", infl({0, 4, 8}), 20}, {"inflated rhombus", infl({0, 1, 6, 7}), 23},
                       {"regular dodecagon side 1", dd, 5827}, {"two edge-sharing dodecagons", twod, 38120461},
                       {"dodecagon + triangle", ddt, 6865}};
  bool ok = true;
  for (auto& r : rs) {
    Seq s; for (int x : r.d) s.push_back((char)x);
    Fill f = fill_count(canon_closed(s));
    bool g = f.c == r.expect; ok &= g;
    printf("%-30s %10llu  expected %10llu  %s\n", r.name, f.c, r.expect, g ? "OK" : "FAIL");
  }
  puts(ok ? "ALL OK" : "SOME FAILED");
  return ok ? 0 : 1;
}

int main(int argc, char** argv) {
  init_tables();
  if (argc < 2) { fprintf(stderr, "usage: tm_gen build|regions|d0 ...\n"); return 1; }
  std::string m = argv[1];
  if (m == "regions") return cmd_regions();
  if (m == "d0") {
    V4 a{atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5])};
    set_period(a);
    Seq d = find_d0(a);
    for (size_t i = 0; i < d.size(); i++) printf("%s%d", i ? "," : "", (int)d[i]);
    puts("");
    return 0;
  }
  if (m == "build") return cmd_build(argc, argv);
  fprintf(stderr, "unknown mode\n");
  return 1;
}
