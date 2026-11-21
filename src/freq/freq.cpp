// toolboxes/control/src/freq/freq.cpp
//
// Frequency-domain responses. Reduces every LTI form to (num, den)
// coefficient rows via the existing toolboxes/builtin (zp2tf) and
// toolboxes/control (ss2tf) primitives, then evaluates the rational
// directly using Horner's method on std::complex<double>.
//
// Continuous: H(jω) = num(jω) / den(jω)
// Discrete  : H(e^{jωTs}) = num(z) / den(z) at z = exp(jωTs)

#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/control/freq/freq.hpp>
#include <numkit/control/conversion/conversion.hpp>
#include <numkit/control/props/props.hpp>

#include <numkit/builtin/polyfun.hpp>

// Compute-only TU: Value substrate + Error, no engine. The evalfr/freqresp/
// bode/nyquist/rlocus builtins (CallContext wrappers) live in freq_reg.cpp.
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <algorithm>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace numkit::control {

namespace {

bool hasKind(const Value &sys, const char *want) {
    if (!sys.isStruct() || !sys.hasField("kind")) return false;
    return sys.field("kind").toString() == want;
}

keel::f64 sampleTime(const Value &sys) {
    if (sys.isStruct() && sys.hasField("Ts"))
        return sys.field("Ts").toScalar();
    return 0.0;
}

keel::Array<keel::f64> coeffsReal(const Value &v) {
    keel::Array<keel::f64> out(v.numel());
    for (keel::usize i = 0; i < v.numel(); ++i) out[i] = v.elemAsDouble(i);
    return out;
}

struct NumDen { keel::Array<keel::f64> num, den; };

NumDen toNumDen(const Value &sys, keel::Allocator *mr) {
    if (hasKind(sys, "tf"))
        return {coeffsReal(sys.field("num")), coeffsReal(sys.field("den"))};
    if (hasKind(sys, "zpk")) {
        auto [num, den] = zp2tf(sys.field("z"), sys.field("p"), sys.field("k"), mr);
        return {coeffsReal(num), coeffsReal(den)};
    }
    if (hasKind(sys, "ss")) {
        auto [num, den] = ss2tf(sys.field("A"), sys.field("B"),
                                 sys.field("C"), sys.field("D"), /*iu=*/1, mr);
        return {coeffsReal(num), coeffsReal(den)};
    }
    throw Error("control freq: expected an LTI struct (tf/zpk/ss)",
                0, 0, "freq", "", "numkit:control:kind");
}

// Horner evaluation of a polynomial whose coefficients are stored in
// MATLAB convention (highest power first).
keel::c64 hornerCx(keel::ArrayRef<keel::f64> p, keel::c64 x) {
    keel::c64 y(0.0, 0.0);
    for (keel::f64 c : p) y = y * x + c;
    return y;
}

// H(s) at s for the given (num, den).
keel::c64 hAt(keel::ArrayRef<keel::f64> num, keel::ArrayRef<keel::f64> den, keel::c64 s) {
    keel::c64 n = hornerCx(num, s);
    keel::c64 d = hornerCx(den, s);
    return n / d;
}

// Default log-spaced ω grid spanning ~ 2 decades around the
// pole/zero magnitudes. For discrete systems the upper bound is the
// Nyquist frequency π / Ts.
keel::Array<keel::f64> pickW(const Value &sys, keel::usize N, keel::Allocator *mr)
{
    Value pV = pole(sys, mr);
    Value zV = zero(sys, mr);
    auto magsOf = [](const Value &v) {
        keel::Array<keel::f64> mags;
        const keel::usize k = v.numel();
        if (v.type() == ValueType::COMPLEX) {
            const keel::c64 *src = v.complexData();
            for (keel::usize i = 0; i < k; ++i) {
                const keel::f64 m = keel::abs(src[i]);
                if (m > 0.0) mags.push_back(m);
            }
        } else {
            for (keel::usize i = 0; i < k; ++i) {
                const keel::f64 m = keel::abs(v.elemAsDouble(i));
                if (m > 0.0) mags.push_back(m);
            }
        }
        return mags;
    };
    auto pm = magsOf(pV);
    auto zm = magsOf(zV);
    pm.append(zm.as_ref());

    keel::f64 lo = 1e-2, hi = 1e2;
    if (!pm.empty()) {
        keel::f64 minM = pm[0], maxM = pm[0];
        for (keel::f64 m : pm) {
            if (m < minM) minM = m;
            if (m > maxM) maxM = m;
        }
        lo = minM / 100.0;
        hi = maxM * 100.0;
    }
    const keel::f64 Ts = sampleTime(sys);
    if (Ts > 0.0) {
        // Discrete: clamp to (0, π/Ts).
        const keel::f64 nyq = M_PI / Ts;
        if (hi > nyq) hi = nyq;
        if (lo >= hi) lo = hi / 1000.0;
    }
    if (lo <= 0.0) lo = 1e-3;
    if (hi <= lo) hi = lo * 100.0;

    keel::Array<keel::f64> w(N);
    const keel::f64 a = std::log10(lo);
    const keel::f64 b = std::log10(hi);
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 t = (N == 1) ? 0.0 : keel::f64(i) / keel::f64(N - 1);
        w[i] = std::pow(10.0, a + (b - a) * t);
    }
    return w;
}

keel::Array<keel::f64> readW(const Value &sys, const Value &wArg, keel::Allocator *mr)
{
    if (wArg.numel() == 0) return pickW(sys, /*N=*/200, mr);
    keel::Array<keel::f64> w(wArg.numel());
    for (keel::usize i = 0; i < wArg.numel(); ++i) w[i] = wArg.elemAsDouble(i);
    return w;
}

Value colDouble(keel::ArrayRef<keel::f64> v, keel::Allocator *mr) {
    Value r = Value::matrix(v.size(), 1, ValueType::DOUBLE, mr);
    if (!v.empty()) std::copy(v.begin(), v.end(), r.doubleDataMut());
    return r;
}

Value colComplex(keel::ArrayRef<keel::c64> v, keel::Allocator *mr) {
    Value r = Value::matrix(v.size(), 1, ValueType::COMPLEX, mr);
    if (!v.empty()) {
        keel::c64 *dst = r.complexDataMut();
        std::copy(v.begin(), v.end(), dst);
    }
    return r;
}

// Map ω → s (continuous) or z (discrete).
keel::c64 freqArg(keel::f64 w, keel::f64 Ts) {
    if (Ts > 0.0) return keel::polar(1.0, w * Ts);   // exp(jωTs)
    return keel::c64(0.0, w);                               // jω
}

void freq_sys_argcheck(const Value &sys)
{
    if (!hasKind(sys, "tf") && !hasKind(sys, "zpk") && !hasKind(sys, "ss"))
        throw Error("control freq: expected an LTI struct (tf/zpk/ss)",
                    0, 0, "freq", "", "numkit:control:kind");
}

void evalfr_argcheck(const Value &sys, keel::f64 f)
{
    (void)f;
    freq_sys_argcheck(sys);
}

void freqresp_argcheck(const Value &sys, const Value &w)
{
    (void)w;
    freq_sys_argcheck(sys);
}

void bode_argcheck(const Value &sys, const Value &wArg)
{
    (void)wArg;
    freq_sys_argcheck(sys);
}

void nyquist_argcheck(const Value &sys, const Value &wArg)
{
    (void)wArg;
    freq_sys_argcheck(sys);
}

void rlocus_argcheck(const Value &sys, const Value &kArg)
{
    (void)kArg;
    freq_sys_argcheck(sys);
}

} // anonymous

Value evalfr(const Value &sys, keel::f64 f, keel::Allocator *mr)
{
    evalfr_argcheck(sys, f);
    auto nd = toNumDen(sys, mr);
    const keel::f64 Ts = sampleTime(sys);
    const keel::c64 h = hAt(nd.num, nd.den, freqArg(f, Ts));
    Value out = Value::matrix(1, 1, ValueType::COMPLEX, mr);
    out.complexDataMut()[0] = h;
    return out;
}

Value freqresp(const Value &sys, const Value &w, keel::Allocator *mr)
{
    freqresp_argcheck(sys, w);
    auto nd = toNumDen(sys, mr);
    const keel::f64 Ts = sampleTime(sys);
    auto wv = readW(sys, w, mr);
    keel::Array<keel::c64> h(wv.size());
    for (keel::usize i = 0; i < wv.size(); ++i)
        h[i] = hAt(nd.num, nd.den, freqArg(wv[i], Ts));
    return colComplex(h, mr);
}

BodeResult bode(const Value &sys, const Value &wArg, keel::Allocator *mr)
{
    bode_argcheck(sys, wArg);
    auto nd = toNumDen(sys, mr);
    const keel::f64 Ts = sampleTime(sys);
    auto w = readW(sys, wArg, mr);
    keel::Array<keel::f64> mag(w.size()), phase(w.size());

    keel::f64 phi0 = 0.0;
    bool hasPhi0 = false;
    if (Ts == 0.0 && !w.empty() && w[0] >= 0.0) {
        auto firstNonZero = [](keel::ArrayRef<keel::f64> v, keel::usize &idx) -> keel::f64 {
            for (keel::usize i = 0; i < v.size(); ++i) {
                if (v[i] != 0.0) { idx = i; return v[i]; }
            }
            idx = v.size();
            return 0.0;
        };
        keel::usize in = 0, id = 0;
        const keel::f64 b0 = firstNonZero(nd.num, in);
        const keel::f64 a0 = firstNonZero(nd.den, id);
        if (b0 != 0.0 && a0 != 0.0) {
            auto polyPhaseAt = [&](keel::ArrayRef<keel::f64> c, keel::usize start, keel::f64 w0) -> keel::f64 {
                keel::usize end = c.size();
                keel::usize q = 0;
                while (end > start + 1 && c[end - 1] == 0.0) {
                    --end;
                    ++q;
                }
                keel::f64 ang = keel::as<keel::f64>(q) * 90.0;
                const keel::usize len = end - start;
                if (len <= 1) return ang;
                Value pv = Value::matrix(1, len, ValueType::DOUBLE, mr);
                std::copy(c.begin() + start, c.begin() + end, pv.doubleDataMut());
                Value rv = numkit::builtin::roots(pv, mr);
                const keel::usize nr = rv.numel();
                for (keel::usize k = 0; k < nr; ++k) {
                    keel::c64 r = (rv.type() == ValueType::COMPLEX)
                        ? rv.complexData()[k]
                        : keel::c64(rv.elemAsDouble(k), 0.0);
                    if (keel::abs(r.imag()) <= 1e-9 * std::max(1.0, keel::abs(r.real()))) {
                        ang += std::atan2(w0, -r.real()) * (180.0 / M_PI);
                    } else if (r.imag() > 0.0) {
                        const keel::f64 mag2 = keel::norm(r);
                        ang += std::atan2(-2.0 * r.real() * w0, mag2 - w0 * w0) * (180.0 / M_PI);
                    }
                }
                return ang;
            };
            const keel::f64 kGain = b0 / a0;
            phi0 = (kGain < 0.0 ? 180.0 : 0.0)
                 + polyPhaseAt(nd.num, in, w[0])
                 - polyPhaseAt(nd.den, id, w[0]);
            hasPhi0 = true;
        }
    }

    keel::f64 prev = 0.0;
    for (keel::usize i = 0; i < w.size(); ++i) {
        keel::c64 h = hAt(nd.num, nd.den, freqArg(w[i], Ts));
        mag[i] = keel::abs(h);
        // Unwrap phase to avoid 2π jumps.
        keel::f64 ph = keel::arg(h) * (180.0 / M_PI);
        if (i == 0 && hasPhi0) {
            while (ph - phi0 > 180.0) ph -= 360.0;
            while (ph - phi0 < -180.0) ph += 360.0;
        } else if (i > 0) {
            while (ph - prev > 180.0) ph -= 360.0;
            while (ph - prev < -180.0) ph += 360.0;
        }
        phase[i] = ph;
        prev = ph;
    }
    return {colDouble(mag, mr),
            colDouble(phase, mr),
            colDouble(w, mr)};
}

NyquistResult nyquist(const Value &sys, const Value &wArg, keel::Allocator *mr)
{
    nyquist_argcheck(sys, wArg);
    auto nd = toNumDen(sys, mr);
    const keel::f64 Ts = sampleTime(sys);
    auto w = readW(sys, wArg, mr);
    keel::Array<keel::f64> re(w.size()), im(w.size());
    for (keel::usize i = 0; i < w.size(); ++i) {
        keel::c64 h = hAt(nd.num, nd.den, freqArg(w[i], Ts));
        re[i] = h.real();
        im[i] = h.imag();
    }
    return {colDouble(re, mr), colDouble(im, mr), colDouble(w, mr)};
}

std::pair<Value, Value>
rlocus(const Value &sys, const Value &kArg, keel::Allocator *mr)
{
    rlocus_argcheck(sys, kArg);
    auto nd = toNumDen(sys, mr);
    // Strip any leading-zero padding so the polynomial sums align on
    // the trailing (constant) term.
    auto strip = [](keel::ArrayRef<keel::f64> v) {
        keel::usize i = 0;
        while (i + 1 < v.size() && v[i] == 0.0) ++i;
        return keel::Array<keel::f64>::from(v.slice(i));
    };
    auto num = strip(nd.num);
    auto den = strip(nd.den);
    if (den.empty())
        throw Error("rlocus: denominator must be non-empty",
                    0, 0, "rlocus", "", "numkit:rlocus:den");
    const keel::usize n = den.size() - 1;   // closed-loop order

    // Build gain vector.
    keel::Array<keel::f64> ks;
    if (kArg.numel() == 0) {
        // 0 + 100 log-spaced points from 1e-2 to 1e3.
        ks.reserve(101);
        ks.push_back(0.0);
        const keel::f64 a = -2.0, b = 3.0;
        for (keel::usize i = 0; i < 100; ++i) {
            const keel::f64 t = keel::f64(i) / 99.0;
            ks.push_back(std::pow(10.0, a + (b - a) * t));
        }
    } else {
        ks.resize(kArg.numel());
        for (keel::usize i = 0; i < kArg.numel(); ++i) ks[i] = kArg.elemAsDouble(i);
    }

    const keel::usize K = ks.size();
    // Allocate K × n complex matrix.
    Value R = Value::matrix(K, n, ValueType::COMPLEX, mr);
    if (n > 0 && K > 0) {
        keel::c64 *rd = R.complexDataMut();

        // Right-align num with den so adding den + k·num is a per-coefficient
        // sum. den has length n+1 (after strip), num has length numNumel ≤ n+1.
        keel::Array<keel::f64> numAligned(den.size(), 0.0);
        const keel::usize pad = den.size() - num.size();
        for (keel::usize i = 0; i < num.size(); ++i)
            numAligned[pad + i] = num[i];

        keel::Array<keel::f64> charPoly(den.size(), 0.0);
        for (keel::usize row = 0; row < K; ++row) {
            const keel::f64 k = ks[row];
            for (keel::usize i = 0; i < den.size(); ++i)
                charPoly[i] = den[i] + k * numAligned[i];
            // Pack as a row Value and run numkit::builtin::roots.
            Value cp = Value::matrix(1, charPoly.size(),
                                     ValueType::DOUBLE, mr);
            std::copy(charPoly.begin(), charPoly.end(),
                      cp.doubleDataMut());
            Value rs = numkit::builtin::roots(cp, mr);
            // rs is a column of length n (for this n-th order polynomial).
            // Pack into row `row` of R (column-major: R[row, j] is at
            // index j*K + row).
            const keel::usize M = rs.numel();
            for (keel::usize j = 0; j < n; ++j) {
                keel::c64 val(0.0, 0.0);
                if (j < M) {
                    if (rs.type() == ValueType::COMPLEX)
                        val = rs.complexData()[j];
                    else
                        val = keel::c64(rs.elemAsDouble(j), 0.0);
                }
                rd[j * K + row] = val;
            }
        }
    }
    return {std::move(R), colDouble(ks, mr)};
}

} // namespace numkit::control
