// toolboxes/control/src/analyze/analyze.cpp
//
// dcgain  — H(0) (continuous) or H(1) (discrete) via evalfr.
// margin  — gain & phase margins from a dense bode grid + linear
//           interpolation around the crossover frequencies.
// stepinfo — rise/settling/peak metrics computed off the existing
//           step response.

#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <keel/string.h>
#include <numkit/control/analyze/analyze.hpp>
#include <numkit/control/freq/freq.hpp>
#include <numkit/control/response/response.hpp>
#include <numkit/control/conversion/conversion.hpp>   // allmargin: open-loop num/den
#include <numkit/builtin/polyfun.hpp>            // allmargin: roots for Stable

// Compute-only TU: Value substrate + Error, no engine. The dcgain/margin/
// stepinfo builtins (CallContext wrappers) live in analyze_reg.cpp.
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace numkit::control {

namespace {

[[maybe_unused]] keel::f64 sampleTime(const Value &sys) {
    if (sys.isStruct() && sys.hasField("Ts"))
        return sys.field("Ts").toScalar();
    return 0.0;
}

keel::Array<keel::f64> readCol(const Value &v) {
    keel::Array<keel::f64> out(v.numel());
    for (keel::usize i = 0; i < v.numel(); ++i) out[i] = v.elemAsDouble(i);
    return out;
}

// Find the first index k where signal `s[k]` crosses level `lvl`
// (linear interpolation between samples). Returns -1 if no crossing.
struct CrossResult { bool found; keel::f64 w; keel::f64 yAtCross; };

CrossResult crossAt(keel::ArrayRef<keel::f64> x,
                    keel::ArrayRef<keel::f64> y,
                    keel::f64 lvl)
{
    for (keel::usize i = 1; i < y.size(); ++i) {
        const keel::f64 a = y[i - 1] - lvl;
        const keel::f64 b = y[i] - lvl;
        if (a == 0.0)
            return {true, x[i - 1], y[i - 1]};
        if ((a < 0.0) != (b < 0.0)) {
            const keel::f64 t = a / (a - b);   // 0..1
            const keel::f64 w = x[i - 1] + t * (x[i] - x[i - 1]);
            const keel::f64 yv = y[i - 1] + t * (y[i] - y[i - 1]);
            return {true, w, yv};
        }
    }
    return {false, 0.0, 0.0};
}

// Interpolate y at x = x0 (assumes x sorted ascending).
keel::f64 interpAt(keel::ArrayRef<keel::f64> x,
                keel::ArrayRef<keel::f64> y,
                keel::f64 x0)
{
    if (x.empty()) return 0.0;
    if (x0 <= x.front()) return y.front();
    if (x0 >= x.back()) return y.back();
    auto it = std::upper_bound(x.begin(), x.end(), x0);
    const keel::usize j = it - x.begin();
    const keel::f64 t = (x0 - x[j - 1]) / (x[j] - x[j - 1]);
    return y[j - 1] + t * (y[j] - y[j - 1]);
}

void dcgain_argcheck(const Value &sys) {
    if (sys.isStruct() && sys.hasField("kind")) {
        const keel::String k = sys.field("kind").toString();
        if (k == "tf" || k == "zpk" || k == "ss") return;
    }
    throw Error("control freq: expected an LTI struct (tf/zpk/ss)",
                0, 0, "freq", "", "numkit:control:kind");
}

void margin_argcheck(const Value &sys) {
    dcgain_argcheck(sys);
}

void allmargin_argcheck(const Value &sys) {
    if (sys.isStruct() && sys.hasField("kind")) {
        const keel::String k = sys.field("kind").toString();
        if (k == "tf" || k == "zpk" || k == "ss") return;
    }
    throw Error("allmargin: expected an LTI struct (tf/zpk/ss)",
                0, 0, "allmargin", "", "numkit:allmargin:kind");
}

void stepinfo_argcheck(const Value &sys) {
    if (sys.isStruct() && sys.hasField("kind")) {
        const keel::String k = sys.field("kind").toString();
        if (k == "tf" || k == "zpk" || k == "ss") return;
        throw Error(keel::format("step: input 1 (sys) has unknown LTI kind '{}'", k),
                    0, 0, "step", "", "numkit:step:kind");
    }
    throw Error("step: input 1 (sys) must be an LTI struct (tf/zpk/ss)",
                0, 0, "step", "", "numkit:step:kind");
}

} // anonymous

Value dcgain(const Value &sys, keel::Allocator *mr)
{
    dcgain_argcheck(sys);
    // evalfr already does the right thing: at f=0 it evaluates s = j0
    // = 0 for continuous, and z = exp(j·0·Ts) = 1 for discrete.
    return evalfr(sys, 0.0, mr);
}

MarginResult margin(const Value &sys, keel::Allocator *mr)
{
    margin_argcheck(sys);
    // Build a dense default Bode grid (the auto-pick chooses ~200
    // points spanning two decades around the dominant poles).
    Value emptyW = Value::matrix(0, 0, ValueType::DOUBLE, mr);
    auto bodeOut = bode(sys, emptyW, mr);

    auto w = readCol(bodeOut.w);
    auto mag = readCol(bodeOut.mag);
    auto phase = readCol(bodeOut.phase);

    // Phase crossover (where phase = -180°). On a typical loop the
    // phase descends through -180; we scan from low to high frequency.
    auto pc = crossAt(w, phase, -180.0);
    // Gain crossover (where |H| = 1, equivalently 0 dB). We scan the
    // *log* magnitude so a linear interpolation matches the bode plot.
    keel::Array<keel::f64> magLog(mag.size());
    for (keel::usize i = 0; i < mag.size(); ++i)
        magLog[i] = (mag[i] > 0.0) ? 20.0 * std::log10(mag[i])
                                   : -300.0;
    auto gc = crossAt(w, magLog, 0.0);

    const keel::f64 inf = std::numeric_limits<keel::f64>::infinity();
    keel::f64 Gm = inf, Pm = inf, Wcg = inf, Wcp = inf;
    if (pc.found) {
        Wcg = pc.w;
        const keel::f64 magAtPc = interpAt(w, mag, Wcg);
        Gm = (magAtPc > 0.0) ? 1.0 / magAtPc : inf;
    }
    if (gc.found) {
        Wcp = gc.w;
        const keel::f64 phAtGc = interpAt(w, phase, Wcp);
        Pm = phAtGc + 180.0;
    }

    return {Value::scalar(Gm,  mr),
            Value::scalar(Pm,  mr),
            Value::scalar(Wcg, mr),
            Value::scalar(Wcp, mr)};
}

namespace {

// Open-loop (num, den) of an LTI struct as descending-power coefficient rows.
std::pair<keel::Array<keel::f64>, keel::Array<keel::f64>>
openLoopNumDen(const Value &sys, keel::Allocator *mr) {
    auto coeffs = [](const Value &v) {
        keel::Array<keel::f64> c(v.numel());
        for (keel::usize i = 0; i < v.numel(); ++i) c[i] = v.elemAsDouble(i);
        return c;
    };
    if (sys.isStruct() && sys.hasField("kind")) {
        const keel::String k = sys.field("kind").toString();
        if (k == "tf")
            return {coeffs(sys.field("num")), coeffs(sys.field("den"))};
        if (k == "zpk") {
            auto [num, den] = zp2tf(sys.field("z"), sys.field("p"), sys.field("k"), mr);
            return {coeffs(num), coeffs(den)};
        }
        if (k == "ss") {
            auto [num, den] = ss2tf(sys.field("A"), sys.field("B"),
                                    sys.field("C"), sys.field("D"), 1, mr);
            return {coeffs(num), coeffs(den)};
        }
    }
    throw Error("allmargin: expected an LTI struct (tf/zpk/ss)",
                0, 0, "allmargin", "", "numkit:allmargin:kind");
}

Value rowFrom(keel::ArrayRef<keel::f64> v, keel::Allocator *mr) {
    Value r = Value::matrix(1, v.size(), ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < v.size(); ++i) r.doubleDataMut()[i] = v[i];
    return r;
}

// Horner evaluation of a descending-power real polynomial at a complex point.
keel::c64 polyvalC(keel::ArrayRef<keel::f64> c, keel::c64 x) {
    keel::c64 r(0.0, 0.0);
    for (keel::f64 a : c) r = r * x + a;
    return r;
}

// Exact open-loop frequency response G(jω) = num(jω) / den(jω).
keel::c64 evalG(keel::ArrayRef<keel::f64> num,
                keel::ArrayRef<keel::f64> den, keel::f64 w) {
    const keel::c64 jw(0.0, w);
    const keel::c64 d = polyvalC(den, jw);
    if (d == keel::c64(0.0, 0.0))
        return {std::numeric_limits<keel::f64>::infinity(), 0.0};
    return polyvalC(num, jw) / d;
}

// Bisect a scalar f(ω) for its root in [a,b] (sign change assumed), ~60 iters.
template <class F>
keel::f64 bisectRoot(F f, keel::f64 a, keel::f64 b) {
    keel::f64 fa = f(a);
    for (keel::i32 it = 0; it < 80; ++it) {
        const keel::f64 m = 0.5 * (a + b);
        const keel::f64 fm = f(m);
        if (fm == 0.0 || (b - a) <= 1e-13 * (1.0 + b)) return m;
        if ((fa < 0.0) != (fm < 0.0)) b = m;
        else { a = m; fa = fm; }
    }
    return 0.5 * (a + b);
}

} // anonymous

Value allmargin(const Value &sys, keel::Allocator *mr)
{
    allmargin_argcheck(sys);
    // Exact open-loop response G(jω) = num(jω)/den(jω) (no bode-grid
    // interpolation): scan a fine log-frequency grid for sign changes of
    // |G|−1 (gain crossovers) and Im(G) with Re(G)<0 (phase crossovers),
    // then bisect each bracket on the exact response.
    auto [num, den] = openLoopNumDen(sys, mr);

    const keel::f64 inf = std::numeric_limits<keel::f64>::infinity();
    const keel::f64 PI = 3.14159265358979323846;

    // Frequency span from the pole/zero magnitudes (±3 decades).
    keel::f64 wlo = 1e-3, whi = 1e3;
    {
        keel::Array<keel::f64> mags;
        auto addRoots = [&](keel::ArrayRef<keel::f64> c) {
            if (c.size() < 2) return;
            Value rts = numkit::builtin::roots(rowFrom(c, mr), mr);
            const keel::usize n = rts.numel();
            if (rts.type() == ValueType::COMPLEX) {
                const keel::c64 *p = rts.complexData();
                for (keel::usize i = 0; i < n; ++i) { keel::f64 m = keel::abs(p[i]); if (m > 1e-9) mags.push_back(m); }
            } else {
                for (keel::usize i = 0; i < n; ++i) { keel::f64 m = keel::abs(rts.elemAsDouble(i)); if (m > 1e-9) mags.push_back(m); }
            }
        };
        addRoots(num);
        addRoots(den);
        if (!mags.empty()) {
            wlo = *std::min_element(mags.begin(), mags.end()) * 1e-3;
            whi = *std::max_element(mags.begin(), mags.end()) * 1e3;
        }
    }

    keel::Array<keel::f64> gmVal, gmFreq, pmVal, pmFreq, dmVal, dmFreq;
    const keel::i32 N = 4000;
    const keel::f64 llo = std::log10(wlo), lhi = std::log10(whi);
    keel::f64 wPrev = wlo;
    keel::c64 gPrev = evalG(num, den, wPrev);
    for (keel::i32 i = 1; i <= N; ++i) {
        const keel::f64 w = std::pow(10.0, llo + (lhi - llo) * (keel::f64)i / N);
        const keel::c64 g = evalG(num, den, w);

        // Gain crossover: |G| − 1 changes sign → phase + delay margin.
        const keel::f64 a = keel::abs(gPrev) - 1.0, b = keel::abs(g) - 1.0;
        if (std::isfinite(a) && std::isfinite(b) && a != 0.0 && (a < 0.0) != (b < 0.0)) {
            const keel::f64 wc = bisectRoot(
                [&](keel::f64 x) { return keel::abs(evalG(num, den, x)) - 1.0; }, wPrev, w);
            const keel::c64 gc = evalG(num, den, wc);
            keel::f64 PM = keel::arg(gc) * 180.0 / PI + 180.0;
            while (PM > 180.0) PM -= 360.0;
            while (PM <= -180.0) PM += 360.0;
            pmVal.push_back(PM);
            pmFreq.push_back(wc);
            keel::f64 pmRad = PM * PI / 180.0;
            while (pmRad < 0.0) pmRad += 2.0 * PI;
            dmVal.push_back(wc > 0.0 ? pmRad / wc : inf);
            dmFreq.push_back(wc);
        }

        // Phase crossover: Im(G) changes sign and Re(G) < 0 → ∠G = ±180°.
        const keel::f64 ia = gPrev.imag(), ib = g.imag();
        if (std::isfinite(ia) && std::isfinite(ib) && ia != 0.0 && (ia < 0.0) != (ib < 0.0)) {
            const keel::f64 wc = bisectRoot(
                [&](keel::f64 x) { return evalG(num, den, x).imag(); }, wPrev, w);
            const keel::c64 gc = evalG(num, den, wc);
            if (gc.real() < 0.0) {
                const keel::f64 m = keel::abs(gc);
                gmVal.push_back(m > 0.0 ? 1.0 / m : inf);
                gmFreq.push_back(wc);
            }
        }
        wPrev = w;
        gPrev = g;
    }

    // Closed-loop (negative unity feedback) stability: roots(den + num).
    bool stable = true;
    {
        const keel::usize L = std::max(num.size(), den.size());
        keel::Array<keel::f64> cl(L, 0.0);
        for (keel::usize i = 0; i < den.size(); ++i) cl[L - den.size() + i] += den[i];
        for (keel::usize i = 0; i < num.size(); ++i) cl[L - num.size() + i] += num[i];
        keel::usize s0 = 0;
        while (s0 + 1 < cl.size() && cl[s0] == 0.0) ++s0;
        keel::ArrayRef<keel::f64> clTrim = cl.slice(s0);
        if (clTrim.size() >= 2) {
            Value rts = numkit::builtin::roots(rowFrom(clTrim, mr), mr);
            const keel::usize n = rts.numel();
            if (rts.type() == ValueType::COMPLEX) {
                const keel::c64 *p = rts.complexData();
                for (keel::usize i = 0; i < n; ++i)
                    if (p[i].real() >= -1e-9) { stable = false; break; }
            } else {
                for (keel::usize i = 0; i < n; ++i)
                    if (rts.elemAsDouble(i) >= -1e-9) { stable = false; break; }
            }
        }
    }

    Value s = Value::structure(mr);
    s.field("GainMargin")  = rowFrom(gmVal,  mr);
    s.field("GMFrequency") = rowFrom(gmFreq, mr);
    s.field("PhaseMargin") = rowFrom(pmVal,  mr);
    s.field("PMFrequency") = rowFrom(pmFreq, mr);
    s.field("DelayMargin") = rowFrom(dmVal,  mr);
    s.field("DMFrequency") = rowFrom(dmFreq, mr);
    s.field("Stable")      = Value::logicalScalar(stable, mr);
    return s;
}

Value stepinfo(const Value &sys, keel::Allocator *mr)
{
    stepinfo_argcheck(sys);
    // Use a deliberately long horizon (multiplier on top of the
    // default Tfinal pickup) so settling-time detection has enough
    // tail to register. We do that by pre-computing the default
    // grid via step(sys), then re-running on a 2× tFinal vector.
    Value emptyT = Value::matrix(0, 0, ValueType::DOUBLE, mr);
    auto [y0, t0] = step_response(sys, emptyT, /*xOut=*/nullptr, mr);
    auto t = readCol(t0);
    auto y = readCol(y0);
    if (t.size() < 2) {
        Value s = Value::structure(mr);
        return s;
    }
    const keel::f64 Tfinal = t.back();
    // Re-grid out to 2× Tfinal.
    Value tArg = Value::scalar(2.0 * Tfinal, mr);
    auto [y1, t1] = step_response(sys, tArg, /*xOut=*/nullptr, mr);
    t = readCol(t1);
    y = readCol(y1);

    const keel::f64 yfinal = y.back();
    const keel::f64 absYf = keel::abs(yfinal);
    const keel::f64 band = 0.02 * absYf;        // 2% settling band

    // Peak / peak time.
    keel::f64 peak = keel::abs(y[0]);
    keel::f64 peakTime = t[0];
    for (keel::usize i = 0; i < y.size(); ++i) {
        const keel::f64 a = keel::abs(y[i]);
        if (a > peak) { peak = a; peakTime = t[i]; }
    }

    // Overshoot / undershoot (only if yfinal != 0).
    keel::f64 overshoot = 0.0;
    keel::f64 undershoot = 0.0;
    if (yfinal > 0.0) {
        keel::f64 maxY = y[0];
        for (keel::f64 v : y) maxY = std::max(maxY, v);
        overshoot = std::max(0.0, 100.0 * (maxY - yfinal) / yfinal);
        keel::f64 minBeforePeak = 0.0;
        for (keel::f64 v : y) {
            if (v < minBeforePeak) minBeforePeak = v;
        }
        undershoot = std::max(0.0, 100.0 * (-minBeforePeak) / yfinal);
    } else if (yfinal < 0.0) {
        keel::f64 minY = y[0];
        for (keel::f64 v : y) minY = std::min(minY, v);
        overshoot = std::max(0.0, 100.0 * (minY - yfinal) / yfinal);
    }

    // Rise time: 10 % → 90 % of yfinal (sign-aware).
    keel::f64 riseTime = std::numeric_limits<keel::f64>::quiet_NaN();
    if (absYf > 0.0) {
        const keel::f64 y10 = 0.1 * yfinal;
        const keel::f64 y90 = 0.9 * yfinal;
        // Find first sample crossing y10, then first subsequent crossing y90.
        keel::f64 t10 = std::numeric_limits<keel::f64>::quiet_NaN();
        keel::f64 t90 = std::numeric_limits<keel::f64>::quiet_NaN();
        for (keel::usize i = 1; i < y.size(); ++i) {
            const keel::f64 a = y[i - 1] - y10;
            const keel::f64 b = y[i] - y10;
            if ((a < 0.0) != (b < 0.0) && std::isnan(t10)) {
                const keel::f64 tt = a / (a - b);
                t10 = t[i - 1] + tt * (t[i] - t[i - 1]);
            }
            const keel::f64 a9 = y[i - 1] - y90;
            const keel::f64 b9 = y[i] - y90;
            if ((a9 < 0.0) != (b9 < 0.0)) {
                const keel::f64 tt = a9 / (a9 - b9);
                t90 = t[i - 1] + tt * (t[i] - t[i - 1]);
                if (!std::isnan(t10)) break;
            }
        }
        if (!std::isnan(t10) && !std::isnan(t90))
            riseTime = t90 - t10;
    }

    // Settling time: last index where |y - yfinal| > band.
    keel::f64 settlingTime = 0.0;
    for (keel::usize i = 0; i < y.size(); ++i) {
        if (keel::abs(y[i] - yfinal) > band)
            settlingTime = t[i];
    }

    // Transient time (MATLAB R2025b's 2nd S field): like SettlingTime but the
    // 2% band is relative to the PEAK deviation max|y(t)-yfinal| rather than
    // |yinit-yfinal|. For a standard step the peak deviation occurs at t=0 and
    // equals |yfinal| (yinit=0), so TransientTime == SettlingTime; it can be
    // smaller when |y-yfinal| overshoots past |yfinal|.
    keel::f64 peakDev = 0.0;
    for (keel::usize i = 0; i < y.size(); ++i)
        peakDev = std::max(peakDev, keel::abs(y[i] - yfinal));
    const keel::f64 transBand = 0.02 * peakDev;
    keel::f64 transientTime = 0.0;
    for (keel::usize i = 0; i < y.size(); ++i) {
        if (keel::abs(y[i] - yfinal) > transBand)
            transientTime = t[i];
    }
    // Min/max within the settling band (after settling).
    keel::f64 sMin = yfinal, sMax = yfinal;
    for (keel::usize i = 0; i < y.size(); ++i) {
        if (t[i] >= settlingTime) {
            sMin = std::min(sMin, y[i]);
            sMax = std::max(sMax, y[i]);
        }
    }

    Value s = Value::structure(mr);
    s.field("RiseTime")      = Value::scalar(riseTime, mr);
    s.field("TransientTime") = Value::scalar(transientTime, mr);
    s.field("SettlingTime")  = Value::scalar(settlingTime, mr);
    s.field("SettlingMin")  = Value::scalar(sMin, mr);
    s.field("SettlingMax")  = Value::scalar(sMax, mr);
    s.field("Overshoot")    = Value::scalar(overshoot, mr);
    s.field("Undershoot")   = Value::scalar(undershoot, mr);
    s.field("Peak")         = Value::scalar(peak, mr);
    s.field("PeakTime")     = Value::scalar(peakTime, mr);
    return s;
}

} // namespace numkit::control
