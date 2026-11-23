// toolboxes/control/src/freq/freq.cpp
//
// Frequency-domain responses (evalfr, freqresp, bode, nyquist, rlocus)
// delegating numerical kernels to keel::control.

#include <keel/array.h>
#include <keel/algo.h>
#include <keel/complex.h>
#include <keel/control.h>
#include <keel/core.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <numkit/control/conversion/conversion.hpp>
#include <numkit/control/freq/freq.hpp>
#include <numkit/control/props/props.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <numkit/control/internal/lti_object.hpp>

namespace numkit::control {

namespace {

bool hasKind(const Value &sys, const char *want) {
    return internal::isLtiKind(sys, want);
}

keel::f64 sampleTime(const Value &sys) {
    return internal::ltiSampleTime(sys);
}

keel::Array<keel::f64> coeffsReal(const Value &v, keel::Allocator *mr) {
    keel::Array<keel::f64> out(v.numel(), 0.0, mr);
    for (keel::usize i = 0; i < v.numel(); ++i) out[i] = v.elemAsDouble(i);
    return out;
}

keel::Array<keel::c64> toComplexVec(const Value &v, keel::Allocator *mr) {
    const keel::usize k = v.numel();
    keel::Array<keel::c64> out(k, keel::c64(0.0, 0.0), mr);
    if (v.type() == ValueType::COMPLEX) {
        const keel::c64 *src = v.complexData();
        for (keel::usize i = 0; i < k; ++i) out[i] = src[i];
    } else {
        for (keel::usize i = 0; i < k; ++i) out[i] = keel::c64(v.elemAsDouble(i), 0.0);
    }
    return out;
}

struct NumDen { keel::Array<keel::f64> num, den; };

NumDen toNumDen(const Value &sys, keel::Allocator *mr) {
    const Value &d = internal::ltiStruct(sys);
    if (hasKind(sys, "tf"))
        return {coeffsReal(d.field("num"), mr), coeffsReal(d.field("den"), mr)};
    if (hasKind(sys, "zpk")) {
        auto [num, den] = zp2tf(d.field("z"), d.field("p"), d.field("k"), mr);
        return {coeffsReal(num, mr), coeffsReal(den, mr)};
    }
    if (hasKind(sys, "ss")) {
        auto [num, den] = ss2tf(d.field("A"), d.field("B"),
                                d.field("C"), d.field("D"), /*iu=*/1, mr);
        return {coeffsReal(num, mr), coeffsReal(den, mr)};
    }
    throw Error("control freq: expected an LTI model (tf/zpk/ss)",
                0, 0, "freq", "", "numkit:control:kind");
}

keel::Array<keel::f64> readW(const Value &sys, const Value &wArg, keel::Allocator *mr) {
    if (wArg.numel() == 0) {
        const auto p = toComplexVec(pole(sys, mr), mr);
        const auto z = toComplexVec(zero(sys, mr), mr);
        keel::Array<keel::f64> w(200, 0.0, mr);
        (void)keel::control::pick_freq_grid(p.as_ref(), z.as_ref(), 200, sampleTime(sys),
                                            w.as_mut_ref());
        return w;
    }
    keel::Array<keel::f64> w(wArg.numel(), 0.0, mr);
    for (keel::usize i = 0; i < wArg.numel(); ++i) w[i] = wArg.elemAsDouble(i);
    return w;
}

Value colDouble(keel::ArrayRef<keel::f64> v, keel::Allocator *mr) {
    Value r = Value::matrix(v.size(), 1, ValueType::DOUBLE, mr);
    if (!v.empty()) keel::copy(v.begin(), v.end(), r.doubleDataMut());
    return r;
}

Value colComplex(keel::ArrayRef<keel::c64> v, keel::Allocator *mr) {
    Value r = Value::matrix(v.size(), 1, ValueType::COMPLEX, mr);
    if (!v.empty()) keel::copy(v.begin(), v.end(), r.complexDataMut());
    return r;
}

void freq_sys_argcheck(const Value &sys) {
    if (!hasKind(sys, "tf") && !hasKind(sys, "zpk") && !hasKind(sys, "ss"))
        throw Error("control freq: expected an LTI struct (tf/zpk/ss)",
                    0, 0, "freq", "", "numkit:control:kind");
}

} // anonymous

Value evalfr(const Value &sys, keel::f64 f, keel::Allocator *mr) {
    freq_sys_argcheck(sys);
    const auto nd = toNumDen(sys, mr);
    const keel::c64 h = keel::control::evalfr_tf(nd.num.as_ref(), nd.den.as_ref(), f, sampleTime(sys));
    Value out = Value::matrix(1, 1, ValueType::COMPLEX, mr);
    out.complexDataMut()[0] = h;
    return out;
}

Value freqresp(const Value &sys, const Value &w, keel::Allocator *mr) {
    freq_sys_argcheck(sys);
    const auto nd = toNumDen(sys, mr);
    const auto wv = readW(sys, w, mr);
    keel::Array<keel::c64> h(wv.size(), keel::c64(0.0, 0.0), mr);
    (void)keel::control::freqresp_tf(nd.num.as_ref(), nd.den.as_ref(), wv.as_ref(),
                                     sampleTime(sys), h.as_mut_ref());
    return colComplex(h.as_ref(), mr);
}

BodeResult bode(const Value &sys, const Value &wArg, keel::Allocator *mr) {
    freq_sys_argcheck(sys);
    const auto nd = toNumDen(sys, mr);
    const auto w = readW(sys, wArg, mr);
    keel::Array<keel::f64> mag(w.size(), 0.0, mr);
    keel::Array<keel::f64> phase(w.size(), 0.0, mr);
    (void)keel::control::bode_tf(nd.num.as_ref(), nd.den.as_ref(), w.as_ref(), sampleTime(sys),
                                 mag.as_mut_ref(), phase.as_mut_ref(), mr);
    return {colDouble(mag.as_ref(), mr),
            colDouble(phase.as_ref(), mr),
            colDouble(w.as_ref(), mr)};
}

NyquistResult nyquist(const Value &sys, const Value &wArg, keel::Allocator *mr) {
    freq_sys_argcheck(sys);
    const auto nd = toNumDen(sys, mr);
    const auto w = readW(sys, wArg, mr);
    keel::Array<keel::f64> re(w.size(), 0.0, mr);
    keel::Array<keel::f64> im(w.size(), 0.0, mr);
    (void)keel::control::nyquist_tf(nd.num.as_ref(), nd.den.as_ref(), w.as_ref(), sampleTime(sys),
                                    re.as_mut_ref(), im.as_mut_ref());
    return {colDouble(re.as_ref(), mr), colDouble(im.as_ref(), mr), colDouble(w.as_ref(), mr)};
}

keel::Pair<Value, Value>
rlocus(const Value &sys, const Value &kArg, keel::Allocator *mr) {
    freq_sys_argcheck(sys);
    const auto nd = toNumDen(sys, mr);

    keel::usize d0 = 0;
    while (d0 + 1 < nd.den.size() && nd.den[d0] == 0.0) ++d0;
    if (d0 >= nd.den.size())
        throw Error("rlocus: denominator must be non-empty",
                    0, 0, "rlocus", "", "numkit:rlocus:den");
    const keel::usize n = nd.den.size() - d0 - 1;

    keel::Array<keel::f64> ks(mr);
    if (kArg.numel() == 0) {
        ks.reserve(101);
        ks.push_back(0.0);
        constexpr keel::f64 a = -2.0;
        constexpr keel::f64 b = 3.0;
        for (keel::usize i = 0; i < 100; ++i) {
            const keel::f64 frac = keel::as<keel::f64>(i) / 99.0;
            ks.push_back(keel::pow(10.0, a + (b - a) * frac));
        }
    } else {
        ks.resize(kArg.numel());
        for (keel::usize i = 0; i < kArg.numel(); ++i) ks[i] = kArg.elemAsDouble(i);
    }

    const keel::usize K = ks.size();
    Value R = Value::matrix(K, n, ValueType::COMPLEX, mr);
    if (n > 0 && K > 0) {
        (void)keel::control::rlocus_tf(nd.num.as_ref(), nd.den.as_ref(), ks.as_ref(), n,
                                       keel::MutArrayRef<keel::c64>(R.complexDataMut(), K * n), mr);
    }
    return {keel::move(R), colDouble(ks.as_ref(), mr)};
}

} // namespace numkit::control
