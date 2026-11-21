// toolboxes/control/src/conversion/conversion.cpp
//
// Inter-form conversions tf <-> zpk <-> ss and minreal delegating numerical
// kernels to keel::control (tf2ss, ss2tf, minreal_tf).

#include <keel/array.h>
#include <keel/complex.h>
#include <keel/control.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <numkit/control/conversion/conversion.hpp>

#include <numkit/builtin/polyfun.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <algorithm>
#include <utility>

namespace numkit::control {

namespace {

keel::Array<keel::f64> coeffsReal(const Value &v, keel::Allocator *mr) {
    keel::Array<keel::f64> out(v.numel(), 0.0, mr);
    if (v.type() == ValueType::COMPLEX) {
        const keel::c64 *c = v.complexData();
        for (keel::usize i = 0; i < v.numel(); ++i) out[i] = c[i].real();
    } else {
        for (keel::usize i = 0; i < v.numel(); ++i) out[i] = v.elemAsDouble(i);
    }
    return out;
}

keel::Array<keel::f64> stripLeading(keel::ArrayRef<keel::f64> v, keel::Allocator *mr) {
    keel::usize i = 0;
    while (i + 1 < v.size() && v[i] == 0.0) ++i;
    keel::Array<keel::f64> out(v.size() - i, 0.0, mr);
    for (keel::usize k = 0; k < out.size(); ++k) out[k] = v[i + k];
    return out;
}

Value rowOfDoubles(keel::ArrayRef<keel::f64> v, keel::Allocator *mr) {
    Value r = Value::matrix(1, v.size(), ValueType::DOUBLE, mr);
    if (!v.empty()) std::copy(v.begin(), v.end(), r.doubleDataMut());
    return r;
}

Value matrixFromVec(keel::usize rows, keel::usize cols, keel::ArrayRef<keel::f64> v,
                    keel::Allocator *mr) {
    Value m = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (!v.empty()) std::copy(v.begin(), v.end(), m.doubleDataMut());
    return m;
}

void tf2zp_argcheck(const Value &num, const Value &den,
                    keel::Array<keel::f64> &numV, keel::Array<keel::f64> &denV,
                    keel::Allocator *mr) {
    const auto nr = coeffsReal(num, mr);
    const auto dr = coeffsReal(den, mr);
    numV = stripLeading(nr.as_ref(), mr);
    denV = stripLeading(dr.as_ref(), mr);
    if (denV.empty() || denV[0] == 0.0)
        throw Error("tf2zp: denominator must have a nonzero leading coefficient",
                    0, 0, "tf2zp", "", "numkit:tf2zp:den");
}

void tf2ss_argcheck(const Value &num, const Value &den,
                    keel::Array<keel::f64> &numV, keel::Array<keel::f64> &denV,
                    keel::Allocator *mr) {
    const auto dr = coeffsReal(den, mr);
    denV = stripLeading(dr.as_ref(), mr);
    if (denV.empty() || denV[0] == 0.0)
        throw Error("tf2ss: invalid denominator",
                    0, 0, "tf2ss", "", "numkit:tf2ss:den");
    numV = coeffsReal(num, mr);
    if (numV.size() > denV.size())
        throw Error("tf2ss: numerator longer than denominator (improper system)",
                    0, 0, "tf2ss", "", "numkit:tf2ss:improper");
}

void ss2tf_argcheck(const Value &A, const Value &B, const Value &C, keel::i32 iu) {
    const keel::usize n = A.dims().rows();
    if (A.dims().cols() != n)
        throw Error("ss2tf: A must be square",
                    0, 0, "ss2tf", "", "numkit:ss2tf:A");
    const keel::usize nin = (B.dims().cols() == 0) ? 1 : B.dims().cols();
    if (iu < 1 || keel::as<keel::usize>(iu) > nin)
        throw Error("ss2tf: iu out of range",
                    0, 0, "ss2tf", "", "numkit:ss2tf:iu");
    if (C.dims().rows() != 1)
        throw Error("ss2tf: only single-output systems supported "
                    "(got C with rows != 1)",
                    0, 0, "ss2tf", "", "numkit:ss2tf:miso");
}

bool isKind(const Value &sys, const char *want) {
    return sys.isStruct() && sys.hasField("kind") &&
           sys.field("kind").toString() == want;
}

keel::f64 sampleTimeOf(const Value &sys) {
    return (sys.isStruct() && sys.hasField("Ts")) ? sys.field("Ts").toScalar() : 0.0;
}

Value taggedStruct(const char *kind, keel::f64 Ts, keel::Allocator *mr) {
    Value s = Value::structure(mr);
    s.field("kind") = Value::fromString(kind, mr);
    s.field("Ts") = Value::scalar(Ts, mr);
    return s;
}

std::pair<Value, Value>
cancelTf(const Value &numV, const Value &denV, keel::f64 tol, keel::Allocator *mr) {
    const auto numR = coeffsReal(numV, mr);
    const auto denR = coeffsReal(denV, mr);
    keel::Array<keel::f64> nn, dd;
    if (keel::control::minreal_tf(numR.as_ref(), denR.as_ref(), tol, nn, dd, mr) !=
        keel::control::ControlStatus::kOk) {
        throw Error("minreal: denominator is identically zero",
                    0, 0, "minreal", "", "numkit:minreal:den");
    }
    return {rowOfDoubles(nn.as_ref(), mr), rowOfDoubles(dd.as_ref(), mr)};
}

void minreal_argcheck(const Value &sys) {
    if (!isKind(sys, "tf") && !isKind(sys, "zpk") && !isKind(sys, "ss"))
        throw Error("minreal: expected an LTI struct (tf / zpk / ss)",
                    0, 0, "minreal", "", "numkit:minreal:kind");
    if (isKind(sys, "ss")) {
        const Value &B = sys.field("B");
        const Value &C = sys.field("C");
        if (C.dims().rows() != 1 || B.dims().cols() != 1)
            throw Error("minreal: only SISO state-space is supported",
                        0, 0, "minreal", "", "numkit:minreal:mimo");
    }
}

} // anonymous

Tf2ZpResult tf2zp(const Value &num, const Value &den, keel::Allocator *mr) {
    keel::Array<keel::f64> numV, denV;
    tf2zp_argcheck(num, den, numV, denV, mr);

    const keel::f64 k = numV.empty() ? 0.0 : numV[0] / denV[0];
    Value zRoots = numkit::builtin::roots(num, mr);
    Value pRoots = numkit::builtin::roots(den, mr);

    return {std::move(zRoots), std::move(pRoots), Value::scalar(k, mr)};
}

std::pair<Value, Value>
zp2tf(const Value &z, const Value &p, const Value &k, keel::Allocator *mr) {
    Value den = numkit::builtin::poly(p, mr);
    Value num = numkit::builtin::poly(z, mr);

    if (num.numel() == 0) {
        num = Value::matrix(1, 1, ValueType::DOUBLE, mr);
        num.doubleDataMut()[0] = 1.0;
    }

    const keel::f64 kVal = k.toScalar();
    if (num.type() == ValueType::COMPLEX) {
        const keel::usize N = num.numel();
        Value scaled = Value::matrix(1, N, ValueType::COMPLEX, mr);
        const keel::c64 *src = num.complexData();
        keel::c64 *dst = scaled.complexDataMut();
        for (keel::usize i = 0; i < N; ++i) dst[i] = src[i] * kVal;
        num = scaled;
    } else {
        const keel::usize N = num.numel();
        Value scaled = Value::matrix(1, N, ValueType::DOUBLE, mr);
        keel::f64 *dst = scaled.doubleDataMut();
        for (keel::usize i = 0; i < N; ++i) dst[i] = num.elemAsDouble(i) * kVal;
        num = scaled;
    }

    return {std::move(num), std::move(den)};
}

StateSpace tf2ss(const Value &num, const Value &den, keel::Allocator *mr) {
    keel::Array<keel::f64> numV, denV;
    tf2ss_argcheck(num, den, numV, denV, mr);

    keel::control::StateSpace res;
    (void)keel::control::tf2ss(numV.as_ref(), denV.as_ref(), res, mr);
    return {matrixFromVec(res.n, res.n, res.a.as_ref(), mr),
            matrixFromVec(res.n, 1, res.b.as_ref(), mr),
            matrixFromVec(1, res.n, res.c.as_ref(), mr),
            Value::scalar(res.d, mr)};
}

std::pair<Value, Value>
ss2tf(const Value &A, const Value &B,
      const Value &C, const Value &D, keel::i32 iu,
      keel::Allocator *mr) {
    ss2tf_argcheck(A, B, C, iu);
    const keel::usize n = A.dims().rows();
    const keel::usize nin = (B.dims().cols() == 0) ? 1 : B.dims().cols();
    const keel::usize inIdx = keel::as<keel::usize>(iu - 1);

    keel::Array<keel::f64> Av(n * n, 0.0, mr);
    for (keel::usize i = 0; i < n * n; ++i) Av[i] = A.elemAsDouble(i);
    keel::Array<keel::f64> Bv(n * nin, 0.0, mr);
    for (keel::usize j = 0; j < nin; ++j)
        for (keel::usize i = 0; i < n; ++i)
            Bv[j * n + i] = (j < B.dims().cols()) ? B.elemAsDouble(j * n + i) : 0.0;
    keel::Array<keel::f64> Cv(n, 0.0, mr);
    for (keel::usize i = 0; i < n; ++i) Cv[i] = C.elemAsDouble(i);

    keel::f64 Dscalar = 0.0;
    if (D.numel() > 0) {
        const keel::usize Drows = (D.dims().rows() == 0) ? 1 : D.dims().rows();
        Dscalar = D.elemAsDouble(inIdx * Drows);
    }

    keel::Array<keel::f64> numOut, denOut;
    (void)keel::control::ss2tf(Av.as_ref(), Bv.as_ref(), Cv.as_ref(), Dscalar,
                               n, nin, inIdx, numOut, denOut, mr);
    return {rowOfDoubles(numOut.as_ref(), mr), rowOfDoubles(denOut.as_ref(), mr)};
}

Value minreal(const Value &sys, keel::f64 tol, keel::Allocator *mr) {
    minreal_argcheck(sys);
    if (isKind(sys, "tf")) {
        auto [n2, d2] = cancelTf(sys.field("num"), sys.field("den"), tol, mr);
        Value s = taggedStruct("tf", sampleTimeOf(sys), mr);
        s.field("num") = std::move(n2);
        s.field("den") = std::move(d2);
        return s;
    }
    if (isKind(sys, "zpk")) {
        auto [num, den] = zp2tf(sys.field("z"), sys.field("p"), sys.field("k"), mr);
        auto [n2, d2] = cancelTf(num, den, tol, mr);
        Value s = taggedStruct("tf", sampleTimeOf(sys), mr);
        s.field("num") = std::move(n2);
        s.field("den") = std::move(d2);
        return s;
    }
    if (isKind(sys, "ss")) {
        const Value &B = sys.field("B");
        const Value &C = sys.field("C");
        auto [num, den] = ss2tf(sys.field("A"), B, C, sys.field("D"), 1, mr);
        auto [n2, d2] = cancelTf(num, den, tol, mr);
        StateSpace r = tf2ss(n2, d2, mr);
        Value s = taggedStruct("ss", sampleTimeOf(sys), mr);
        s.field("A") = std::move(r.A);
        s.field("B") = std::move(r.B);
        s.field("C") = std::move(r.C);
        s.field("D") = std::move(r.D);
        return s;
    }
    throw Error("minreal: expected an LTI struct (tf / zpk / ss)",
                0, 0, "minreal", "", "numkit:minreal:kind");
}

} // namespace numkit::control
