// toolboxes/control/src/state/state.cpp
//
// ctrb, obsv, gram, and covar delegating numerical kernels to keel::control.

#include <keel/array.h>
#include <keel/complex.h>
#include <keel/control.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/control/conversion/conversion.hpp>
#include <numkit/control/internal/type_gates.hpp>
#include <numkit/control/state/state.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <algorithm>

namespace numkit::control {

namespace {

bool hasKind(const Value &sys, const char *want) {
    return internal::isLtiKind(sys, want);
}

struct ABC { Value A, B, C, D; };
ABC pullABC(const Value &sys, const char *fn, keel::Allocator *mr) {
    const Value &d = internal::ltiStruct(sys);
    ABC out;
    if (hasKind(sys, "ss")) {
        out.A = d.field("A"); out.B = d.field("B");
        out.C = d.field("C"); out.D = d.field("D");
    } else if (hasKind(sys, "tf")) {
        auto ss = tf2ss(d.field("num"), d.field("den"), mr);
        out.A = keel::move(ss.A); out.B = keel::move(ss.B);
        out.C = keel::move(ss.C); out.D = keel::move(ss.D);
    } else if (hasKind(sys, "zpk")) {
        auto [num, den] = zp2tf(d.field("z"), d.field("p"), d.field("k"), mr);
        auto ss = tf2ss(num, den, mr);
        out.A = keel::move(ss.A); out.B = keel::move(ss.B);
        out.C = keel::move(ss.C); out.D = keel::move(ss.D);
    } else {
        throw Error(keel::format("{}: input 1 (sys) must be an LTI model (tf/zpk/ss)", fn),
                    0, 0, fn, "", keel::format("numkit:{}:kind", fn));
    }
    return out;
}

keel::Array<keel::f64> readMat(const Value &v, keel::usize r, keel::usize c, keel::Allocator *mr) {
    keel::Array<keel::f64> M(r * c, 0.0, mr);
    for (keel::usize i = 0; i < r * c; ++i) M[i] = v.elemAsDouble(i);
    return M;
}

keel::Array<keel::c64> readCMat(const Value &v, keel::usize r, keel::usize c, keel::Allocator *mr) {
    keel::Array<keel::c64> M(r * c, keel::c64(0.0, 0.0), mr);
    if (v.isComplex()) {
        const keel::c64 *cd = v.complexData();
        for (keel::usize i = 0; i < r * c; ++i) M[i] = cd[i];
    } else {
        for (keel::usize i = 0; i < r * c; ++i) M[i] = keel::c64(v.elemAsDouble(i), 0.0);
    }
    return M;
}

Value matFromVec(keel::usize r, keel::usize c, keel::ArrayRef<keel::f64> v, keel::Allocator *mr) {
    Value m = Value::matrix(r, c, ValueType::DOUBLE, mr);
    if (!v.empty()) std::copy(v.begin(), v.end(), m.doubleDataMut());
    return m;
}

Value cmatFromVec(keel::usize r, keel::usize c, keel::ArrayRef<keel::c64> v, keel::Allocator *mr) {
    Value m = Value::matrix(r, c, ValueType::COMPLEX, mr);
    if (!v.empty()) std::copy(v.begin(), v.end(), m.complexDataMut());
    return m;
}

void ctrb_obsv_argcheck(const Value &A, const Value &X, const char *fn,
                        const char *xName, keel::i32 xNum) {
    internal::requireCoercibleMatrix(A, fn, "A", 1);
    internal::requireNonIntegerMatrix(A, fn, "A", 1);
    internal::requireCoercibleMatrix(X, fn, xName, xNum);
    internal::requireNonIntegerMatrix(X, fn, xName, xNum);
}

char gram_argcheck(const Value &type) {
    if (!type.isChar() && !type.isString())
        throw Error(keel::format("gram: input 2 (type) must be 'c' (controllability) or 'o' (observability), as char or string (got a {} {} array).",
                                 internal::shapeOf(type), mtypeName(type.type())),
                    0, 0, "gram", "", "numkit:gram:type");
    const keel::String ts = type.toString();
    const char t = ts.empty() ? 'c' : keel::to_ascii_lower(ts[0]);
    if (t != 'c' && t != 'o')
        throw Error(keel::format("gram: input 2 (type) must be 'c' (controllability) or 'o' (observability), as char or string (got '{}').", ts),
                    0, 0, "gram", "", "numkit:gram:type");
    return t;
}

void throwLyapError(bool discrete, keel::control::ControlStatus st) {
    if (st == keel::control::ControlStatus::kOk) return;
    const char *fn = discrete ? "dlyap" : "lyap";
    throw Error(discrete
                    ? "dlyap: system is singular (A has eigenvalues with product 1)"
                    : "lyap: system is singular (A has eigenvalues with zero sum)",
                0, 0, fn, "", keel::format("numkit:{}:singular", fn));
}

} // anonymous

Value ctrb_AB(const Value &Av, const Value &Bv, keel::Allocator *mr) {
    ctrb_obsv_argcheck(Av, Bv, "ctrb", "B", 2);
    const keel::usize n = Av.dims().rows();
    if (Av.dims().cols() != n)
        throw Error("ctrb: A must be square", 0, 0, "ctrb", "", "numkit:ctrb:A");
    const keel::usize m = Bv.dims().cols();
    if (Bv.dims().rows() != n)
        throw Error("ctrb: B must have the same row count as A", 0, 0, "ctrb", "", "numkit:ctrb:B");

    if (Av.isComplex() || Bv.isComplex()) {
        const auto A = readCMat(Av, n, n, mr);
        const auto B = readCMat(Bv, n, m, mr);
        keel::Array<keel::c64> Co(n * (n * m), keel::c64(0.0, 0.0), mr);
        (void)keel::control::ctrb_c64(A.as_ref(), n, B.as_ref(), m, Co.as_mut_ref(), mr);
        return cmatFromVec(n, n * m, Co.as_ref(), mr);
    }

    const auto A = readMat(Av, n, n, mr);
    const auto B = readMat(Bv, n, m, mr);
    keel::Array<keel::f64> Co(n * (n * m), 0.0, mr);
    (void)keel::control::ctrb(A.as_ref(), n, B.as_ref(), m, Co.as_mut_ref(), mr);
    return matFromVec(n, n * m, Co.as_ref(), mr);
}

Value obsv_AC(const Value &Av, const Value &Cv, keel::Allocator *mr) {
    ctrb_obsv_argcheck(Av, Cv, "obsv", "C", 2);
    const keel::usize n = Av.dims().rows();
    if (Av.dims().cols() != n)
        throw Error("obsv: A must be square", 0, 0, "obsv", "", "numkit:obsv:A");
    const keel::usize p = Cv.dims().rows();
    if (Cv.dims().cols() != n)
        throw Error("obsv: C must have the same column count as A", 0, 0, "obsv", "", "numkit:obsv:C");

    if (Av.isComplex() || Cv.isComplex()) {
        const auto A = readCMat(Av, n, n, mr);
        const auto C = readCMat(Cv, p, n, mr);
        keel::Array<keel::c64> Ob((n * p) * n, keel::c64(0.0, 0.0), mr);
        (void)keel::control::obsv_c64(A.as_ref(), n, C.as_ref(), p, Ob.as_mut_ref(), mr);
        return cmatFromVec(n * p, n, Ob.as_ref(), mr);
    }

    const auto A = readMat(Av, n, n, mr);
    const auto C = readMat(Cv, p, n, mr);
    keel::Array<keel::f64> Ob((n * p) * n, 0.0, mr);
    (void)keel::control::obsv(A.as_ref(), n, C.as_ref(), p, Ob.as_mut_ref(), mr);
    return matFromVec(n * p, n, Ob.as_ref(), mr);
}

Value ctrb_sys(const Value &sys, keel::Allocator *mr) {
    auto abc = pullABC(sys, "ctrb", mr);
    return ctrb_AB(abc.A, abc.B, mr);
}

Value obsv_sys(const Value &sys, keel::Allocator *mr) {
    auto abc = pullABC(sys, "obsv", mr);
    return obsv_AC(abc.A, abc.C, mr);
}

Value gram(const Value &sys, const Value &type, keel::Allocator *mr) {
    auto abc = pullABC(sys, "gram", mr);
    const char t = gram_argcheck(type);
    const keel::usize n = abc.A.dims().rows();
    const bool discrete = (internal::ltiSampleTime(sys) != 0.0);

    const auto A = readMat(abc.A, n, n, mr);
    keel::Array<keel::f64> W(n * n, 0.0, mr);
    keel::control::ControlStatus st = keel::control::ControlStatus::kOk;
    if (t == 'c') {
        const keel::usize m = abc.B.dims().cols();
        const auto B = readMat(abc.B, n, m, mr);
        st = keel::control::gram(A.as_ref(), B.as_ref(), n, m,
                                 keel::control::GramType::kControllability, discrete,
                                 W.as_mut_ref(), mr);
    } else {
        const keel::usize p = abc.C.dims().rows();
        const auto C = readMat(abc.C, p, n, mr);
        st = keel::control::gram(A.as_ref(), C.as_ref(), n, p,
                                 keel::control::GramType::kObservability, discrete,
                                 W.as_mut_ref(), mr);
    }
    throwLyapError(discrete, st);
    return matFromVec(n, n, W.as_ref(), mr);
}

CovarResult covar(const Value &sys, const Value &Wv, keel::Allocator *mr) {
    auto abc = pullABC(sys, "covar", mr);
    const keel::usize n = abc.A.dims().rows();
    const keel::usize m = abc.B.dims().cols();
    const keel::usize p = abc.C.dims().rows();
    const bool discrete = (internal::ltiSampleTime(sys) != 0.0);

    const auto A = readMat(abc.A, n, n, mr);
    const auto B = readMat(abc.B, n, m, mr);
    const auto C = readMat(abc.C, p, n, mr);
    const auto D = readMat(abc.D, p, m, mr);

    const bool wScalar = (Wv.numel() == 1);
    const keel::Array<keel::f64> W = wScalar
        ? keel::Array<keel::f64>(1, Wv.toScalar(), mr)
        : readMat(Wv, m, m, mr);

    keel::Array<keel::f64> P(p * p, 0.0, mr);
    keel::Array<keel::f64> Q(n * n, 0.0, mr);
    const auto st = keel::control::covar(A.as_ref(), B.as_ref(), C.as_ref(), D.as_ref(),
                                         n, m, p, W.as_ref(), wScalar, discrete,
                                         P.as_mut_ref(), Q.as_mut_ref(), mr);
    throwLyapError(discrete, st);

    CovarResult out;
    out.P = matFromVec(p, p, P.as_ref(), mr);
    out.Q = matFromVec(n, n, Q.as_ref(), mr);
    return out;
}

} // namespace numkit::control
