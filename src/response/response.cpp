// toolboxes/control/src/response/response.cpp
//
// step / impulse / lsim / initial time-domain responses delegating numerical
// simulation and grid selection to keel::control.

#include <keel/array.h>
#include <keel/complex.h>
#include <keel/control.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/string.h>
#include <numkit/control/conversion/conversion.hpp>
#include <numkit/control/internal/type_gates.hpp>
#include <numkit/control/props/props.hpp>
#include <numkit/control/response/response.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <algorithm>
#include <utility>

namespace numkit::control {

namespace {

using Vec = keel::Array<keel::f64>;

struct SS {
    keel::usize n = 0;
    Vec A;
    Vec B;
    Vec C;
    keel::f64 D = 0.0;
    keel::f64 Ts = 0.0;
};

SS toSSiso(const Value &sys, const char *fn, keel::Allocator *mr) {
    Value Av, Bv, Cv, Dv;
    keel::f64 Ts = 0.0;
    if (sys.isStruct() && sys.hasField("kind")) {
        Ts = sys.field("Ts").toScalar();
        const keel::String k = sys.field("kind").toString();
        if (k == "ss") {
            Av = sys.field("A"); Bv = sys.field("B");
            Cv = sys.field("C"); Dv = sys.field("D");
        } else if (k == "tf") {
            auto s = tf2ss(sys.field("num"), sys.field("den"), mr);
            Av = std::move(s.A); Bv = std::move(s.B);
            Cv = std::move(s.C); Dv = std::move(s.D);
        } else if (k == "zpk") {
            auto [num, den] = zp2tf(sys.field("z"), sys.field("p"), sys.field("k"), mr);
            auto s = tf2ss(num, den, mr);
            Av = std::move(s.A); Bv = std::move(s.B);
            Cv = std::move(s.C); Dv = std::move(s.D);
        } else {
            throw Error(keel::format("{}: input 1 (sys) has unknown LTI kind '{}'", fn, k),
                        0, 0, fn, "", keel::format("numkit:{}:kind", fn));
        }
    } else {
        throw Error(keel::format("{}: input 1 (sys) must be an LTI struct (tf/zpk/ss)", fn),
                    0, 0, fn, "", keel::format("numkit:{}:kind", fn));
    }

    SS s;
    s.n = Av.dims().rows();
    s.Ts = Ts;
    s.A = Vec(s.n * s.n, 0.0, mr);
    for (keel::usize i = 0; i < s.n * s.n; ++i) s.A[i] = Av.elemAsDouble(i);
    s.B = Vec(s.n, 0.0, mr);
    for (keel::usize i = 0; i < s.n; ++i) s.B[i] = Bv.elemAsDouble(i);
    s.C = Vec(s.n, 0.0, mr);
    for (keel::usize i = 0; i < s.n; ++i) s.C[i] = Cv.elemAsDouble(i);
    s.D = (Dv.numel() == 0) ? 0.0 : Dv.toScalar();
    return s;
}

keel::Array<keel::c64> polesAsComplex(const Value &sys, keel::Allocator *mr) {
    Value pV = pole(sys, mr);
    const keel::usize np = pV.numel();
    keel::Array<keel::c64> ps(np, keel::c64(0.0, 0.0), mr);
    if (pV.type() == ValueType::COMPLEX) {
        const keel::c64 *src = pV.complexData();
        for (keel::usize i = 0; i < np; ++i) ps[i] = src[i];
    } else {
        for (keel::usize i = 0; i < np; ++i) ps[i] = keel::c64(pV.elemAsDouble(i), 0.0);
    }
    return ps;
}

Vec readTimeArg(const Value &sys, const Value &tArg, keel::Allocator *mr) {
    if (tArg.numel() == 0 || tArg.numel() == 1) {
        const auto ps = polesAsComplex(sys, mr);
        const keel::f64 tOverride = (tArg.numel() == 1) ? tArg.toScalar() : 0.0;
        Vec t(mr);
        (void)keel::control::pick_time_grid(ps.as_ref(), tOverride, t, mr);
        return t;
    }
    Vec t(tArg.numel(), 0.0, mr);
    for (keel::usize i = 0; i < tArg.numel(); ++i) t[i] = tArg.elemAsDouble(i);
    return t;
}

Value colFromVec(keel::ArrayRef<keel::f64> v, keel::Allocator *mr) {
    Value r = Value::matrix(v.size(), 1, ValueType::DOUBLE, mr);
    if (!v.empty()) std::copy(v.begin(), v.end(), r.doubleDataMut());
    return r;
}

Value matFromTraj(keel::ArrayRef<keel::f64> xTraj, keel::usize N, keel::usize n,
                  keel::Allocator *mr) {
    Value m = Value::matrix(N, n, ValueType::DOUBLE, mr);
    if (!xTraj.empty()) std::copy(xTraj.begin(), xTraj.end(), m.doubleDataMut());
    return m;
}

Vec runSimSiso(const SS &sys, const Vec &t, const Vec &u, const Vec &x0,
               Vec *xTraj, keel::Allocator *mr) {
    const keel::usize N = t.size();
    Vec y(N, 0.0, mr);
    keel::MutArrayRef<keel::f64> trajRef{};
    if (xTraj) {
        *xTraj = Vec(N * sys.n, 0.0, mr);
        trajRef = xTraj->as_mut_ref();
    }
    (void)keel::control::simulate_siso(sys.A.as_ref(), sys.B.as_ref(), sys.C.as_ref(), sys.D,
                                       sys.n, sys.Ts, t.as_ref(), u.as_ref(), x0.as_ref(),
                                       y.as_mut_ref(), trajRef, mr);
    return y;
}

void response_sys_argcheck(const Value &sys, const char *fn) {
    if (sys.isStruct() && sys.hasField("kind")) {
        const keel::String k = sys.field("kind").toString();
        if (k == "ss" || k == "tf" || k == "zpk") return;
        throw Error(keel::format("{}: input 1 (sys) has unknown LTI kind '{}'", fn, k),
                    0, 0, fn, "", keel::format("numkit:{}:kind", fn));
    }
    throw Error(keel::format("{}: input 1 (sys) must be an LTI struct (tf/zpk/ss)", fn),
                0, 0, fn, "", keel::format("numkit:{}:kind", fn));
}

void lsim_argcheck(const Value &uIn, const Value &tIn, const Value &x0In) {
    internal::requireNumericInput(uIn, "lsim", "u", 2);
    const ValueType tt = tIn.type();
    if (tt != ValueType::DOUBLE && tt != ValueType::SINGLE)
        throw Error(keel::format("lsim: input 3 (t) must be a real double or single time vector (got a {} {} array).",
                                 internal::shapeOf(tIn), mtypeName(tt)),
                    0, 0, "lsim", "", "numkit:lsim:t");
    internal::requireNumericInput(x0In, "lsim", "x0", 4);
    if (tIn.numel() < 2)
        throw Error("lsim: t must have at least 2 samples",
                    0, 0, "lsim", "", "numkit:lsim:t");
}

} // anonymous

std::pair<Value, Value>
step_response(const Value &sys, const Value &tArg, Value *xOut, keel::Allocator *mr) {
    response_sys_argcheck(sys, "step");
    const SS s = toSSiso(sys, "step", mr);
    const Vec t = readTimeArg(sys, tArg, mr);
    const Vec u(t.size(), 1.0, mr);
    const Vec x0(s.n, 0.0, mr);
    Vec xTraj;
    const Vec y = runSimSiso(s, t, u, x0, xOut ? &xTraj : nullptr, mr);
    if (xOut) *xOut = matFromTraj(xTraj.as_ref(), t.size(), s.n, mr);
    return {colFromVec(y.as_ref(), mr), colFromVec(t.as_ref(), mr)};
}

std::pair<Value, Value>
impulse_response(const Value &sys, const Value &tArg, Value *xOut, keel::Allocator *mr) {
    response_sys_argcheck(sys, "impulse");
    const SS s = toSSiso(sys, "impulse", mr);
    const Vec t = readTimeArg(sys, tArg, mr);
    Vec x0(s.n, 0.0, mr);
    Vec u(t.size(), 0.0, mr);
    if (s.Ts > 0.0) {
        if (!u.empty()) u[0] = 1.0;
    } else {
        x0 = s.B.clone();
    }
    Vec xTraj;
    const Vec y = runSimSiso(s, t, u, x0, xOut ? &xTraj : nullptr, mr);
    if (xOut) *xOut = matFromTraj(xTraj.as_ref(), t.size(), s.n, mr);
    return {colFromVec(y.as_ref(), mr), colFromVec(t.as_ref(), mr)};
}

Value lsim(const Value &sys, const Value &uIn, const Value &tIn,
           const Value &x0In, Value *xOut, keel::Allocator *mr) {
    lsim_argcheck(uIn, tIn, x0In);
    Vec t(tIn.numel(), 0.0, mr);
    for (keel::usize i = 0; i < tIn.numel(); ++i) t[i] = tIn.elemAsDouble(i);

    if (sys.isStruct() && sys.hasField("kind") && sys.field("kind").toString() == "ss") {
        const Value &Av = sys.field("A");
        const Value &Bv = sys.field("B");
        const Value &Cv = sys.field("C");
        const Value &Dv = sys.field("D");
        const keel::usize n = Av.dims().rows();
        const keel::usize nu = Bv.dims().cols();
        const keel::usize ny = Cv.dims().rows();
        if ((nu > 1 || ny > 1) && !uIn.isComplex() && !x0In.isComplex()) {
            const keel::usize N = t.size();
            bool uTransposed = false;
            if (uIn.dims().rows() == N && uIn.dims().cols() == nu) {
                uTransposed = false;
            } else if (uIn.dims().cols() == N && uIn.dims().rows() == nu) {
                uTransposed = true;
            } else {
                throw Error("lsim: u dimensions must match time vector length and input count",
                            0, 0, "lsim", "", "numkit:lsim:size");
            }
            if (x0In.numel() != 0 && x0In.numel() != n)
                throw Error(keel::format("lsim: input 4 (x0) must have one entry per state (got {} entries for {} states).",
                                         x0In.numel(), n),
                            0, 0, "lsim", "", "numkit:lsim:x0");

            Vec A(n * n, 0.0, mr), B(n * nu, 0.0, mr), C(ny * n, 0.0, mr);
            for (keel::usize i = 0; i < n * n; ++i) A[i] = Av.elemAsDouble(i);
            for (keel::usize i = 0; i < n * nu; ++i) B[i] = Bv.elemAsDouble(i);
            for (keel::usize i = 0; i < ny * n; ++i) C[i] = Cv.elemAsDouble(i);
            Vec D(Dv.numel(), 0.0, mr);
            for (keel::usize i = 0; i < Dv.numel(); ++i) D[i] = Dv.elemAsDouble(i);

            Vec uCol(N * nu, 0.0, mr);
            for (keel::usize j = 0; j < nu; ++j)
                for (keel::usize k = 0; k < N; ++k)
                    uCol[j * N + k] = uTransposed ? uIn.elemAsDouble(k * nu + j)
                                                  : uIn.elemAsDouble(j * N + k);

            Vec x0(mr);
            if (x0In.numel() == n) {
                x0.resize(n);
                for (keel::usize i = 0; i < n; ++i) x0[i] = x0In.elemAsDouble(i);
            }

            Value yOut = Value::matrix(N, ny, ValueType::DOUBLE, mr);
            keel::MutArrayRef<keel::f64> xRef{};
            if (xOut) {
                *xOut = Value::matrix(N, n, ValueType::DOUBLE, mr);
                xRef = keel::MutArrayRef<keel::f64>(xOut->doubleDataMut(), N * n);
            }
            const keel::f64 Ts = sys.field("Ts").toScalar();
            (void)keel::control::simulate_mimo(A.as_ref(), B.as_ref(), C.as_ref(), D.as_ref(),
                                               n, nu, ny, Ts, t.as_ref(), uCol.as_ref(),
                                               x0.as_ref(),
                                               keel::MutArrayRef<keel::f64>(yOut.doubleDataMut(), N * ny),
                                               xRef, mr);
            return yOut;
        }
    }

    const SS s = toSSiso(sys, "lsim", mr);
    if (uIn.numel() != t.size())
        throw Error("lsim: u and t must be the same length",
                    0, 0, "lsim", "", "numkit:lsim:size");
    if (x0In.numel() != 0 && x0In.numel() != s.n)
        throw Error(keel::format("lsim: input 4 (x0) must have one entry per state (got {} entries for {} states).",
                                 x0In.numel(), s.n),
                    0, 0, "lsim", "", "numkit:lsim:x0");

    if (uIn.isComplex() || x0In.isComplex()) {
        const keel::usize N = t.size();
        Vec uRe(N, 0.0, mr), uIm(N, 0.0, mr);
        if (uIn.isComplex()) {
            const keel::c64 *cd = uIn.complexData();
            for (keel::usize i = 0; i < N; ++i) { uRe[i] = cd[i].real(); uIm[i] = cd[i].imag(); }
        } else {
            for (keel::usize i = 0; i < N; ++i) uRe[i] = uIn.elemAsDouble(i);
        }
        Vec x0Re(s.n, 0.0, mr), x0Im(s.n, 0.0, mr);
        if (x0In.numel() == s.n) {
            if (x0In.isComplex()) {
                const keel::c64 *cd = x0In.complexData();
                for (keel::usize i = 0; i < s.n; ++i) { x0Re[i] = cd[i].real(); x0Im[i] = cd[i].imag(); }
            } else {
                for (keel::usize i = 0; i < s.n; ++i) x0Re[i] = x0In.elemAsDouble(i);
            }
        }
        Vec xTrajRe, xTrajIm;
        const Vec yRe = runSimSiso(s, t, uRe, x0Re, xOut ? &xTrajRe : nullptr, mr);
        const Vec yIm = runSimSiso(s, t, uIm, x0Im, xOut ? &xTrajIm : nullptr, mr);
        if (xOut) {
            Value xm = Value::matrix(N, s.n, ValueType::COMPLEX, mr);
            keel::c64 *xd = xm.complexDataMut();
            for (keel::usize i = 0; i < N * s.n; ++i) xd[i] = keel::c64(xTrajRe[i], xTrajIm[i]);
            *xOut = std::move(xm);
        }
        Value yOut = Value::matrix(N, 1, ValueType::COMPLEX, mr);
        keel::c64 *yd = yOut.complexDataMut();
        for (keel::usize i = 0; i < N; ++i) yd[i] = keel::c64(yRe[i], yIm[i]);
        return yOut;
    }

    Vec u(t.size(), 0.0, mr);
    for (keel::usize i = 0; i < t.size(); ++i) u[i] = uIn.elemAsDouble(i);
    Vec x0(s.n, 0.0, mr);
    if (x0In.numel() == s.n) {
        for (keel::usize i = 0; i < s.n; ++i) x0[i] = x0In.elemAsDouble(i);
    }
    Vec xTraj;
    const Vec y = runSimSiso(s, t, u, x0, xOut ? &xTraj : nullptr, mr);
    if (xOut) *xOut = matFromTraj(xTraj.as_ref(), t.size(), s.n, mr);
    return colFromVec(y.as_ref(), mr);
}

std::pair<Value, Value>
initial_response(const Value &sys, const Value &x0In, const Value &tArg,
                 Value *xOut, keel::Allocator *mr) {
    response_sys_argcheck(sys, "initial");
    const SS s = toSSiso(sys, "initial", mr);
    const Vec t = readTimeArg(sys, tArg, mr);
    const Vec u(t.size(), 0.0, mr);
    Vec x0(s.n, 0.0, mr);
    if (x0In.numel() == s.n)
        for (keel::usize i = 0; i < s.n; ++i) x0[i] = x0In.elemAsDouble(i);
    else if (x0In.numel() != 0)
        throw Error("initial: x0 length must equal the number of states",
                    0, 0, "initial", "", "numkit:initial:x0");
    Vec xTraj;
    const Vec y = runSimSiso(s, t, u, x0, xOut ? &xTraj : nullptr, mr);
    if (xOut) *xOut = matFromTraj(xTraj.as_ref(), t.size(), s.n, mr);
    return {colFromVec(y.as_ref(), mr), colFromVec(t.as_ref(), mr)};
}

} // namespace numkit::control
