// toolboxes/control/src/lti/lti.cpp
//
// LTI constructors. Each builds a nk struct value tagged by the
// `kind` field. Discrete-time systems carry Ts > 0; continuous = 0.
// We don't enforce algebraic invariants here (e.g. real coefficients,
// proper rational form) — that's left to the user / converters which
// check on-demand.

#include <keel/array.h>
#include <keel/algo.h>
#include <keel/tuple.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/complex.h>
#include <keel/linalg.h>
#include <keel/math.h>
#include <keel/string.h>
#include <numkit/control/lti/lti.hpp>
#include <numkit/control/conversion/conversion.hpp>
#include <numkit/control/internal/lti_object.hpp>
#include <numkit/control/internal/type_gates.hpp>

// Compute-only TU: Value substrate + Error, no engine. The tf/zpk/ss/filt/frd
// constructors + *data extractors (CallContext wrappers) live in lti_reg.cpp.
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>


namespace numkit::control {

namespace {

Value rowVec(const Value &v, keel::Allocator *mr) {
    // Always represent coefficient lists as a row vector, copying
    // through the user memory resource so the struct fields don't
    // alias caller-side scratch memory.
    const keel::usize N = v.numel();
    if (v.type() == ValueType::COMPLEX) {
        Value r = Value::matrix(1, N, ValueType::COMPLEX, mr);
        const keel::c64 *src = v.complexData();
        keel::c64 *rd = r.complexDataMut();
        for (keel::usize i = 0; i < N; ++i) rd[i] = src[i];
        return r;
    }
    Value r = Value::matrix(1, N, ValueType::DOUBLE, mr);
    keel::f64 *rd = r.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) rd[i] = v.elemAsDouble(i);
    return r;
}

Value tagStruct(const char *kind, keel::f64 Ts, keel::Allocator *mr) {
    Value s = Value::structure(mr);
    s.field("kind") = Value::fromString(kind, mr);
    s.field("Ts") = Value::scalar(Ts, mr);
    return s;
}

// ──────────────────────────────────────────────────────────────────────
// tf/filt coefficient argcheck (argcheck separation,
// dev-docs/todo/argcheck_separation.md). Rejection classes
// reference-probed R2025b 2026-09-18 (build/probe9..11_control_ref.m):
// both constructors coerce single/integer coefficients to double
// storage; tf rejects logical/char/string while filt's NUMERATOR
// coerces even those (filt's denominator rejects them like tf); a
// 1x1 cell is the SISO-in-cell form and unwraps, any larger cell is
// the MIMO form (unsupported stub).
// ──────────────────────────────────────────────────────────────────────
Value tfCoeffArg(const Value &v, const char *fn, const char *argName,
                 bool broadNum, keel::Allocator *mr)
{
    using internal::shapeOf;
    Value u = v;
    // Copy to a temp first: cellAt returns a reference into u's own
    // buffer, and a direct u = u.cellAt(0) would assign from freed
    // storage after u releases it.
    while (u.isCell() && u.numel() == 1) {
        Value inner = u.cellAt(0);
        u = keel::move(inner);
    }
    const ValueType t = u.type();
    const keel::String tag = keel::format("numkit:{}:{}", fn, argName);
    if (u.isCell())
        throw Error(keel::format("{}: input ({}) uses the cell (MIMO) coefficient form — only SISO models are supported in this build.",
                                 fn, argName),
                    0, 0, fn, "", keel::format("numkit:{}:mimoUnsupported", fn));
    if (u.numel() == 0)
        throw Error(keel::format("{}: input ({}) must not be empty (got a {} {} array).",
                                 fn, argName, shapeOf(u), mtypeName(t)),
                    0, 0, fn, "", tag.as_ref());
    if (t == ValueType::DOUBLE || t == ValueType::COMPLEX)
        return u;
    const bool numericCoercible = (t == ValueType::SINGLE || isIntegerType(t));
    if (numericCoercible || (broadNum && (t == ValueType::LOGICAL
                                          || t == ValueType::CHAR
                                          || t == ValueType::STRING)))
        return internal::coerceToDoubleMatrix(u, mr);
    throw Error(keel::format("{}: input ({}) must be a numeric or complex coefficient vector (got a {} {} array).",
                             fn, argName, shapeOf(u), mtypeName(t)),
                0, 0, fn, "", keel::format("numkit:{}:inputType", fn));
}

// Build a MIMO cell grid when either num or den is a multi-element cell.
bool buildMimoCellTf(const Value &num, const Value &den, const char *fn,
                     bool broadNum, Value &numOut, Value &denOut,
                     keel::Allocator *mr)
{
    Value uNum = num;
    while (uNum.isCell() && uNum.numel() == 1) {
        Value inner = uNum.cellAt(0);
        uNum = keel::move(inner);
    }
    Value uDen = den;
    while (uDen.isCell() && uDen.numel() == 1) {
        Value inner = uDen.cellAt(0);
        uDen = keel::move(inner);
    }
    if (!uNum.isCell() && !uDen.isCell()) return false;

    keel::usize r = 1, c = 1;
    if (uNum.isCell() && uDen.isCell()) {
        if (uNum.dims().rows() != uDen.dims().rows() ||
            uNum.dims().cols() != uDen.dims().cols())
            throw Error(keel::format("{}: cell arrays num and den must have matching dimensions", fn),
                        0, 0, fn, "", keel::format("numkit:{}:mimoSize", fn));
        r = uNum.dims().rows();
        c = uNum.dims().cols();
    } else if (uNum.isCell()) {
        r = uNum.dims().rows();
        c = uNum.dims().cols();
    } else {
        r = uDen.dims().rows();
        c = uDen.dims().cols();
    }

    numOut = Value::cell(r, c, mr);
    denOut = Value::cell(r, c, mr);
    const keel::usize total = r * c;
    for (keel::usize i = 0; i < total; ++i) {
        const Value &ni = uNum.isCell() ? uNum.cellAt(i) : uNum;
        const Value &di = uDen.isCell() ? uDen.cellAt(i) : uDen;
        numOut.cellAt(i) = rowVec(tfCoeffArg(ni, fn, "num", broadNum, mr), mr);
        denOut.cellAt(i) = rowVec(tfCoeffArg(di, fn, "den", false, mr), mr);
    }
    return true;
}

bool tf_argcheck(const Value &num, const Value &den, Value &numOut, Value &denOut,
                 keel::Allocator *mr)
{
    if (buildMimoCellTf(num, den, "tf", /*broadNum=*/false, numOut, denOut, mr))
        return true;
    numOut = tfCoeffArg(num, "tf", "num", /*broadNum=*/false, mr);
    denOut = tfCoeffArg(den, "tf", "den", /*broadNum=*/false, mr);
    return false;
}

void zpk_argcheck(const Value &z, const Value &p, keel::f64 k, keel::f64 Ts)
{
    (void)z;
    (void)p;
    (void)k;
    (void)Ts;
}

bool filt_argcheck(const Value &num, const Value &den, Value &numOut, Value &denOut,
                   keel::Allocator *mr)
{
    if (buildMimoCellTf(num, den, "filt", /*broadNum=*/true, numOut, denOut, mr))
        return true;
    numOut = tfCoeffArg(num, "filt", "num", /*broadNum=*/true, mr);
    denOut = tfCoeffArg(den, "filt", "den", /*broadNum=*/false, mr);
    return false;
}

void frd_argcheck(const Value &response, const Value &frequency)
{
    if (response.numel() != frequency.numel())
        throw Error("frd: response and frequency must have the same length",
                    0, 0, "frd", "", "numkit:frd:size");
}

} // anonymous

Value tf(const Value &num, const Value &den, keel::f64 Ts, keel::Allocator *mr)
{
    Value numC, denC;
    if (tf_argcheck(num, den, numC, denC, mr)) {
        Value s = tagStruct("tf", Ts, mr);
        s.field("num") = keel::move(numC);
        s.field("den") = keel::move(denC);
        return internal::wrapLtiObject("tf", keel::move(s), mr);
    }
    Value s = tagStruct("tf", Ts, mr);
    s.field("num") = rowVec(numC, mr);
    s.field("den") = rowVec(denC, mr);
    return internal::wrapLtiObject("tf", keel::move(s), mr);
}

Value zpk(const Value &z, const Value &p, keel::f64 k, keel::f64 Ts,
          keel::Allocator *mr)
{
    zpk_argcheck(z, p, k, Ts);
    Value s = tagStruct("zpk", Ts, mr);
    s.field("z") = rowVec(z, mr);
    s.field("p") = rowVec(p, mr);
    s.field("k") = Value::scalar(k, mr);
    return internal::wrapLtiObject("zpk", keel::move(s), mr);
}

namespace {

// ss matrix argcheck: the reference coerces single/integer matrices to
// double storage but rejects logical/char/cell/string (probed R2025b —
// Control:ltiobject:ssProperties6).
Value ssMatArg(const Value &m, const char *fn, const char *argName,
               keel::Allocator *mr)
{
    using internal::shapeOf;
    const ValueType t = m.type();
    if (t == ValueType::DOUBLE || t == ValueType::COMPLEX)
        return m;
    if (t == ValueType::SINGLE || isIntegerType(t))
        return internal::coerceToDoubleMatrix(m, mr);
    throw Error(keel::format("{}: input ({}) must be a numeric or complex matrix (got a {} {} array).",
                             fn, argName, shapeOf(m), mtypeName(t)),
                0, 0, fn, "", keel::format("numkit:{}:inputType", fn));
}

// Expand a scalar block to r×c; pass everything else through. Scalars
// expand exactly like the reference's documented D-gain expansion.
Value expandScalarBlock(const Value &m, keel::usize r, keel::usize c,
                        keel::Allocator *mr)
{
    if (m.numel() != 1 || (r == 1 && c == 1) || r == 0 || c == 0)
        return m;
    if (m.type() == ValueType::COMPLEX) {
        Value out = Value::matrix(r, c, ValueType::COMPLEX, mr);
        const keel::c64 src = m.complexData()[0];
        keel::c64 *od = out.complexDataMut();
        for (keel::usize i = 0; i < r * c; ++i) od[i] = src;
        return out;
    }
    Value out = Value::matrix(r, c, ValueType::DOUBLE, mr);
    const keel::f64 src = m.elemAsDouble(0);
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize i = 0; i < r * c; ++i) od[i] = src;
    return out;
}

void ss_argcheck(const Value &A, const Value &B, const Value &C, const Value &D,
                 Value &Av, Value &Bv, Value &Cv, Value &Dv,
                 keel::Allocator *mr)
{
    Av = ssMatArg(A, "ss", "A", mr);
    Bv = ssMatArg(B, "ss", "B", mr);
    Cv = ssMatArg(C, "ss", "C", mr);
    Dv = ssMatArg(D, "ss", "D", mr);

    using internal::shapeOf;
    const auto blockErr = [&](const char *arg, const Value &v) {
        throw Error(keel::format("ss: input ({}) has shape {}, which is inconsistent with the other A/B/C/D blocks.",
                                 arg, shapeOf(v)),
                    0, 0, "ss", "", keel::format("numkit:ss:{}", arg));
    };
    if (Av.numel() > 1 && Av.dims().rows() != Av.dims().cols())
        blockErr("A", Av);
    const keel::usize n = (Av.numel() == 0) ? 0 : Av.dims().rows();
    const keel::usize m = (Bv.numel() == 0) ? 0 : ((Bv.numel() == 1) ? 1 : Bv.dims().cols());
    const keel::usize p = (Cv.numel() == 0) ? 0 : ((Cv.numel() == 1) ? 1 : Cv.dims().rows());
    if (Bv.numel() > 1 && Bv.dims().rows() != n) blockErr("B", Bv);
    if (Cv.numel() > 1 && Cv.dims().cols() != n) blockErr("C", Cv);
    if (Dv.numel() > 1
        && (Dv.dims().rows() != p || Dv.dims().cols() != m))
        blockErr("D", Dv);
    Bv = expandScalarBlock(Bv, n, m, mr);
    Cv = expandScalarBlock(Cv, p, n, mr);
    Dv = expandScalarBlock(Dv, p ? p : 1, m ? m : 1, mr);
}

} // anonymous

Value ss(const Value &A, const Value &B, const Value &C, const Value &D, keel::f64 Ts, keel::Allocator *mr)
{
    Value Av, Bv, Cv, Dv;
    ss_argcheck(A, B, C, D, Av, Bv, Cv, Dv, mr);

    Value s = tagStruct("ss", Ts, mr);
    // Copy state-space matrices through the user resource.
    auto copyMat = [&](const Value &m) {
        const keel::usize r = m.dims().rows();
        const keel::usize c = m.dims().cols();
        if (m.type() == ValueType::COMPLEX) {
            Value out = Value::matrix(r, c, ValueType::COMPLEX, mr);
            const keel::c64 *src = m.complexData();
            keel::c64 *od = out.complexDataMut();
            for (keel::usize i = 0; i < r * c; ++i) od[i] = src[i];
            return out;
        }
        Value out = Value::matrix(r, c, ValueType::DOUBLE, mr);
        keel::f64 *od = out.doubleDataMut();
        for (keel::usize i = 0; i < r * c; ++i) od[i] = m.elemAsDouble(i);
        return out;
    };
    s.field("A") = copyMat(Av);
    s.field("B") = copyMat(Bv);
    s.field("C") = copyMat(Cv);
    s.field("D") = copyMat(Dv);
    return internal::wrapLtiObject("ss", keel::move(s), mr);
}

Value filt(const Value &num, const Value &den, keel::f64 Ts, keel::Allocator *mr)
{
    Value numC, denC;
    if (filt_argcheck(num, den, numC, denC, mr)) {
        Value s = tagStruct("tf", Ts, mr);
        s.field("num") = keel::move(numC);
        s.field("den") = keel::move(denC);
        s.field("variable") = Value::fromString("z^-1", mr);
        return internal::wrapLtiObject("tf", keel::move(s), mr);
    }
    Value s = tagStruct("tf", Ts, mr);
    s.field("num") = rowVec(numC, mr);
    s.field("den") = rowVec(denC, mr);
    s.field("variable") = Value::fromString("z^-1", mr);
    return internal::wrapLtiObject("tf", keel::move(s), mr);
}

Value frd(const Value &response, const Value &frequency, keel::f64 Ts, keel::Allocator *mr)
{
    frd_argcheck(response, frequency);
    Value s = tagStruct("frd", Ts, mr);
    const keel::usize N = frequency.numel();
    // resp may be complex; copy as column vector.
    if (response.type() == ValueType::COMPLEX) {
        Value rv = Value::matrix(N, 1, ValueType::COMPLEX, mr);
        const keel::c64 *src = response.complexData();
        keel::c64 *rd = rv.complexDataMut();
        for (keel::usize i = 0; i < N; ++i) rd[i] = src[i];
        s.field("resp") = keel::move(rv);
    } else {
        Value rv = Value::matrix(N, 1, ValueType::DOUBLE, mr);
        keel::f64 *rd = rv.doubleDataMut();
        for (keel::usize i = 0; i < N; ++i) rd[i] = response.elemAsDouble(i);
        s.field("resp") = keel::move(rv);
    }
    Value fv = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::f64 *fd = fv.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) fd[i] = frequency.elemAsDouble(i);
    s.field("freq") = keel::move(fv);
    return internal::wrapLtiObject("frd", keel::move(s), mr);
}

// ──────────────────────────────────────────────────────────────────────
// Matrix helpers (used by ss2ss for the similarity transform).
// ──────────────────────────────────────────────────────────────────────

namespace {

template <typename T>
void matmul_n(const keel::Array<T> &A, keel::usize aR, keel::usize aC,
              const keel::Array<T> &B, keel::usize bR, keel::usize bC,
              keel::Array<T> &C)
{
    C.assign(aR * bC, T(0));
    if (aR == 0 || bC == 0 || aC == 0) return;
    keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans,
                       keel::linalg::MatrixTranspose::NoTrans,
                       aR, bC, aC, T(1),
                       A.data(), aR, B.data(), bR,
                       T(0), C.data(), aR);
}

keel::Array<keel::f64> readMat(const Value &v) {
    const keel::usize r = v.dims().rows();
    const keel::usize c = v.dims().cols();
    keel::Array<keel::f64> out(r * c);
    for (keel::usize i = 0; i < r * c; ++i) out[i] = v.elemAsDouble(i);
    return out;
}

keel::Array<keel::c64> readCMat(const Value &v) {
    const keel::usize r = v.dims().rows();
    const keel::usize c = v.dims().cols();
    keel::Array<keel::c64> out(r * c);
    if (v.isComplex()) {
        const keel::c64 *cd = v.complexData();
        for (keel::usize i = 0; i < r * c; ++i) out[i] = cd[i];
    } else {
        for (keel::usize i = 0; i < r * c; ++i) out[i] = keel::c64(v.elemAsDouble(i), 0.0);
    }
    return out;
}

bool inverse_n(const Value &T, keel::Array<keel::f64> &out, keel::usize n, keel::Allocator *mr) {
    if (T.dims().rows() != n || T.dims().cols() != n) return false;
    out.assign(n * n, 0.0);
    if (n == 0) return true;
    const auto Tm = readMat(T);
    return keel::linalg::matrix_inv(Tm.data(), n, out.data(), mr);
}

bool inverse_n_cplx(const Value &T, keel::Array<keel::c64> &out, keel::usize n, keel::Allocator *mr) {
    if (T.dims().rows() != n || T.dims().cols() != n) return false;
    out.assign(n * n, keel::c64(0.0, 0.0));
    if (n == 0) return true;
    const auto Tm = readCMat(T);
    return keel::linalg::matrix_inv(Tm.data(), n, out.data(), mr);
}

} // anonymous

// ──────────────────────────────────────────────────────────────────────
// Extractors: tfdata / zpkdata / ssdata / frdata
// ──────────────────────────────────────────────────────────────────────

namespace {

keel::String kindOf(const Value &sys) {
    return internal::ltiKindOf(sys);
}

// Pad num with leading zeros to match den length (MATLAB tfdata convention).
Value padToLen(const Value &row, keel::usize targetLen, keel::Allocator *mr) {
    const keel::usize N = row.numel();
    if (N >= targetLen) return rowVec(row, mr);
    Value out = Value::matrix(1, targetLen, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    const keel::usize pad = targetLen - N;
    for (keel::usize i = 0; i < pad; ++i)     od[i] = 0.0;
    for (keel::usize i = 0; i < N;   ++i)     od[pad + i] = row.elemAsDouble(i);
    return out;
}

// Wrap a single value in a 1×1 cell.
Value wrapCell(const Value &v, keel::Allocator *mr) {
    Value c = Value::cell(1, 1, mr);
    c.cellAt(0) = v;
    return c;
}

// Column-vector copy (used for zpkdata's z/p outputs).
Value colVec(const Value &v, keel::Allocator *mr) {
    const keel::usize N = v.numel();
    if (v.type() == ValueType::COMPLEX) {
        Value r = Value::matrix(N, 1, ValueType::COMPLEX, mr);
        const keel::c64 *src = v.complexData();
        keel::c64 *rd = r.complexDataMut();
        for (keel::usize i = 0; i < N; ++i) rd[i] = src[i];
        return r;
    }
    Value r = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::f64 *rd = r.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) rd[i] = v.elemAsDouble(i);
    return r;
}

void tfdata_argcheck(const Value &sys)
{
    const keel::String k = kindOf(sys);
    if (k != "tf" && k != "zpk" && k != "ss")
        throw Error("tfdata: input must be tf / zpk / ss",
                    0, 0, "tfdata", "", "numkit:tfdata:kind");
}

void zpkdata_argcheck(const Value &sys)
{
    const keel::String kind = kindOf(sys);
    if (kind != "zpk" && kind != "tf" && kind != "ss")
        throw Error("zpkdata: input must be tf / zpk / ss",
                    0, 0, "zpkdata", "", "numkit:zpkdata:kind");
}

void ssdata_argcheck(const Value &sys)
{
    const keel::String kind = kindOf(sys);
    if (kind != "ss" && kind != "tf" && kind != "zpk")
        throw Error("ssdata: input must be tf / zpk / ss",
                    0, 0, "ssdata", "", "numkit:ssdata:kind");
}

void frdata_argcheck(const Value &sys)
{
    if (kindOf(sys) != "frd")
        throw Error("frdata: input must be an frd model",
                    0, 0, "frdata", "", "numkit:frdata:kind");
}

void ss2ss_argcheck(const Value &sys, const Value &T, Value &Tc,
                    keel::Allocator *mr)
{
    if (kindOf(sys) != "ss")
        throw Error("ss2ss: input must be an ss model",
                    0, 0, "ss2ss", "", "numkit:ss2ss:kind");
    internal::requireCoercibleMatrix(T, "ss2ss", "T", 2);
    internal::requireNonIntegerMatrix(T, "ss2ss", "T", 2);
    Tc = internal::coerceToDoubleMatrix(T, mr);
    const Value &s = internal::ltiStruct(sys);
    const Value &Av = s.field("A");
    const keel::usize n = Av.dims().rows();
    if (Av.dims().cols() != n || Tc.dims().rows() != n || Tc.dims().cols() != n)
        throw Error("ss2ss: T must be n×n with n = order(sys)",
                    0, 0, "ss2ss", "", "numkit:ss2ss:size");
}

} // anonymous

keel::Tuple<Value, Value>
tfdata(const Value &sys, bool asVector, keel::Allocator *mr)
{
    tfdata_argcheck(sys);
    const Value &s = internal::ltiStruct(sys);
    Value num, den;
    const keel::String k = kindOf(sys);
    if (k == "tf") {
        num = s.field("num");
        den = s.field("den");
        if (num.isCell() && den.isCell()) {
            const keel::usize ny = num.dims().rows();
            const keel::usize nu = num.dims().cols();
            if (ny * nu == 1 && asVector) {
                num = num.cellAt(0);
                den = den.cellAt(0);
            } else {
                Value numOut = Value::cell(ny, nu, mr);
                Value denOut = Value::cell(ny, nu, mr);
                for (keel::usize i = 0; i < ny * nu; ++i) {
                    const Value &ni = num.cellAt(i);
                    const Value &di = den.cellAt(i);
                    const keel::usize L = keel::max(ni.numel(), di.numel());
                    numOut.cellAt(i) = padToLen(ni, L, mr);
                    denOut.cellAt(i) = padToLen(di, L, mr);
                }
                return {keel::move(numOut), keel::move(denOut)};
            }
        }
    } else if (k == "zpk") {
        keel::tie(num, den) = zp2tf(s.field("z"), s.field("p"), s.field("k"), mr);
    } else {
        const Value &A = s.field("A");
        const Value &B = s.field("B");
        const Value &C = s.field("C");
        const Value &D = s.field("D");
        const keel::usize ny = (C.dims().rows() == 0) ? 1 : C.dims().rows();
        const keel::usize nu = (B.dims().cols() == 0) ? 1 : B.dims().cols();
        if (ny > 1 || nu > 1) {
            const keel::usize n = A.dims().rows();
            Value numOut = Value::cell(ny, nu, mr);
            Value denOut = Value::cell(ny, nu, mr);
            for (keel::usize iy = 0; iy < ny; ++iy) {
                Value Crow = Value::matrix(1, n, ValueType::DOUBLE, mr);
                for (keel::usize j = 0; j < n; ++j)
                    Crow.doubleDataMut()[j] = C.elemAsDouble(iy + j * ny);
                Value Drow = Value::matrix(1, nu, ValueType::DOUBLE, mr);
                for (keel::usize j = 0; j < nu; ++j)
                    Drow.doubleDataMut()[j] = (D.numel() > 0) ? D.elemAsDouble(iy + j * ny) : 0.0;
                for (keel::usize iu = 0; iu < nu; ++iu) {
                    auto [nij, dij] = ss2tf(A, B, Crow, Drow, keel::as<keel::i32>(iu + 1), mr);
                    const keel::usize L = keel::max(nij.numel(), dij.numel());
                    numOut.cellAt(iy + iu * ny) = padToLen(nij, L, mr);
                    denOut.cellAt(iy + iu * ny) = padToLen(dij, L, mr);
                }
            }
            return {keel::move(numOut), keel::move(denOut)};
        }
        keel::tie(num, den) = ss2tf(A, B, C, D, 1, mr);
    }
    const keel::usize L = keel::max(num.numel(), den.numel());
    Value np = padToLen(num, L, mr);
    Value dp = padToLen(den, L, mr);
    if (asVector) return {keel::move(np), keel::move(dp)};
    return {wrapCell(np, mr), wrapCell(dp, mr)};
}

keel::Tuple<Value, Value, Value>
zpkdata(const Value &sys, bool asVector, keel::Allocator *mr)
{
    zpkdata_argcheck(sys);
    const Value &s = internal::ltiStruct(sys);
    Value z, p, k;
    const keel::String kind = kindOf(sys);
    if (kind == "zpk") {
        z = colVec(s.field("z"), mr);
        p = colVec(s.field("p"), mr);
        k = Value::scalar(s.field("k").toScalar(), mr);
    } else if (kind == "tf") {
        auto r = tf2zp(s.field("num"), s.field("den"), mr);
        z = colVec(r.z, mr);
        p = colVec(r.p, mr);
        k = Value::scalar(r.k.toScalar(), mr);
    } else {
        auto [n, d] = ss2tf(s.field("A"), s.field("B"),
                            s.field("C"), s.field("D"), 1, mr);
        auto r = tf2zp(n, d, mr);
        z = colVec(r.z, mr);
        p = colVec(r.p, mr);
        k = Value::scalar(r.k.toScalar(), mr);
    }
    if (asVector) return {keel::move(z), keel::move(p), keel::move(k)};
    return {wrapCell(z, mr), wrapCell(p, mr), keel::move(k)};
}

keel::Tuple<Value, Value, Value, Value>
ssdata(const Value &sys, keel::Allocator *mr)
{
    ssdata_argcheck(sys);
    const Value &s = internal::ltiStruct(sys);
    const keel::String kind = kindOf(sys);
    Value A, B, C, D;
    if (kind == "ss") {
        A = s.field("A");
        B = s.field("B");
        C = s.field("C");
        D = s.field("D");
    } else if (kind == "tf") {
        auto ss = tf2ss(s.field("num"), s.field("den"), mr);
        A = keel::move(ss.A); B = keel::move(ss.B);
        C = keel::move(ss.C); D = keel::move(ss.D);
    } else {
        auto [n, d] = zp2tf(s.field("z"), s.field("p"), s.field("k"), mr);
        auto ss = tf2ss(n, d, mr);
        A = keel::move(ss.A); B = keel::move(ss.B);
        C = keel::move(ss.C); D = keel::move(ss.D);
    }
    return {keel::move(A), keel::move(B), keel::move(C), keel::move(D)};
}

keel::Tuple<Value, Value>
frdata(const Value &sys, keel::Allocator *mr)
{
    frdata_argcheck(sys);
    const Value &s = internal::ltiStruct(sys);
    // Shallow copy through the user resource; resp / freq are already
    // stored as column vectors by frd().
    Value resp = s.field("resp");
    Value freq = s.field("freq");
    // Defensive copy so callers don't observe aliasing into the struct.
    auto copyV = [&](const Value &v) -> Value {
        const keel::usize N = v.numel();
        if (v.type() == ValueType::COMPLEX) {
            Value out = Value::matrix(N, 1, ValueType::COMPLEX, mr);
            const keel::c64 *src = v.complexData();
            keel::c64 *od = out.complexDataMut();
            for (keel::usize i = 0; i < N; ++i) od[i] = src[i];
            return out;
        }
        Value out = Value::matrix(N, 1, ValueType::DOUBLE, mr);
        keel::f64 *od = out.doubleDataMut();
        for (keel::usize i = 0; i < N; ++i) od[i] = v.elemAsDouble(i);
        return out;
    };
    return {copyV(resp), copyV(freq)};
}

Value ss2ss(const Value &sys, const Value &T, keel::Allocator *mr)
{
    Value Tc;
    ss2ss_argcheck(sys, T, Tc, mr);
    const Value &s = internal::ltiStruct(sys);
    const Value &Av = s.field("A");
    const Value &Bv = s.field("B");
    const Value &Cv = s.field("C");
    const Value &Dv = s.field("D");
    const keel::usize n = Av.dims().rows();
    const keel::usize m = Bv.dims().cols();
    const keel::usize p = Cv.dims().rows();

    if (Tc.isComplex() || Av.isComplex() || Bv.isComplex() || Cv.isComplex()) {
        keel::Array<keel::c64> Tinv;
        if (!inverse_n_cplx(Tc, Tinv, n, mr))
            throw Error("ss2ss: T is singular",
                        0, 0, "ss2ss", "", "numkit:ss2ss:singular");
        auto Tm = readCMat(Tc);
        auto Am = readCMat(Av);
        auto Bm = readCMat(Bv);
        auto Cm = readCMat(Cv);
        keel::Array<keel::c64> M, Anew, Bnew, Cnew;
        matmul_n(Am, n, n, Tinv, n, n, M);
        matmul_n(Tm, n, n, M, n, n, Anew);
        matmul_n(Tm, n, n, Bm, n, m, Bnew);
        matmul_n(Cm, p, n, Tinv, n, n, Cnew);
        auto putCMat = [&](keel::ArrayRef<keel::c64> v, keel::usize rr, keel::usize cc) {
            Value out = Value::matrix(rr, cc, ValueType::COMPLEX, mr);
            keel::c64 *od = out.complexDataMut();
            for (keel::usize i = 0; i < rr * cc; ++i) od[i] = v[i];
            return out;
        };
        Value out = tagStruct("ss", s.field("Ts").toScalar(), mr);
        out.field("A") = putCMat(Anew, n, n);
        out.field("B") = putCMat(Bnew, n, m);
        out.field("C") = putCMat(Cnew, p, n);
        out.field("D") = Dv;
        return internal::wrapLtiObject("ss", keel::move(out), mr);
    }

    keel::Array<keel::f64> Tinv;
    if (!inverse_n(Tc, Tinv, n, mr))
        throw Error("ss2ss: T is singular",
                    0, 0, "ss2ss", "", "numkit:ss2ss:singular");

    keel::Array<keel::f64> Tm = readMat(Tc);
    keel::Array<keel::f64> Am = readMat(Av);
    keel::Array<keel::f64> Bm = readMat(Bv);
    keel::Array<keel::f64> Cm = readMat(Cv);

    keel::Array<keel::f64> M, Anew, Bnew, Cnew;
    matmul_n(Am, n, n, Tinv, n, n, M);
    matmul_n(Tm, n, n, M, n, n, Anew);
    matmul_n(Tm, n, n, Bm, n, m, Bnew);
    matmul_n(Cm, p, n, Tinv, n, n, Cnew);

    auto putMat = [&](keel::ArrayRef<keel::f64> v, keel::usize rr, keel::usize cc) {
        Value out = Value::matrix(rr, cc, ValueType::DOUBLE, mr);
        keel::f64 *od = out.doubleDataMut();
        for (keel::usize i = 0; i < rr * cc; ++i) od[i] = v[i];
        return out;
    };

    Value out = tagStruct("ss", s.field("Ts").toScalar(), mr);
    out.field("A") = putMat(Anew, n, n);
    out.field("B") = putMat(Bnew, n, m);
    out.field("C") = putMat(Cnew, p, n);
    out.field("D") = Dv;
    return internal::wrapLtiObject("ss", keel::move(out), mr);
}

} // namespace numkit::control
