// toolboxes/control/src/state/state.cpp
//
// ctrb / obsv. Pure matrix algebra — no expm, no eigensolver. The
// 2-arg forms accept (A, B) / (A, C) coefficient pairs directly, so
// they work without an LTI struct; the 1-arg forms peel A,B,C out
// of the cycle-31 sys struct (going through ss form for tf/zpk).

#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/complex.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/control/state/state.hpp>
#include <numkit/control/conversion/conversion.hpp>
#include <numkit/control/lyapunov/lyapunov.hpp>   // gram delegates to lyap / dlyap
#include <numkit/control/internal/type_gates.hpp>

// Compute-only TU: Value substrate + Error, no engine. The ctrb / obsv
// builtins (CallContext wrappers) live in state/state_reg.cpp.
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace numkit::control {

namespace {

bool hasKind(const Value &sys, const char *want) {
    if (!sys.isStruct() || !sys.hasField("kind")) return false;
    return sys.field("kind").toString() == want;
}

// (A, B, C) extraction: returns A,B,C as plain Values regardless of
// the input form. The shapes are ss-canonical:
//   A : n×n, B : n×m, C : p×n.
struct ABC { Value A, B, C, D; };
ABC pullABC(const Value &sys, const char *fn, keel::Allocator *mr) {
    ABC out;
    if (hasKind(sys, "ss")) {
        out.A = sys.field("A"); out.B = sys.field("B");
        out.C = sys.field("C"); out.D = sys.field("D");
    } else if (hasKind(sys, "tf")) {
        auto ss = tf2ss(sys.field("num"), sys.field("den"), mr);
        out.A = std::move(ss.A); out.B = std::move(ss.B);
        out.C = std::move(ss.C); out.D = std::move(ss.D);
    } else if (hasKind(sys, "zpk")) {
        auto [num, den] = zp2tf(sys.field("z"), sys.field("p"), sys.field("k"), mr);
        auto ss = tf2ss(num, den, mr);
        out.A = std::move(ss.A); out.B = std::move(ss.B);
        out.C = std::move(ss.C); out.D = std::move(ss.D);
    } else {
        throw Error(keel::format("{}: input 1 (sys) must be an LTI struct (tf/zpk/ss)", fn),
                    0, 0, fn, "", keel::format("numkit:{}:kind", fn));
    }
    return out;
}

// Read a matrix into a column-major double buffer.
keel::Array<double> readMat(const Value &v, keel::usize r, keel::usize c) {
    keel::Array<double> M(r * c, 0.0);
    for (keel::usize i = 0; i < r * c; ++i) M[i] = v.elemAsDouble(i);
    return M;
}

keel::Array<keel::c64> readCMat(const Value &v, keel::usize r, keel::usize c) {
    keel::Array<keel::c64> M(r * c, keel::c64(0.0, 0.0));
    if (v.isComplex()) {
        const keel::c64 *cd = v.complexData();
        for (keel::usize i = 0; i < r * c; ++i) M[i] = cd[i];
    } else {
        for (keel::usize i = 0; i < r * c; ++i) M[i] = keel::c64(v.elemAsDouble(i), 0.0);
    }
    return M;
}

Value matFromVec(keel::usize r, keel::usize c, keel::ArrayRef<double> v, keel::Allocator *mr) {
    Value m = Value::matrix(r, c, ValueType::DOUBLE, mr);
    if (!v.empty()) std::copy(v.begin(), v.end(), m.doubleDataMut());
    return m;
}

Value cmatFromVec(keel::usize r, keel::usize c, keel::ArrayRef<keel::c64> v, keel::Allocator *mr) {
    Value m = Value::matrix(r, c, ValueType::COMPLEX, mr);
    if (!v.empty()) std::copy(v.begin(), v.end(), m.complexDataMut());
    return m;
}

// ctrb/obsv argcheck (argcheck separation). Classes probed R2025b:
// double/single/logical/char coerce; integer matrices fail inside the
// reference's matmul (integerNotSupported) — mirrored with its own
// identifier; complex operands produce a complex result; cell/string
// are inputType.
void ctrb_obsv_argcheck(const Value &A, const Value &X, const char *fn,
                        const char *xName, int xNum)
{
    internal::requireCoercibleMatrix(A, fn, "A", 1);
    internal::requireNonIntegerMatrix(A, fn, "A", 1);
    internal::requireCoercibleMatrix(X, fn, xName, xNum);
    internal::requireNonIntegerMatrix(X, fn, xName, xNum);
}

char gram_argcheck(const Value &type)
{
    if (!type.isChar() && !type.isString())
        throw Error(keel::format("gram: input 2 (type) must be 'c' (controllability) or 'o' (observability), as char or string (got a {} {} array).",
                                 internal::shapeOf(type), mtypeName(type.type())),
                    0, 0, "gram", "", "numkit:gram:type");
    const keel::String ts = type.toString();
    const char t = ts.empty()
                 ? 'c'
                 : keel::to_ascii_lower(ts[0]);
    if (t != 'c' && t != 'o')
        throw Error(keel::format("gram: input 2 (type) must be 'c' (controllability) or 'o' (observability), as char or string (got '{}').", ts),
                    0, 0, "gram", "", "numkit:gram:type");
    return t;
}

void covar_argcheck(const Value &sys, const Value &Wv)
{
    (void)sys;
    (void)Wv;
}

} // anonymous

Value ctrb_AB(const Value &Av, const Value &Bv, keel::Allocator *mr)
{
    ctrb_obsv_argcheck(Av, Bv, "ctrb", "B", 2);
    const keel::usize n = Av.dims().rows();
    if (Av.dims().cols() != n)
        throw Error("ctrb: A must be square",
                    0, 0, "ctrb", "", "numkit:ctrb:A");
    const keel::usize m = Bv.dims().cols();
    if (Bv.dims().rows() != n)
        throw Error("ctrb: B must have the same row count as A",
                    0, 0, "ctrb", "", "numkit:ctrb:B");

    if (Av.isComplex() || Bv.isComplex()) {
        auto A = readCMat(Av, n, n);
        auto B = readCMat(Bv, n, m);
        keel::Array<keel::c64> Co(n * (n * m), keel::c64(0.0, 0.0));
        for (keel::usize j = 0; j < m; ++j)
            for (keel::usize i = 0; i < n; ++i)
                Co[j * n + i] = B[j * n + i];
        auto prev = B.clone();
        for (keel::usize k = 1; k < n; ++k) {
            keel::Array<keel::c64> next(n * m, keel::c64(0.0, 0.0));
            for (keel::usize j = 0; j < m; ++j)
                for (keel::usize i = 0; i < n; ++i) {
                    keel::c64 s(0.0, 0.0);
                    for (keel::usize l = 0; l < n; ++l)
                        s += A[l * n + i] * prev[j * n + l];
                    next[j * n + i] = s;
                }
            const keel::usize colOffset = k * m;
            for (keel::usize j = 0; j < m; ++j)
                for (keel::usize i = 0; i < n; ++i)
                    Co[(colOffset + j) * n + i] = next[j * n + i];
            prev = keel::move(next);
        }
        return cmatFromVec(n, n * m, Co, mr);
    }

    auto A = readMat(Av, n, n);
    auto B = readMat(Bv, n, m);

    // Output Co is n × (n·m), column-major. Block k (k=0..n-1) holds
    // A^k · B at columns [k·m … k·m + m − 1].
    keel::Array<double> Co(n * (n * m), 0.0);
    // Block 0 = B.
    for (keel::usize j = 0; j < m; ++j)
        for (keel::usize i = 0; i < n; ++i)
            Co[j * n + i] = B[j * n + i];
    // Blocks 1..n-1 by repeated multiplication: prev = A^k · B,
    // next = A · prev.
    keel::Array<double> prev = B.clone();
    for (keel::usize k = 1; k < n; ++k) {
        keel::Array<double> next(n * m, 0.0);
        for (keel::usize j = 0; j < m; ++j)
            for (keel::usize i = 0; i < n; ++i) {
                double s = 0.0;
                for (keel::usize l = 0; l < n; ++l)
                    s += A[l * n + i] * prev[j * n + l];
                next[j * n + i] = s;
            }
        // Place next into block k of Co.
        const keel::usize colOffset = k * m;
        for (keel::usize j = 0; j < m; ++j)
            for (keel::usize i = 0; i < n; ++i)
                Co[(colOffset + j) * n + i] = next[j * n + i];
        prev = keel::move(next);
    }
    return matFromVec(n, n * m, Co, mr);
}

Value obsv_AC(const Value &Av, const Value &Cv, keel::Allocator *mr)
{
    ctrb_obsv_argcheck(Av, Cv, "obsv", "C", 2);
    const keel::usize n = Av.dims().rows();
    if (Av.dims().cols() != n)
        throw Error("obsv: A must be square",
                    0, 0, "obsv", "", "numkit:obsv:A");
    const keel::usize p = Cv.dims().rows();
    if (Cv.dims().cols() != n)
        throw Error("obsv: C must have the same column count as A",
                    0, 0, "obsv", "", "numkit:obsv:C");

    if (Av.isComplex() || Cv.isComplex()) {
        auto A = readCMat(Av, n, n);
        auto C = readCMat(Cv, p, n);
        keel::Array<keel::c64> Ob((n * p) * n, keel::c64(0.0, 0.0));
        for (keel::usize j = 0; j < n; ++j)
            for (keel::usize i = 0; i < p; ++i)
                Ob[j * (n * p) + i] = C[j * p + i];
        auto prev = C.clone();
        for (keel::usize k = 1; k < n; ++k) {
            keel::Array<keel::c64> next(p * n, keel::c64(0.0, 0.0));
            for (keel::usize j = 0; j < n; ++j)
                for (keel::usize i = 0; i < p; ++i) {
                    keel::c64 s(0.0, 0.0);
                    for (keel::usize l = 0; l < n; ++l)
                        s += prev[l * p + i] * A[j * n + l];
                    next[j * p + i] = s;
                }
            const keel::usize rowOffset = k * p;
            for (keel::usize j = 0; j < n; ++j)
                for (keel::usize i = 0; i < p; ++i)
                    Ob[j * (n * p) + (rowOffset + i)] = next[j * p + i];
            prev = keel::move(next);
        }
        return cmatFromVec(n * p, n, Ob, mr);
    }

    auto A = readMat(Av, n, n);
    auto C = readMat(Cv, p, n);

    // Output Ob is (n·p) × n, column-major. Block k (k=0..n-1) holds
    // C · A^k at rows [k·p … k·p + p − 1].
    keel::Array<double> Ob((n * p) * n, 0.0);
    // Block 0 = C.
    for (keel::usize j = 0; j < n; ++j)
        for (keel::usize i = 0; i < p; ++i)
            Ob[j * (n * p) + i] = C[j * p + i];
    // Walk k = 1..n-1: next = prev · A.
    keel::Array<double> prev = C.clone();
    for (keel::usize k = 1; k < n; ++k) {
        keel::Array<double> next(p * n, 0.0);
        for (keel::usize j = 0; j < n; ++j)
            for (keel::usize i = 0; i < p; ++i) {
                double s = 0.0;
                for (keel::usize l = 0; l < n; ++l)
                    s += prev[l * p + i] * A[j * n + l];
                next[j * p + i] = s;
            }
        const keel::usize rowOffset = k * p;
        for (keel::usize j = 0; j < n; ++j)
            for (keel::usize i = 0; i < p; ++i)
                Ob[j * (n * p) + (rowOffset + i)] = next[j * p + i];
        prev = keel::move(next);
    }
    return matFromVec(n * p, n, Ob, mr);
}

Value ctrb_sys(const Value &sys, keel::Allocator *mr) {
    auto abc = pullABC(sys, "ctrb", mr);
    return ctrb_AB(abc.A, abc.B, mr);
}

Value obsv_sys(const Value &sys, keel::Allocator *mr) {
    auto abc = pullABC(sys, "obsv", mr);
    return obsv_AC(abc.A, abc.C, mr);
}

Value gram(const Value &sys, const Value &type, keel::Allocator *mr)
{
    auto abc = pullABC(sys, "gram", mr);
    const char t = gram_argcheck(type);
    const keel::usize n = abc.A.dims().rows();
    const bool discrete =
        (sys.isStruct() && sys.hasField("Ts") && sys.field("Ts").toScalar() != 0.0);

    // gram solves a Lyapunov equation; the controllability / observability
    // gramian differs only in which (A, Q) pair is fed to lyap / dlyap:
    //   'c': A·Wc + Wc·Aᵀ + B·Bᵀ = 0   → lyap(A,  B·Bᵀ)
    //   'o': Aᵀ·Wo + Wo·A + Cᵀ·C = 0   → lyap(Aᵀ, Cᵀ·C)
    auto A = readMat(abc.A, n, n);
    keel::Array<double> Q(n * n, 0.0);
    Value Ause;
    if (t == 'c') {
        const keel::usize m = abc.B.dims().cols();
        auto B = readMat(abc.B, n, m);
        for (keel::usize j = 0; j < n; ++j)
            for (keel::usize i = 0; i < n; ++i) {
                double s = 0.0;
                for (keel::usize k = 0; k < m; ++k) s += B[k * n + i] * B[k * n + j];  // (B·Bᵀ)[i,j]
                Q[j * n + i] = s;
            }
        Ause = abc.A;
    } else {
        const keel::usize p = abc.C.dims().rows();
        auto C = readMat(abc.C, p, n);   // p×n column-major: C[i,j] = C[j*p + i]
        for (keel::usize j = 0; j < n; ++j)
            for (keel::usize i = 0; i < n; ++i) {
                double s = 0.0;
                for (keel::usize k = 0; k < p; ++k) s += C[i * p + k] * C[j * p + k];  // (Cᵀ·C)[i,j]
                Q[j * n + i] = s;
            }
        keel::Array<double> At(n * n, 0.0);
        for (keel::usize j = 0; j < n; ++j)
            for (keel::usize i = 0; i < n; ++i) At[j * n + i] = A[i * n + j];   // Aᵀ
        Ause = matFromVec(n, n, At, mr);
    }
    Value Qv = matFromVec(n, n, Q, mr);
    return discrete ? dlyap(Ause, Qv, mr) : lyap(Ause, Qv, mr);
}

CovarResult covar(const Value &sys, const Value &Wv, keel::Allocator *mr)
{
    covar_argcheck(sys, Wv);
    auto abc = pullABC(sys, "covar", mr);
    const keel::usize n = abc.A.dims().rows();
    const keel::usize m = abc.B.dims().cols();
    const keel::usize p = abc.C.dims().rows();
    const bool discrete =
        (sys.isStruct() && sys.hasField("Ts") && sys.field("Ts").toScalar() != 0.0);

    auto B = readMat(abc.B, n, m);
    auto C = readMat(abc.C, p, n);
    auto D = readMat(abc.D, p, m);

    // Noise intensity: scalar W → W·I_m, or a full m×m matrix.
    const bool wScalar = (Wv.numel() == 1);
    const keel::Array<double> W = wScalar ? keel::Array<double>{} : readMat(Wv, m, m);
    const double wval = wScalar ? Wv.toScalar() : 0.0;
    auto Wij = [&](keel::usize i, keel::usize j) -> double {
        return wScalar ? (i == j ? wval : 0.0) : W[j * m + i];
    };

    // State covariance Q solves the gramian Lyapunov equation with B·W·Bᵀ.
    keel::Array<double> BWBt(n * n, 0.0);
    for (keel::usize a = 0; a < n; ++a)
        for (keel::usize b = 0; b < n; ++b) {
            double s = 0.0;
            for (keel::usize i = 0; i < m; ++i)
                for (keel::usize j = 0; j < m; ++j)
                    s += B[i * n + a] * Wij(i, j) * B[j * n + b];   // B[a,i]·W[i,j]·B[b,j]
            BWBt[b * n + a] = s;
        }
    Value BWBtV = matFromVec(n, n, BWBt, mr);
    Value Q = discrete ? dlyap(abc.A, BWBtV, mr) : lyap(abc.A, BWBtV, mr);

    // Output covariance P = C·Q·Cᵀ (+ D·W·Dᵀ discrete; ∞ if continuous & D≠0).
    auto Qm = readMat(Q, n, n);
    keel::Array<double> P(p * p, 0.0);
    for (keel::usize a = 0; a < p; ++a)
        for (keel::usize b = 0; b < p; ++b) {
            double s = 0.0;
            for (keel::usize i = 0; i < n; ++i)
                for (keel::usize j = 0; j < n; ++j)
                    s += C[i * p + a] * Qm[j * n + i] * C[j * p + b];   // C[a,i]·Q[i,j]·C[b,j]
            P[b * p + a] = s;
        }
    bool dNonzero = false;
    for (double d : D) if (d != 0.0) { dNonzero = true; break; }
    if (discrete) {
        for (keel::usize a = 0; a < p; ++a)
            for (keel::usize b = 0; b < p; ++b) {
                double s = 0.0;
                for (keel::usize i = 0; i < m; ++i)
                    for (keel::usize j = 0; j < m; ++j)
                        s += D[i * p + a] * Wij(i, j) * D[j * p + b];   // D·W·Dᵀ
                P[b * p + a] += s;
            }
    } else if (dNonzero) {
        std::fill(P.begin(), P.end(), std::numeric_limits<double>::infinity());
    }

    CovarResult out;
    out.P = matFromVec(p, p, P, mr);
    out.Q = std::move(Q);
    return out;
}

} // namespace numkit::control
