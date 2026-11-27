// toolboxes/control/src/riccati/riccati.cpp
//
// Algebraic Riccati solvers (care / dare), delegated to keel::control.

#include <keel/core.h>
#include <keel/algo.h>
#include <keel/control.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <numkit/control/riccati/riccati.hpp>
#include <numkit/control/internal/numerics.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/builtin/polyfun.hpp>   // numkit::builtin::roots


namespace numkit::control {

namespace {

using Mat = internal::Mat;
using internal::charPoly;

Mat readMat(const Value &v, keel::usize r, keel::usize c) {
    Mat M(r * c, 0.0);
    for (keel::usize i = 0; i < r * c; ++i) M[i] = v.elemAsDouble(i);
    return M;
}

keel::f64 frobNorm(const Mat &M) noexcept {
    keel::f64 mx = 0.0;
    for (keel::usize i = 0; i < M.size(); ++i) {
        const keel::f64 v = keel::abs(M[i]);
        if (!keel::is_finite(v)) return keel::Limits<keel::f64>::quiet_nan();
        if (v > mx) mx = v;
    }
    if (mx == 0.0) return 0.0;
    const keel::f64 inv = 1.0 / mx;
    keel::f64 s = 0.0;
    for (keel::usize i = 0; i < M.size(); ++i) {
        const keel::f64 u = M[i] * inv;
        s += u * u;
    }
    return mx * keel::sqrt(s);
}

Value closedLoopEig(const Mat &A, const Mat &B, const keel::f64 *G,
                    keel::usize n, keel::usize m, keel::Allocator *mr) {
    Mat BG(n * n, 0.0);
    (void)keel::control::matmul(B, n, m, {G, m * n}, n, BG);
    Mat Acl(n * n, 0.0);
    for (keel::usize i = 0; i < n * n; ++i) Acl[i] = A[i] - BG[i];
    auto cp = charPoly(Acl, n);
    Value row = Value::matrix(1, cp.size(), ValueType::DOUBLE, mr);
    if (!cp.empty()) keel::copy(cp.begin(), cp.end(), row.doubleDataMut());
    Value eigs = numkit::builtin::roots(row, mr);
    // Canonicalize adjacent complex conjugate pairs to (+imag, -imag) order (LAPACK convention).
    if (eigs.type() == ValueType::COMPLEX && eigs.numel() >= 2) {
        keel::c64 *d = eigs.complexDataMut();
        for (keel::usize i = 0; i + 1 < eigs.numel(); ++i) {
            if (d[i].imag() < 0.0 && d[i + 1].imag() > 0.0
                && keel::abs(d[i].real() - d[i + 1].real()) <= 1e-9 * (1.0 + keel::abs(d[i].real()))
                && keel::abs(d[i].imag() + d[i + 1].imag()) <= 1e-9 * (1.0 + keel::abs(d[i].imag()))) {
                keel::swap(d[i], d[i + 1]);
                ++i;
            }
        }
    }
    return eigs;
}

void riccati_argcheck(const Value &A, const Value &B, const Value &Q,
                      const Value &R, const char *name,
                      keel::usize &n, keel::usize &m) {
    if (A.dims().rows() != A.dims().cols())
        throw Error(keel::format("{}: A must be square", name),
                    0, 0, name, "", "numkit:care:A");
    n = A.dims().rows();
    m = B.dims().cols();
    if (B.dims().rows() != n)
        throw Error(keel::format("{}: B must have the same row count as A", name),
                    0, 0, name, "", "numkit:care:B");
    if (Q.dims().rows() != n || Q.dims().cols() != n)
        throw Error(keel::format("{}: Q must be n×n", name),
                    0, 0, name, "", "numkit:care:Q");
    if (R.dims().rows() != m || R.dims().cols() != m)
        throw Error(keel::format("{}: R must be m×m (m = columns of B)", name),
                    0, 0, name, "", "numkit:care:R");
}

void care_argcheck(const Value &A, const Value &B, const Value &Q,
                   const Value &R, keel::usize &n, keel::usize &m) {
    riccati_argcheck(A, B, Q, R, "care", n, m);
}

void dare_argcheck(const Value &A, const Value &B, const Value &Q,
                   const Value &R, keel::usize &n, keel::usize &m) {
    riccati_argcheck(A, B, Q, R, "dare", n, m);
}

} // anonymous

RiccatiResult care(const Value &Av, const Value &Bv, const Value &Qv,
                   const Value &Rv, keel::Allocator *mr)
{
    keel::usize n = 0, m = 0;
    care_argcheck(Av, Bv, Qv, Rv, n, m);

    auto A = readMat(Av, n, n);
    auto B = readMat(Bv, n, m);
    auto Q = readMat(Qv, n, n);
    auto R = readMat(Rv, m, m);

    // Symplectic diagonal scaling: X = gamma * X_tilde transforms
    //   A'X + XA - X*B*R^-1*B'*X + Q = 0
    // into
    //   A'*X_tilde + X_tilde*A - X_tilde*B*(R/gamma)^-1*B'*X_tilde + (Q/gamma) = 0,
    // preserving the gain G = R^-1*B'*X = (R/gamma)^-1*B'*X_tilde while making
    // ||Q/gamma||_F == ||gamma*B*R^-1*B'||_F for the Hamiltonian matrix-sign iteration.
    keel::f64 gamma = 1.0;
    bool scaled = false;
    if (n > 0 && m > 0) {
        const keel::f64 qNorm = frobNorm(Q);
        Mat rLu = R.clone();
        Mat bt(m * n, 0.0);
        for (keel::usize j = 0; j < m; ++j)
            for (keel::usize i = 0; i < n; ++i)
                bt[i * m + j] = B[j * n + i];
        if (keel::control::solve_in_place(rLu, bt, m, n) == keel::control::ControlStatus::kOk) {
            Mat gbr(n * n, 0.0);
            (void)keel::control::matmul(B, n, m, bt, n, gbr);
            const keel::f64 gNorm = frobNorm(gbr);
            if (keel::is_finite(qNorm) && keel::is_finite(gNorm) && qNorm > 0.0 && gNorm > 0.0) {
                const keel::f64 candidate = keel::sqrt(qNorm / gNorm);
                if (keel::is_finite(candidate) && (candidate > 10.0 || candidate < 0.1)) {
                    gamma = candidate;
                    const keel::f64 invGamma = 1.0 / gamma;
                    for (keel::usize i = 0; i < Q.size(); ++i) Q[i] *= invGamma;
                    for (keel::usize i = 0; i < R.size(); ++i) R[i] *= invGamma;
                    scaled = true;
                }
            }
        }
    }

    RiccatiResult out;
    out.X = Value::matrix(n, n, ValueType::DOUBLE, mr);
    out.G = Value::matrix(m, n, ValueType::DOUBLE, mr);

    const auto st = keel::control::care(A, B, Q, R, n, m,
                                        {out.X.doubleDataMut(), n * n},
                                        {out.G.doubleDataMut(), m * n}, mr);
    using S = keel::control::ControlStatus;
    if (st == S::kSingularR)
        throw Error("care: R must be nonsingular", 0, 0, "care", "", "numkit:care:Rsingular");
    if (st == S::kSignFailed)
        throw Error("care: Hamiltonian sign iteration failed (singular embedding)",
                    0, 0, "care", "", "numkit:care:sign");
    if (st != S::kOk)
        throw Error("care: stable-subspace solve is singular",
                    0, 0, "care", "", "numkit:care:subspace");

    if (scaled) {
        keel::f64 *xd = out.X.doubleDataMut();
        for (keel::usize i = 0; i < n * n; ++i) xd[i] *= gamma;
    }

    out.L = closedLoopEig(A, B, out.G.doubleData(), n, m, mr);
    return out;
}

RiccatiResult dare(const Value &Av, const Value &Bv, const Value &Qv,
                   const Value &Rv, keel::Allocator *mr)
{
    keel::usize n = 0, m = 0;
    dare_argcheck(Av, Bv, Qv, Rv, n, m);

    auto A = readMat(Av, n, n);
    auto B = readMat(Bv, n, m);
    auto Q = readMat(Qv, n, n);
    auto R = readMat(Rv, m, m);

    RiccatiResult out;
    out.X = Value::matrix(n, n, ValueType::DOUBLE, mr);
    out.G = Value::matrix(m, n, ValueType::DOUBLE, mr);

    const auto st = keel::control::dare(A, B, Q, R, n, m,
                                        {out.X.doubleDataMut(), n * n},
                                        {out.G.doubleDataMut(), m * n}, mr);
    using S = keel::control::ControlStatus;
    if (st == S::kSingularR)
        throw Error("dare: R must be nonsingular", 0, 0, "dare", "", "numkit:dare:Rsingular");
    if (st == S::kSingularA)
        throw Error("dare: A must be nonsingular for the symplectic form "
                    "(singular-A path needs a QZ solver, not yet implemented)",
                    0, 0, "dare", "", "numkit:dare:Asingular");
    if (st == S::kSingularCayley)
        throw Error("dare: Cayley transform singular (Sp has an eigenvalue at −1)",
                    0, 0, "dare", "", "numkit:dare:cayley");
    if (st == S::kSignFailed)
        throw Error("dare: symplectic sign iteration failed (singular embedding)",
                    0, 0, "dare", "", "numkit:dare:sign");
    if (st == S::kSingularGain)
        throw Error("dare: gain solve singular (R + BᵀXB not invertible)",
                    0, 0, "dare", "", "numkit:dare:gain");
    if (st != S::kOk)
        throw Error("dare: stable-subspace solve is singular",
                    0, 0, "dare", "", "numkit:dare:subspace");

    out.L = closedLoopEig(A, B, out.G.doubleData(), n, m, mr);
    return out;
}

} // namespace numkit::control
