// toolboxes/control/src/lyapunov/lyapunov.cpp
//
// Direct Kronecker-vectorisation Lyapunov solver. The continuous
// equation
//     A·X + X·Aᵀ + Q = 0
// vectorises to
//     (I ⊗ A  +  A ⊗ I) · vec(X) = −vec(Q),
// while the discrete version
//     A·X·Aᵀ − X + Q = 0
// becomes
//     (A ⊗ A  −  I) · vec(X) = −vec(Q).
//
// The matrices on the left are n²·n² with full-rank algebra for
// stable A; we solve them directly via partial-pivot LU. This is
// O(n⁶) work — perfectly fine up to n ≈ 32 (typical for textbook
// control problems). Bartels–Stewart is the textbook upgrade for
// larger systems but would need a Schur decomposition we don't
// have yet.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/complex.h>
#include <numkit/control/lyapunov/lyapunov.hpp>
#include <numkit/control/internal/numerics.hpp>
#include <numkit/control/internal/type_gates.hpp>

// Compute-only TU: Value substrate + Error, no engine. The lyap / dlyap
// builtins (CallContext wrappers) live in lyapunov/lyapunov_reg.cpp.
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace numkit::control {

namespace {

using Mat = internal::Mat;
using Vec = internal::Vec;
using CMat = internal::CMat;
using CVec = internal::CVec;
using internal::solveInPlace;

Mat readMat(const Value &v, keel::usize r, keel::usize c) {
    Mat M(r * c, 0.0);
    for (keel::usize i = 0; i < r * c; ++i) M[i] = v.elemAsDouble(i);
    return M;
}

CMat readCMat(const Value &v, keel::usize r, keel::usize c) {
    CMat M(r * c, keel::c64(0.0, 0.0));
    if (v.isComplex()) {
        const keel::c64 *cd = v.complexData();
        for (keel::usize i = 0; i < r * c; ++i) M[i] = cd[i];
    } else {
        for (keel::usize i = 0; i < r * c; ++i) M[i] = keel::c64(v.elemAsDouble(i), 0.0);
    }
    return M;
}

Value matFromVec(keel::usize r, keel::usize c, const Mat &v, keel::Allocator *mr) {
    Value m = Value::matrix(r, c, ValueType::DOUBLE, mr);
    if (!v.empty()) std::copy(v.begin(), v.end(), m.doubleDataMut());
    return m;
}

Value cmatFromVec(keel::usize r, keel::usize c, const CMat &v, keel::Allocator *mr) {
    Value m = Value::matrix(r, c, ValueType::COMPLEX, mr);
    if (!v.empty()) std::copy(v.begin(), v.end(), m.complexDataMut());
    return m;
}

// lyap/dlyap argcheck (argcheck separation). Type classes probed
// R2025b: A and Q coerce every numeric class AND char (lyap(int8 A),
// lyap('12' A) both compute in the reference); cell/string are the
// inputType rejection; complex A or Q yields a complex solution.
// Square/shape checks follow type.
void lyap_argcheck(const Value &A, const Value &Q, const char *name)
{
    internal::requireCoercibleMatrix(A, name, "A", 1);
    internal::requireCoercibleMatrix(Q, name, "Q", 2);
    if (A.dims().rows() != A.dims().cols())
        throw Error(std::string(name) + ": A must be square",
                    0, 0, name, "", std::string("numkit:") + name + ":A");
    if (Q.dims().rows() != A.dims().rows() ||
        Q.dims().cols() != A.dims().cols())
        throw Error(std::string(name) + ": Q must match A in shape",
                    0, 0, name, "", std::string("numkit:") + name + ":Q");
}

// Solve a Kronecker-form linear system M·x = b, where M is n²×n²
// in column-major. Inputs are pre-built.
//
// Used by both lyap and dlyap; the only difference is how M is
// constructed. `fn` names the entry in the singular-system voice.
template <typename MatT>
MatT solveKron(MatT &M, MatT &rhs, keel::usize n2, const char *fn)
{
    MatT rhsMat = rhs;  // single-rhs solveInPlace expects column-major
    if (!solveInPlace(M, rhsMat, n2, 1))
        throw Error(std::string(fn) + ": the Lyapunov equation is "
                    "singular (check that A is stable / the equation "
                    "is solvable)",
                    0, 0, fn, "",
                    std::string("numkit:") + fn + ":singular");
    return rhsMat;
}

} // anonymous

Value lyap(const Value &Av, const Value &Qv, keel::Allocator *mr)
{
    lyap_argcheck(Av, Qv, "lyap");
    const keel::usize n = Av.dims().rows();
    const bool isCplx = Av.isComplex() || Qv.isComplex();
    if (n == 0) return Value::matrix(0, 0, isCplx ? ValueType::COMPLEX : ValueType::DOUBLE, mr);

    const keel::usize n2 = n * n;
    if (isCplx) {
        auto A = readCMat(Av, n, n);
        auto Q = readCMat(Qv, n, n);
        // Continuous Hermitian Lyapunov: A·X + X·A^H + Q = 0
        // vectorises to (I ⊗ A + conj(A) ⊗ I) · vec(X) = −vec(Q).
        CMat M(n2 * n2, keel::c64(0.0, 0.0));
        auto setEntry = [&](keel::usize row, keel::usize col, keel::c64 v) {
            M[col * n2 + row] += v;
        };
        for (keel::usize p = 0; p < n; ++p)
            for (keel::usize i = 0; i < n; ++i)
                for (keel::usize j = 0; j < n; ++j)
                    setEntry(i + n * p, j + n * p, A[j * n + i]);
        for (keel::usize p = 0; p < n; ++p)
            for (keel::usize q = 0; q < n; ++q)
                for (keel::usize i = 0; i < n; ++i)
                    setEntry(i + n * p, i + n * q, keel::conj(A[q * n + p]));
        CVec rhs(n2);
        for (keel::usize j = 0; j < n; ++j)
            for (keel::usize i = 0; i < n; ++i)
                rhs[i + n * j] = -Q[j * n + i];
        auto x = solveKron(M, rhs, n2, "lyap");
        CMat X(n2);
        for (keel::usize j = 0; j < n; ++j)
            for (keel::usize i = 0; i < n; ++i)
                X[j * n + i] = x[i + n * j];
        return cmatFromVec(n, n, X, mr);
    }

    auto A = readMat(Av, n, n);
    auto Q = readMat(Qv, n, n);

    // Build M = I⊗A + A⊗I, of size n²×n² (column-major).
    // Kronecker convention used throughout: (P⊗R)[i+np, j+nq]
    // = P[p,q] · R[i,j]. The vec(X) layout is column-major so
    // vec(X)[i + n·j] = X[i,j].
    Mat M(n2 * n2, 0.0);

    auto setEntry = [&](keel::usize row, keel::usize col, double v) {
        M[col * n2 + row] += v;
    };

    // (I⊗A): row = i + n·p, col = j + n·q with q == p, contribution A[i,j].
    for (keel::usize p = 0; p < n; ++p)
        for (keel::usize i = 0; i < n; ++i)
            for (keel::usize j = 0; j < n; ++j) {
                const keel::usize row = i + n * p;
                const keel::usize col = j + n * p;
                setEntry(row, col, A[j * n + i]);  // A[i, j] in col-major: A[j*n + i]
            }

    // (A⊗I): row = i + n·p, col = j + n·q with i == j, contribution A[p,q].
    for (keel::usize p = 0; p < n; ++p)
        for (keel::usize q = 0; q < n; ++q)
            for (keel::usize i = 0; i < n; ++i) {
                const keel::usize row = i + n * p;
                const keel::usize col = i + n * q;
                setEntry(row, col, A[q * n + p]);   // A[p, q]
            }

    // RHS = -vec(Q), column-major.
    Vec rhs(n2, 0.0);
    for (keel::usize j = 0; j < n; ++j)
        for (keel::usize i = 0; i < n; ++i)
            rhs[i + n * j] = -Q[j * n + i];

    auto x = solveKron(M, rhs, n2, "lyap");

    // Reshape vec back into n×n column-major X.
    Mat X(n2, 0.0);
    for (keel::usize j = 0; j < n; ++j)
        for (keel::usize i = 0; i < n; ++i)
            X[j * n + i] = x[i + n * j];
    return matFromVec(n, n, X, mr);
}

Value dlyap(const Value &Av, const Value &Qv, keel::Allocator *mr)
{
    lyap_argcheck(Av, Qv, "dlyap");
    const keel::usize n = Av.dims().rows();
    const bool isCplx = Av.isComplex() || Qv.isComplex();
    if (n == 0) return Value::matrix(0, 0, isCplx ? ValueType::COMPLEX : ValueType::DOUBLE, mr);

    const keel::usize n2 = n * n;
    if (isCplx) {
        auto A = readCMat(Av, n, n);
        auto Q = readCMat(Qv, n, n);
        // Discrete Hermitian Lyapunov: A·X·A^H − X + Q = 0
        // vectorises to (conj(A) ⊗ A − I) · vec(X) = −vec(Q).
        CMat M(n2 * n2, keel::c64(0.0, 0.0));
        for (keel::usize p = 0; p < n; ++p)
            for (keel::usize q = 0; q < n; ++q)
                for (keel::usize i = 0; i < n; ++i)
                    for (keel::usize j = 0; j < n; ++j) {
                        const keel::usize row = i + n * p;
                        const keel::usize col = j + n * q;
                        M[col * n2 + row] += keel::conj(A[q * n + p]) * A[j * n + i];
                    }
        for (keel::usize k = 0; k < n2; ++k) M[k * n2 + k] -= 1.0;
        CVec rhs(n2);
        for (keel::usize j = 0; j < n; ++j)
            for (keel::usize i = 0; i < n; ++i)
                rhs[i + n * j] = -Q[j * n + i];
        auto x = solveKron(M, rhs, n2, "dlyap");
        CMat X(n2);
        for (keel::usize j = 0; j < n; ++j)
            for (keel::usize i = 0; i < n; ++i)
                X[j * n + i] = x[i + n * j];
        return cmatFromVec(n, n, X, mr);
    }

    auto A = readMat(Av, n, n);
    auto Q = readMat(Qv, n, n);

    Mat M(n2 * n2, 0.0);

    // (A⊗A): row = i + n·p, col = j + n·q, contribution A[p,q] · A[i,j].
    for (keel::usize p = 0; p < n; ++p)
        for (keel::usize q = 0; q < n; ++q)
            for (keel::usize i = 0; i < n; ++i)
                for (keel::usize j = 0; j < n; ++j) {
                    const keel::usize row = i + n * p;
                    const keel::usize col = j + n * q;
                    M[col * n2 + row] += A[q * n + p] * A[j * n + i];
                }
    // − I (n² × n²)
    for (keel::usize k = 0; k < n2; ++k) M[k * n2 + k] -= 1.0;

    Vec rhs(n2, 0.0);
    for (keel::usize j = 0; j < n; ++j)
        for (keel::usize i = 0; i < n; ++i)
            rhs[i + n * j] = -Q[j * n + i];

    auto x = solveKron(M, rhs, n2, "dlyap");

    Mat X(n2, 0.0);
    for (keel::usize j = 0; j < n; ++j)
        for (keel::usize i = 0; i < n; ++i)
            X[j * n + i] = x[i + n * j];
    return matFromVec(n, n, X, mr);
}

} // namespace numkit::control
