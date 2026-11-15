// toolboxes/control/src/internal/numerics.cpp
//
// Shared numerical kernels — see internal/numerics.hpp.

#include <keel/core.h>
#include <keel/math.h>
#include <numkit/control/internal/numerics.hpp>

#include <algorithm>
#include <cmath>

namespace numkit::control::internal {

template <typename T>
bool solveInPlaceTpl(std::vector<T> &A, std::vector<T> &B, keel::usize n, keel::usize nrhs)
{
    for (keel::usize k = 0; k < n; ++k) {
        keel::usize pk = k;
        double bestAbs = keel::abs(A[k * n + k]);
        for (keel::usize i = k + 1; i < n; ++i) {
            const double v = keel::abs(A[k * n + i]);
            if (v > bestAbs) { bestAbs = v; pk = i; }
        }
        if (bestAbs < 1e-14) return false;
        if (pk != k) {
            for (keel::usize j = 0; j < n; ++j)
                std::swap(A[j * n + k], A[j * n + pk]);
            for (keel::usize j = 0; j < nrhs; ++j)
                std::swap(B[j * n + k], B[j * n + pk]);
        }
        const T diag = A[k * n + k];
        for (keel::usize i = k + 1; i < n; ++i) {
            const T f = A[k * n + i] / diag;
            A[k * n + i] = f;
            for (keel::usize j = k + 1; j < n; ++j)
                A[j * n + i] -= f * A[j * n + k];
            for (keel::usize j = 0; j < nrhs; ++j)
                B[j * n + i] -= f * B[j * n + k];
        }
    }
    for (keel::usize j = 0; j < nrhs; ++j) {
        for (keel::usize i = n; i-- > 0;) {
            T s = B[j * n + i];
            for (keel::usize k = i + 1; k < n; ++k)
                s -= A[k * n + i] * B[j * n + k];
            B[j * n + i] = s / A[i * n + i];
        }
    }
    return true;
}

template <typename T>
std::vector<T> matmulSqTpl(const std::vector<T> &A, const std::vector<T> &B, keel::usize n)
{
    std::vector<T> C(n * n, T(0));
    for (keel::usize j = 0; j < n; ++j)
        for (keel::usize i = 0; i < n; ++i) {
            T s(0);
            for (keel::usize k = 0; k < n; ++k)
                s += A[k * n + i] * B[j * n + k];
            C[j * n + i] = s;
        }
    return C;
}

bool solveInPlace(Mat &A, Mat &B, keel::usize n, keel::usize nrhs)
{
    return solveInPlaceTpl(A, B, n, nrhs);
}

bool solveInPlace(CMat &A, CMat &B, keel::usize n, keel::usize nrhs)
{
    return solveInPlaceTpl(A, B, n, nrhs);
}

Mat matmulSq(const Mat &A, const Mat &B, keel::usize n)
{
    return matmulSqTpl(A, B, n);
}

CMat matmulSq(const CMat &A, const CMat &B, keel::usize n)
{
    return matmulSqTpl(A, B, n);
}

namespace {

double matInfNorm(const Mat &A, keel::usize n) {
    double m = 0.0;
    for (keel::usize i = 0; i < n; ++i) {
        double s = 0.0;
        for (keel::usize j = 0; j < n; ++j) s += keel::abs(A[j * n + i]);
        m = std::max(m, s);
    }
    return m;
}

Mat eyeM(keel::usize n) {
    Mat I(n * n, 0.0);
    for (keel::usize i = 0; i < n; ++i) I[i * n + i] = 1.0;
    return I;
}

Mat zerosM(keel::usize r, keel::usize c) {
    return Mat(r * c, 0.0);
}

} // anonymous

Mat expm(const Mat &Ain, keel::usize n)
{
    if (n == 0) return Mat{};
    Mat A = Ain;
    const double normA = matInfNorm(A, n);
    int s = 0;
    if (normA > 0.5) {
        const double l2 = std::log2(normA / 0.5);
        s = static_cast<int>(std::ceil(std::max(l2, 0.0)));
    }
    if (s > 0) {
        const double scale = std::pow(0.5, s);
        for (auto &v : A) v *= scale;
    }
    // Canonical [6/6] Padé coefficients for exp(x):
    //   c_k = (12 − k)! · 6! / (12! · k! · (6 − k)!)
    static const double c[7] = {
        1.0,
        1.0 /  2.0,
        5.0 / 44.0,
        1.0 / 66.0,
        1.0 / 792.0,
        1.0 / 15840.0,
        1.0 / 665280.0
    };
    Mat I  = eyeM(n);
    Mat A2 = matmulSq(A,  A,  n);
    Mat A3 = matmulSq(A2, A,  n);
    Mat A4 = matmulSq(A2, A2, n);
    Mat A5 = matmulSq(A4, A,  n);
    Mat A6 = matmulSq(A4, A2, n);

    auto axpy = [&](Mat &dst, const Mat &src, double a) {
        for (keel::usize i = 0; i < n * n; ++i) dst[i] += a * src[i];
    };
    Mat N = zerosM(n, n), D = zerosM(n, n);
    axpy(N, I,   c[0]); axpy(D, I,   c[0]);
    axpy(N, A,   c[1]); axpy(D, A,  -c[1]);
    axpy(N, A2,  c[2]); axpy(D, A2,  c[2]);
    axpy(N, A3,  c[3]); axpy(D, A3, -c[3]);
    axpy(N, A4,  c[4]); axpy(D, A4,  c[4]);
    axpy(N, A5,  c[5]); axpy(D, A5, -c[5]);
    axpy(N, A6,  c[6]); axpy(D, A6,  c[6]);

    Mat Dcopy = D;
    Mat X     = N;
    if (!solveInPlace(Dcopy, X, n, n)) {
        // Truncated power series fallback.
        Mat E    = I;
        Mat term = I;
        for (int k = 1; k < 30; ++k) {
            term = matmulSq(term, A, n);
            const double inv = 1.0 / static_cast<double>(k);
            for (auto &v : term) v *= inv;
            for (keel::usize i = 0; i < n * n; ++i) E[i] += term[i];
        }
        X = E;
    }
    for (int k = 0; k < s; ++k) X = matmulSq(X, X, n);
    return X;
}

Vec charPoly(const Mat &A, keel::usize n)
{
    Mat M(n * n, 0.0);
    for (keel::usize i = 0; i < n; ++i) M[i * n + i] = 1.0;   // I
    Vec coeff(n + 1, 0.0);
    coeff[0] = 1.0;

    for (keel::usize k = 1; k <= n; ++k) {
        Mat AM = matmulSq(A, M, n);
        double tr = 0.0;
        for (keel::usize i = 0; i < n; ++i) tr += AM[i * n + i];
        const double ck = -tr / static_cast<double>(k);
        coeff[k] = ck;
        for (keel::usize i = 0; i < n; ++i) AM[i * n + i] += ck;
        M = std::move(AM);
    }
    return coeff;
}

} // namespace numkit::control::internal
