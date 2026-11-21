// toolboxes/control/src/internal/numerics.cpp
//
// Shared numerical kernels delegated to keel::control — see internal/numerics.hpp.

#include <keel/core.h>
#include <keel/control.h>
#include <numkit/control/internal/numerics.hpp>

namespace numkit::control::internal {

bool solveInPlace(Mat &A, Mat &B, keel::usize n, keel::usize nrhs)
{
    return keel::control::solve_in_place(A, B, n, nrhs) == keel::control::ControlStatus::kOk;
}

bool solveInPlace(CMat &A, CMat &B, keel::usize n, keel::usize nrhs)
{
    return keel::control::solve_in_place_c64(A, B, n, nrhs) == keel::control::ControlStatus::kOk;
}

Mat matmulSq(const Mat &A, const Mat &B, keel::usize n)
{
    Mat C(n * n, 0.0);
    (void)keel::control::matmul(A, n, n, B, n, C);
    return C;
}

CMat matmulSq(const CMat &A, const CMat &B, keel::usize n)
{
    CMat C(n * n, keel::c64(0.0, 0.0));
    (void)keel::control::matmul_c64(A, n, n, B, n, C);
    return C;
}

Mat expm(const Mat &Ain, keel::usize n)
{
    if (n == 0) return Mat{};
    Mat X(n * n, 0.0);
    (void)keel::control::expm(Ain, n, X);
    return X;
}

Vec charPoly(const Mat &A, keel::usize n)
{
    Vec coeff(n + 1, 0.0);
    (void)keel::control::char_poly(A, n, coeff);
    return coeff;
}

} // namespace numkit::control::internal
