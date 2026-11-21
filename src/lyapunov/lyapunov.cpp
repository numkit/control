// toolboxes/control/src/lyapunov/lyapunov.cpp
//
// Continuous and discrete Lyapunov solvers, delegated to keel::control.

#include <keel/core.h>
#include <keel/control.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/complex.h>
#include <numkit/control/lyapunov/lyapunov.hpp>
#include <numkit/control/internal/numerics.hpp>
#include <numkit/control/internal/type_gates.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <algorithm>

namespace numkit::control {

namespace {

using Mat = internal::Mat;
using CMat = internal::CMat;

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

void lyap_argcheck(const Value &A, const Value &Q, const char *name)
{
    internal::requireCoercibleMatrix(A, name, "A", 1);
    internal::requireCoercibleMatrix(Q, name, "Q", 2);
    if (A.dims().rows() != A.dims().cols())
        throw Error(keel::format("{}: A must be square", name),
                    0, 0, name, "", keel::format("numkit:{}:A", name));
    if (Q.dims().rows() != A.dims().rows() ||
        Q.dims().cols() != A.dims().cols())
        throw Error(keel::format("{}: Q must match A in shape", name),
                    0, 0, name, "", keel::format("numkit:{}:Q", name));
}

void checkLyapStatus(keel::control::ControlStatus st, const char *fn)
{
    if (st != keel::control::ControlStatus::kOk)
        throw Error(keel::format("{}: the Lyapunov equation is singular (check that A is stable / the equation is solvable)", fn),
                    0, 0, fn, "", keel::format("numkit:{}:singular", fn));
}

} // anonymous

Value lyap(const Value &Av, const Value &Qv, keel::Allocator *mr)
{
    lyap_argcheck(Av, Qv, "lyap");
    const keel::usize n = Av.dims().rows();
    const bool isCplx = Av.isComplex() || Qv.isComplex();
    if (n == 0) return Value::matrix(0, 0, isCplx ? ValueType::COMPLEX : ValueType::DOUBLE, mr);

    if (isCplx) {
        auto A = readCMat(Av, n, n);
        auto Q = readCMat(Qv, n, n);
        Value out = Value::matrix(n, n, ValueType::COMPLEX, mr);
        checkLyapStatus(keel::control::lyap_c64(A, Q, n, {out.complexDataMut(), n * n}, mr), "lyap");
        return out;
    }

    auto A = readMat(Av, n, n);
    auto Q = readMat(Qv, n, n);
    Value out = Value::matrix(n, n, ValueType::DOUBLE, mr);
    checkLyapStatus(keel::control::lyap(A, Q, n, {out.doubleDataMut(), n * n}, mr), "lyap");
    return out;
}

Value dlyap(const Value &Av, const Value &Qv, keel::Allocator *mr)
{
    lyap_argcheck(Av, Qv, "dlyap");
    const keel::usize n = Av.dims().rows();
    const bool isCplx = Av.isComplex() || Qv.isComplex();
    if (n == 0) return Value::matrix(0, 0, isCplx ? ValueType::COMPLEX : ValueType::DOUBLE, mr);

    if (isCplx) {
        auto A = readCMat(Av, n, n);
        auto Q = readCMat(Qv, n, n);
        Value out = Value::matrix(n, n, ValueType::COMPLEX, mr);
        checkLyapStatus(keel::control::dlyap_c64(A, Q, n, {out.complexDataMut(), n * n}, mr), "dlyap");
        return out;
    }

    auto A = readMat(Av, n, n);
    auto Q = readMat(Qv, n, n);
    Value out = Value::matrix(n, n, ValueType::DOUBLE, mr);
    checkLyapStatus(keel::control::dlyap(A, Q, n, {out.doubleDataMut(), n * n}, mr), "dlyap");
    return out;
}

} // namespace numkit::control
