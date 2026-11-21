// toolboxes/control/src/state/place.cpp
//
// Ackermann pole placement for SISO systems, delegated to keel::control.

#include <keel/array.h>
#include <keel/control.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/complex.h>
#include <numkit/control/state/place.hpp>
#include <numkit/control/internal/numerics.hpp>
#include <numkit/control/internal/type_gates.hpp>

#include <numkit/builtin/polyfun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <algorithm>

namespace numkit::control {

namespace {

using Mat = internal::Mat;
using Vec = internal::Vec;
using CMat = internal::CMat;
using CVec = internal::CVec;

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

Value rowFromVec(const Vec &v, bool asSingle, keel::Allocator *mr) {
    if (asSingle) {
        Value r = Value::matrix(1, v.size(), ValueType::SINGLE, mr);
        float *fd = r.singleDataMut();
        for (keel::usize i = 0; i < v.size(); ++i) fd[i] = static_cast<float>(v[i]);
        return r;
    }
    Value r = Value::matrix(1, v.size(), ValueType::DOUBLE, mr);
    if (!v.empty()) std::copy(v.begin(), v.end(), r.doubleDataMut());
    return r;
}

void acker_argcheck(const Value &Av, const Value &Bv, const Value &pv)
{
    internal::requireCoercibleMatrix(Av, "acker", "A", 1);
    internal::requireNonIntegerMatrix(Av, "acker", "A", 1);
    internal::requireCoercibleMatrix(Bv, "acker", "B", 2);
    internal::requireNonIntegerMatrix(Bv, "acker", "B", 2);
    internal::requireNumericInput(pv, "acker", "p", 3);
    internal::requireNonIntegerMatrix(pv, "acker", "p", 3);

    const keel::usize n = Av.dims().rows();
    if (Av.dims().cols() != n)
        throw Error("acker: A must be square",
                    0, 0, "acker", "", "numkit:acker:A");
    if (Bv.dims().rows() != n || Bv.dims().cols() != 1)
        throw Error("acker: B must be n×1 (SISO only)",
                    0, 0, "acker", "", "numkit:acker:B");
    if (pv.numel() != n)
        throw Error("acker: pole list must have length n",
                    0, 0, "acker", "", "numkit:acker:p");
}

void place_argcheck(const Value &A, const Value &B, const Value &p)
{
    acker_argcheck(A, B, p);
}

} // anonymous

Value acker(const Value &Av, const Value &Bv, const Value &pv, keel::Allocator *mr)
{
    acker_argcheck(Av, Bv, pv);
    const keel::usize n = Av.dims().rows();

    const bool wantSingle = (Av.type() == ValueType::SINGLE ||
                             Bv.type() == ValueType::SINGLE ||
                             pv.type() == ValueType::SINGLE);
    const bool wantComplex = (Av.isComplex() || Bv.isComplex());

    Value pC = internal::coerceToDoubleMatrix(pv, mr);
    Value coeffs = numkit::builtin::poly(pC, mr);
    Vec c(coeffs.numel());
    for (keel::usize i = 0; i < coeffs.numel(); ++i) c[i] = coeffs.elemAsDouble(i);
    if (c.size() != n + 1)
        throw Error("acker: poly(p) returned unexpected length",
                    0, 0, "acker", "", "numkit:acker:poly");

    if (wantComplex) {
        auto A = readCMat(Av, n, n);
        auto B = readCMat(Bv, n, 1);
        Value out = Value::matrix(1, n, ValueType::COMPLEX, mr);
        if (keel::control::acker_from_poly_c64(A, B, c, n, {out.complexDataMut(), n}, mr) !=
            keel::control::ControlStatus::kOk)
            throw Error("acker: controllability matrix is singular "
                        "(system is uncontrollable)",
                        0, 0, "acker", "", "numkit:acker:singular");
        return out;
    }

    auto A = readMat(Av, n, n);
    auto B = readMat(Bv, n, 1);
    Vec K(n, 0.0);
    if (keel::control::acker_from_poly(A, B, c, n, K, mr) != keel::control::ControlStatus::kOk)
        throw Error("acker: controllability matrix is singular "
                    "(system is uncontrollable)",
                    0, 0, "acker", "", "numkit:acker:singular");
    return rowFromVec(K, wantSingle, mr);
}

Value place(const Value &A, const Value &B, const Value &p, keel::Allocator *mr)
{
    place_argcheck(A, B, p);
    return acker(A, B, p, mr);
}

} // namespace numkit::control
