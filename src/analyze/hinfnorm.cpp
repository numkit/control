// toolboxes/control/src/analyze/hinfnorm.cpp
//
// H-infinity norm wrapper delegating to keel::control::hinfnorm.

#include <keel/array.h>
#include <keel/complex.h>
#include <keel/control.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <numkit/control/analyze/analyze.hpp>
#include <numkit/control/conversion/conversion.hpp>
#include <numkit/control/internal/type_gates.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

namespace numkit::control {

namespace {

bool hasKind(const Value &sys, const char *want) {
    return internal::isLtiKind(sys, want);
}

keel::f64 sampleTime(const Value &sys) {
    return internal::ltiSampleTime(sys);
}

StateSpace pullABCD(const Value &sys, keel::Allocator *mr) {
    const Value &d = internal::ltiStruct(sys);
    StateSpace out;
    if (hasKind(sys, "ss")) {
        out.A = d.field("A"); out.B = d.field("B");
        out.C = d.field("C"); out.D = d.field("D");
    } else if (hasKind(sys, "tf")) {
        out = tf2ss(d.field("num"), d.field("den"), mr);
    } else if (hasKind(sys, "zpk")) {
        auto [num, den] = zp2tf(d.field("z"), d.field("p"), d.field("k"), mr);
        out = tf2ss(num, den, mr);
    } else {
        throw Error("hinfnorm: expected an LTI model (tf/zpk/ss)",
                    0, 0, "hinfnorm", "", "numkit:hinfnorm:kind");
    }
    return out;
}

keel::Array<keel::f64> readMat(const Value &v, keel::usize r, keel::usize c, keel::Allocator *mr) {
    keel::Array<keel::f64> m(r * c, 0.0, mr);
    for (keel::usize i = 0; i < r * c; ++i) m[i] = v.elemAsDouble(i);
    return m;
}

bool hinfnorm_argcheck(const Value &sys) {
    const ValueType t = sys.type();
    const bool numericScalar = sys.numel() == 1
        && (t == ValueType::DOUBLE || t == ValueType::COMPLEX
            || t == ValueType::SINGLE || t == ValueType::LOGICAL
            || isIntegerType(t));
    if (numericScalar) return true;
    if (!internal::isLtiModel(sys))
        throw Error(keel::format("hinfnorm: input 1 (sys) must be an LTI "
                                 "model (tf/zpk/ss) or a static-gain scalar "
                                 "(got a {} {} array).",
                                 internal::shapeOf(sys), mtypeName(t)),
                    0, 0, "hinfnorm", "", "numkit:hinfnorm:inputType");
    return false;
}

} // anonymous

Value hinfnorm(const Value &sys, keel::Allocator *mr) {
    if (hinfnorm_argcheck(sys)) {
        const keel::f64 g = sys.isComplex()
            ? keel::abs(sys.complexData()[0]) : keel::abs(sys.elemAsDouble(0));
        Value out = Value::matrix(1, 3, ValueType::DOUBLE, mr);
        keel::f64 *od = out.doubleDataMut();
        od[0] = g; od[1] = g; od[2] = 0.0;
        return out;
    }

    const bool discrete = (sampleTime(sys) != 0.0);
    StateSpace ss = pullABCD(sys, mr);
    const keel::usize n = ss.A.dims().rows();
    const keel::usize m = ss.B.dims().cols();
    const keel::usize p = ss.C.dims().rows();
    if (ss.A.dims().cols() != n)
        throw Error("hinfnorm: A must be square",
                    0, 0, "hinfnorm", "", "numkit:hinfnorm:A");

    const keel::Array<keel::f64> A = (n ? readMat(ss.A, n, n, mr) : keel::Array<keel::f64>(mr));
    const keel::Array<keel::f64> B = (n ? readMat(ss.B, n, m, mr) : keel::Array<keel::f64>(mr));
    const keel::Array<keel::f64> C = (n ? readMat(ss.C, p, n, mr) : keel::Array<keel::f64>(mr));
    const keel::Array<keel::f64> D = readMat(ss.D, p, m, mr);

    keel::f64 norm_val = 0.0;
    (void)keel::control::hinfnorm(A.as_ref(), B.as_ref(), C.as_ref(), D.as_ref(),
                                  n, m, p, discrete, norm_val, mr);
    return Value::scalar(norm_val, mr);
}

} // namespace numkit::control
