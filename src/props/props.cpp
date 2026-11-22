// toolboxes/control/src/props/props.cpp
//
// LTI model predicates and analytic properties. Every entry point
// dispatches on the struct's `kind` tag ('tf' / 'zpk' / 'ss') so a
// script can call e.g. `isstable(tf(num,den))` or `pole(ss(A,B,C,D))`
// uniformly.
//
// State-space poles are computed via the Faddeev–LeVerrier expansion
// of det(sI − A) (already used by ss2tf in toolboxes/control/conversion);
// the resulting characteristic polynomial is then handed to
// numkit::builtin::roots. State-space *zeros* fall back to converting to a
// transfer function via toolboxes/signal's ss2tf and taking roots(num) —
// works for SISO systems, which is what `zero(sys)` is most useful
// for in a function-form environment.

#include <keel/array.h>
#include <keel/control.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/control/props/props.hpp>
#include <numkit/control/internal/numerics.hpp>
#include <numkit/control/internal/type_gates.hpp>
#include <numkit/control/conversion/conversion.hpp>

#include <numkit/builtin/polyfun.hpp>

// Compute-only TU: Value substrate + Error, no engine. The is*/order/pole/
// zero/damp/pzmap/tzero builtins (CallContext wrappers) live in props_reg.cpp.
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <algorithm>

namespace numkit::control {

namespace {

bool hasKind(const Value &sys, const char *want) {
    return internal::isLtiKind(sys, want);
}

keel::f64 sampleTime(const Value &sys) {
    return internal::ltiSampleTime(sys);
}

Value boolValue(bool b, keel::Allocator *mr) {
    return Value::logicalScalar(b, mr);
}

// Char poly thin wrapper around the shared Faddeev–LeVerrier kernel.
// Pulls A out of the LTI struct as a plain column-major double buffer
// and forwards.
keel::Array<keel::f64> charPoly(const Value &A) {
    const keel::usize n = A.dims().rows();
    keel::Array<keel::f64> Av(n * n);
    for (keel::usize i = 0; i < n * n; ++i) Av[i] = A.elemAsDouble(i);
    return internal::charPoly(Av, n);
}

// Pull a Value (either real or complex) into a vector of complex
// numbers. Convenient for stability checks where everything compares
// uniformly.
keel::Array<keel::c64> toComplexVec(const Value &v) {
    const keel::usize N = v.numel();
    keel::Array<keel::c64> out(N);
    if (v.type() == ValueType::COMPLEX) {
        const keel::c64 *src = v.complexData();
        for (keel::usize i = 0; i < N; ++i) out[i] = src[i];
    } else {
        for (keel::usize i = 0; i < N; ++i) out[i] = keel::c64(v.elemAsDouble(i), 0.0);
    }
    return out;
}

Value rowFromCoeffs(keel::ArrayRef<keel::f64> v, keel::Allocator *mr) {
    Value r = Value::matrix(1, v.size(), ValueType::DOUBLE, mr);
    if (!v.empty()) std::copy(v.begin(), v.end(), r.doubleDataMut());
    return r;
}

// Helper: poles as a column Value, regardless of input form.
Value polesOf(const Value &sys, keel::Allocator *mr) {
    const Value &d = internal::ltiStruct(sys);
    if (hasKind(sys, "tf"))
        return numkit::builtin::roots(d.field("den"), mr);
    if (hasKind(sys, "zpk"))
        return d.field("p");
    if (hasKind(sys, "ss")) {
        auto cp = charPoly(d.field("A"));
        return numkit::builtin::roots(rowFromCoeffs(cp, mr), mr);
    }
    if ((sys.isNumeric() || sys.isComplex()) && !sys.isObject()
        && sys.dims().rows() == sys.dims().cols()) {
        auto cp = charPoly(sys);
        return numkit::builtin::roots(rowFromCoeffs(cp, mr), mr);
    }
    throw Error("expected an LTI model (tf/zpk/ss)",
                0, 0, "lti", "", "numkit:lti:kind");
}

Value zerosOf(const Value &sys, keel::Allocator *mr) {
    const Value &d = internal::ltiStruct(sys);
    if (hasKind(sys, "tf"))
        return numkit::builtin::roots(d.field("num"), mr);
    if (hasKind(sys, "zpk"))
        return d.field("z");
    if (hasKind(sys, "ss")) {
        const Value &A = d.field("A");
        const Value &B = d.field("B");
        const Value &C = d.field("C");
        const Value &D = d.field("D");
        const bool one_in  = (B.dims().cols() == 1);
        const bool one_out = (C.dims().rows() == 1);
        if (!one_in || !one_out) {
            throw Error("zero(sys): MIMO state-space transmission zeros "
                        "require a generalised eigenvalue (QZ) solver; "
                        "not yet implemented in this build.",
                        0, 0, "zero", "", "numkit:zero:miso");
        }
        auto [num, den] = ss2tf(A, B, C, D, /*iu=*/1, mr);
        return numkit::builtin::roots(num, mr);
    }
    throw Error("expected an LTI struct (tf/zpk/ss)",
                0, 0, "lti", "", "numkit:lti:kind");
}

void isct_argcheck(const Value &sys) { (void)sys; }
void isdt_argcheck(const Value &sys) { (void)sys; }

void issiso_argcheck(const Value &sys) {
    internal::requireLtiModel(sys, "issiso", /*acceptFrD=*/true);
}

void isproper_argcheck(const Value &sys) {
    internal::requireLtiModel(sys, "isproper", /*acceptFrD=*/true);
}

void pole_argcheck(const Value &sys) {
    if (hasKind(sys, "tf") || hasKind(sys, "zpk") || hasKind(sys, "ss")) return;
    if ((sys.isNumeric() || sys.isComplex()) && !sys.isObject()
        && sys.dims().rows() == sys.dims().cols()) return;
    throw Error("expected an LTI struct (tf/zpk/ss)",
                0, 0, "lti", "", "numkit:lti:kind");
}

void isstable_argcheck(const Value &sys) { pole_argcheck(sys); }
void damp_argcheck(const Value &sys) { pole_argcheck(sys); }

void order_argcheck(const Value &sys) {
    internal::requireLtiModel(sys, "order", /*acceptFrD=*/false);
}

void isstatic_argcheck(const Value &sys) { order_argcheck(sys); }

void zero_argcheck(const Value &sys) {
    if (hasKind(sys, "tf") || hasKind(sys, "zpk")) return;
    if (hasKind(sys, "ss")) {
        const Value &d = internal::ltiStruct(sys);
        const Value &B = d.field("B");
        const Value &C = d.field("C");
        if (B.dims().cols() != 1 || C.dims().rows() != 1) {
            throw Error("zero(sys): MIMO state-space transmission zeros "
                        "require a generalised eigenvalue (QZ) solver; "
                        "not yet implemented in this build.",
                        0, 0, "zero", "", "numkit:zero:miso");
        }
        return;
    }
    throw Error("expected an LTI model (tf/zpk/ss)",
                0, 0, "lti", "", "numkit:lti:kind");
}

void pzmap_argcheck(const Value &sys) {
    pole_argcheck(sys);
    zero_argcheck(sys);
}

void tzero_argcheck(const Value &sys) {
    internal::requireLtiModel(sys, "issiso", /*acceptFrD=*/true);
}

} // anonymous

Value isct(const Value &sys, keel::Allocator *mr) {
    isct_argcheck(sys);
    return boolValue(sampleTime(sys) == 0.0, mr);
}

Value isdt(const Value &sys, keel::Allocator *mr) {
    isdt_argcheck(sys);
    const keel::f64 Ts = sampleTime(sys);
    return boolValue(Ts > 0.0 || Ts == -1.0, mr);
}

Value issiso(const Value &sys, keel::Allocator *mr) {
    // Un-typed arguments are rejected, mirroring the reference's
    // UndefinedFunction dispatch (probed R2025b); frd models are
    // always SISO here and answer true.
    issiso_argcheck(sys);
    const Value &d = internal::ltiStruct(sys);
    if (hasKind(sys, "tf")) {
        const Value &num = d.field("num");
        if (num.isCell() && num.numel() > 1)
            return boolValue(false, mr);
        return boolValue(true, mr);
    }
    if (hasKind(sys, "zpk") || hasKind(sys, "frd"))
        return boolValue(true, mr);   // single-output by construction here
    if (hasKind(sys, "ss")) {
        const Value &B = d.field("B");
        const Value &C = d.field("C");
        const bool one_in  = (B.dims().cols() == 1);
        const bool one_out = (C.dims().rows() == 1);
        return boolValue(one_in && one_out, mr);
    }
    return boolValue(false, mr);
}

Value isproper(const Value &sys, keel::Allocator *mr) {
    isproper_argcheck(sys);
    const Value &d = internal::ltiStruct(sys);
    if (hasKind(sys, "tf"))
        return boolValue(d.field("num").numel() <= d.field("den").numel(), mr);
    if (hasKind(sys, "zpk"))
        return boolValue(d.field("z").numel() <= d.field("p").numel(), mr);
    if (hasKind(sys, "frd") || hasKind(sys, "ss"))
        return boolValue(true, mr);
    return boolValue(false, mr);
}

Value isstable(const Value &sys, keel::Allocator *mr) {
    isstable_argcheck(sys);
    Value p = polesOf(sys, mr);
    auto pv = toComplexVec(p);
    const bool discrete = (sampleTime(sys) > 0.0);
    for (const auto &z : pv) {
        if (discrete) {
            if (keel::abs(z) >= 1.0)
                return boolValue(false, mr);
        } else {
            if (z.real() >= 0.0)
                return boolValue(false, mr);
        }
    }
    return boolValue(true, mr);
}

Value order(const Value &sys, keel::Allocator *mr) {
    // frd models are REJECTED here — the reference answers
    // NotSupportedModelsofClass for order(frd) while issiso/isproper
    // accept it (probed R2025b).
    order_argcheck(sys);
    const Value &sd = internal::ltiStruct(sys);
    if (hasKind(sys, "tf")) {
        // order = max(deg num, deg den) = max(numel-1).
        const keel::usize n = sd.field("num").numel();
        const keel::usize d = sd.field("den").numel();
        const keel::usize m = (n > d) ? n : d;
        return Value::scalar((m == 0) ? 0.0 : keel::f64(m - 1), mr);
    }
    if (hasKind(sys, "zpk")) {
        const keel::usize z = sd.field("z").numel();
        const keel::usize p = sd.field("p").numel();
        return Value::scalar(keel::f64((z > p) ? z : p), mr);
    }
    if (hasKind(sys, "ss"))
        return Value::scalar(keel::f64(sd.field("A").dims().rows()), mr);
    return Value::scalar(0.0, mr);
}

Value pole(const Value &sys, keel::Allocator *mr) {
    pole_argcheck(sys);
    return polesOf(sys, mr);
}

Value zero(const Value &sys, keel::Allocator *mr) {
    zero_argcheck(sys);
    return zerosOf(sys, mr);
}

DampResult damp(const Value &sys, keel::Allocator *mr)
{
    damp_argcheck(sys);
    Value p = polesOf(sys, mr);
    auto pv = toComplexVec(p);
    const keel::f64 Ts = sampleTime(sys);
    const keel::usize N = pv.size();

    Value wn = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    Value zeta = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    (void)keel::control::damp(pv.as_ref(), Ts,
                              keel::MutArrayRef<keel::f64>(wn.doubleDataMut(), N),
                              keel::MutArrayRef<keel::f64>(zeta.doubleDataMut(), N));
    return {std::move(wn), std::move(zeta), std::move(p)};
}

std::pair<Value, Value>
pzmap(const Value &sys, keel::Allocator *mr)
{
    pzmap_argcheck(sys);
    return {polesOf(sys, mr), zerosOf(sys, mr)};
}

Value isstatic(const Value &sys, keel::Allocator *mr)
{
    isstatic_argcheck(sys);
    Value ord = order(sys, mr);
    return Value::logicalScalar(ord.toScalar() == 0.0, mr);
}

Value tzero(const Value &sys, keel::Allocator *mr)
{
    tzero_argcheck(sys);
    // SISO build: transmission zeros == ordinary zeros.
    Value siso = issiso(sys, mr);
    if (siso.toScalar() == 0.0)
        throw Error("tzero: only SISO systems supported in this build "
                    "(MIMO transmission zeros need a generalized eigenproblem)",
                    0, 0, "tzero", "", "numkit:tzero:miso");
    return zerosOf(sys, mr);
}

} // namespace numkit::control
