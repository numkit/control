// toolboxes/control/src/connect/connect.cpp
//
// LTI interconnections in transfer-function arithmetic. Each entry
// point first converts the inputs to (num, den) coefficient rows
// via the existing toolboxes/builtin (zp2tf) and toolboxes/control (ss2tf
// helper) primitives, then performs the polynomial arithmetic, and
// finally hands a `tf` struct back to the caller.
//
// Polynomial multiplication uses signal::conv. Polynomial addition
// is done locally (right-aligned, longer length wins).

#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <numkit/control/connect/connect.hpp>
#include <numkit/control/lti/lti.hpp>
#include <numkit/control/conversion/conversion.hpp>

#include <numkit/builtin/polyfun.hpp>
#include <numkit/signal/convolution/convolution.hpp>

// Compute-only TU: Value substrate + Error, no engine. The series / parallel
// / feedback builtins (CallContext wrappers) live in connect/connect_reg.cpp.
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <algorithm>
#include <cmath>

namespace numkit::control {

namespace {

bool hasKind(const Value &sys, const char *want) {
    if (!sys.isStruct() || !sys.hasField("kind")) return false;
    return sys.field("kind").toString() == want;
}

keel::f64 sampleTime(const Value &sys) {
    if (sys.isStruct() && sys.hasField("Ts"))
        return sys.field("Ts").toScalar();
    // A plain numeric is a static gain with UNSPECIFIED sample time (-1):
    // it inherits the other operand's Ts via Ts_combine.
    if (!sys.isStruct()) return -1.0;
    return 0.0;
}

keel::Array<keel::f64> coeffsReal(const Value &v) {
    keel::Array<keel::f64> out(v.numel());
    for (keel::usize i = 0; i < v.numel(); ++i) out[i] = v.elemAsDouble(i);
    return out;
}

Value rowOf(keel::ArrayRef<keel::f64> v, keel::Allocator *mr) {
    Value r = Value::matrix(1, v.size(), ValueType::DOUBLE, mr);
    if (!v.empty()) std::copy(v.begin(), v.end(), r.doubleDataMut());
    return r;
}

// Bring an LTI struct (or a raw 2-arg call: (num, den) split across two
// args via `series(num1, den1, num2, den2)` — not yet supported) into
// a (num, den) coefficient pair.
struct NumDen { keel::Array<keel::f64> num, den; };

NumDen toNumDen(const Value &sys, keel::Allocator *mr) {
    // A plain numeric scalar is a STATIC gain system K = K/1, i.e.
    // num = [K], den = [1]. MATLAB accepts this in every interconnection
    // (e.g. feedback(sys, 1) for unity feedback, series(sys, 2), ...).
    if (!sys.isStruct()) {
        if (sys.numel() == 1)
            return {keel::Array<keel::f64>{sys.toScalar()}, keel::Array<keel::f64>{1.0}};
        throw Error("control connect: a numeric system must be a scalar gain",
                    0, 0, "control", "", "numkit:control:kind");
    }
    if (hasKind(sys, "tf")) {
        return {coeffsReal(sys.field("num")), coeffsReal(sys.field("den"))};
    }
    if (hasKind(sys, "zpk")) {
        auto [num, den] = zp2tf(sys.field("z"), sys.field("p"), sys.field("k"), mr);
        return {coeffsReal(num), coeffsReal(den)};
    }
    if (hasKind(sys, "ss")) {
        auto [num, den] = ss2tf(sys.field("A"), sys.field("B"),
                                 sys.field("C"), sys.field("D"), /*iu=*/1, mr);
        return {coeffsReal(num), coeffsReal(den)};
    }
    throw Error("control connect: expected tf/zpk/ss struct",
                0, 0, "control", "", "numkit:control:kind");
}

// Polynomial addition right-aligned (highest order first, MATLAB style).
keel::Array<keel::f64> polyAdd(keel::ArrayRef<keel::f64> a,
                            keel::ArrayRef<keel::f64> b,
                            keel::f64 bScale = 1.0)
{
    const keel::usize na = a.size();
    const keel::usize nb = b.size();
    const keel::usize n = std::max(na, nb);
    keel::Array<keel::f64> y(n, 0.0);
    // Right-align: a[na-1] aligns with y[n-1], same for b.
    for (keel::usize i = 0; i < na; ++i) y[n - na + i] += a[i];
    for (keel::usize i = 0; i < nb; ++i) y[n - nb + i] += bScale * b[i];
    return y;
}

// Polynomial multiplication via signal::conv.
keel::Array<keel::f64> polyMul(keel::ArrayRef<keel::f64> a, keel::ArrayRef<keel::f64> b, keel::Allocator *mr)
{
    if (a.empty() || b.empty()) return {};
    Value av = rowOf(a, mr);
    Value bv = rowOf(b, mr);
    Value y = signal::conv(av, bv, "full", mr);
    return coeffsReal(y);
}

keel::f64 Ts_combine(keel::f64 Ts1, keel::f64 Ts2, const char *op) {
    // MATLAB rules: continuous + continuous → continuous; same Ts →
    // same Ts; mismatched Ts → error.
    if (Ts1 == 0.0 && Ts2 == 0.0) return 0.0;
    if (Ts1 == Ts2) return Ts1;
    // Allow mixing only when one side is "unspecified" (Ts == -1).
    if (Ts1 == -1.0) return Ts2;
    if (Ts2 == -1.0) return Ts1;
    throw Error(keel::format("control {}: sample times must match (Ts mismatch)", op),
                0, 0, op, "", "numkit:control:Ts");
}

void connect_sys_argcheck(const Value &sys) {
    if (!sys.isStruct()) {
        if (sys.numel() == 1) return;
        throw Error("control connect: a numeric system must be a scalar gain",
                    0, 0, "control", "", "numkit:control:kind");
    }
    if (!hasKind(sys, "tf") && !hasKind(sys, "zpk") && !hasKind(sys, "ss"))
        throw Error("control connect: expected tf/zpk/ss struct",
                    0, 0, "control", "", "numkit:control:kind");
}

void series_argcheck(const Value &sys1, const Value &sys2) {
    connect_sys_argcheck(sys1);
    connect_sys_argcheck(sys2);
}

void parallel_argcheck(const Value &sys1, const Value &sys2) {
    connect_sys_argcheck(sys1);
    connect_sys_argcheck(sys2);
}

void feedback_argcheck(const Value &G, const Value &H, int sign) {
    if (sign != 1 && sign != -1) {
        throw Error("feedback: sign must be 1 or -1", 0, 0, "feedback", "",
                    "numkit:feedback:signNotChoice");
    }
    connect_sys_argcheck(G);
    connect_sys_argcheck(H);
}

} // anonymous

Value series(const Value &sys1, const Value &sys2, keel::Allocator *mr)
{
    series_argcheck(sys1, sys2);
    auto a = toNumDen(sys1, mr);
    auto b = toNumDen(sys2, mr);
    auto num = polyMul(a.num, b.num, mr);
    auto den = polyMul(a.den, b.den, mr);
    const keel::f64 Ts = Ts_combine(sampleTime(sys1), sampleTime(sys2),
                                 "series");
    return tf(rowOf(num, mr), rowOf(den, mr), Ts, mr);
}

Value parallel(const Value &sys1, const Value &sys2, keel::Allocator *mr)
{
    parallel_argcheck(sys1, sys2);
    auto a = toNumDen(sys1, mr);
    auto b = toNumDen(sys2, mr);
    auto t1 = polyMul(a.num, b.den, mr); // n1·d2
    auto t2 = polyMul(b.num, a.den, mr); // n2·d1
    auto num = polyAdd(t1, t2);
    auto den = polyMul(a.den, b.den, mr);
    const keel::f64 Ts = Ts_combine(sampleTime(sys1), sampleTime(sys2),
                                 "parallel");
    return tf(rowOf(num, mr), rowOf(den, mr), Ts, mr);
}

Value feedback(const Value &G, const Value &H, int sign, keel::Allocator *mr)
{
    feedback_argcheck(G, H, sign);
    auto g = toNumDen(G, mr);
    auto h = toNumDen(H, mr);
    // T(s) = num_G · den_H / (den_G · den_H − sign · num_G · num_H)
    auto numT = polyMul(g.num, h.den, mr);
    auto loop = polyMul(g.num, h.num, mr);
    auto denBase = polyMul(g.den, h.den, mr);
    // sign convention: MATLAB feedback default is -1 (negative
    // feedback), giving denBase + loop. sign = +1 ⇒ denBase − loop.
    auto denT = polyAdd(denBase, loop, -keel::as<keel::f64>(sign));
    const keel::f64 Ts = Ts_combine(sampleTime(G), sampleTime(H), "feedback");
    return tf(rowOf(numT, mr), rowOf(denT, mr), Ts, mr);
}

} // namespace numkit::control
