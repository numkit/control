// toolboxes/control/src/analyze/analyze.cpp
//
// dcgain, margin, allmargin, and stepinfo delegating numerical kernels to keel::control.

#include <keel/array.h>
#include <keel/control.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/string.h>
#include <numkit/control/analyze/analyze.hpp>
#include <numkit/control/conversion/conversion.hpp>
#include <numkit/control/freq/freq.hpp>
#include <numkit/control/response/response.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <utility>

namespace numkit::control {

namespace {

keel::Array<keel::f64> readCol(const Value &v, keel::Allocator *mr) {
    keel::Array<keel::f64> out(v.numel(), 0.0, mr);
    for (keel::usize i = 0; i < v.numel(); ++i) out[i] = v.elemAsDouble(i);
    return out;
}

Value rowFrom(keel::ArrayRef<keel::f64> v, keel::Allocator *mr) {
    Value r = Value::matrix(1, v.size(), ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < v.size(); ++i) r.doubleDataMut()[i] = v[i];
    return r;
}

void dcgain_argcheck(const Value &sys) {
    if (sys.isStruct() && sys.hasField("kind")) {
        const keel::String k = sys.field("kind").toString();
        if (k == "tf" || k == "zpk" || k == "ss") return;
    }
    throw Error("control freq: expected an LTI struct (tf/zpk/ss)",
                0, 0, "freq", "", "numkit:control:kind");
}

void margin_argcheck(const Value &sys) {
    dcgain_argcheck(sys);
}

void allmargin_argcheck(const Value &sys) {
    if (sys.isStruct() && sys.hasField("kind")) {
        const keel::String k = sys.field("kind").toString();
        if (k == "tf" || k == "zpk" || k == "ss") return;
    }
    throw Error("allmargin: expected an LTI struct (tf/zpk/ss)",
                0, 0, "allmargin", "", "numkit:allmargin:kind");
}

void stepinfo_argcheck(const Value &sys) {
    if (sys.isStruct() && sys.hasField("kind")) {
        const keel::String k = sys.field("kind").toString();
        if (k == "tf" || k == "zpk" || k == "ss") return;
        throw Error(keel::format("step: input 1 (sys) has unknown LTI kind '{}'", k),
                    0, 0, "step", "", "numkit:step:kind");
    }
    throw Error("step: input 1 (sys) must be an LTI struct (tf/zpk/ss)",
                0, 0, "step", "", "numkit:step:kind");
}

std::pair<keel::Array<keel::f64>, keel::Array<keel::f64>>
openLoopNumDen(const Value &sys, keel::Allocator *mr) {
    if (sys.isStruct() && sys.hasField("kind")) {
        const keel::String k = sys.field("kind").toString();
        if (k == "tf")
            return {readCol(sys.field("num"), mr), readCol(sys.field("den"), mr)};
        if (k == "zpk") {
            auto [num, den] = zp2tf(sys.field("z"), sys.field("p"), sys.field("k"), mr);
            return {readCol(num, mr), readCol(den, mr)};
        }
        if (k == "ss") {
            auto [num, den] = ss2tf(sys.field("A"), sys.field("B"),
                                    sys.field("C"), sys.field("D"), 1, mr);
            return {readCol(num, mr), readCol(den, mr)};
        }
    }
    throw Error("allmargin: expected an LTI struct (tf/zpk/ss)",
                0, 0, "allmargin", "", "numkit:allmargin:kind");
}

} // anonymous

Value dcgain(const Value &sys, keel::Allocator *mr) {
    dcgain_argcheck(sys);
    return evalfr(sys, 0.0, mr);
}

MarginResult margin(const Value &sys, keel::Allocator *mr) {
    margin_argcheck(sys);
    Value emptyW = Value::matrix(0, 0, ValueType::DOUBLE, mr);
    auto bodeOut = bode(sys, emptyW, mr);

    const auto w = readCol(bodeOut.w, mr);
    const auto mag = readCol(bodeOut.mag, mr);
    const auto phase = readCol(bodeOut.phase, mr);

    keel::control::MarginResult res;
    (void)keel::control::margin_from_bode(w.as_ref(), mag.as_ref(), phase.as_ref(), res, mr);

    return {Value::scalar(res.gm,  mr),
            Value::scalar(res.pm,  mr),
            Value::scalar(res.wcg, mr),
            Value::scalar(res.wcp, mr)};
}

Value allmargin(const Value &sys, keel::Allocator *mr) {
    allmargin_argcheck(sys);
    auto [num, den] = openLoopNumDen(sys, mr);

    keel::control::AllMarginResult res;
    (void)keel::control::allmargin_tf(num.as_ref(), den.as_ref(), res, mr);

    Value s = Value::structure(mr);
    s.field("GainMargin")  = rowFrom(res.gain_margin.as_ref(),  mr);
    s.field("GMFrequency") = rowFrom(res.gm_frequency.as_ref(), mr);
    s.field("PhaseMargin") = rowFrom(res.phase_margin.as_ref(), mr);
    s.field("PMFrequency") = rowFrom(res.pm_frequency.as_ref(), mr);
    s.field("DelayMargin") = rowFrom(res.delay_margin.as_ref(), mr);
    s.field("DMFrequency") = rowFrom(res.dm_frequency.as_ref(), mr);
    s.field("Stable")      = Value::logicalScalar(res.stable, mr);
    return s;
}

Value stepinfo(const Value &sys, keel::Allocator *mr) {
    stepinfo_argcheck(sys);
    Value emptyT = Value::matrix(0, 0, ValueType::DOUBLE, mr);
    auto [y0, t0] = step_response(sys, emptyT, /*xOut=*/nullptr, mr);
    auto t = readCol(t0, mr);
    if (t.size() < 2) {
        return Value::structure(mr);
    }
    const keel::f64 Tfinal = t.back();
    Value tArg = Value::scalar(2.0 * Tfinal, mr);
    auto [y1, t1] = step_response(sys, tArg, /*xOut=*/nullptr, mr);
    t = readCol(t1, mr);
    const auto y = readCol(y1, mr);

    keel::control::StepInfo info;
    if (keel::control::step_info(t.as_ref(), y.as_ref(), info) != keel::control::ControlStatus::kOk) {
        return Value::structure(mr);
    }

    Value s = Value::structure(mr);
    s.field("RiseTime")      = Value::scalar(info.rise_time, mr);
    s.field("TransientTime") = Value::scalar(info.transient_time, mr);
    s.field("SettlingTime")  = Value::scalar(info.settling_time, mr);
    s.field("SettlingMin")   = Value::scalar(info.settling_min, mr);
    s.field("SettlingMax")   = Value::scalar(info.settling_max, mr);
    s.field("Overshoot")     = Value::scalar(info.overshoot, mr);
    s.field("Undershoot")    = Value::scalar(info.undershoot, mr);
    s.field("Peak")          = Value::scalar(info.peak, mr);
    s.field("PeakTime")      = Value::scalar(info.peak_time, mr);
    return s;
}

} // namespace numkit::control
