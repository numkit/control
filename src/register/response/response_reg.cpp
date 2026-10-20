// toolboxes/control/src/response/response_reg.cpp
//
// Register half of the time-response builtins: the CallContext wrappers
// step / impulse / lsim that delegate to the engine-free compute in
// response.cpp. library.cpp forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <numkit/control/response/response.hpp>
#include <numkit/control/lti/lti.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, ctx.engine->resource()
#include <numkit/value/value.hpp>   // Value::matrix, ValueType (default t/x0 args)
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <cstddef>
#include <utility>

namespace numkit::control {
namespace detail {

namespace {
Value defaultDemoSys(std::pmr::memory_resource *mr) {
    Value num = Value::scalar(1.0, mr);
    Value den = Value::matrix(1, 3, ValueType::DOUBLE, mr);
    den.doubleDataMut()[0] = 1.0;
    den.doubleDataMut()[1] = 1.0;
    den.doubleDataMut()[2] = 1.0;
    return tf(num, den, 0.0, mr);
}
} // anonymous

void step_reg(Span<const Value> a, size_t /*nargout*/, Span<Value> outs,
              CallContext &c)
{
    argcheck::checkNargin(a.size(), 0, 6, "step");
    auto *mr = c.engine->resource();
    Value sys;
    Value tArg;
    if (a.empty()) {
        sys = defaultDemoSys(mr);
        tArg = Value::matrix(1, 0, ValueType::DOUBLE, mr);
    } else if (a.size() >= 2 && !a[0].isStruct() && !a[0].isObject()
               && (a[0].isNumeric() || a[0].isComplex())
               && (a[1].isNumeric() || a[1].isComplex())) {
        sys = tf(a[0], a[1], 0.0, mr);
        tArg = (a.size() >= 3) ? a[2] : Value::matrix(1, 0, ValueType::DOUBLE, mr);
    } else {
        sys = a[0];
        tArg = (a.size() >= 2) ? a[1] : Value::matrix(1, 0, ValueType::DOUBLE, mr);
    }
    Value xOut;
    auto [y, t] = step_response(sys, tArg,
                                outs.size() >= 3 ? &xOut : nullptr,
                                mr);
    if (outs.size() >= 1) outs[0] = std::move(y);
    if (outs.size() >= 2) outs[1] = std::move(t);
    if (outs.size() >= 3) outs[2] = std::move(xOut);
}

void impulse_reg(Span<const Value> a, size_t /*nargout*/, Span<Value> outs,
                 CallContext &c)
{
    argcheck::checkNargin(a.size(), 0, 6, "impulse");
    auto *mr = c.engine->resource();
    Value sys;
    Value tArg;
    if (a.empty()) {
        sys = defaultDemoSys(mr);
        tArg = Value::matrix(1, 0, ValueType::DOUBLE, mr);
    } else if (a.size() >= 2 && !a[0].isStruct() && !a[0].isObject()
               && (a[0].isNumeric() || a[0].isComplex())
               && (a[1].isNumeric() || a[1].isComplex())) {
        sys = tf(a[0], a[1], 0.0, mr);
        tArg = (a.size() >= 3) ? a[2] : Value::matrix(1, 0, ValueType::DOUBLE, mr);
    } else {
        sys = a[0];
        tArg = (a.size() >= 2) ? a[1] : Value::matrix(1, 0, ValueType::DOUBLE, mr);
    }
    Value xOut;
    auto [y, t] = impulse_response(sys, tArg,
                                   outs.size() >= 3 ? &xOut : nullptr,
                                   mr);
    if (outs.size() >= 1) outs[0] = std::move(y);
    if (outs.size() >= 2) outs[1] = std::move(t);
    if (outs.size() >= 3) outs[2] = std::move(xOut);
}

void initial_reg(Span<const Value> a, size_t /*nargout*/, Span<Value> outs,
                 CallContext &c)
{
    argcheck::checkNargin(a.size(), 2, 6, "initial");
    Value tArg = (a.size() >= 3) ? a[2] : Value::matrix(1, 0, ValueType::DOUBLE, c.engine->resource());
    Value xOut;
    auto [y, t] = initial_response(a[0], a[1], tArg,
                                   outs.size() >= 3 ? &xOut : nullptr,
                                   c.engine->resource());
    if (outs.size() >= 1) outs[0] = std::move(y);
    if (outs.size() >= 2) outs[1] = std::move(t);
    if (outs.size() >= 3) outs[2] = std::move(xOut);
}

void lsim_reg(Span<const Value> a, size_t /*nargout*/, Span<Value> outs,
              CallContext &c)
{
    argcheck::checkNargin(a.size(), 2, 7, "lsim");
    auto *mr = c.engine->resource();
    if (!a[0].isStruct() && !a[0].isObject()
        && (a[0].isNumeric() || a[0].isComplex())
        && (a[1].isNumeric() || a[1].isComplex())) {
        if (a.size() >= 6) {
            Value sys = ss(a[0], a[1], a[2], a[3], 0.0, mr);
            Value x0 = (a.size() >= 7) ? a[6] : Value::matrix(0, 0, ValueType::DOUBLE, mr);
            Value xOut;
            outs[0] = lsim(sys, a[4], a[5], x0,
                           outs.size() >= 2 ? &xOut : nullptr, mr);
            if (outs.size() >= 2) outs[1] = std::move(xOut);
            return;
        }
        if (a.size() >= 4) {
            Value sys = tf(a[0], a[1], 0.0, mr);
            Value x0 = (a.size() >= 5) ? a[4] : Value::matrix(0, 0, ValueType::DOUBLE, mr);
            Value xOut;
            outs[0] = lsim(sys, a[2], a[3], x0,
                           outs.size() >= 2 ? &xOut : nullptr, mr);
            if (outs.size() >= 2) outs[1] = std::move(xOut);
            return;
        }
    }
    Value tIn;
    if (a.size() >= 3 && !a[2].isEmpty()) {
        tIn = a[2];
    } else {
        const double Ts = (a[0].isStruct() && a[0].hasField("Ts"))
            ? a[0].field("Ts").toScalar() : 0.0;
        if (Ts == 0.0)
            throw Error("lsim: time vector t is required for continuous-time models.",
                        0, 0, "lsim", "", "numkit:lsim:t");
        const double dt = (Ts > 0.0) ? Ts : 1.0;
        const std::size_t N = a[1].numel();
        tIn = Value::matrix(N, 1, ValueType::DOUBLE, mr);
        for (std::size_t i = 0; i < N; ++i)
            tIn.doubleDataMut()[i] = static_cast<double>(i) * dt;
    }
    Value x0 = (a.size() >= 4 && !a[3].isChar() && !a[3].isString())
        ? a[3] : Value::matrix(0, 0, ValueType::DOUBLE, mr);
    Value xOut;
    outs[0] = lsim(a[0], a[1], tIn, x0,
                   outs.size() >= 3 ? &xOut : nullptr,
                   mr);
    // MATLAB: [y, t, x] = lsim(...). t echoes the input time grid.
    if (outs.size() >= 2) {
        Value t = Value::matrix(tIn.numel(), 1, ValueType::DOUBLE, mr);
        for (std::size_t i = 0; i < tIn.numel(); ++i)
            t.doubleDataMut()[i] = tIn.elemAsDouble(i);
        outs[1] = std::move(t);
    }
    if (outs.size() >= 3) outs[2] = std::move(xOut);
}

} // namespace detail
} // namespace numkit::control

