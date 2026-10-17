// toolboxes/control/src/freq/freq_reg.cpp
//
// Register half of the frequency-response builtins: the CallContext wrappers
// evalfr / freqresp / bode / nyquist / rlocus that delegate to the engine-free
// compute in freq.cpp. library.cpp forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <numkit/control/freq/freq.hpp>
#include <numkit/control/lti/lti.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, ctx.engine->resource()
#include <numkit/value/value.hpp>   // Value::matrix, ValueType (default w/k args)
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

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

void evalfr_reg(Span<const Value> a, size_t, Span<Value> o, CallContext &c)
{
    argcheck::checkNargin(a.size(), 2, 2, "evalfr");
    o[0] = evalfr(a[0], a[1].toScalar(), c.engine->resource());
}

void freqresp_reg(Span<const Value> a, size_t, Span<Value> o, CallContext &c)
{
    argcheck::checkNargin(a.size(), 1, 6, "freqresp");
    auto *mr = c.engine->resource();
    Value wArg = (a.size() >= 2) ? a[1]
                : Value::matrix(0, 0, ValueType::DOUBLE, mr);
    o[0] = freqresp(a[0], wArg, mr);
}

void bode_reg(Span<const Value> a, size_t, Span<Value> o, CallContext &c)
{
    argcheck::checkNargin(a.size(), 0, 6, "bode");
    auto *mr = c.engine->resource();
    Value sys = a.empty() ? defaultDemoSys(mr) : a[0];
    Value wArg = (a.size() >= 2) ? a[1]
                : Value::matrix(0, 0, ValueType::DOUBLE, mr);
    auto b = bode(sys, wArg, mr);
    if (o.size() >= 1) o[0] = std::move(b.mag);
    if (o.size() >= 2) o[1] = std::move(b.phase);
    if (o.size() >= 3) o[2] = std::move(b.w);
}

void nyquist_reg(Span<const Value> a, size_t, Span<Value> o, CallContext &c)
{
    argcheck::checkNargin(a.size(), 0, 6, "nyquist");
    auto *mr = c.engine->resource();
    Value sys = a.empty() ? defaultDemoSys(mr) : a[0];
    Value wArg = (a.size() >= 2) ? a[1]
                : Value::matrix(0, 0, ValueType::DOUBLE, mr);
    auto n = nyquist(sys, wArg, mr);
    if (o.size() >= 1) o[0] = std::move(n.re);
    if (o.size() >= 2) o[1] = std::move(n.im);
    if (o.size() >= 3) o[2] = std::move(n.w);
}

void rlocus_reg(Span<const Value> a, size_t, Span<Value> o, CallContext &c)
{
    argcheck::checkNargin(a.size(), 0, 5, "rlocus");
    auto *mr = c.engine->resource();
    Value sys = a.empty() ? defaultDemoSys(mr) : a[0];
    Value kArg = (a.size() >= 2) ? a[1]
                : Value::matrix(0, 0, ValueType::DOUBLE, mr);
    auto [r, k] = rlocus(sys, kArg, mr);
    if (o.size() >= 1) o[0] = std::move(r);
    if (o.size() >= 2) o[1] = std::move(k);
}

} // namespace detail
} // namespace numkit::control

