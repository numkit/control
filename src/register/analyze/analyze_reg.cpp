// toolboxes/control/src/analyze/analyze_reg.cpp
//
// Register half of the analysis builtins: the CallContext wrappers dcgain /
// margin / stepinfo that delegate to the engine-free compute in analyze.cpp.
// library.cpp forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <numkit/control/analyze/analyze.hpp>
#include <numkit/control/lti/lti.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, ctx.engine->resource()
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

void dcgain_reg(Span<const Value> a, size_t, Span<Value> o, CallContext &c)
{
    argcheck::checkNargin(a.size(), 1, 4, "dcgain");
    o[0] = dcgain(a[0], c.engine->resource());
}

void margin_reg(Span<const Value> a, size_t, Span<Value> o, CallContext &c)
{
    argcheck::checkNargin(a.size(), 0, 4, "margin");
    auto *mr = c.engine->resource();
    Value sys = a.empty() ? defaultDemoSys(mr) : a[0];
    auto m = margin(sys, mr);
    if (o.size() >= 1) o[0] = std::move(m.Gm);
    if (o.size() >= 2) o[1] = std::move(m.Pm);
    if (o.size() >= 3) o[2] = std::move(m.Wcg);
    if (o.size() >= 4) o[3] = std::move(m.Wcp);
}

void stepinfo_reg(Span<const Value> a, size_t, Span<Value> o, CallContext &c)
{
    argcheck::checkNargin(a.size(), 1, "stepinfo");  // no max: name-value tail in the reference (no count wall to 12 args)
    o[0] = stepinfo(a[0], c.engine->resource());
}

void hinfnorm_reg(Span<const Value> a, size_t, Span<Value> o, CallContext &c)
{
    argcheck::checkNargin(a.size(), 1, 3, "hinfnorm");
    o[0] = hinfnorm(a[0], c.engine->resource());
}

void allmargin_reg(Span<const Value> a, size_t, Span<Value> o, CallContext &c)
{
    argcheck::checkNargin(a.size(), 1, 6, "allmargin");
    o[0] = allmargin(a[0], c.engine->resource());
}

} // namespace detail
} // namespace numkit::control
