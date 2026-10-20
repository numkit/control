// toolboxes/control/src/discretize/discretize_reg.cpp
//
// Register half of the discretisation builtins: the CallContext wrappers
// c2d / d2c that delegate to the engine-free compute in discretize.cpp.
// library.cpp forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <numkit/control/discretize/discretize.hpp>
#include <numkit/control/lti/lti.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, ctx.engine->resource()
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <string>

namespace numkit::control {
namespace detail {

static std::string argString(const Value &v) {
    if (!v.isChar() && !v.isString())
        throw Error("control discretize: expected a string method",
                    0, 0, "discretize", "", "numkit:discretize:type");
    return v.toString();
}

void c2d_reg(Span<const Value> a, size_t, Span<Value> o, CallContext &c)
{
    argcheck::checkNargin(a.size(), 2, 3, "c2d");
    auto *mr = c.engine->resource();
    if (a.size() == 3 && !a[0].isStruct() && !a[0].isObject()
        && a[0].isNumeric() && a[1].isNumeric() && a[2].isNumeric()) {
        const size_t n = a[0].dims().rows();
        const size_t m = a[1].dims().cols();
        Value C = Value::matrix(1, n, ValueType::DOUBLE, mr);
        Value D = Value::matrix(1, m, ValueType::DOUBLE, mr);
        Value sysc = ss(a[0], a[1], C, D, 0.0, mr);
        Value sysd = c2d(sysc, a[2].toScalar(), "zoh", mr);
        o[0] = sysd.field("A");
        if (o.size() >= 2) o[1] = sysd.field("B");
        return;
    }
    std::string method;
    if (a.size() >= 3 && !a[2].isEmpty()) method = argString(a[2]);
    Value sysd = c2d(a[0], a[1].toScalar(), method, mr);
    if (o.size() >= 2) {
        if (sysd.isStruct() && sysd.hasField("G"))
            o[1] = sysd.field("G");
        else
            o[1] = Value::matrix(0, 0, ValueType::DOUBLE, mr);
    }
    o[0] = std::move(sysd);
}

void d2c_reg(Span<const Value> a, size_t, Span<Value> o, CallContext &c)
{
    argcheck::checkNargin(a.size(), 1, 3, "d2c");
    std::string method;
    if (a.size() >= 2 && !a[1].isEmpty()) method = argString(a[1]);
    o[0] = d2c(a[0], method, c.engine->resource());
}

} // namespace detail
} // namespace numkit::control
