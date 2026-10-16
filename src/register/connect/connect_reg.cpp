// toolboxes/control/src/connect/connect_reg.cpp
//
// Register half of the interconnection builtins: the CallContext wrappers
// series / parallel / feedback that delegate to the engine-free compute in
// connect.cpp. library.cpp forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <numkit/control/connect/connect.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, ctx.engine->resource()
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>

namespace numkit::control {
namespace detail {

void series_reg(Span<const Value> a, size_t, Span<Value> o, CallContext &c) {
    argcheck::checkNargin(a.size(), 2, 10, "series");
    o[0] = series(a[0], a[1], c.engine->resource());
}

void parallel_reg(Span<const Value> a, size_t, Span<Value> o, CallContext &c) {
    argcheck::checkNargin(a.size(), 2, "parallel");  // no max: name-value tail in the reference (no count wall to 12 args)
    o[0] = parallel(a[0], a[1], c.engine->resource());
}

void feedback_reg(Span<const Value> a, size_t, Span<Value> o, CallContext &c) {
    argcheck::checkNargin(a.size(), 2, 10, "feedback");
    int sign = -1;
    if (a.size() >= 3 && !a[2].isEmpty()) {
        const double sv = argcheck::expectedReal(a[2], 3, "sign", "feedback");
        argcheck::expectedInteger(sv, 3, "sign", "feedback");
        if (sv != 1.0 && sv != -1.0)
            throw Error("feedback: sign must be 1 or -1",
                        0, 0, "feedback", "", "numkit:feedback:signNotChoice");
        sign = static_cast<int>(sv);
    }
    o[0] = feedback(a[0], a[1], sign, c.engine->resource());
}

} // namespace detail
} // namespace numkit::control
