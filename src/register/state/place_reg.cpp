// toolboxes/control/src/state/place_reg.cpp
//
// Register half of the pole-placement builtins: the CallContext wrappers
// acker / place that delegate to the engine-free compute in place.cpp.
// library.cpp forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <numkit/control/state/place.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, ctx.engine->resource()
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

namespace numkit::control {
namespace detail {

void acker_reg(Span<const Value> a, size_t, Span<Value> o, CallContext &c)
{
    argcheck::checkNargin(a.size(), 3, 3, "acker");
    o[0] = acker(a[0], a[1], a[2], c.engine->resource());
}

void place_reg(Span<const Value> a, size_t, Span<Value> o, CallContext &c)
{
    argcheck::checkNargin(a.size(), 3, 3, "place");
    o[0] = place(a[0], a[1], a[2], c.engine->resource());
}

} // namespace detail
} // namespace numkit::control
