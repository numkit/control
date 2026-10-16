// toolboxes/control/src/lyapunov/lyapunov_reg.cpp
//
// Register half of the Lyapunov solvers: the CallContext builtins lyap /
// dlyap that delegate to the engine-free compute in lyapunov.cpp.
// library.cpp forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <numkit/control/lyapunov/lyapunov.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, ctx.engine->resource()
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

namespace numkit::control {
namespace detail {

void lyap_reg(Span<const Value> a, size_t, Span<Value> o, CallContext &c)
{
    argcheck::checkNargin(a.size(), 2, 4, "lyap");
    o[0] = lyap(a[0], a[1], c.engine->resource());
}

void dlyap_reg(Span<const Value> a, size_t, Span<Value> o, CallContext &c)
{
    argcheck::checkNargin(a.size(), 2, 4, "dlyap");
    o[0] = dlyap(a[0], a[1], c.engine->resource());
}

} // namespace detail
} // namespace numkit::control
