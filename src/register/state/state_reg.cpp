// toolboxes/control/src/state/state_reg.cpp
//
// Register half of the controllability/observability builtins: the
// CallContext wrappers ctrb / obsv that delegate to the engine-free
// compute in state.cpp. library.cpp forward-declares + registers these
// by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/control/state/state.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, ctx.engine->resource()
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

namespace numkit::control {
namespace detail {

void ctrb_reg(keel::ArrayRef<Value> a, keel::usize, keel::MutArrayRef<Value> o, CallContext &c)
{
    argcheck::checkNargin(a.size(), 1, 2, "ctrb");
    auto *mr = c.engine->allocator();
    if (a.size() == 1) {
        // Single-arg form: must be a sys struct.
        o[0] = ctrb_sys(a[0], mr);
        return;
    }
    o[0] = ctrb_AB(a[0], a[1], mr);
}

void obsv_reg(keel::ArrayRef<Value> a, keel::usize, keel::MutArrayRef<Value> o, CallContext &c)
{
    argcheck::checkNargin(a.size(), 1, 2, "obsv");
    auto *mr = c.engine->allocator();
    if (a.size() == 1) {
        o[0] = obsv_sys(a[0], mr);
        return;
    }
    o[0] = obsv_AC(a[0], a[1], mr);
}

void gram_reg(keel::ArrayRef<Value> a, keel::usize, keel::MutArrayRef<Value> o, CallContext &c)
{
    argcheck::checkNargin(a.size(), 2, 2, "gram");
    o[0] = gram(a[0], a[1], c.engine->allocator());
}

void covar_reg(keel::ArrayRef<Value> a, keel::usize, keel::MutArrayRef<Value> o, CallContext &c)
{
    argcheck::checkNargin(a.size(), 2, 5, "covar");
    if (a[0].isStruct())
        argcheck::checkNargin(a.size(), 2, 2, "covar");
    auto r = covar(a[0], a[1], c.engine->allocator());
    if (o.size() >= 1) o[0] = std::move(r.P);
    if (o.size() >= 2) o[1] = std::move(r.Q);
}

} // namespace detail
} // namespace numkit::control
