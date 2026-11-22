// toolboxes/control/src/props/props_reg.cpp
//
// Register half of the system-property builtins: the CallContext wrappers
// (isct/isdt/issiso/isproper/isstable/order/pole/zero/damp/pzmap/isstatic/
// tzero) that delegate to the engine-free compute in props.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/control/internal/lti_object.hpp>
#include <numkit/control/props/props.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, ctx.engine->resource()
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <utility>

namespace numkit::control {
namespace detail {

void isct_reg(keel::ArrayRef<Value> a, keel::usize, keel::MutArrayRef<Value> o, CallContext &c)
{ argcheck::checkNargin(a.size(), 1, 1, "isct");
  o[0] = isct(a[0], c.engine->allocator()); }

void isdt_reg(keel::ArrayRef<Value> a, keel::usize, keel::MutArrayRef<Value> o, CallContext &c)
{ argcheck::checkNargin(a.size(), 1, 1, "isdt");
  o[0] = isdt(a[0], c.engine->allocator()); }

void issiso_reg(keel::ArrayRef<Value> a, keel::usize, keel::MutArrayRef<Value> o, CallContext &c)
{ argcheck::checkNargin(a.size(), 1, 1, "issiso");
  o[0] = issiso(a[0], c.engine->allocator()); }

void isproper_reg(keel::ArrayRef<Value> a, keel::usize, keel::MutArrayRef<Value> o, CallContext &c)
{ argcheck::checkNargin(a.size(), 1, "isproper");  // no max: accepts multiple models in the reference (no count wall to 7 args)
  o[0] = isproper(a[0], c.engine->allocator()); }

void isstable_reg(keel::ArrayRef<Value> a, keel::usize nargout, keel::MutArrayRef<Value> o, CallContext &c)
{ argcheck::checkNargin(a.size(), 1, 3, "isstable");
  if (!internal::isLtiModel(a[0]) && (a.size() >= 2 || a[0].dims().rows() != a[0].dims().cols())) {
      if (const auto *fn = c.engine->findExternal("signal.filter_analysis.isstable")) {
          (*fn)(a, nargout, o, c);
          return;
      }
  }
  o[0] = isstable(a[0], c.engine->allocator()); }

void order_reg(keel::ArrayRef<Value> a, keel::usize, keel::MutArrayRef<Value> o, CallContext &c)
{ argcheck::checkNargin(a.size(), 1, 1, "order");
  o[0] = order(a[0], c.engine->allocator()); }

void pole_reg(keel::ArrayRef<Value> a, keel::usize, keel::MutArrayRef<Value> o, CallContext &c)
{ argcheck::checkNargin(a.size(), 1, "pole");  // no max: model-array tail in the reference (no count wall to 7 args)
  o[0] = pole(a[0], c.engine->allocator()); }

void zero_reg(keel::ArrayRef<Value> a, keel::usize, keel::MutArrayRef<Value> o, CallContext &c)
{ argcheck::checkNargin(a.size(), 1, "zero");  // no max: model-array tail in the reference (no count wall to 7 args)
  o[0] = zero(a[0], c.engine->allocator()); }

void damp_reg(keel::ArrayRef<Value> a, keel::usize, keel::MutArrayRef<Value> o, CallContext &c)
{
    argcheck::checkNargin(a.size(), 1, 2, "damp");
    auto d = damp(a[0], c.engine->allocator());
    if (o.size() >= 1) o[0] = std::move(d.wn);
    if (o.size() >= 2) o[1] = std::move(d.zeta);
    if (o.size() >= 3) o[2] = std::move(d.p);
}

void pzmap_reg(keel::ArrayRef<Value> a, keel::usize, keel::MutArrayRef<Value> o, CallContext &c)
{
    argcheck::checkNargin(a.size(), 1, 4, "pzmap");
    auto [p, z] = pzmap(a[0], c.engine->allocator());
    if (o.size() >= 1) o[0] = std::move(p);
    if (o.size() >= 2) o[1] = std::move(z);
}

void isstatic_reg(keel::ArrayRef<Value> a, keel::usize, keel::MutArrayRef<Value> o, CallContext &c)
{ argcheck::checkNargin(a.size(), 1, 2, "isstatic");
  o[0] = isstatic(a[0], c.engine->allocator()); }

void tzero_reg(keel::ArrayRef<Value> a, keel::usize, keel::MutArrayRef<Value> o, CallContext &c)
{ argcheck::checkNargin(a.size(), 1, 6, "tzero");
  o[0] = tzero(a[0], c.engine->allocator()); }

} // namespace detail
} // namespace numkit::control
