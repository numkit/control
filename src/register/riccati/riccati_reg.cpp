// toolboxes/control/src/riccati/riccati_reg.cpp
//
// Register half of the algebraic Riccati solvers: the CallContext
// builtins care / dare that delegate to the engine-free compute in
// riccati.cpp. library.cpp forward-declares + registers these by name.
//
// MATLAB output order is [X, L, G]: solution, closed-loop eigenvalues,
// gain. R defaults to the m×m identity when omitted (care(A,B,Q)).
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <numkit/control/riccati/riccati.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, ctx.engine->resource()
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>

namespace numkit::control {
namespace detail {

namespace {

// m×m identity Value (default R for the 3-argument forms).
Value eyeVal(size_t m, std::pmr::memory_resource *mr) {
    Value I = Value::matrix(m, m, ValueType::DOUBLE, mr);
    double *d = I.doubleDataMut();
    for (size_t i = 0; i < m; ++i) d[i * m + i] = 1.0;
    return I;
}

void emit(const RiccatiResult &r, Span<Value> o) {
    if (o.size() >= 1) o[0] = r.X;
    if (o.size() >= 2) o[1] = r.L;
    if (o.size() >= 3) o[2] = r.G;
}

} // anonymous

void care_reg(Span<const Value> a, size_t, Span<Value> o, CallContext &c)
{
    argcheck::checkNargin(a.size(), 3, "care");  // no max: name-value tail in the reference (no count wall to 12 args)
    auto *mr = c.engine->resource();
    Value R = (a.size() >= 4) ? a[3] : eyeVal(a[1].dims().cols(), mr);
    emit(care(a[0], a[1], a[2], R, mr), o);
}

void dare_reg(Span<const Value> a, size_t, Span<Value> o, CallContext &c)
{
    argcheck::checkNargin(a.size(), 3, "dare");  // no max: name-value tail in the reference (no count wall to 12 args)
    auto *mr = c.engine->resource();
    Value R = (a.size() >= 4) ? a[3] : eyeVal(a[1].dims().cols(), mr);
    emit(dare(a[0], a[1], a[2], R, mr), o);
}

namespace {
// lqr / dlqr are thin wrappers on care / dare. MATLAB returns [K, S, P]:
// optimal gain, Riccati solution, closed-loop poles — i.e. care's
// {G, X, L} re-ordered. (Cross-term N and the lqr(sys,…) form are
// deferred; the (A,B,Q[,R]) signature covers the common case.)
void emitLqr(const RiccatiResult &r, Span<Value> o) {
    if (o.size() >= 1) o[0] = r.G;   // K — optimal gain
    if (o.size() >= 2) o[1] = r.X;   // S — Riccati solution
    if (o.size() >= 3) o[2] = r.L;   // P — closed-loop poles
}
} // anonymous

void lqr_reg(Span<const Value> a, size_t, Span<Value> o, CallContext &c)
{
    // R2025b matrix form requires R: lqr(A,B,Q,R[,N]). The model form
    // lqr(sys,Q,R) is deferred — bugs/opened/control/arity-min-forms.md
    // (the old nk 3-arg matrix form with R=I was a divergence, removed).
    argcheck::checkNargin(a.size(), 4, 5, "lqr");
    auto *mr = c.engine->resource();
    emitLqr(care(a[0], a[1], a[2], a[3], mr), o);
}

void dlqr_reg(Span<const Value> a, size_t, Span<Value> o, CallContext &c)
{
    // R2025b matrix form requires R: dlqr(A,B,Q,R[,N]); model form
    // dlqr(sys,Q,R) deferred — see lqr above.
    argcheck::checkNargin(a.size(), 4, 5, "dlqr");
    auto *mr = c.engine->resource();
    emitLqr(dare(a[0], a[1], a[2], a[3], mr), o);
}

} // namespace detail
} // namespace numkit::control
