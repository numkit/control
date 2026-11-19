// toolboxes/control/src/library.cpp
//
// Registration hub for the Control System Toolbox builtins.
// Namespace: control.<sub>.<name>; most entry points are also aliased
// into `compat.<name>` so MATLAB-style scripts can call them flat.
// A few names (e.g. `isstable`) collide with builtins from another
// toolbox — for those we register only the qualified form via
// `regOnly` so engine startup doesn't trip the duplicate-registration
// guard.

#include <keel/core.h>
#include <keel/format.h>
#include <keel/ref.h>
#include <numkit/control/library.hpp>

#include <numkit/sys/plugin.hpp>
#include <utility>

namespace numkit::control::detail {
// lti/lti.cpp
void tf_reg       (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void zpk_reg      (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void ss_reg       (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void tfdata_reg   (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void zpkdata_reg  (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void ssdata_reg   (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void filt_reg     (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void frd_reg      (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void frdata_reg   (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void ss2ss_reg    (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void minreal_reg  (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void tf2ss_reg    (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void ss2tf_reg    (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void tf2zp_reg    (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void zp2tf_reg    (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void ord2_reg     (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void modred_reg   (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
// props/props.cpp
void isct_reg     (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void isdt_reg     (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void issiso_reg   (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void isproper_reg (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void isstable_reg (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void order_reg    (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void pole_reg     (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void zero_reg     (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void damp_reg     (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void pzmap_reg    (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void isstatic_reg (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void tzero_reg    (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
// connect/connect.cpp
void series_reg   (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void parallel_reg (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void feedback_reg (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
// response/response.cpp
void step_reg     (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void impulse_reg  (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void initial_reg  (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void lsim_reg     (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
// freq/freq.cpp
void evalfr_reg   (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void freqresp_reg (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void bode_reg     (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void nyquist_reg  (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void rlocus_reg   (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
// analyze/analyze.cpp
void dcgain_reg   (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void margin_reg   (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void stepinfo_reg (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void hinfnorm_reg (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void allmargin_reg(keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
// discretize/discretize.cpp
void c2d_reg      (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void d2c_reg      (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
// state/state.cpp
void ctrb_reg     (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void obsv_reg     (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void gram_reg     (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void covar_reg    (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
// state/place.cpp
void acker_reg    (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void place_reg    (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
// lyapunov/lyapunov.cpp
void lyap_reg     (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void dlyap_reg    (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
// riccati/riccati.cpp
void care_reg     (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void dare_reg     (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void lqr_reg      (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void dlqr_reg     (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
} // namespace numkit::control::detail

namespace numkit {

void ControlLibrary::install(PluginContext &engine)
{
    auto reg = [&](const char *sub, const char *name, ExternalFunc fn) {
        keel::String ns = keel::format("control.{}", sub);
        engine.registerFunction(ns.as_ref(), name, std::move(fn));
        // compat.* registration removed — bare-name resolver replaces it
    };
    auto regOnly = [&](const char *sub, const char *name, ExternalFunc fn) {
        keel::String ns = keel::format("control.{}", sub);
        engine.registerFunction(ns.as_ref(), name, std::move(fn));
    };

    reg("lti", "tf",      &control::detail::tf_reg);
    reg("lti", "zpk",     &control::detail::zpk_reg);
    reg("lti", "ss",      &control::detail::ss_reg);
    reg("lti", "tfdata",  &control::detail::tfdata_reg);
    reg("lti", "zpkdata", &control::detail::zpkdata_reg);
    reg("lti", "ssdata",  &control::detail::ssdata_reg);
    reg("lti", "filt",    &control::detail::filt_reg);
    reg("lti", "frd",     &control::detail::frd_reg);
    reg("lti", "frdata",  &control::detail::frdata_reg);
    reg("lti", "ord2",    &control::detail::ord2_reg);

    reg("convert", "ss2ss",   &control::detail::ss2ss_reg);
    reg("convert", "minreal", &control::detail::minreal_reg);
    reg("convert", "modred",  &control::detail::modred_reg);
    reg("convert", "tf2ss",   &control::detail::tf2ss_reg);
    reg("convert", "ss2tf",   &control::detail::ss2tf_reg);
    reg("convert", "tf2zp",   &control::detail::tf2zp_reg);
    reg("convert", "zp2tf",   &control::detail::zp2tf_reg);

    reg("props", "isct",     &control::detail::isct_reg);
    reg("props", "isdt",     &control::detail::isdt_reg);
    reg("props", "issiso",   &control::detail::issiso_reg);
    reg("props", "isproper", &control::detail::isproper_reg);
    // isstable already lives in toolboxes/signal (operates on coefficient
    // pairs (b,a)). Keep ours qualified so a script can opt-in via
    // `import control.props.*` when working with sys structs.
    regOnly("props", "isstable", &control::detail::isstable_reg);
    reg("props", "order",    &control::detail::order_reg);
    reg("props", "pole",     &control::detail::pole_reg);
    reg("props", "zero",     &control::detail::zero_reg);
    reg("props", "damp",     &control::detail::damp_reg);
    reg("props", "pzmap",    &control::detail::pzmap_reg);
    reg("props", "isstatic", &control::detail::isstatic_reg);
    reg("props", "tzero",    &control::detail::tzero_reg);

    reg("connect", "series",   &control::detail::series_reg);
    reg("connect", "parallel", &control::detail::parallel_reg);
    reg("connect", "feedback", &control::detail::feedback_reg);

    reg("response", "step",     &control::detail::step_reg);
    reg("response", "impulse",  &control::detail::impulse_reg);
    // Internal alias so the m-source dispatcher (math_integration_reg.cpp)
    // can delegate struct/LTI inputs to the C++ path — the m-source and
    // the reg share the "impulse" name, the m-source wins.
    reg("response", "impulse_lti",  &control::detail::impulse_reg);
    reg("response", "initial",  &control::detail::initial_reg);
    reg("response", "lsim",     &control::detail::lsim_reg);
    reg("response", "evalfr",   &control::detail::evalfr_reg);
    reg("response", "freqresp", &control::detail::freqresp_reg);
    reg("response", "bode",     &control::detail::bode_reg);
    reg("response", "nyquist",  &control::detail::nyquist_reg);
    reg("response", "rlocus",   &control::detail::rlocus_reg);

    reg("analyze", "dcgain",   &control::detail::dcgain_reg);
    reg("analyze", "margin",   &control::detail::margin_reg);
    reg("analyze", "stepinfo", &control::detail::stepinfo_reg);
    reg("analyze", "hinfnorm", &control::detail::hinfnorm_reg);
    reg("analyze", "allmargin", &control::detail::allmargin_reg);

    reg("discretize", "c2d", &control::detail::c2d_reg);
    reg("discretize", "d2c", &control::detail::d2c_reg);

    reg("state", "ctrb",  &control::detail::ctrb_reg);
    reg("state", "obsv",  &control::detail::obsv_reg);
    reg("state", "gram",  &control::detail::gram_reg);
    reg("state", "covar", &control::detail::covar_reg);
    reg("state", "acker", &control::detail::acker_reg);
    reg("state", "place", &control::detail::place_reg);

    reg("lyap", "lyap",  &control::detail::lyap_reg);
    reg("lyap", "dlyap", &control::detail::dlyap_reg);

    reg("riccati", "care", &control::detail::care_reg);
    reg("riccati", "dare", &control::detail::dare_reg);
    reg("riccati", "lqr",  &control::detail::lqr_reg);
    reg("riccati", "dlqr", &control::detail::dlqr_reg);
}

} // namespace numkit
