// toolboxes/control/tests/known_bugs_test.cpp
//
// One DISABLED_ test per OPEN bug in bugs/control/*.md. Disabled until
// fixed; remove `DISABLED_` to turn into a live regression guard.
// MATLAB R2025b reference values.

#include <numkit/control/connect/connect.hpp>
#include <numkit/core/standard_engine.hpp>
#include <numkit/core/standard_engine.hpp>
#include <gtest/gtest.h>

using namespace numkit;

class ControlKnownBug : public ::testing::Test
{
public:
    StandardEngine engine;
    void SetUp() override {}
    Value eval(const std::string &c) { return engine.eval(c); }
    double evalScalar(const std::string &c) { return eval(c).toScalar(); }
};

// bugs/control/lqr-hinfnorm.md — LQR gain via the CARE.
// FIXED 2026-06-19 (wraps care) — promoted live.
TEST_F(ControlKnownBug, Lqr)
{
    eval("K = lqr([0 1; 0 0], [0; 1], eye(2), 1);");
    EXPECT_NEAR(evalScalar("K(1)"), 1.000000, 1e-5);
    EXPECT_NEAR(evalScalar("K(2)"), 1.732051, 1e-5);
}

// bugs/control/hinfnorm.md — H-infinity norm (Inf for poles on jω axis).
// FIXED 2026-06-19 (Hamiltonian bisection) — promoted live.
TEST_F(ControlKnownBug, Hinfnorm)
{
    eval("g = hinfnorm(ss([0 1; -1 0], [0; 1], [1 0], 0));");
    EXPECT_TRUE(std::isinf(evalScalar("g")));
}

// bugs/control/lqr-hinfnorm.md — dlqr (discrete LQR) missing.
// FIXED 2026-06-19 (wraps dare) — promoted live.
TEST_F(ControlKnownBug, Dlqr)
{
    eval("K = dlqr([0.9 0.1; 0 0.8], [0; 1], eye(2), 1);");
    EXPECT_NEAR(evalScalar("sum(K)"), 0.71004388, 1e-5);
}

// bugs/control/lqr-hinfnorm.md — gram (controllability/observability gramian).
// FIXED 2026-06-19 (wraps lyap/dlyap) — promoted live.
TEST_F(ControlKnownBug, Gram)
{
    eval("W = gram(ss([-1 0; 0 -2], [1; 1], [1 1], 0), 'c');");
    EXPECT_NEAR(evalScalar("sum(W(:))"), 1.4166667, 1e-5);
}

// bugs/control/care-dare.md — continuous/discrete algebraic Riccati solvers.
// FIXED 2026-06-18 (matrix sign-function method) — promoted live.
TEST_F(ControlKnownBug, CareDare)
{
    eval("X = care([0 1; 0 0], [0; 1], eye(2));");
    EXPECT_NEAR(evalScalar("trace(X)"), 3.46410161513776, 1e-8);
    EXPECT_NEAR(evalScalar("X(1,1)"),   1.73205080756888, 1e-8);  // sqrt(3)
    eval("Y = dare([1 1; 0 1], [0; 1], eye(2), 1);");
    EXPECT_NEAR(evalScalar("trace(Y)"), 7.56025722770319, 1e-7);
}

// bugs/control/minreal.md — minimal realization (pole/zero cancellation).
// FIXED 2026-06-19 (root cancellation) — promoted live.
TEST_F(ControlKnownBug, Minreal)
{
    // (s+1)/(s+1)^2 -> 1/(s+1)
    eval("sysr = minreal(tf([1 1], [1 2 1])); [n, d] = tfdata(sysr, 'v');");
    EXPECT_NEAR(evalScalar("n(end)"), 1.0, 1e-9);
    EXPECT_NEAR(evalScalar("d(1)"),   1.0, 1e-9);
    EXPECT_NEAR(evalScalar("d(end)"), 1.0, 1e-9);
    EXPECT_EQ(static_cast<int>(evalScalar("numel(d)")), 2);   // reduced to first order
}

// bugs/control/initial.md — initial-condition response (zero input).
// FIXED 2026-06-19 (zero-input simulate + x0) — promoted live.
TEST_F(ControlKnownBug, Initial)
{
    eval("[y, t] = initial(ss(-2, 0, 1, 0), 1);");   // y = e^{-2t}
    EXPECT_NEAR(evalScalar("y(1)"),   1.0, 1e-9);
    EXPECT_NEAR(evalScalar("y(end)"), 0.00301995172040398, 1e-6);
}

// bugs/control/allmargin.md — all stability margins as a struct.
// FIXED 2026-06-19 (exact G(jω) scan + bisection) — promoted live.
TEST_F(ControlKnownBug, Allmargin)
{
    eval("S = allmargin(tf(1, [1 6 11 6]));");   // 1/((s+1)(s+2)(s+3))
    EXPECT_DOUBLE_EQ(evalScalar("double(S.Stable)"), 1.0);
    EXPECT_NEAR(evalScalar("S.GainMargin(1)"),  60.0,            1e-2);
    EXPECT_NEAR(evalScalar("S.GMFrequency(1)"), 3.31662561934,   1e-6);  // sqrt(11)
}

// bugs/control/zpk-empty-zeros.md — zpk with no finite zeros drops gain k.
// FIXED 2026-06-19 (zp2tf empty-zero → num=[k]) — promoted live.
TEST_F(ControlKnownBug, ZpkEmptyZerosGain)
{
    eval("[n, d] = tfdata(zpk([], [-1 -2], 2), 'v');");
    EXPECT_NEAR(evalScalar("n(end)"), 2.0, 1e-12);   // nk: 0 (gain dropped)
    EXPECT_NEAR(evalScalar("d(1)"),   1.0, 1e-12);
    EXPECT_NEAR(evalScalar("d(end)"), 2.0, 1e-12);
}

// bugs/control/covar.md — output covariance from white-noise input.
// FIXED 2026-06-19 (lyap/dlyap gramian + C·Q·Cᵀ) — promoted live.
TEST_F(ControlKnownBug, Covar)
{
    EXPECT_NEAR(evalScalar("covar(ss(-1, 1, 1, 0), 1)"), 0.5, 1e-9);
}

// bugs/closed/control/tf-arithmetic.md — tf arithmetic operators
// (*, +, scalar*) compose transfer functions.
TEST_F(ControlKnownBug, TfArithmeticOperators)
{
    eval("G1 = tf([1], [1 1]); G2 = tf([1], [1 2]);");
    eval("G = G1 * G2;");                            // 1/(s^2+3s+2)
    EXPECT_NEAR(evalScalar("G.num"),    1.0, 1e-12);
    EXPECT_NEAR(evalScalar("G.den(2)"), 3.0, 1e-12);
    EXPECT_NEAR(evalScalar("G.den(3)"), 2.0, 1e-12);
    eval("P = G1 + G2;");                            // (2s+3)/(s^2+3s+2)
    EXPECT_NEAR(evalScalar("P.num(1)"), 2.0, 1e-12);
    EXPECT_NEAR(evalScalar("P.num(2)"), 3.0, 1e-12);
    eval("S = 2 * G1;");                             // 2/(s+1)
    EXPECT_NEAR(evalScalar("S.num"),    2.0, 1e-12);
    EXPECT_NEAR(evalScalar("S.den(2)"), 1.0, 1e-12);
}

// ─── argcheck migration findings (2026-09-17), bugs/opened/control/*.md ──

// bugs/opened/control/arity-min-forms.md — reference-legal short forms:
// empty/static-gain constructors, default-grid freqresp, Ts-derived lsim,
// model-form lqr, 0-arg demo plots.
TEST_F(ControlKnownBug, DISABLED_ControlShortForms)
{
    EXPECT_NO_THROW(eval("t = tf();"));
    EXPECT_NO_THROW(eval("g = tf(1);"));
    EXPECT_NO_THROW(eval("fr = freqresp(tf([1], [1 1]));"));
    EXPECT_NO_THROW(eval("y = lsim(tf([1], [1 1], 0.1), [1 2 3]);"));
    EXPECT_NO_THROW(eval("[K, S, P] = lqr(ss([0 1; -1 -2], [1; 0], [1 0], 0), eye(2), 1);"));
    EXPECT_NO_THROW(eval("step();"));   // demo-plot form
}

// bugs/closed/control/type-dependent-walls.md — the reference's model
// path rejects extra args the numeric-dummy wall allows.
TEST_F(ControlKnownBug, CovarModelPathStrict)
{
    EXPECT_ANY_THROW(eval("covar(ss(-1, 1, 1, 0), 1, 1);"));
    EXPECT_ANY_THROW(eval("ss2ss(ss([0 1; -1 -2], [1; 0], [1 0], 0), eye(2), 1, 1, 1);"));
}

// bugs/opened/lang/reg-domain-logic.md (control site): the sign +/-1
// content check lives in feedback_reg; the compute accepts any int.
// Move it to a compute-side feedback_argcheck so both call paths answer
// nk:feedback:signNotChoice.
TEST_F(ControlKnownBug, FeedbackSignCppPathUnified)
{
    eval("G = tf(1, [1 1]); H = tf(1, 1);");
    // Interpreter path (the reg validates).
    try {
        engine.eval("feedback(G, H, 2);");
        FAIL() << "eval path should reject sign=2";
    } catch (const numkit::Error &e) {
        EXPECT_EQ(e.identifier(), "numkit:feedback:signNotChoice") << e.what();
    }
    // Direct C++ public-API path (the compute does not validate yet).
    try {
        control::feedback(*engine.getVariable("G"), *engine.getVariable("H"),
                          2, engine.resource());
        FAIL() << "C++ path should reject sign=2";
    } catch (const numkit::Error &e) {
        EXPECT_EQ(e.identifier(), "numkit:feedback:signNotChoice") << e.what();
    }
}

// bugs/closed/control/d2c-default-and-tustin.md — the reference default
// is 'zoh'; nk defaults to tustin, which errors on a plain stable pole.
TEST_F(ControlKnownBug, D2cDefaultZoh)
{
    eval("d = d2c(tf([1], [1 1], 0.1));");
    eval("d2 = d2c(tf([1], [1 1], 0.1), 'zoh');");
    eval("[num, den] = tfdata(d, 'v');");
    EXPECT_NEAR(evalScalar("den(3)"), 986.9603, 1e-2);
}

// ──────────────────────────────────────────────────────────────────────
// Type-contract portion (2026-09-18) — reference-probed gaps in
// bugs/opened/control/{complex-kernels,tf-filt-mimo-cell,
// single-float-inputs,hinfnorm-discrete}.md. All values from
// MATLAB R2025b (build/probe9..11_control_ref.m).
// ──────────────────────────────────────────────────────────────────────

// complex-kernels.md — lyap with complex A computes a complex solution
// in the reference; nk stubs with nk:lyap:complexUnsupported.
TEST_F(ControlKnownBug, DISABLED_LyapComplexKernel)
{
    eval("X = lyap(complex([-1 0; 0 -2], 1), eye(2));");
    EXPECT_NEAR(evalScalar("real(X(1,1))"), 0.4444444444, 1e-9);
    EXPECT_NEAR(evalScalar("imag(X(1,2))"), -0.0555555556, 1e-9);
    EXPECT_NEAR(evalScalar("real(X(2,2))"), 0.2777777778, 1e-9);
}

// complex-kernels.md — dlyap with complex A (reference value probed).
TEST_F(ControlKnownBug, DISABLED_DlyapComplexKernel)
{
    eval("X = dlyap(complex([-1 0; 0 -2], 1), eye(2));");
    EXPECT_NEAR(evalScalar("real(X(1,1))"), -0.7777777778, 1e-9);
    EXPECT_NEAR(evalScalar("imag(X(1,2))"), 0.2222222222, 1e-9);
    EXPECT_NEAR(evalScalar("real(X(2,2))"), -0.4444444444, 1e-9);
}

// complex-kernels.md — ctrb/obsv with complex A produce complex
// matrices in the reference.
TEST_F(ControlKnownBug, DISABLED_CtrbObsvComplexKernel)
{
    eval("Co = ctrb(complex([2 1; 1 3], 1), [1; 0]);");
    EXPECT_NEAR(evalScalar("real(Co(1,2))"), 2.0, 1e-9);
    EXPECT_NEAR(evalScalar("imag(Co(1,2))"), 1.0, 1e-9);
    eval("Ob = obsv(complex([2 1; 1 3], 1), [1 0]);");
    EXPECT_NEAR(evalScalar("imag(Ob(2,1))"), 1.0, 1e-9);
}

// complex-kernels.md — acker with complex A returns a complex K
// (reference probed: [8+2i 15-4i] + a pole-location warning).
TEST_F(ControlKnownBug, DISABLED_AckerComplexKernel)
{
    eval("K = acker(complex([2 1; 1 3], 1), [1; 0], [-1 -2]);");
    EXPECT_NEAR(evalScalar("real(K(1))"), 8.0, 1e-9);
    EXPECT_NEAR(evalScalar("imag(K(1))"), 2.0, 1e-9);
    EXPECT_NEAR(evalScalar("imag(K(2))"), -4.0, 1e-9);
}

// complex-kernels.md — ss2ss with a complex T builds a complex model
// in the reference.
TEST_F(ControlKnownBug, DISABLED_Ss2ssComplexKernel)
{
    eval("s = ss2ss(ss([0 1; -1 -2], [1; 0], [1 0], 0), complex(eye(2), 1));");
    EXPECT_EQ(eval("s.kind").toString(), "ss");
}

// complex-kernels.md — lsim with complex u / complex x0 computes a
// complex response in the reference.
TEST_F(ControlKnownBug, DISABLED_LsimComplexKernel)
{
    eval("sysd = ss([0 1; -1 -2], [1; 0], [1 0], 0);");
    eval("y = lsim(sysd, complex(ones(1, 11), 1), 0:0.1:1);");
    EXPECT_NEAR(evalScalar("imag(y(2))"), 0.09984142212, 1e-9);
    eval("y2 = lsim(sysd, ones(1, 11), 0:0.1:1, complex([0; 0], 1));");
    EXPECT_NEAR(evalScalar("imag(y2(1))"), 1.0, 1e-9);
}

// tf-filt-mimo-cell.md — the cell (MIMO) coefficient form builds a
// MIMO model in the reference; nk's struct model is SISO-only (1x1
// cells unwrap; larger cells stub with nk:tf:mimoUnsupported).
TEST_F(ControlKnownBug, DISABLED_TfMimoCellForm)
{
    eval("sys = tf({1 2; 5 6}, {1 3; 1 4});");
    EXPECT_EQ(evalScalar("issiso(sys)"), 0.0);
}

// single-float-inputs.md — acker with single inputs returns a
// single-class result in the reference; nk's kernel is double-only
// and stubs with nk:acker:singleUnsupported.
TEST_F(ControlKnownBug, DISABLED_AckerSingleResult)
{
    eval("K = acker(single([2 1; 1 3]), [1; 0], [-1 -2]);");
    EXPECT_EQ(eval("class(K)").toString(), "single");
    EXPECT_NEAR(evalScalar("K(1)"), 8.0, 1e-6);
    EXPECT_NEAR(evalScalar("K(2)"), 21.0, 1e-6);
}

// hinfnorm-discrete.md — discrete-time models have a finite H-infinity
// norm in the reference (probed Inf for this marginally-stable one);
// nk's kernel is continuous-only.
TEST_F(ControlKnownBug, DISABLED_HinfnormDiscrete)
{
    eval("g = hinfnorm(ss([2 1; 1 3], [1; 0], [1 0], 0, 0.1));");
    EXPECT_TRUE(std::isinf(evalScalar("g")));
}
