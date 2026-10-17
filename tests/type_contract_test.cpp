// toolboxes/control/tests/type_contract_test.cpp
//
// Type-contract guards for the 16 value-contract control entries
// (argcheck separation, dev-docs/todo/argcheck_separation.md).
// Rejection classes + coercion results reference-probed against
// MATLAB R2025b on 2026-09-18 (build/probe9..11_control_ref.m):
// the reference coerces broad, per-entry-different numeric sets
// (lyap takes int8/char; ctrb rejects integer at its matmul; tf
// rejects logical while filt's numerator takes even strings), and
// answers garbage sys-args with clean rejections where nk used to
// crash raw (cell/string) or silently drop imaginary parts.

#include <numkit/bundle/standard_engine.hpp>
#include <numkit/core/engine.hpp>
#include <gtest/gtest.h>

using namespace numkit;

class ControlTypeContractTest : public ::testing::Test
{
public:
    StandardEngine engine;
    Value eval(const std::string &c) { return engine.eval(c); }
    double evalScalar(const std::string &c) { return eval(c).toScalar(); }
    // Assert that `expr` throws with exactly this nk identifier.
    template <typename Expr>
    void expectId(const char *id, Expr expr) {
        try {
            expr();
            FAIL() << "expected numkit::Error with id " << id;
        } catch (const Error &e) {
            EXPECT_EQ(e.identifier(), id);
        }
    }
};

// ──────────────────────────────────────────────────────────────────────
// lyap / dlyap
// ──────────────────────────────────────────────────────────────────────

TEST_F(ControlTypeContractTest, LyapCoercesNumericAndChar)
{
    eval("As = [-1 0; 0 -2];");
    eval("X0 = lyap(As, eye(2));");
    EXPECT_NEAR(evalScalar("X0(1,1)"), 0.5, 1e-12);
    // int8 / logical / single / char all coerce (reference-probed).
    eval("X8 = lyap(int8(As), eye(2));");
    EXPECT_NEAR(evalScalar("X8(1,1)"), 0.5, 1e-12);
    eval("Xl = lyap(logical(As), eye(2));");
    EXPECT_NEAR(evalScalar("Xl(1,1)"), -0.5, 1e-12);
    eval("Xs = lyap(single(As), eye(2));");
    EXPECT_NEAR(evalScalar("Xs(1,1)"), 0.5, 1e-12);
    eval("Xq = lyap(As, int8(eye(2)));");
    EXPECT_NEAR(evalScalar("Xq(1,1)"), 0.5, 1e-12);
    // char ['12';'34'] — reference computes the same code-value matrix.
    eval("Xc = lyap(['12';'34'], eye(2));");
    EXPECT_NEAR(evalScalar("Xc(1,1)"), 12.8762376237624, 1e-9);
    EXPECT_NEAR(evalScalar("Xc(1,2)"), -12.6287128712871, 1e-9);
}

TEST_F(ControlTypeContractTest, LyapDlyapRejectionClasses)
{
    eval("As = [-1 0; 0 -2];");
    expectId("numkit:lyap:inputType", [&] { eval("lyap({1 2; 3 4}, eye(2))"); });
    expectId("numkit:lyap:inputType", [&] { eval("lyap(\"x\", eye(2))"); });
    expectId("numkit:lyap:inputType", [&] { eval("lyap(As, {1 2; 3 4})"); });
    expectId("numkit:dlyap:inputType", [&] { eval("dlyap({1 2; 3 4}, eye(2))"); });
    // complex A/Q compute complex results in the reference.
    eval("X1 = lyap(complex(As, 1), eye(2));");
    EXPECT_NEAR(evalScalar("real(X1(1,1))"), 0.4444444444, 1e-9);
    eval("X2 = lyap(As, complex(eye(2), 1));");
    EXPECT_NEAR(evalScalar("imag(X2(1,1))"), 0.5, 1e-9);
    eval("X3 = dlyap(complex(As, 1), eye(2));");
    EXPECT_NEAR(evalScalar("real(X3(1,1))"), -0.7777777778, 1e-9);
}

TEST_F(ControlTypeContractTest, DlyapSingularVoiceAndValue)
{
    // A with an eigenvalue on the unit circle: no Stein solution —
    // voiced with dlyap's OWN identifier (was nk:lyap:singular).
    expectId("numkit:dlyap:singular",
             [&] { eval("dlyap([-1 0; 0 -2], eye(2))"); });
    // Unit-disc-stable reference value (MATLAB R2025b probed).
    eval("X = dlyap([-0.5 0; 0 -0.25], eye(2));");
    EXPECT_NEAR(evalScalar("X(1,1)"), 1.3333333333333, 1e-10);
    EXPECT_NEAR(evalScalar("X(2,2)"), 1.0666666666667, 1e-10);
}

// ──────────────────────────────────────────────────────────────────────
// ctrb / obsv
// ──────────────────────────────────────────────────────────────────────

TEST_F(ControlTypeContractTest, CtrbObsvCoercionAndRejection)
{
    eval("Ad = [2 1; 1 3]; B = [1; 0]; C = [1 0];");
    // single coerces to a double result (reference-probed class=double).
    eval("Co = ctrb(single(Ad), B);");
    EXPECT_EQ(eval("class(Co)").toString(), "double");
    EXPECT_NEAR(evalScalar("Co(1,2)"), 2.0, 1e-12);
    // char coerces with code values (['12';'34'] → [49 50;51 52]).
    eval("Coc = ctrb(['12';'34'], B);");
    EXPECT_NEAR(evalScalar("Coc(1,2)"), 49.0, 1e-12);
    EXPECT_NEAR(evalScalar("Coc(2,2)"), 51.0, 1e-12);
    // integer matrices fail at the reference's matmul layer.
    expectId("numkit:ctrb:integerUnsupported", [&] { eval("ctrb(int8(Ad), B)"); });
    expectId("numkit:obsv:integerUnsupported", [&] { eval("obsv(int8(Ad), C)"); });
    // cell / complex.
    expectId("numkit:ctrb:inputType", [&] { eval("ctrb({1 2; 3 4}, B)"); });
    eval("Cocplx = ctrb(complex(Ad, 1), B);");
    EXPECT_NEAR(evalScalar("imag(Cocplx(1,2))"), 1.0, 1e-12);
    eval("Obcplx = obsv(complex(Ad, 1), C);");
    EXPECT_NEAR(evalScalar("imag(Obcplx(2,1))"), 1.0, 1e-12);
}

// ──────────────────────────────────────────────────────────────────────
// acker
// ──────────────────────────────────────────────────────────────────────

TEST_F(ControlTypeContractTest, AckerContracts)
{
    eval("Ad = [2 1; 1 3]; B = [1; 0];");
    eval("K = acker(Ad, B, [-1 -2]);");
    EXPECT_NEAR(evalScalar("K(1)"), 8.0, 1e-12);
    EXPECT_NEAR(evalScalar("K(2)"), 21.0, 1e-12);
    // logical p coerces (was a raw "Not a double array" via poly).
    eval("Kl = acker(Ad, B, logical([1 1]));");
    EXPECT_NEAR(evalScalar("Kl(1)"), 3.0, 1e-12);
    EXPECT_NEAR(evalScalar("Kl(2)"), 5.0, 1e-12);
    // complex conjugate pair → real K (reference-identical [7 18]);
    // a non-conjugate complex set degenerates to the same real answer
    // the reference computes (with its own warning).
    eval("Kc = acker(Ad, B, [-1-1i, -1+1i]);");
    EXPECT_NEAR(evalScalar("Kc(1)"), 7.0, 1e-12);
    EXPECT_NEAR(evalScalar("Kc(2)"), 18.0, 1e-12);
    eval("Kn = acker(Ad, B, [-1-1i, -2]);");
    EXPECT_NEAR(evalScalar("Kn(1)"), 8.0, 1e-12);
    EXPECT_NEAR(evalScalar("Kn(2)"), 21.0, 1e-12);
    // char A coerces (reference computes the same code-value answer).
    eval("Ka = acker(['12';'34'], B, [-1 -2]);");
    EXPECT_NEAR(evalScalar("Ka(1)"), 104.0, 1e-9);
    // single → single-class result in the reference.
    eval("Ks1 = acker(single(Ad), B, [-1 -2]);");
    EXPECT_EQ(eval("class(Ks1)").toString(), "single");
    eval("Ks2 = acker(Ad, B, single([-1 -2]));");
    EXPECT_EQ(eval("class(Ks2)").toString(), "single");
    expectId("numkit:acker:integerUnsupported", [&] { eval("acker(int8(Ad), B, [-1 -2])"); });
    expectId("numkit:acker:integerUnsupported", [&] { eval("acker(Ad, B, int8([-1 -2]))"); });
    expectId("numkit:acker:inputType", [&] { eval("acker(Ad, B, 'ab')"); });
    expectId("numkit:acker:inputType", [&] { eval("acker(Ad, B, {1 2})"); });
    eval("Kcplx = acker(complex(Ad, 1), B, [-1 -2]);");
    EXPECT_NEAR(evalScalar("real(Kcplx(1))"), 8.0, 1e-9);
    EXPECT_NEAR(evalScalar("imag(Kcplx(1))"), 2.0, 1e-9);
}

// ──────────────────────────────────────────────────────────────────────
// ss2ss
// ──────────────────────────────────────────────────────────────────────

TEST_F(ControlTypeContractTest, Ss2ssTContract)
{
    eval("sysd = ss([0 1; -1 -2], [1; 0], [1 0], 0);");
    // logical / char / single T coerce (reference-probed acceptances).
    eval("s1 = ss2ss(sysd, logical(eye(2)));");
    eval("s2 = ss2ss(sysd, single(eye(2)));");
    eval("s3 = ss2ss(sysd, ['12';'34']);");
    EXPECT_EQ(eval("s1.kind").toString(), "ss");
    // integer T fails at the reference's matmul; complex T builds a
    // complex model.
    expectId("numkit:ss2ss:integerUnsupported",
             [&] { eval("ss2ss(sysd, int8(eye(2)))"); });
    eval("sc = ss2ss(sysd, complex(eye(2), 1));");
    EXPECT_EQ(eval("sc.kind").toString(), "ss");
    expectId("numkit:ss2ss:inputType",
             [&] { eval("ss2ss(sysd, {1 2; 3 4})"); });
    expectId("numkit:ss2ss:kind", [&] { eval("ss2ss(1, eye(2))"); });
}

// ──────────────────────────────────────────────────────────────────────
// tf / filt coefficient contracts
// ──────────────────────────────────────────────────────────────────────

TEST_F(ControlTypeContractTest, TfCoefficientContract)
{
    // int8 coerces to double storage (reference-probed: tf(int8,int8)
    // OK; tfdata content double).
    eval("s = tf(int8([1 2]), int8([1 3]));");
    EXPECT_EQ(eval("class(s.num)").toString(), "double");
    // 1x1 cell is the SISO-in-cell form: unwraps (reference-probed OK).
    eval("sc = tf({1}, [1 3]);");
    EXPECT_NEAR(evalScalar("sc.num"), 1.0, 1e-12);
    // complex coefficients are kept.
    eval("sx = tf(complex([1 2], 1), [1 3]);");
    EXPECT_EQ(eval("class(sx.num)").toString(), "double"); // complex-double
    // logical / char / string / cell-MIMO / empties are rejected.
    expectId("numkit:tf:inputType", [&] { eval("tf(logical([1 0]), [1 1])"); });
    expectId("numkit:tf:inputType", [&] { eval("tf([1 2], logical([1 1]))"); });
    expectId("numkit:tf:inputType", [&] { eval("tf('ab', [1 3])"); });
    expectId("numkit:tf:inputType", [&] { eval("tf(\"ab\", [1 3])"); });
    expectId("numkit:tf:inputType", [&] { eval("tf([1 2], \"13\")"); });
    expectId("numkit:tf:num", [&] { eval("tf([], [1 3])"); });
    expectId("numkit:tf:den", [&] { eval("tf([1 2], [])"); });
    eval("mimoTf = tf({1 2; 5 6}, {1 3; 1 4});");
    EXPECT_EQ(evalScalar("issiso(mimoTf)"), 0.0);
    expectId("numkit:tf:mimoSize",
             [&] { eval("tf({1 2; 5 6}, {1 3})"); });
}

TEST_F(ControlTypeContractTest, FiltBroadNumeratorContract)
{
    // filt's NUMERATOR coerces even logical/char/string (probed);
    // its DENOMINATOR rejects them exactly like tf's.
    eval("f1 = filt(logical([1 0]), [1 3]);");
    EXPECT_NEAR(evalScalar("f1.num(1)"), 1.0, 1e-12);
    eval("f2 = filt('ab', [1 3]);");
    EXPECT_NEAR(evalScalar("f2.num(1)"), 97.0, 1e-12);
    eval("f3 = filt(\"ab\", [1 3]);");
    EXPECT_NEAR(evalScalar("f3.num(1)"), 97.0, 1e-12);
    eval("f4 = filt({[1 2]}, [1 3]);");
    EXPECT_NEAR(evalScalar("f4.num(2)"), 2.0, 1e-12);
    expectId("numkit:filt:inputType", [&] { eval("filt([1 2], logical([1 1]))"); });
    expectId("numkit:filt:den", [&] { eval("filt([1 2], [])"); });
    expectId("numkit:filt:num", [&] { eval("filt([], [1 3])"); });
}

// ──────────────────────────────────────────────────────────────────────
// ss block contract
// ──────────────────────────────────────────────────────────────────────

TEST_F(ControlTypeContractTest, SsClassAndShapeContract)
{
    eval("Ad = [2 1; 1 3]; B = [1; 0]; C = [1 0];");
    // int8 coerces; ssdata reads back double (reference-probed).
    eval("s = ss(int8(Ad), B, C, 0);");
    EXPECT_NEAR(evalScalar("s.A(1,1)"), 2.0, 1e-12);
    // static / empty forms stay valid.
    eval("ss([], [], [], 5);");
    eval("ss(Ad, [], [], 5);");
    // logical / char / cell rejected (reference ssProperties6).
    expectId("numkit:ss:inputType", [&] { eval("ss(logical(Ad), B, C, 0)"); });
    expectId("numkit:ss:inputType", [&] { eval("ss(['12';'34'], B, C, 0)"); });
    expectId("numkit:ss:inputType", [&] { eval("ss({1 2; 3 4}, B, C, 0)"); });
    // shape consistency (previously built silently-broken models).
    expectId("numkit:ss:A", [&] { eval("ss([1 2 3], B, C, 0)"); });
    expectId("numkit:ss:B", [&] { eval("ss(Ad, [1 0 0], C, 0)"); });
    expectId("numkit:ss:C", [&] { eval("ss(Ad, B, [1 0 0], 0)"); });
    expectId("numkit:ss:D", [&] { eval("ss(Ad, B, C, eye(2))"); });
    expectId("numkit:ss:D", [&] { eval("ss(Ad, B, C, ones(2, 1))"); });
}

// ──────────────────────────────────────────────────────────────────────
// minreal tol / gram type
// ──────────────────────────────────────────────────────────────────────

TEST_F(ControlTypeContractTest, MinrealTolContract)
{
    eval("sysd = ss([0 1; -1 -2], [1; 0], [1 0], 0);");
    // numeric scalars coerce (int8/single accepted by the reference);
    // logical / char / cell are its minreal3 rejection.
    eval("minreal(sysd, int8(1));");
    eval("minreal(sysd, single(0.1));");
    expectId("numkit:minreal:tol", [&] { eval("minreal(sysd, true)"); });
    expectId("numkit:minreal:tol", [&] { eval("minreal(sysd, 'x')"); });
    expectId("numkit:minreal:tol", [&] { eval("minreal(sysd, {1})"); });
}

TEST_F(ControlTypeContractTest, GramTypeContract)
{
    eval("sysd = ss([0 1; -1 -2], [1; 0], [1 0], 0);");
    eval("Wc = gram(sysd, 'c');");
    EXPECT_NEAR(evalScalar("Wc(1,1)"), 1.25, 1e-12);
    eval("Ws = gram(sysd, \"c\");");   // string form accepted (probed)
    EXPECT_NEAR(evalScalar("Ws(2,2)"), 0.25, 1e-12);
    // non-char/string type args were RAW "Not a char array" crashes.
    expectId("numkit:gram:type", [&] { eval("gram(sysd, 'x')"); });
    expectId("numkit:gram:type", [&] { eval("gram(sysd, 1)"); });
    expectId("numkit:gram:type", [&] { eval("gram(sysd, {})"); });
    expectId("numkit:gram:kind", [&] { eval("gram(1, 'c')"); });
}

// ──────────────────────────────────────────────────────────────────────
// predicates + order
// ──────────────────────────────────────────────────────────────────────

TEST_F(ControlTypeContractTest, PredicateGarbageRejected)
{
    eval("sysd = ss([0 1; -1 -2], [1; 0], [1 0], 0);");
    EXPECT_EQ(evalScalar("issiso(sysd)"), 1.0);
    EXPECT_EQ(evalScalar("isproper(sysd)"), 1.0);
    EXPECT_NEAR(evalScalar("order(sysd)"), 2.0, 1e-12);
    // Garbage used to answer silent false/0 — the reference rejects
    // with UndefinedFunction; nk voices an inputType rejection.
    expectId("numkit:issiso:inputType", [&] { eval("issiso(1)"); });
    expectId("numkit:issiso:inputType", [&] { eval("issiso(struct('a', 1))"); });
    expectId("numkit:issiso:inputType", [&] { eval("issiso({})"); });
    expectId("numkit:isproper:inputType", [&] { eval("isproper(1)"); });
    expectId("numkit:isproper:inputType", [&] { eval("isproper({})"); });
    expectId("numkit:order:inputType", [&] { eval("order(1)"); });
    expectId("numkit:order:inputType", [&] { eval("order(struct('a', 1))"); });
    expectId("numkit:order:inputType", [&] { eval("order({})"); });
}

TEST_F(ControlTypeContractTest, FrdPredicates)
{
    // frd answers true for issiso/isproper and is REJECTED by order —
    // all three reference-probed (NotSupportedModelsofClass for order).
    eval("fd = frd([1; 2], [0.1; 0.2]);");
    EXPECT_EQ(evalScalar("issiso(fd)"), 1.0);
    EXPECT_EQ(evalScalar("isproper(fd)"), 1.0);
    expectId("numkit:order:inputType", [&] { eval("order(fd)"); });
}

// ──────────────────────────────────────────────────────────────────────
// hinfnorm
// ──────────────────────────────────────────────────────────────────────

TEST_F(ControlTypeContractTest, HinfnormContracts)
{
    eval("sysd = ss([0 1; -1 -2], [1; 0], [1 0], 0);");
    EXPECT_NEAR(evalScalar("hinfnorm(sysd)"), 2.0, 1e-9);
    // Static-gain scalar form: [|x|, |x|, 0] (reference-probed shape).
    eval("h = hinfnorm(1);");
    EXPECT_NEAR(evalScalar("h(1)"), 1.0, 1e-12);
    EXPECT_NEAR(evalScalar("h(3)"), 0.0, 1e-12);
    eval("hm = hinfnorm(-3);");
    EXPECT_NEAR(evalScalar("hm(1)"), 3.0, 1e-12);
    eval("hc = hinfnorm(2i);");
    EXPECT_NEAR(evalScalar("hc(1)"), 2.0, 1e-12);
    expectId("numkit:hinfnorm:inputType", [&] { eval("hinfnorm(struct('a', 1))"); });
    expectId("numkit:hinfnorm:inputType", [&] { eval("hinfnorm({})"); });
    EXPECT_TRUE(std::isinf(evalScalar("hinfnorm(ss([0 1; -1 -2], [1; 0], [1 0], 0, 0.1))")));
}

// ──────────────────────────────────────────────────────────────────────
// lsim
// ──────────────────────────────────────────────────────────────────────

TEST_F(ControlTypeContractTest, LsimInputContracts)
{
    eval("sysd = ss([0 1; -1 -2], [1; 0], [1 0], 0);");
    eval("t = 0:0.1:1; u = ones(1, 11);");
    // int8 / logical / single u coerce (reference-probed).
    eval("y0 = lsim(sysd, u, t);");
    eval("y8 = lsim(sysd, int8(u), t);");
    EXPECT_NEAR(evalScalar("y8(end)"), evalScalar("y0(end)"), 1e-12);
    eval("ys = lsim(sysd, single(u), single(t));");
    EXPECT_NEAR(evalScalar("ys(end)"), evalScalar("y0(end)"), 1e-6);
    // char u / cell u rejected (reference rfinputs02/18); complex u
    // computes a complex response.
    expectId("numkit:lsim:inputType",
             [&] { eval("lsim(sysd, repmat('1', 1, 11), t)"); });
    expectId("numkit:lsim:inputType", [&] { eval("lsim(sysd, {u}, t)"); });
    eval("yc1 = lsim(sysd, complex(u, 1), t);");
    EXPECT_NEAR(evalScalar("imag(yc1(2))"), 0.09984142212, 1e-9);
    // t must be real double/single (int8/complex t rejected — the
    // old code accepted int8 t and produced a garbage trajectory).
    expectId("numkit:lsim:t", [&] { eval("lsim(sysd, u, int8(t))"); });
    expectId("numkit:lsim:t", [&] { eval("lsim(sysd, u, complex(t, 1))"); });
    expectId("numkit:lsim:t", [&] { eval("lsim(sysd, u, {t})"); });
    // x0: wrong length was SILENTLY IGNORED (reference: timeresp1);
    // complex x0 computes a complex response; cell rejected; empty x0 valid.
    expectId("numkit:lsim:x0", [&] { eval("lsim(sysd, u, t, [0])"); });
    expectId("numkit:lsim:x0", [&] { eval("lsim(sysd, u, t, [0; 0; 0])"); });
    eval("yc2 = lsim(sysd, u, t, complex([0; 0], 1));");
    EXPECT_NEAR(evalScalar("imag(yc2(1))"), 1.0, 1e-9);
    expectId("numkit:lsim:inputType", [&] { eval("lsim(sysd, u, t, {[0; 0]})"); });
    eval("ye = lsim(sysd, u, t, []);");
    EXPECT_NEAR(evalScalar("ye(end)"), evalScalar("y0(end)"), 1e-12);
    expectId("numkit:lsim:kind", [&] { eval("lsim(1, u, t)"); });
}

// Nonzero x0 actually reaches the simulation now that the length is
// strictly checked (reference trajectory, MATLAB R2025b).
TEST_F(ControlTypeContractTest, LsimNonzeroX0Trajectory)
{
    eval("sysd = ss([0 1; -1 -2], [1; 0], [1 0], 0);");
    eval("y = lsim(sysd, ones(1, 11), 0:0.1:1, [0.5; -0.25]);");
    EXPECT_NEAR(evalScalar("y(1)"), 0.5, 1e-10);
    EXPECT_NEAR(evalScalar("y(end)"), 1.1722712574, 1e-8);
}
