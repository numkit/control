// toolboxes/control/tests/arity_bounds_test.cpp
//
// Max-arity rejection guards for the control domain (argcheck migration,
// 2026-09-17). Every bounded contract probed against R2025b gets a
// max+1 call that must fail with nk:<fn>:maxnargin. Two names need a
// routed call: bare `impulse` resolves to the m-source dispatcher, so
// the shared reg is guarded through its impulse_lti alias; `isstable`
// is registered qualified-only (bare belongs to signal), so the guard
// imports it first.

#include <numkit/bundle/standard_engine.hpp>
#include <numkit/core/engine.hpp>
#include <numkit/value/error.hpp>
#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace {

struct Case
{
    const char *fn;
    const char *call;   // NULL -> same as fn
    int maxArgs;
};

const std::vector<Case> kBounds = {
    {"acker", nullptr, 3},
    {"allmargin", nullptr, 6},
    {"bode", nullptr, 6},
    {"c2d", nullptr, 3},
    {"covar", nullptr, 5},
    {"ctrb", nullptr, 2},
    {"d2c", nullptr, 3},
    {"damp", nullptr, 2},
    {"dcgain", nullptr, 4},
    {"dlqr", nullptr, 5},
    {"dlyap", nullptr, 4},
    {"evalfr", nullptr, 2},
    {"feedback", nullptr, 10},
    {"freqresp", nullptr, 6},
    {"gram", nullptr, 2},
    {"hinfnorm", nullptr, 3},
    {"initial", nullptr, 6},
    {"isct", nullptr, 1},
    {"isdt", nullptr, 1},
    {"issiso", nullptr, 1},
    {"isstable", nullptr, 3},
    {"isstatic", nullptr, 2},
    {"lqr", nullptr, 5},
    {"lsim", nullptr, 7},
    {"lyap", nullptr, 4},
    {"margin", nullptr, 4},
    {"minreal", nullptr, 5},
    {"nyquist", nullptr, 6},
    {"obsv", nullptr, 2},
    {"order", nullptr, 1},
    {"place", nullptr, 3},
    {"pzmap", nullptr, 4},
    {"rlocus", nullptr, 5},
    {"series", nullptr, 10},
    {"ss2ss", nullptr, 5},
    {"step", nullptr, 6},
    {"tzero", nullptr, 6},
    {"impulse", "impulse_lti", 6},
};

std::string callExpr(const Case &c)
{
    std::string s;
    if (std::string(c.fn) == "isstable")
        s = "import control.props.isstable; ";
    s += c.call ? c.call : c.fn;
    s += '(';
    for (int i = 0; i <= c.maxArgs; ++i) {
        if (i)
            s += ", ";
        s += '1';
    }
    s += ')';
    return s;
}

} // namespace

class ControlArityBoundsTest : public ::testing::TestWithParam<Case>
{
public:
    numkit::StandardEngine engine;
};

TEST_P(ControlArityBoundsTest, RejectsMaxPlusOne)
{
    const Case &c = GetParam();
    try {
        engine.eval(callExpr(c));
        FAIL() << "expected maxnargin error: " << callExpr(c);
    } catch (const numkit::Error &e) {
        EXPECT_EQ(e.identifier(), std::string("numkit:") + c.fn + ":maxnargin")
            << callExpr(c) << "\n  got: " << e.what();
    }
}

INSTANTIATE_TEST_SUITE_P(ControlArityBounds, ControlArityBoundsTest,
                         ::testing::ValuesIn(kBounds));
