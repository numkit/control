// toolboxes/control/src/lti/lti_reg.cpp
//
// Register half of the LTI-model builtins: the CallContext wrappers for the
// constructors (tf/zpk/ss/filt/frd), the data extractors (tfdata/zpkdata/
// ssdata/frdata) and ss2ss, all delegating to the engine-free compute in
// lti.cpp. library.cpp forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <numkit/control/lti/lti.hpp>
#include <numkit/control/conversion/conversion.hpp>   // minreal

#include <numkit/sys/plugin.hpp>   // CallContext, ctx.engine->resource()
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/value_type.hpp>   // ValueType, mtypeName (tol class gate)

#include <cctype>
#include <string>
#include <utility>

namespace numkit::control {
namespace detail {

static double argTs(Span<const Value> args, size_t pos) {
    if (args.size() <= pos || args[pos].isEmpty()) return 0.0;
    return args[pos].toScalar();
}

void tf_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs,
            CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 0, "tf");
    auto *mr = ctx.engine->resource();
    if (args.empty()) {
        Value s = Value::structure(mr);
        s.field("kind") = Value::fromString("tf", mr);
        s.field("Ts") = Value::scalar(0.0, mr);
        s.field("num") = Value::matrix(0, 0, ValueType::DOUBLE, mr);
        s.field("den") = Value::matrix(0, 0, ValueType::DOUBLE, mr);
        outs[0] = std::move(s);
        return;
    }
    if (args.size() == 1) {
        if (args[0].isStruct()) {
            if (args[0].hasField("kind") && args[0].field("kind").toString() == "tf") {
                outs[0] = args[0];
                return;
            }
            auto [num, den] = tfdata(args[0], true, mr);
            const double Ts = args[0].hasField("Ts") ? args[0].field("Ts").toScalar() : 0.0;
            outs[0] = tf(num, den, Ts, mr);
            return;
        }
        outs[0] = tf(args[0], Value::scalar(1.0, mr), 0.0, mr);
        return;
    }
    outs[0] = tf(args[0], args[1], argTs(args, 2), mr);
}

void zpk_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs,
             CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 0, "zpk");
    auto *mr = ctx.engine->resource();
    if (args.empty()) {
        Value s = Value::structure(mr);
        s.field("kind") = Value::fromString("zpk", mr);
        s.field("Ts") = Value::scalar(0.0, mr);
        s.field("z") = Value::matrix(0, 0, ValueType::DOUBLE, mr);
        s.field("p") = Value::matrix(0, 0, ValueType::DOUBLE, mr);
        s.field("k") = Value::matrix(0, 0, ValueType::DOUBLE, mr);
        outs[0] = std::move(s);
        return;
    }
    if (args.size() == 1) {
        if (args[0].isStruct()) {
            auto [z, p, k] = zpkdata(args[0], true, mr);
            const double Ts = args[0].hasField("Ts") ? args[0].field("Ts").toScalar() : 0.0;
            outs[0] = zpk(z, p, k.toScalar(), Ts, mr);
            return;
        }
        outs[0] = zpk(Value::matrix(0, 0, ValueType::DOUBLE, mr),
                      Value::matrix(0, 0, ValueType::DOUBLE, mr),
                      args[0].toScalar(), 0.0, mr);
        return;
    }
    argcheck::checkNargin(args.size(), 3, "zpk");
    outs[0] = zpk(args[0], args[1], args[2].toScalar(), argTs(args, 3), mr);
}

void ss_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs,
            CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 0, "ss");
    auto *mr = ctx.engine->resource();
    if (args.empty()) {
        Value s = Value::structure(mr);
        s.field("kind") = Value::fromString("ss", mr);
        s.field("Ts") = Value::scalar(0.0, mr);
        s.field("A") = Value::matrix(0, 0, ValueType::DOUBLE, mr);
        s.field("B") = Value::matrix(0, 0, ValueType::DOUBLE, mr);
        s.field("C") = Value::matrix(0, 0, ValueType::DOUBLE, mr);
        s.field("D") = Value::matrix(0, 0, ValueType::DOUBLE, mr);
        outs[0] = std::move(s);
        return;
    }
    if (args.size() == 1) {
        if (args[0].isStruct()) {
            auto [A, B, C, D] = ssdata(args[0], mr);
            const double Ts = args[0].hasField("Ts") ? args[0].field("Ts").toScalar() : 0.0;
            outs[0] = ss(A, B, C, D, Ts, mr);
            return;
        }
        outs[0] = ss(Value::matrix(0, 0, ValueType::DOUBLE, mr),
                     Value::matrix(0, 0, ValueType::DOUBLE, mr),
                     Value::matrix(0, 0, ValueType::DOUBLE, mr),
                     args[0], 0.0, mr);
        return;
    }
    argcheck::checkNargin(args.size(), 4, "ss");
    outs[0] = ss(args[0], args[1], args[2], args[3], argTs(args, 4), mr);
}

void filt_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs,
              CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 0, "filt");
    auto *mr = ctx.engine->resource();
    if (args.empty()) {
        Value s = Value::structure(mr);
        s.field("kind") = Value::fromString("tf", mr);
        s.field("Ts") = Value::scalar(-1.0, mr);
        s.field("num") = Value::matrix(0, 0, ValueType::DOUBLE, mr);
        s.field("den") = Value::matrix(0, 0, ValueType::DOUBLE, mr);
        s.field("variable") = Value::fromString("z^-1", mr);
        outs[0] = std::move(s);
        return;
    }
    if (args.size() == 1) {
        outs[0] = filt(args[0], Value::scalar(1.0, mr), -1.0, mr);
        return;
    }
    // MATLAB default Ts for filt is -1 (unspecified discrete).
    const double Ts = (args.size() >= 3 && !args[2].isEmpty())
                      ? args[2].toScalar() : -1.0;
    outs[0] = filt(args[0], args[1], Ts, mr);
}

void frd_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs,
             CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 0, "frd");
    auto *mr = ctx.engine->resource();
    if (args.empty()) {
        Value s = Value::structure(mr);
        s.field("kind") = Value::fromString("frd", mr);
        s.field("Ts") = Value::scalar(0.0, mr);
        s.field("resp") = Value::matrix(0, 0, ValueType::DOUBLE, mr);
        s.field("freq") = Value::matrix(0, 0, ValueType::DOUBLE, mr);
        outs[0] = std::move(s);
        return;
    }
    argcheck::checkNargin(args.size(), 2, "frd");
    outs[0] = frd(args[0], args[1], argTs(args, 2), mr);
}

static bool wantVector(Span<const Value> args, size_t pos) {
    if (args.size() <= pos) return false;
    if (!args[pos].isChar() && !args[pos].isString()) return false;
    std::string s = args[pos].toString();
    for (auto &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s == "v" || s == "vector";
}

void tfdata_reg(Span<const Value> args, size_t nargout, Span<Value> outs,
                CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "tfdata");  // no max: model-array tail in the reference (no count wall to 7 args)
    auto [num, den] = tfdata(args[0], wantVector(args, 1), ctx.engine->resource());
    outs[0] = std::move(num);
    if (nargout > 1) outs[1] = std::move(den);
}

void zpkdata_reg(Span<const Value> args, size_t nargout, Span<Value> outs,
                 CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "zpkdata");  // no max: model-array tail in the reference (no count wall to 7 args)
    auto [z, p, k] = zpkdata(args[0], wantVector(args, 1), ctx.engine->resource());
    outs[0] = std::move(z);
    if (nargout > 1) outs[1] = std::move(p);
    if (nargout > 2) outs[2] = std::move(k);
}

void ssdata_reg(Span<const Value> args, size_t nargout, Span<Value> outs,
                CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "ssdata");  // no max: model-array tail in the reference (no count wall to 7 args)
    auto [A, B, C, D] = ssdata(args[0], ctx.engine->resource());
    outs[0] = std::move(A);
    if (nargout > 1) outs[1] = std::move(B);
    if (nargout > 2) outs[2] = std::move(C);
    if (nargout > 3) outs[3] = std::move(D);
}

void ss2ss_reg(Span<const Value> args, size_t /*nargout*/,
               Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 5, "ss2ss");
    if (args[0].isStruct())
        argcheck::checkNargin(args.size(), 2, 2, "ss2ss");
    outs[0] = ss2ss(args[0], args[1], ctx.engine->resource());
}

void minreal_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs,
                 CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 5, "minreal");
    // tol extraction — class contract probed R2025b: numeric scalars
    // coerce (int8 tol and even complex tol are accepted there; the
    // real part carries); logical/char/cell/string are rejected
    // (Control:transformation:minreal3). The compute API takes a plain
    // double, so this is where the Value→native crossing happens.
    double tol = 0.0;
    if (args.size() >= 2 && !args[1].isEmpty()) {
        const Value &tv = args[1];
        const ValueType t = tv.type();
        const bool numericScalar = tv.isScalar()
            && (t == ValueType::DOUBLE || t == ValueType::SINGLE
                || t == ValueType::COMPLEX || isIntegerType(t));
        if (!numericScalar)
            throw Error(std::string("minreal: input 2 (tol) must be a ")
                        + "numeric scalar tolerance (got a "
                        + argcheck::detail::dimsStr(tv) + " "
                        + mtypeName(t) + " array).",
                        0, 0, "minreal", "", "numkit:minreal:tol");
        tol = tv.isComplex() ? tv.complexData()[0].real()
                             : tv.elemAsDouble(0);
    }
    outs[0] = minreal(args[0], tol, ctx.engine->resource());
}

void frdata_reg(Span<const Value> args, size_t nargout, Span<Value> outs,
                CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "frdata");  // no max: model-array tail in the reference (no count wall to 7 args)
    // 'v' flag accepted for MATLAB compatibility; frdata always returns
    // column vectors regardless (we don't model SISO 1×1×N tensors).
    (void)wantVector(args, 1);
    auto [resp, freq] = frdata(args[0], ctx.engine->resource());
    outs[0] = std::move(resp);
    if (nargout > 1) outs[1] = std::move(freq);
}

void tf2ss_reg(Span<const Value> args, size_t nargout, Span<Value> outs,
               CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "tf2ss");
    auto r = tf2ss(args[0], args[1], ctx.engine->resource());
    outs[0] = std::move(r.A);
    if (nargout > 1) outs[1] = std::move(r.B);
    if (nargout > 2) outs[2] = std::move(r.C);
    if (nargout > 3) outs[3] = std::move(r.D);
}

void ss2tf_reg(Span<const Value> args, size_t nargout, Span<Value> outs,
               CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 4, 5, "ss2tf");
    int iu = 1;
    if (args.size() >= 5 && !args[4].isEmpty()) {
        argcheck::expectedScalar(args[4], 5, "ss2tf", "iu");
        argcheck::expectedPositive(args[4], 5, "ss2tf", "iu");
        argcheck::expectedInteger(args[4], 5, "ss2tf", "iu");
        iu = static_cast<int>(args[4].elemAsDouble(0));
    }
    auto [num, den] = ss2tf(args[0], args[1], args[2], args[3], iu, ctx.engine->resource());
    outs[0] = std::move(num);
    if (nargout > 1) outs[1] = std::move(den);
}

void tf2zp_reg(Span<const Value> args, size_t nargout, Span<Value> outs,
               CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "tf2zp");
    auto r = tf2zp(args[0], args[1], ctx.engine->resource());
    outs[0] = std::move(r.z);
    if (nargout > 1) outs[1] = std::move(r.p);
    if (nargout > 2) outs[2] = std::move(r.k);
}

void zp2tf_reg(Span<const Value> args, size_t nargout, Span<Value> outs,
               CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 3, "zp2tf");
    auto [num, den] = zp2tf(args[0], args[1], args[2], ctx.engine->resource());
    outs[0] = std::move(num);
    if (nargout > 1) outs[1] = std::move(den);
}

} // namespace detail
} // namespace numkit::control
