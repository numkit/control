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
    if (args.size() == 1 || (args.size() == 2 && (args[0].isChar() || args[0].isString()))) {
        if (args[0].isChar() || args[0].isString()) {
            std::string v = args[0].toString();
            if (v == "s" || v == "p" || v == "z" || v == "q" || v == "z^-1" || v == "q^-1") {
                const double Ts = (v == "s" || v == "p")
                    ? 0.0
                    : ((args.size() >= 2 && !args[1].isEmpty()) ? args[1].toScalar() : -1.0);
                Value num = Value::matrix(1, 2, ValueType::DOUBLE, mr);
                if (v == "z^-1" || v == "q^-1") {
                    num.doubleDataMut()[0] = 0.0;
                    num.doubleDataMut()[1] = 1.0;
                    outs[0] = filt(num, Value::scalar(1.0, mr), Ts, mr);
                } else {
                    num.doubleDataMut()[0] = 1.0;
                    num.doubleDataMut()[1] = 0.0;
                    outs[0] = tf(num, Value::scalar(1.0, mr), Ts, mr);
                }
                return;
            }
        }
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
    if (args.size() == 1 || (args.size() == 2 && (args[0].isChar() || args[0].isString()))) {
        if (args[0].isChar() || args[0].isString()) {
            std::string v = args[0].toString();
            if (v == "s" || v == "p" || v == "z" || v == "q") {
                const double Ts = (v == "s" || v == "p")
                    ? 0.0
                    : ((args.size() >= 2 && !args[1].isEmpty()) ? args[1].toScalar() : -1.0);
                outs[0] = zpk(Value::scalar(0.0, mr),
                              Value::matrix(0, 0, ValueType::DOUBLE, mr),
                              1.0, Ts, mr);
                return;
            }
        }
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
        const double iuVal = argcheck::expectedReal(args[4], 5, "iu", "ss2tf");
        iu = argcheck::expectedPositiveInteger(iuVal, 5, "iu", "ss2tf");
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

void ord2_reg(Span<const Value> args, size_t nargout, Span<Value> outs,
              CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "ord2");
    const double wn = argcheck::expectedReal(args[0], 1, "wn", "ord2");
    const double z  = argcheck::expectedReal(args[1], 2, "z", "ord2");
    auto *mr = ctx.engine->resource();
    if (nargout <= 2) {
        outs[0] = Value::scalar(1.0, mr);
        if (outs.size() >= 2) {
            Value den = Value::matrix(1, 3, ValueType::DOUBLE, mr);
            den.doubleDataMut()[0] = 1.0;
            den.doubleDataMut()[1] = 2.0 * z * wn;
            den.doubleDataMut()[2] = wn * wn;
            outs[1] = std::move(den);
        }
        return;
    }
    Value A = Value::matrix(2, 2, ValueType::DOUBLE, mr);
    A.doubleDataMut()[0] = 0.0;
    A.doubleDataMut()[1] = -wn * wn;
    A.doubleDataMut()[2] = 1.0;
    A.doubleDataMut()[3] = -2.0 * z * wn;
    Value B = Value::matrix(2, 1, ValueType::DOUBLE, mr);
    B.doubleDataMut()[0] = 0.0;
    B.doubleDataMut()[1] = 1.0;
    Value C = Value::matrix(1, 2, ValueType::DOUBLE, mr);
    C.doubleDataMut()[0] = 1.0;
    C.doubleDataMut()[1] = 0.0;
    outs[0] = std::move(A);
    if (outs.size() >= 2) outs[1] = std::move(B);
    if (outs.size() >= 3) outs[2] = std::move(C);
    if (outs.size() >= 4) outs[3] = Value::scalar(0.0, mr);
}

void modred_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs,
                CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "modred");
    auto *mr = ctx.engine->resource();
    auto [A, B, C, D] = ssdata(args[0], mr);
    const double Ts = (args[0].isStruct() && args[0].hasField("Ts"))
        ? args[0].field("Ts").toScalar() : 0.0;
    const size_t n = A.dims().rows();
    const size_t nu = B.dims().cols();
    const size_t ny = C.dims().rows();

    std::vector<bool> isElim(n, false);
    for (size_t i = 0; i < args[1].numel(); ++i) {
        const double v = args[1].elemAsDouble(i);
        const int idx = static_cast<int>(v) - 1;
        if (idx < 0 || static_cast<size_t>(idx) >= n)
            throw Error("modred: state elimination index out of range.",
                        0, 0, "modred", "", "numkit:modred:elim");
        isElim[static_cast<size_t>(idx)] = true;
    }
    std::vector<size_t> keep, elim;
    for (size_t i = 0; i < n; ++i) {
        if (isElim[i]) elim.push_back(i);
        else keep.push_back(i);
    }

    std::string method = "matchdc";
    if (args.size() >= 3 && (args[2].isChar() || args[2].isString())) {
        method = args[2].toString();
        for (char &ch : method) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }

    const size_t nk = keep.size();
    Value Ar = Value::matrix(nk, nk, ValueType::DOUBLE, mr);
    Value Br = Value::matrix(nk, nu, ValueType::DOUBLE, mr);
    Value Cr = Value::matrix(ny, nk, ValueType::DOUBLE, mr);
    Value Dr = Value::matrix(ny, nu, ValueType::DOUBLE, mr);
    for (size_t j = 0; j < nk; ++j)
        for (size_t i = 0; i < nk; ++i)
            Ar.doubleDataMut()[j * nk + i] = A.elemAsDouble(keep[j] * n + keep[i]);
    for (size_t j = 0; j < nu; ++j)
        for (size_t i = 0; i < nk; ++i)
            Br.doubleDataMut()[j * nk + i] = B.elemAsDouble(j * n + keep[i]);
    for (size_t j = 0; j < nk; ++j)
        for (size_t i = 0; i < ny; ++i)
            Cr.doubleDataMut()[j * ny + i] = C.elemAsDouble(keep[j] * ny + i);
    for (size_t j = 0; j < nu; ++j)
        for (size_t i = 0; i < ny; ++i)
            Dr.doubleDataMut()[j * ny + i] = (D.numel() == ny * nu)
                ? D.elemAsDouble(j * ny + i) : D.elemAsDouble(0);

    outs[0] = ss(Ar, Br, Cr, Dr, Ts, mr);
}

} // namespace detail
} // namespace numkit::control
