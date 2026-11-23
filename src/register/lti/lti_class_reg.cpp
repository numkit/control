// modules/control/src/register/lti/lti_class_reg.cpp
//
// BuiltinClass definitions for LTI models (tf, ss, zpk, frd).
// Encapsulates property access, 1D/2D indexing (subsref), `end` (__end),
// `size`, LTI arithmetic operators (+, -, *, /, ^, uminus, uplus), and
// class-dispatched methods (impulse, isstable) inside the control plugin so
// core VM / TreeWalker / ops contain zero control-toolbox knowledge.

#include <keel/array.h>
#include <keel/algo.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>

#include <numkit/control/internal/lti_object.hpp>
#include <numkit/sys/plugin.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/object.hpp>
#include <numkit/value/value.hpp>


namespace numkit::control::detail {

void tf_reg      (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void zpk_reg     (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void ss_reg      (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void frd_reg     (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void impulse_reg (keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);
void isstable_reg(keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value>, CallContext &);

namespace {

using internal::isLtiKind;
using internal::isLtiModel;
using internal::ltiDataMutPtr;
using internal::ltiStruct;
using internal::wrapLtiObject;

const Value *ltiDataPtr(const Value &v) noexcept
{
    return isLtiModel(v) ? &ltiStruct(v) : nullptr;
}

// ── I/O dimensions (ny outputs x nu inputs) ─────────────────────────────────
void ltiIoDims(const Value &v, keel::usize &ny, keel::usize &nu)
{
    ny = 1;
    nu = 1;
    const Value *d = ltiDataPtr(v);
    if (!d || !d->hasField("kind"))
        return;
    const keel::String kind = d->field("kind").toString();
    if (kind == "tf" && d->hasField("num") && d->field("num").isCell()) {
        ny = d->field("num").dims().rows();
        nu = d->field("num").dims().cols();
    } else if (kind == "ss" && d->hasField("B") && d->hasField("C")) {
        ny = d->field("C").dims().rows();
        nu = d->field("B").dims().cols();
    }
}

// ── SISO polynomial extraction for LTI operators (+, -, *, /, ^) ────────────
struct LtiPoly
{
    keel::Array<keel::f64> num;
    keel::Array<keel::f64> den;
    keel::f64 Ts = 0.0;
};

bool isSisoTf(const Value &v)
{
    const Value *d = ltiDataPtr(v);
    return d && d->hasField("kind") && d->field("kind").toString() == "tf"
           && d->hasField("num") && d->hasField("den")
           && !d->field("num").isCell() && !d->field("den").isCell();
}

bool isSisoSs(const Value &v)
{
    const Value *d = ltiDataPtr(v);
    if (!d || !d->hasField("kind") || d->field("kind").toString() != "ss"
        || !d->hasField("A") || !d->hasField("B") || !d->hasField("C") || !d->hasField("D")) {
        return false;
    }
    const Value &B = d->field("B");
    const Value &C = d->field("C");
    const Value &D = d->field("D");
    return B.isNumeric() && C.isNumeric() && D.isNumeric()
           && !B.isComplex() && !C.isComplex() && !D.isComplex()
           && C.dims().rows() == 1 && B.dims().cols() == 1
           && D.dims().rows() == 1 && D.dims().cols() == 1;
}

bool toLtiPoly(const Value &v, LtiPoly &out)
{
    if (isSisoTf(v)) {
        const Value &d = *ltiDataPtr(v);
        const Value &nv = d.field("num");
        const Value &dv = d.field("den");
        if (!nv.isNumeric() || !dv.isNumeric() || nv.isComplex() || dv.isComplex())
            return false;
        out.num.resize(nv.numel());
        for (keel::usize i = 0; i < nv.numel(); ++i) out.num[i] = nv.elemAsDouble(i);
        out.den.resize(dv.numel());
        for (keel::usize i = 0; i < dv.numel(); ++i) out.den[i] = dv.elemAsDouble(i);
        out.Ts = d.hasField("Ts") ? d.field("Ts").toScalar() : 0.0;
        return true;
    }
    if (isSisoSs(v)) {
        const Value &d = *ltiDataPtr(v);
        const Value &Av = d.field("A");
        const Value &Bv = d.field("B");
        const Value &Cv = d.field("C");
        const Value &Dv = d.field("D");
        if (!Av.isNumeric() || Av.isComplex() || Av.dims().rows() != Av.dims().cols())
            return false;
        const keel::usize n = Av.dims().rows();
        const keel::f64 dVal = Dv.elemAsDouble(0);
        out.Ts = d.hasField("Ts") ? d.field("Ts").toScalar() : 0.0;
        if (n == 0) {
            out.num = {dVal};
            out.den = {1.0};
            return true;
        }
        // Fadeev-LeVerrier: den(s) = det(sI - A), num(s) = C * adj(sI - A) * B + D * den(s).
        auto Aat = [&](keel::usize r, keel::usize c) { return Av.elemAsDouble(c * n + r); };
        keel::Array<keel::f64> M(n * n, 0.0);
        for (keel::usize i = 0; i < n; ++i) M[i * n + i] = 1.0;
        out.den.assign(n + 1, 0.0);
        out.num.assign(n + 1, 0.0);
        out.den[0] = 1.0;
        out.num[0] = dVal;
        keel::Array<keel::f64> P(n * n, 0.0);
        for (keel::usize k = 1; k <= n; ++k) {
            keel::f64 cMB = 0.0;
            for (keel::usize r = 0; r < n; ++r) {
                const keel::f64 cr = Cv.elemAsDouble(r);
                for (keel::usize c = 0; c < n; ++c)
                    cMB += cr * M[r * n + c] * Bv.elemAsDouble(c);
            }
            for (keel::usize r = 0; r < n; ++r) {
                for (keel::usize c = 0; c < n; ++c) {
                    keel::f64 acc = 0.0;
                    for (keel::usize j = 0; j < n; ++j)
                        acc += Aat(r, j) * M[j * n + c];
                    P[r * n + c] = acc;
                }
            }
            keel::f64 tr = 0.0;
            for (keel::usize i = 0; i < n; ++i) tr += P[i * n + i];
            const keel::f64 ck = -tr / keel::as<keel::f64>(k);
            out.den[k] = ck;
            out.num[k] = cMB + dVal * ck;
            for (keel::usize r = 0; r < n; ++r) {
                for (keel::usize c = 0; c < n; ++c)
                    M[r * n + c] = P[r * n + c] + (r == c ? ck : 0.0);
            }
        }
        return true;
    }
    if (v.isScalar() && v.isNumeric() && !v.isComplex()) {
        out.num = {v.toScalar()};
        out.den = {1.0};
        out.Ts = -1.0; // static gain inherits sample time from LTI operand
        return true;
    }
    return false;
}

keel::Array<keel::f64> polyStrip(keel::ArrayRef<keel::f64> p)
{
    keel::usize i = 0;
    while (i + 1 < p.size() && p[i] == 0.0) ++i;
    if (i >= p.size()) return {0.0};
    return keel::Array<keel::f64>(p.slice(i));
}

keel::Array<keel::f64> polyMul(keel::ArrayRef<keel::f64> a, keel::ArrayRef<keel::f64> b)
{
    if (a.empty() || b.empty()) return {0.0};
    keel::Array<keel::f64> c(a.size() + b.size() - 1, 0.0);
    for (keel::usize i = 0; i < a.size(); ++i)
        for (keel::usize j = 0; j < b.size(); ++j)
            c[i + j] += a[i] * b[j];
    return polyStrip(c);
}

keel::Array<keel::f64> polyAdd(keel::ArrayRef<keel::f64> a, keel::ArrayRef<keel::f64> b, keel::f64 sign)
{
    const keel::usize n = keel::max(a.size(), b.size());
    keel::Array<keel::f64> c(n, 0.0);
    for (keel::usize i = 0; i < a.size(); ++i)
        c[n - a.size() + i] += a[i];
    for (keel::usize i = 0; i < b.size(); ++i)
        c[n - b.size() + i] += sign * b[i];
    return polyStrip(c);
}

Value buildTfObject(keel::ArrayRef<keel::f64> num, keel::ArrayRef<keel::f64> den,
                    keel::f64 Ts, keel::Allocator *mr)
{
    keel::Array<keel::f64> ns = polyStrip(num);
    keel::Array<keel::f64> ds = polyStrip(den);
    const keel::usize len = keel::max(ns.size(), ds.size());
    Value nv = Value::matrix(1, len, ValueType::DOUBLE, mr);
    Value dv = Value::matrix(1, len, ValueType::DOUBLE, mr);
    keel::f64 *nd = nv.doubleDataMut();
    keel::f64 *dd = dv.doubleDataMut();
    keel::fill(nd, nd + len, 0.0);
    keel::fill(dd, dd + len, 0.0);
    for (keel::usize i = 0; i < ns.size(); ++i) nd[len - ns.size() + i] = ns[i];
    for (keel::usize i = 0; i < ds.size(); ++i) dd[len - ds.size() + i] = ds[i];
    Value s = Value::structure(mr);
    s.field("kind") = Value::fromString("tf", mr);
    s.field("Ts") = Value::scalar(Ts, mr);
    s.field("num") = keel::move(nv);
    s.field("den") = keel::move(dv);
    return wrapLtiObject("tf", keel::move(s), mr);
}

Value evalLtiBinaryOp(char op, const Value &a, const Value &b, keel::Allocator *mr)
{
    LtiPoly pa, pb;
    if (!toLtiPoly(a, pa))
        throw numkit::Error(keel::format("Undefined operator '{}' for LTI model", op).c_str());
    const bool aIsLti = (isSisoTf(a) || isSisoSs(a));
    const bool bIsLti = (isSisoTf(b) || isSisoSs(b));
    if (op == '^') {
        if (!aIsLti || !b.isScalar() || !b.isNumeric() || b.isComplex())
            throw numkit::Error("Undefined operator '^' for LTI model");
        const keel::f64 expVal = b.toScalar();
        const keel::i32 k = keel::as<keel::i32>(expVal);
        if (keel::as<keel::f64>(k) != expVal)
            throw numkit::Error("LTI power exponent must be an integer");
        keel::Array<keel::f64> num = {1.0}, den = {1.0};
        const keel::i32 ak = keel::abs(k);
        const auto &baseN = (k >= 0) ? pa.num : pa.den;
        const auto &baseD = (k >= 0) ? pa.den : pa.num;
        for (keel::i32 i = 0; i < ak; ++i) {
            num = polyMul(num, baseN);
            den = polyMul(den, baseD);
        }
        return buildTfObject(num, den, pa.Ts, mr);
    }
    if (!toLtiPoly(b, pb))
        throw numkit::Error(keel::format("Undefined operator '{}' for LTI model", op).c_str());
    const keel::f64 Ts = aIsLti ? pa.Ts : (bIsLti ? pb.Ts : 0.0);
    if (op == '+' || op == '-') {
        const keel::f64 sgn = (op == '+') ? 1.0 : -1.0;
        if (pa.den == pb.den)
            return buildTfObject(polyAdd(pa.num, pb.num, sgn), pa.den, Ts, mr);
        return buildTfObject(
            polyAdd(polyMul(pa.num, pb.den), polyMul(pb.num, pa.den), sgn),
            polyMul(pa.den, pb.den), Ts, mr);
    }
    if (op == '*')
        return buildTfObject(polyMul(pa.num, pb.num), polyMul(pa.den, pb.den), Ts, mr);
    if (op == '/')
        return buildTfObject(polyMul(pa.num, pb.den), polyMul(pa.den, pb.num), Ts, mr);
    throw numkit::Error(keel::format("Undefined operator '{}' for LTI model", op).c_str());
}

// ── MIMO tf / ss indexing helpers ───────────────────────────────────────────
Value wrapIndexedTf(const Value &srcData, Value subNum, Value subDen, keel::Allocator *mr)
{
    Value out = Value::structure(mr);
    out.field("kind") = Value::fromString("tf", mr);
    if (srcData.hasField("Ts"))
        out.field("Ts") = srcData.field("Ts");
    if (srcData.hasField("variable"))
        out.field("variable") = srcData.field("variable");
    if (subNum.isCell() && subNum.numel() == 1) {
        out.field("num") = subNum.cellAt(0);
        out.field("den") = subDen.cellAt(0);
    } else {
        out.field("num") = keel::move(subNum);
        out.field("den") = keel::move(subDen);
    }
    return wrapLtiObject("tf", keel::move(out), mr);
}

Value indexSs2D(const Value &srcData,
                const keel::usize *rowIds, keel::usize rowCount,
                const keel::usize *colIds, keel::usize colCount,
                keel::Allocator *mr)
{
    const Value &A = srcData.field("A");
    const Value &B = srcData.field("B");
    const Value &C = srcData.field("C");
    const Value &D = srcData.field("D");
    const keel::usize nx = A.dims().rows();
    keel::Array<keel::usize> states(nx);
    for (keel::usize i = 0; i < nx; ++i) states[i] = i;
    Value subB = (nx > 0) ? B.indexGet2D(states.data(), nx, colIds, colCount, mr)
                          : Value::matrix(0, colCount, ValueType::DOUBLE, mr);
    Value subC = (nx > 0) ? C.indexGet2D(rowIds, rowCount, states.data(), nx, mr)
                          : Value::matrix(rowCount, 0, ValueType::DOUBLE, mr);
    Value subD = D.indexGet2D(rowIds, rowCount, colIds, colCount, mr);
    Value out = Value::structure(mr);
    out.field("kind") = Value::fromString("ss", mr);
    out.field("A") = A;
    out.field("B") = keel::move(subB);
    out.field("C") = keel::move(subC);
    out.field("D") = keel::move(subD);
    if (srcData.hasField("Ts"))
        out.field("Ts") = srcData.field("Ts");
    return wrapLtiObject("ss", keel::move(out), mr);
}

BuiltinClass makeCommonLtiClass(const char *clsName,
                                void (*ctorFn)(keel::ArrayRef<Value>, keel::usize,
                                               keel::MutArrayRef<Value>, CallContext &))
{
    BuiltinClass cls;
    cls.name = keel::String::from(clsName);
    cls.isHandle = false;

    cls.construct = [ctorFn](keel::ArrayRef<Value> args, CallContext &ctx) -> Value {
        Value out[1];
        ctorFn(args, 1, keel::MutArrayRef<Value>(out, 1), ctx);
        return keel::move(out[0]);
    };

    cls.propGet = [clsName](const Value &self, keel::StringRef name, Value &out, CallContext &) -> bool {
        const Value *d = ltiDataPtr(self);
        if (!d)
            return false;
        if (d->hasField(name)) {
            out = d->field(name);
            return true;
        }
        // State-space lowercase matrix aliases: sys.a/b/c/d -> sys.A/B/C/D.
        if (keel::StringRef(clsName) == "ss" && name.size() == 1) {
            const char up = keel::to_ascii_upper(name[0]);
            char upBuf[2] = {up, '\0'};
            keel::StringRef upField(upBuf, 1);
            if ((up == 'A' || up == 'B' || up == 'C' || up == 'D') && d->hasField(upField)) {
                out = d->field(upField);
                return true;
            }
        }
        return false;
    };

    cls.propSet = [](Value &self, keel::StringRef name, const Value &val, CallContext &) -> bool {
        Value *d = ltiDataMutPtr(self);
        if (!d)
            return false;
        d->field(name) = val;
        return true;
    };

    cls.dispText = [clsName](const Value &self) -> keel::String {
        keel::usize ny = 1, nu = 1;
        ltiIoDims(self, ny, nu);
        const Value *d = ltiDataPtr(self);
        const keel::f64 Ts = (d && d->hasField("Ts")) ? d->field("Ts").toScalar() : 0.0;
        if (Ts == 0.0)
            return keel::format("  {}x{} continuous-time {} model", ny, nu, clsName);
        return keel::format("  {}x{} discrete-time {} model (Ts = {})", ny, nu, clsName, Ts);
    };

    cls.methods.put(keel::String::from("__end"),
                    [](Value &self, keel::ArrayRef<Value> args, keel::usize,
                       keel::MutArrayRef<Value> outs, CallContext &ctx) {
        keel::Allocator *mr = ctx.engine->allocator();
        keel::usize ny = 1, nu = 1;
        ltiIoDims(self, ny, nu);
        const keel::i32 dim = args.size() >= 1 ? keel::as<keel::i32>(args[0].toScalar()) : 0;
        const keel::i32 ndims = args.size() >= 2 ? keel::as<keel::i32>(args[1].toScalar()) : 1;
        keel::usize sz = 1;
        if (ndims <= 1)
            sz = ny * nu;
        else if (dim == 0)
            sz = ny;
        else if (dim == 1)
            sz = nu;
        outs[0] = Value::scalar(keel::as<keel::f64>(sz), mr);
    });

    cls.methods.put(keel::String::from("size"),
                    [](Value &self, keel::ArrayRef<Value> rest, keel::usize nargout,
                       keel::MutArrayRef<Value> outs, CallContext &ctx) {
        keel::Allocator *mr = ctx.engine->allocator();
        keel::usize ny = 1, nu = 1;
        ltiIoDims(self, ny, nu);
        if (!rest.empty()) {
            const keel::i32 dim = argcheck::dimArg(rest[0], "size");
            const keel::usize d = (dim == 1) ? ny : (dim == 2 ? nu : 1);
            outs[0] = Value::scalar(keel::as<keel::f64>(d), mr);
            return;
        }
        if (nargout > 1) {
            for (keel::usize i = 0; i < nargout && i < outs.size(); ++i) {
                const keel::usize d = (i == 0) ? ny : (i == 1 ? nu : 1);
                outs[i] = Value::scalar(keel::as<keel::f64>(d), mr);
            }
            return;
        }
        Value out = Value::matrix(1, 2, ValueType::DOUBLE, mr);
        out.doubleDataMut()[0] = keel::as<keel::f64>(ny);
        out.doubleDataMut()[1] = keel::as<keel::f64>(nu);
        outs[0] = keel::move(out);
    });

    auto prependSelf = [](const Value &self, keel::ArrayRef<Value> rest) {
        keel::Array<Value> all;
        all.reserve(1 + rest.size());
        all.push_back(self);
        for (keel::usize i = 0; i < rest.size(); ++i)
            all.push_back(rest[i]);
        return all;
    };

    cls.methods.put(keel::String::from("impulse"),
                    [prependSelf](Value &self, keel::ArrayRef<Value> rest, keel::usize nargout,
                                  keel::MutArrayRef<Value> outs, CallContext &ctx) {
        auto all = prependSelf(self, rest);
        impulse_reg(all.as_ref(), nargout, outs, ctx);
    });

    cls.methods.put(keel::String::from("isstable"),
                    [prependSelf](Value &self, keel::ArrayRef<Value> rest, keel::usize nargout,
                                  keel::MutArrayRef<Value> outs, CallContext &ctx) {
        auto all = prependSelf(self, rest);
        isstable_reg(all.as_ref(), nargout, outs, ctx);
    });

    auto makeBinOp = [](char op) {
        return [op](Value &, keel::ArrayRef<Value> args, keel::usize,
                    keel::MutArrayRef<Value> outs, CallContext &ctx) {
            outs[0] = evalLtiBinaryOp(op, args[0], args[1], ctx.engine->allocator());
        };
    };
    cls.ops.put(keel::String::from("plus"),     makeBinOp('+'));
    cls.ops.put(keel::String::from("minus"),    makeBinOp('-'));
    cls.ops.put(keel::String::from("mtimes"),   makeBinOp('*'));
    cls.ops.put(keel::String::from("times"),    makeBinOp('*'));
    cls.ops.put(keel::String::from("mrdivide"), makeBinOp('/'));
    cls.ops.put(keel::String::from("rdivide"),  makeBinOp('/'));
    cls.ops.put(keel::String::from("mpower"),   makeBinOp('^'));
    cls.ops.put(keel::String::from("power"),    makeBinOp('^'));

    cls.ops.put(keel::String::from("uminus"),
                [](Value &self, keel::ArrayRef<Value>, keel::usize,
                   keel::MutArrayRef<Value> outs, CallContext &ctx) {
        keel::Allocator *mr = ctx.engine->allocator();
        LtiPoly p;
        if (!toLtiPoly(self, p))
            throw numkit::Error("Undefined unary '-' for LTI model");
        for (keel::f64 &x : p.num) x = -x;
        outs[0] = buildTfObject(p.num, p.den, p.Ts, mr);
    });

    cls.ops.put(keel::String::from("uplus"),
                [](Value &self, keel::ArrayRef<Value>, keel::usize,
                   keel::MutArrayRef<Value> outs, CallContext &) {
        outs[0] = self;
    });

    return cls;
}

} // namespace

BuiltinClass makeTfBuiltinClass()
{
    BuiltinClass cls = makeCommonLtiClass("tf", &tf_reg);
    cls.subsref = [](Value &self, keel::ArrayRef<Value> idxArgs, keel::usize,
                     keel::MutArrayRef<Value> outs, CallContext &ctx) {
        keel::Allocator *mr = ctx.engine->allocator();
        const Value *d = ltiDataPtr(self);
        if (!d || !d->hasField("num") || !d->hasField("den"))
            throw numkit::Error("tf: invalid model state");
        const Value &num = d->field("num");
        const Value &den = d->field("den");
        if (!num.isCell() || !den.isCell()) {
            // SISO tf: allow H(1) or H(1, 1)
            if (idxArgs.size() == 1) {
                auto ids = Value::resolveIndices(idxArgs[0], 1);
                if (ids.size() == 1 && ids[0] == 0) {
                    outs[0] = self;
                    return;
                }
            } else if (idxArgs.size() == 2) {
                auto rids = Value::resolveIndices(idxArgs[0], 1);
                auto cids = Value::resolveIndices(idxArgs[1], 1);
                if (rids.size() == 1 && cids.size() == 1 && rids[0] == 0 && cids[0] == 0) {
                    outs[0] = self;
                    return;
                }
            }
            throw numkit::Error("Index exceeds array dimensions");
        }
        if (idxArgs.size() == 1) {
            auto ids = Value::resolveIndices(idxArgs[0], num.numel());
            Value subNum = num.indexGet(ids.data(), ids.size(), mr);
            Value subDen = den.indexGet(ids.data(), ids.size(), mr);
            outs[0] = wrapIndexedTf(*d, keel::move(subNum), keel::move(subDen), mr);
            return;
        }
        if (idxArgs.size() == 2) {
            auto rids = Value::resolveIndices(idxArgs[0], num.dims().rows());
            auto cids = Value::resolveIndices(idxArgs[1], num.dims().cols());
            Value subNum = num.indexGet2D(rids.data(), rids.size(), cids.data(), cids.size(), mr);
            Value subDen = den.indexGet2D(rids.data(), rids.size(), cids.data(), cids.size(), mr);
            outs[0] = wrapIndexedTf(*d, keel::move(subNum), keel::move(subDen), mr);
            return;
        }
        throw numkit::Error("tf: expected 1 or 2 subscripts");
    };
    return cls;
}

BuiltinClass makeSsBuiltinClass()
{
    BuiltinClass cls = makeCommonLtiClass("ss", &ss_reg);
    cls.subsref = [](Value &self, keel::ArrayRef<Value> idxArgs, keel::usize,
                     keel::MutArrayRef<Value> outs, CallContext &ctx) {
        keel::Allocator *mr = ctx.engine->allocator();
        const Value *d = ltiDataPtr(self);
        if (!d || !d->hasField("A") || !d->hasField("B") || !d->hasField("C") || !d->hasField("D"))
            throw numkit::Error("ss: invalid model state");
        const Value &D = d->field("D");
        const keel::usize ny = D.dims().rows();
        const keel::usize nu = D.dims().cols();
        if (idxArgs.size() == 1) {
            if (ny == 1) {
                auto cols = Value::resolveIndices(idxArgs[0], nu);
                keel::usize r0 = 0;
                outs[0] = indexSs2D(*d, &r0, 1, cols.data(), cols.size(), mr);
                return;
            }
            if (nu == 1) {
                auto rows = Value::resolveIndices(idxArgs[0], ny);
                keel::usize c0 = 0;
                outs[0] = indexSs2D(*d, rows.data(), rows.size(), &c0, 1, mr);
                return;
            }
            auto lin = Value::resolveIndices(idxArgs[0], ny * nu);
            if (lin.size() == 1) {
                keel::usize r = lin[0] % ny;
                keel::usize c = lin[0] / ny;
                outs[0] = indexSs2D(*d, &r, 1, &c, 1, mr);
                return;
            }
            throw numkit::Error("ss: multi-element linear indexing on MIMO ss requires 2D subscripts");
        }
        if (idxArgs.size() == 2) {
            auto rids = Value::resolveIndices(idxArgs[0], ny);
            auto cids = Value::resolveIndices(idxArgs[1], nu);
            outs[0] = indexSs2D(*d, rids.data(), rids.size(), cids.data(), cids.size(), mr);
            return;
        }
        throw numkit::Error("ss: expected 1 or 2 subscripts");
    };
    return cls;
}

BuiltinClass makeZpkBuiltinClass()
{
    return makeCommonLtiClass("zpk", &zpk_reg);
}

BuiltinClass makeFrdBuiltinClass()
{
    return makeCommonLtiClass("frd", &frd_reg);
}

} // namespace numkit::control::detail
