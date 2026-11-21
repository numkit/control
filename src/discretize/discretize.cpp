// toolboxes/control/src/discretize/discretize.cpp
//
// c2d / d2c sample-time conversions delegating numerical kernels to keel::control.

#include <keel/array.h>
#include <keel/control.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/string.h>
#include <numkit/control/conversion/conversion.hpp>
#include <numkit/control/discretize/discretize.hpp>
#include <numkit/control/lti/lti.hpp>

#include <numkit/builtin/polyfun.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <algorithm>
#include <utility>

namespace numkit::control {

namespace {

bool hasKind(const Value &sys, const char *want) {
    if (!sys.isStruct() || !sys.hasField("kind")) return false;
    return sys.field("kind").toString() == want;
}

keel::f64 sampleTime(const Value &sys) {
    if (sys.isStruct() && sys.hasField("Ts"))
        return sys.field("Ts").toScalar();
    return 0.0;
}

keel::Array<keel::f64> coeffsReal(const Value &v, keel::Allocator *mr) {
    keel::Array<keel::f64> out(v.numel(), 0.0, mr);
    for (keel::usize i = 0; i < v.numel(); ++i) out[i] = v.elemAsDouble(i);
    return out;
}

Value matFromVec(keel::usize r, keel::usize c, keel::ArrayRef<keel::f64> v, keel::Allocator *mr) {
    Value m = Value::matrix(r, c, ValueType::DOUBLE, mr);
    if (!v.empty()) std::copy(v.begin(), v.end(), m.doubleDataMut());
    return m;
}

struct SS {
    keel::usize n = 0;
    keel::Array<keel::f64> A;
    keel::Array<keel::f64> B;
    keel::Array<keel::f64> C;
    keel::f64 D = 0.0;
    keel::f64 Ts = 0.0;
};

SS toSSiso(const Value &sys, keel::Allocator *mr) {
    Value Av, Bv, Cv, Dv;
    const keel::f64 Ts = sampleTime(sys);
    if (hasKind(sys, "ss")) {
        Av = sys.field("A"); Bv = sys.field("B");
        Cv = sys.field("C"); Dv = sys.field("D");
    } else if (hasKind(sys, "tf")) {
        auto s = tf2ss(sys.field("num"), sys.field("den"), mr);
        Av = std::move(s.A); Bv = std::move(s.B);
        Cv = std::move(s.C); Dv = std::move(s.D);
    } else if (hasKind(sys, "zpk")) {
        auto [num, den] = zp2tf(sys.field("z"), sys.field("p"), sys.field("k"), mr);
        auto s = tf2ss(num, den, mr);
        Av = std::move(s.A); Bv = std::move(s.B);
        Cv = std::move(s.C); Dv = std::move(s.D);
    } else {
        throw Error("c2d/d2c: expected an LTI struct (tf/zpk/ss)",
                    0, 0, "discretize", "", "numkit:control:kind");
    }
    SS s;
    s.n = Av.dims().rows();
    s.Ts = Ts;
    s.A = keel::Array<keel::f64>(s.n * s.n, 0.0, mr);
    for (keel::usize i = 0; i < s.n * s.n; ++i) s.A[i] = Av.elemAsDouble(i);
    s.B = keel::Array<keel::f64>(s.n, 0.0, mr);
    for (keel::usize i = 0; i < s.n; ++i) s.B[i] = Bv.elemAsDouble(i);
    s.C = keel::Array<keel::f64>(s.n, 0.0, mr);
    for (keel::usize i = 0; i < s.n; ++i) s.C[i] = Cv.elemAsDouble(i);
    s.D = (Dv.numel() == 0) ? 0.0 : Dv.toScalar();
    return s;
}

Value packResult(keel::ArrayRef<keel::f64> Ad, keel::ArrayRef<keel::f64> Bd,
                 keel::ArrayRef<keel::f64> Cd, keel::f64 Dd, keel::usize n, keel::f64 Ts,
                 keel::StringRef origKind, keel::Allocator *mr) {
    Value Av = matFromVec(n, n, Ad, mr);
    Value Bv = matFromVec(n, 1, Bd, mr);
    Value Cv = matFromVec(1, n, Cd, mr);
    Value Dv = Value::scalar(Dd, mr);

    if (origKind == "ss") {
        return ss(Av, Bv, Cv, Dv, Ts, mr);
    }
    auto [numV, denV] = ss2tf(Av, Bv, Cv, Dv, /*iu=*/1, mr);
    if (origKind == "tf") {
        return tf(numV, denV, Ts, mr);
    }
    if (origKind == "zpk") {
        Value zV = numkit::builtin::roots(numV, mr);
        Value pV = numkit::builtin::roots(denV, mr);
        const auto numVec = coeffsReal(numV, mr);
        const auto denVec = coeffsReal(denV, mr);
        keel::usize in = 0; while (in + 1 < numVec.size() && numVec[in] == 0.0) ++in;
        keel::usize id = 0; while (id + 1 < denVec.size() && denVec[id] == 0.0) ++id;
        const keel::f64 k = (in < numVec.size() && id < denVec.size())
                         ? numVec[in] / denVec[id] : 0.0;
        return zpk(zV, pV, k, Ts, mr);
    }
    return ss(Av, Bv, Cv, Dv, Ts, mr);
}

void c2d_argcheck(const Value &sys, keel::f64 Ts, keel::StringRef method) {
    if (Ts <= 0.0)
        throw Error("c2d: Ts must be positive",
                    0, 0, "c2d", "", "numkit:c2d:Ts");
    if (!hasKind(sys, "ss") && !hasKind(sys, "tf") && !hasKind(sys, "zpk"))
        throw Error("c2d/d2c: expected an LTI struct (tf/zpk/ss)",
                    0, 0, "discretize", "", "numkit:control:kind");
    if (sampleTime(sys) > 0.0)
        throw Error("c2d: input system is already discrete (Ts > 0)",
                    0, 0, "c2d", "", "numkit:c2d:already_discrete");
    if (!method.empty() && method != "zoh" && method != "foh" &&
        method != "tustin" && method != "tutsin" && method != "bilinear")
        throw Error("c2d: method must be 'zoh', 'foh', or 'tustin'",
                    0, 0, "c2d", "", "numkit:c2d:method");
}

void d2c_argcheck(const Value &sys, keel::StringRef method) {
    if (!hasKind(sys, "ss") && !hasKind(sys, "tf") && !hasKind(sys, "zpk"))
        throw Error("c2d/d2c: expected an LTI struct (tf/zpk/ss)",
                    0, 0, "discretize", "", "numkit:control:kind");
    if (sampleTime(sys) <= 0.0)
        throw Error("d2c: input system is already continuous (Ts == 0)",
                    0, 0, "d2c", "", "numkit:d2c:already_continuous");
    const bool isZoh = method.empty() || method == "zoh";
    const bool isTustin = method == "tustin" || method == "bilinear";
    if (!isZoh && !isTustin)
        throw Error("d2c: method must be 'zoh' or 'tustin'",
                    0, 0, "d2c", "", "numkit:d2c:method");
}

} // anonymous

Value c2d(const Value &sys, keel::f64 Ts, keel::StringRef method, keel::Allocator *mr) {
    c2d_argcheck(sys, Ts, method);

    if (hasKind(sys, "ss") && (method.empty() || method == "zoh" || method == "foh")) {
        const Value &Av = sys.field("A");
        const Value &Bv = sys.field("B");
        const Value &Cv = sys.field("C");
        const Value &Dv = sys.field("D");
        const keel::usize n = Av.dims().rows();
        const keel::usize nu = Bv.dims().cols();
        const keel::usize ny = Cv.dims().rows();

        keel::Array<keel::f64> A(n * n, 0.0, mr);
        for (keel::usize i = 0; i < n * n; ++i) A[i] = Av.elemAsDouble(i);
        keel::Array<keel::f64> B(n * nu, 0.0, mr);
        for (keel::usize i = 0; i < n * nu; ++i) B[i] = Bv.elemAsDouble(i);
        keel::Array<keel::f64> C(ny * n, 0.0, mr);
        for (keel::usize i = 0; i < ny * n; ++i) C[i] = Cv.elemAsDouble(i);
        keel::Array<keel::f64> D(ny * nu, 0.0, mr);
        for (keel::usize j = 0; j < nu; ++j)
            for (keel::usize p = 0; p < ny; ++p)
                D[j * ny + p] =
                    (Dv.numel() == ny * nu) ? Dv.elemAsDouble(j * ny + p)
                    : ((Dv.numel() == 1) ? Dv.elemAsDouble(0) : 0.0);

        keel::Array<keel::f64> Ad(n * n, 0.0, mr);
        keel::Array<keel::f64> Bd(n * nu, 0.0, mr);
        keel::Array<keel::f64> Cd(ny * n, 0.0, mr);
        keel::Array<keel::f64> Dd(ny * nu, 0.0, mr);
        Value GV = Value::matrix(n, n + nu, ValueType::DOUBLE, mr);
        keel::MutArrayRef<keel::f64> gRef(GV.doubleDataMut(), n * (n + nu));

        if (method.empty() || method == "zoh") {
            (void)keel::control::c2d_zoh(A.as_ref(), B.as_ref(), n, nu, Ts,
                                         Ad.as_mut_ref(), Bd.as_mut_ref(), mr);
            Cd = C.clone();
            Dd = D.clone();
            for (keel::usize i = 0; i < gRef.size(); ++i) gRef[i] = 0.0;
            for (keel::usize i = 0; i < n; ++i) gRef[i * n + i] = 1.0;
        } else {
            (void)keel::control::c2d_foh(A.as_ref(), B.as_ref(), C.as_ref(), D.as_ref(),
                                         n, nu, ny, Ts, Ad.as_mut_ref(), Bd.as_mut_ref(),
                                         Cd.as_mut_ref(), Dd.as_mut_ref(), gRef, mr);
        }

        Value AdV = matFromVec(n, n, Ad.as_ref(), mr);
        Value BdV = matFromVec(n, nu, Bd.as_ref(), mr);
        Value CdV = (method.empty() || method == "zoh") ? Cv : matFromVec(ny, n, Cd.as_ref(), mr);
        Value DdV = matFromVec(ny, nu, Dd.as_ref(), mr);
        Value res = ss(AdV, BdV, CdV, DdV, Ts, mr);
        res.field("G") = std::move(GV);
        return res;
    }

    SS s = toSSiso(sys, mr);
    if (s.Ts > 0.0)
        throw Error("c2d: input system is already discrete (Ts > 0)",
                    0, 0, "c2d", "", "numkit:c2d:already_discrete");

    keel::Array<keel::f64> Ad(s.n * s.n, 0.0, mr);
    keel::Array<keel::f64> Bd(s.n, 0.0, mr);
    keel::Array<keel::f64> Cd(s.n, 0.0, mr);
    keel::f64 Dd = s.D;
    const keel::f64 D_in[1] = {s.D};
    keel::f64 D_out[1] = {s.D};

    if (method.empty() || method == "zoh") {
        (void)keel::control::c2d_zoh(s.A.as_ref(), s.B.as_ref(), s.n, 1, Ts,
                                     Ad.as_mut_ref(), Bd.as_mut_ref(), mr);
        Cd = s.C.clone();
    } else if (method == "tustin" || method == "tutsin" || method == "bilinear") {
        if (keel::control::c2d_tustin(s.A.as_ref(), s.B.as_ref(), s.C.as_ref(),
                                      keel::ArrayRef<keel::f64>(D_in, 1), s.n, 1, 1, Ts,
                                      Ad.as_mut_ref(), Bd.as_mut_ref(), Cd.as_mut_ref(),
                                      keel::MutArrayRef<keel::f64>(D_out, 1), mr) !=
            keel::control::ControlStatus::kOk) {
            throw Error("c2d (tustin): I − A·Ts/2 is singular",
                        0, 0, "c2d", "", "numkit:c2d:singular");
        }
        Dd = D_out[0];
    } else if (method == "foh") {
        (void)keel::control::c2d_foh(s.A.as_ref(), s.B.as_ref(), s.C.as_ref(),
                                     keel::ArrayRef<keel::f64>(D_in, 1), s.n, 1, 1, Ts,
                                     Ad.as_mut_ref(), Bd.as_mut_ref(), Cd.as_mut_ref(),
                                     keel::MutArrayRef<keel::f64>(D_out, 1), {}, mr);
        Dd = D_out[0];
    } else {
        throw Error("c2d: method must be 'zoh', 'foh', or 'tustin'",
                    0, 0, "c2d", "", "numkit:c2d:method");
    }

    const keel::String origKind = sys.isStruct() && sys.hasField("kind")
                                  ? sys.field("kind").toString()
                                  : keel::String::from("ss");
    return packResult(Ad.as_ref(), Bd.as_ref(), Cd.as_ref(), Dd, s.n, Ts, origKind.as_ref(), mr);
}

Value d2c(const Value &sys, keel::StringRef method, keel::Allocator *mr) {
    d2c_argcheck(sys, method);
    SS s = toSSiso(sys, mr);
    const bool isZoh = method.empty() || method == "zoh";
    const keel::f64 Ts = s.Ts;
    const keel::String origKind = sys.isStruct() && sys.hasField("kind")
                                  ? sys.field("kind").toString()
                                  : keel::String::from("ss");

    if (isZoh) {
        keel::control::StateSpace res;
        if (keel::control::d2c_zoh(s.A.as_ref(), s.B.as_ref(), s.C.as_ref(), s.D,
                                   s.n, Ts, res, mr) != keel::control::ControlStatus::kOk) {
            throw Error("d2c (zoh): matrix logarithm failed (singular or zero pole)",
                        0, 0, "d2c", "", "numkit:d2c:singular");
        }
        return packResult(res.a.as_ref(), res.b.as_ref(), res.c.as_ref(), res.d,
                          res.n, 0.0, origKind.as_ref(), mr);
    }

    keel::Array<keel::f64> Ac(s.n * s.n, 0.0, mr);
    keel::Array<keel::f64> Bc(s.n, 0.0, mr);
    keel::Array<keel::f64> Cc(s.n, 0.0, mr);
    keel::f64 Dc = s.D;
    if (keel::control::d2c_tustin(s.A.as_ref(), s.B.as_ref(), s.C.as_ref(), s.D,
                                  s.n, Ts, Ac.as_mut_ref(), Bc.as_mut_ref(),
                                  Cc.as_mut_ref(), Dc, mr) !=
        keel::control::ControlStatus::kOk) {
        throw Error("d2c (tustin): A_d + I is singular",
                    0, 0, "d2c", "", "numkit:d2c:singular");
    }
    return packResult(Ac.as_ref(), Bc.as_ref(), Cc.as_ref(), Dc, s.n, 0.0, origKind.as_ref(), mr);
}

} // namespace numkit::control
