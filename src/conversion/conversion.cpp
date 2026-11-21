// toolboxes/control/src/conversion/conversion.cpp
//
// Inter-form conversions tf ↔ zpk ↔ ss. Built on the existing
// numkit::builtin::roots / numkit::builtin::poly polynomial primitives, plus a local
// Faddeev–LeVerrier expansion for ss2tf.

#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/control/conversion/conversion.hpp>

#include <numkit/builtin/polyfun.hpp>

// Compute-only TU: Value substrate + Error, no engine. conversion.cpp has no
// CallContext builtins of its own (the tf2ss/ss2tf/etc. register wrappers live
// in their caller toolboxes); it is pure inter-form numeric conversion.
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace numkit::control {

namespace {

// Read a row/column polynomial coefficient vector into a vector of
// doubles. Complex coefficients are reduced to their real parts (the
// caller verifies the inputs are real for the SS path).
keel::Array<keel::f64> coeffsReal(const Value &v) {
    keel::Array<keel::f64> out(v.numel());
    if (v.type() == ValueType::COMPLEX) {
        const keel::c64 *c = v.complexData();
        for (keel::usize i = 0; i < v.numel(); ++i) out[i] = c[i].real();
    } else {
        for (keel::usize i = 0; i < v.numel(); ++i) out[i] = v.elemAsDouble(i);
    }
    return out;
}

// Strip leading zeros (preserve at least 1 entry).
keel::Array<keel::f64> stripLeading(keel::ArrayRef<keel::f64> v) {
    keel::usize i = 0;
    while (i + 1 < v.size() && v[i] == 0.0) ++i;
    return keel::Array<keel::f64>::from(v.slice(i));
}

Value rowOfDoubles(keel::ArrayRef<keel::f64> v, keel::Allocator *mr) {
    Value r = Value::matrix(1, v.size(), ValueType::DOUBLE, mr);
    if (!v.empty()) std::copy(v.begin(), v.end(), r.doubleDataMut());
    return r;
}

Value matrixFromVec(keel::usize rows, keel::usize cols, keel::ArrayRef<keel::f64> v, keel::Allocator *mr) {
    Value m = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (!v.empty()) std::copy(v.begin(), v.end(), m.doubleDataMut());
    return m;
}

void tf2zp_argcheck(const Value &num, const Value &den,
                    keel::Array<keel::f64> &numV, keel::Array<keel::f64> &denV)
{
    numV = stripLeading(coeffsReal(num));
    denV = stripLeading(coeffsReal(den));
    if (denV.empty() || denV[0] == 0.0)
        throw Error("tf2zp: denominator must have a nonzero leading coefficient",
                    0, 0, "tf2zp", "", "numkit:tf2zp:den");
}

void zp2tf_argcheck(const Value &z, const Value &p, const Value &k)
{
    (void)z;
    (void)p;
    (void)k;
}

void tf2ss_argcheck(const Value &num, const Value &den,
                    keel::Array<keel::f64> &numV, keel::Array<keel::f64> &denV)
{
    denV = stripLeading(coeffsReal(den));
    if (denV.empty() || denV[0] == 0.0)
        throw Error("tf2ss: invalid denominator",
                    0, 0, "tf2ss", "", "numkit:tf2ss:den");
    numV = coeffsReal(num);
    if (numV.size() > denV.size())
        throw Error("tf2ss: numerator longer than denominator (improper system)",
                    0, 0, "tf2ss", "", "numkit:tf2ss:improper");
}

void ss2tf_argcheck(const Value &A, const Value &B, const Value &C, keel::i32 iu)
{
    const keel::usize n = A.dims().rows();
    if (A.dims().cols() != n)
        throw Error("ss2tf: A must be square",
                    0, 0, "ss2tf", "", "numkit:ss2tf:A");
    const keel::usize nin = (B.dims().cols() == 0) ? 1 : B.dims().cols();
    if (iu < 1 || keel::as<keel::usize>(iu) > nin)
        throw Error("ss2tf: iu out of range",
                    0, 0, "ss2tf", "", "numkit:ss2tf:iu");
    if (C.dims().rows() != 1)
        throw Error("ss2tf: only single-output systems supported "
                    "(got C with rows != 1)",
                    0, 0, "ss2tf", "", "numkit:ss2tf:miso");
}

} // anonymous

Tf2ZpResult tf2zp(const Value &num, const Value &den,
                  keel::Allocator *mr)
{
    keel::Array<keel::f64> numV, denV;
    tf2zp_argcheck(num, den, numV, denV);

    // Gain is num(1)/den(1) (after stripping leading zeros).
    const keel::f64 k = numV.empty() ? 0.0 : numV[0] / denV[0];

    // Roots of num and den. The existing numkit::builtin::roots already
    // handles both real-and-complex output and trailing-zero roots.
    Value zRoots = numkit::builtin::roots(num, mr);
    Value pRoots = numkit::builtin::roots(den, mr);

    return {std::move(zRoots), std::move(pRoots),
            Value::scalar(k, mr)};
}

std::pair<Value, Value>
zp2tf(const Value &z, const Value &p, const Value &k,
      keel::Allocator *mr)
{
    zp2tf_argcheck(z, p, k);
    // num = k * poly(z), den = poly(p). numkit::builtin::poly takes a column
    // vector of roots and returns the row of polynomial coefficients
    // (leading 1, descending powers).
    Value den = numkit::builtin::poly(p, mr);
    Value num = numkit::builtin::poly(z, mr);

    // poly([]) of no zeros is the constant polynomial 1 (MATLAB: poly([])==1).
    // numkit::builtin::poly returns an empty row for empty input, which would
    // collapse num to [] and silently drop the gain (bugs/control/zpk-empty-zeros).
    // Normalize the no-zero case to [1] so num scales to [k].
    if (num.numel() == 0) {
        num = Value::matrix(1, 1, ValueType::DOUBLE, mr);
        num.doubleDataMut()[0] = 1.0;
    }

    // Multiply num by the gain (real scalar in our zpk; we handle the
    // complex-product case by promoting num if k or num is complex).
    const keel::f64 kVal = k.toScalar();
    if (num.type() == ValueType::COMPLEX) {
        const keel::usize N = num.numel();
        Value scaled = Value::matrix(1, N, ValueType::COMPLEX, mr);
        const keel::c64 *src = num.complexData();
        keel::c64 *dst = scaled.complexDataMut();
        for (keel::usize i = 0; i < N; ++i) dst[i] = src[i] * kVal;
        num = scaled;
    } else {
        const keel::usize N = num.numel();
        Value scaled = Value::matrix(1, N, ValueType::DOUBLE, mr);
        keel::f64 *dst = scaled.doubleDataMut();
        for (keel::usize i = 0; i < N; ++i) dst[i] = num.elemAsDouble(i) * kVal;
        num = scaled;
    }

    return {std::move(num), std::move(den)};
}

StateSpace tf2ss(const Value &num, const Value &den,
                 keel::Allocator *mr)
{
    keel::Array<keel::f64> numV, denV;
    tf2ss_argcheck(num, den, numV, denV);
    const keel::f64 a0 = denV[0];
    // Normalise so leading coefficient of den is 1.
    keel::Array<keel::f64> a(denV.size());
    for (keel::usize i = 0; i < denV.size(); ++i) a[i] = denV[i] / a0;

    // Left-pad numerator with zeros so it has the same length as `a`.
    keel::Array<keel::f64> b(a.size(), 0.0);
    const keel::usize pad = a.size() - numV.size();
    for (keel::usize i = 0; i < numV.size(); ++i) b[pad + i] = numV[i] / a0;

    const keel::usize n = a.size() - 1;  // state-space order
    if (n == 0) {
        // Pure feedthrough: A=0×0, B=0×1, C=1×0, D=b[0].
        return {Value::matrix(0, 0, ValueType::DOUBLE, mr),
                Value::matrix(0, 1, ValueType::DOUBLE, mr),
                Value::matrix(1, 0, ValueType::DOUBLE, mr),
                Value::scalar(b.empty() ? 0.0 : b[0], mr)};
    }

    // Controllable canonical form (column-major storage).
    // A = [0 1 0 ... 0;
    //      0 0 1 ... 0;
    //      ...
    //      0 0 0 ... 1;
    //      -a_n -a_{n-1} ... -a_1]
    keel::Array<keel::f64> Avec(n * n, 0.0);
    for (keel::usize i = 0; i + 1 < n; ++i) {
        // column j = i+1, row i.
        Avec[(i + 1) * n + i] = 1.0;
    }
    // Last row: -a_n .. -a_1 (a[1..n] are the non-leading coeffs;
    // we want them reversed because A(n-1, j) = -a[n-j]).
    for (keel::usize j = 0; j < n; ++j) {
        Avec[j * n + (n - 1)] = -a[n - j];
    }

    // B = [0; ...; 0; 1] (column).
    keel::Array<keel::f64> Bvec(n, 0.0);
    Bvec[n - 1] = 1.0;

    // C row of length n. With proper rational form b[0] is the D term.
    // For G(s) = b(s)/a(s) with deg(b) <= deg(a), after dividing,
    // G(s) = b[0] + (b[1] - b[0]·a[1]) s^{n-1}/a(s) + ... + (b[n] - b[0]·a[n])/a(s).
    // C[k] = b[n - k] - b[0] · a[n - k]   for k = 0..n-1
    keel::Array<keel::f64> Cvec(n, 0.0);
    for (keel::usize k = 0; k < n; ++k) {
        Cvec[k] = b[n - k] - b[0] * a[n - k];
    }

    return {matrixFromVec(n, n, Avec, mr),
            matrixFromVec(n, 1, Bvec, mr),
            matrixFromVec(1, n, Cvec, mr),
            Value::scalar(b[0], mr)};
}

std::pair<Value, Value>
ss2tf(const Value &A, const Value &B,
      const Value &C, const Value &D, keel::i32 iu,
      keel::Allocator *mr)
{
    ss2tf_argcheck(A, B, C, iu);
    const keel::usize n = A.dims().rows();
    const keel::usize inIdx = keel::as<keel::usize>(iu - 1);

    // Read matrices into flat column-major double buffers.
    auto readMat = [&](const Value &v, keel::usize r, keel::usize c) {
        keel::Array<keel::f64> M(r * c, 0.0);
        for (keel::usize i = 0; i < r * c; ++i) M[i] = v.elemAsDouble(i);
        return M;
    };
    auto Av = readMat(A, n, n);
    // B column iu.
    keel::Array<keel::f64> Bv(n, 0.0);
    for (keel::usize i = 0; i < n; ++i) Bv[i] = B.elemAsDouble(inIdx * n + i);
    // C row 0 (single-output).
    keel::Array<keel::f64> Cv(n, 0.0);
    for (keel::usize i = 0; i < n; ++i) Cv[i] = C.elemAsDouble(i * 1 + 0);
    // D scalar at (0, iu-1).
    keel::f64 Dscalar = 0.0;
    if (D.numel() > 0) {
        // D is ny × nin; we read element (0, inIdx).
        const keel::usize Drows = (D.dims().rows() == 0) ? 1 : D.dims().rows();
        Dscalar = D.elemAsDouble(inIdx * Drows + 0);
    }

    // Faddeev–LeVerrier.  M_0 = I,  c_k = -(1/k) trace(A·M_{k-1}),
    // M_k = A·M_{k-1} + c_k I.  char poly = s^n + c_1 s^{n-1} + ... + c_n.
    keel::Array<keel::f64> M(n * n, 0.0); // current M_{k-1}
    for (keel::usize i = 0; i < n; ++i) M[i * n + i] = 1.0; // I
    keel::Array<keel::f64> coeff(n + 1, 0.0);
    coeff[0] = 1.0;

    // num accumulates from C·M_k·B (length n) + D·char(s).
    keel::Array<keel::f64> numAcc(n, 0.0); // s^{n-1} ... s^0

    auto matMul = [&](keel::ArrayRef<keel::f64> X,
                      keel::ArrayRef<keel::f64> Y) {
        keel::Array<keel::f64> Z(n * n, 0.0);
        for (keel::usize j = 0; j < n; ++j)
            for (keel::usize i = 0; i < n; ++i) {
                keel::f64 s = 0.0;
                for (keel::usize l = 0; l < n; ++l)
                    s += X[l * n + i] * Y[j * n + l];
                Z[j * n + i] = s;
            }
        return Z;
    };
    auto matVec = [&](keel::ArrayRef<keel::f64> X,
                      keel::ArrayRef<keel::f64> y) {
        keel::Array<keel::f64> z(n, 0.0);
        for (keel::usize i = 0; i < n; ++i) {
            keel::f64 s = 0.0;
            for (keel::usize j = 0; j < n; ++j) s += X[j * n + i] * y[j];
            z[i] = s;
        }
        return z;
    };

    // numAcc[k] gets C·M_k·B for k = 0..n-1.
    {
        // k = 0: M_0 = I → C·I·B = C·B.
        keel::f64 cb = 0.0;
        for (keel::usize i = 0; i < n; ++i) cb += Cv[i] * Bv[i];
        numAcc[0] = cb;
    }

    for (keel::usize k = 1; k <= n; ++k) {
        // A·M_{k-1}
        auto AM = matMul(Av, M);
        // trace(A·M_{k-1})
        keel::f64 tr = 0.0;
        for (keel::usize i = 0; i < n; ++i) tr += AM[i * n + i];
        const keel::f64 ck = -tr / keel::as<keel::f64>(k);
        coeff[k] = ck;
        // M_k = A·M_{k-1} + ck · I
        keel::Array<keel::f64> Mk(n * n, 0.0);
        for (keel::usize i = 0; i < n * n; ++i) Mk[i] = AM[i];
        for (keel::usize i = 0; i < n; ++i) Mk[i * n + i] += ck;
        M = keel::move(Mk);

        if (k < n) {
            auto MB = matVec(M, Bv);
            keel::f64 cmb = 0.0;
            for (keel::usize i = 0; i < n; ++i) cmb += Cv[i] * MB[i];
            numAcc[k] = cmb;
        }
    }

    // Final num = numAcc + D·char_poly.
    // num is degree n (length n+1) when D != 0; otherwise degree n-1.
    keel::Array<keel::f64> numOutVec;
    if (Dscalar != 0.0) {
        numOutVec.assign(n + 1, 0.0);
        for (keel::usize i = 0; i <= n; ++i) numOutVec[i] = Dscalar * coeff[i];
        // Add C·M_k·B contributions (degree n-1 .. 0) into positions 1..n.
        for (keel::usize k = 0; k < n; ++k) numOutVec[k + 1] += numAcc[k];
    } else {
        numOutVec = keel::move(numAcc); // length n
    }
    // den is the char poly: length n+1, leading 1.
    keel::Array<keel::f64> denOutVec = keel::move(coeff);

    return {rowOfDoubles(numOutVec, mr),
            rowOfDoubles(denOutVec, mr)};
}

// --- minreal: pole/zero cancellation -----------------------------------

namespace {

bool isKind(const Value &sys, const char *want) {
    return sys.isStruct() && sys.hasField("kind") &&
           sys.field("kind").toString() == want;
}

keel::f64 sampleTimeOf(const Value &sys) {
    return (sys.isStruct() && sys.hasField("Ts")) ? sys.field("Ts").toScalar() : 0.0;
}

keel::Array<keel::c64>
rootsComplex(const Value &polyV, keel::Allocator *mr) {
    Value r = numkit::builtin::roots(polyV, mr);
    const keel::usize n = r.numel();
    keel::Array<keel::c64> out(n);
    if (r.type() == ValueType::COMPLEX) {
        const keel::c64 *s = r.complexData();
        for (keel::usize i = 0; i < n; ++i) out[i] = s[i];
    } else {
        for (keel::usize i = 0; i < n; ++i) out[i] = {r.elemAsDouble(i), 0.0};
    }
    return out;
}

// Monic polynomial whose roots are `rts`, expanded as ∏(s − r_i) in complex
// arithmetic; the real part is returned (conjugate pairs cancel the imaginary
// part exactly, up to round-off noise).
keel::Array<keel::f64> polyFromRoots(keel::ArrayRef<keel::c64> rts) {
    keel::Array<keel::c64> c{keel::c64{1.0, 0.0}};
    for (const auto &r : rts) {
        keel::Array<keel::c64> nc(c.size() + 1, keel::c64{0.0, 0.0});
        for (keel::usize i = 0; i < c.size(); ++i) { nc[i] += c[i]; nc[i + 1] -= c[i] * r; }
        c = keel::move(nc);
    }
    keel::Array<keel::f64> out(c.size());
    for (keel::usize i = 0; i < c.size(); ++i) out[i] = c[i].real();
    return out;
}

Value taggedStruct(const char *kind, keel::f64 Ts, keel::Allocator *mr) {
    Value s = Value::structure(mr);
    s.field("kind") = Value::fromString(kind, mr);
    s.field("Ts") = Value::scalar(Ts, mr);
    return s;
}

// Cancel common roots of (num, den) within relative tolerance; returns the
// reduced (num, den) as real coefficient rows. Greedy: each pole claims the
// nearest surviving zero within tol — symmetric, so conjugate pairs cancel
// together and the rebuilt polynomials stay real.
std::pair<Value, Value>
cancelTf(const Value &numV, const Value &denV, keel::f64 tol,
         keel::Allocator *mr) {
    auto numS = stripLeading(coeffsReal(numV));
    auto denS = stripLeading(coeffsReal(denV));
    if (denS.empty())
        throw Error("minreal: denominator is identically zero",
                    0, 0, "minreal", "", "numkit:minreal:den");
    const keel::f64 numLead = numS.empty() ? 0.0 : numS.front();
    const keel::f64 denLead = denS.front();

    auto z = rootsComplex(numV, mr);
    auto p = rootsComplex(denV, mr);

    keel::Array<char> zused(z.size(), 0);
    keel::Array<keel::c64> remP, remZ;
    for (const auto &pj : p) {
        keel::i32 best = -1;
        keel::f64 bd = std::numeric_limits<keel::f64>::max();
        for (keel::usize i = 0; i < z.size(); ++i)
            if (!zused[i]) {
                const keel::f64 d = keel::abs(z[i] - pj);
                if (d < bd) { bd = d; best = keel::as<keel::i32>(i); }
            }
        if (best >= 0 && bd <= tol * std::max(1.0, keel::abs(pj))) zused[best] = 1;
        else remP.push_back(pj);
    }
    for (keel::usize i = 0; i < z.size(); ++i) if (!zused[i]) remZ.push_back(z[i]);

    auto nn = polyFromRoots(remZ);
    for (auto &c : nn) c *= numLead;
    auto dd = polyFromRoots(remP);
    for (auto &c : dd) c *= denLead;
    return {rowOfDoubles(nn, mr), rowOfDoubles(dd, mr)};
}

void minreal_argcheck(const Value &sys)
{
    if (!isKind(sys, "tf") && !isKind(sys, "zpk") && !isKind(sys, "ss"))
        throw Error("minreal: expected an LTI struct (tf / zpk / ss)",
                    0, 0, "minreal", "", "numkit:minreal:kind");
    if (isKind(sys, "ss")) {
        const Value &B = sys.field("B");
        const Value &C = sys.field("C");
        if (C.dims().rows() != 1 || B.dims().cols() != 1)
            throw Error("minreal: only SISO state-space is supported",
                        0, 0, "minreal", "", "numkit:minreal:mimo");
    }
}

} // anonymous

Value minreal(const Value &sys, keel::f64 tol, keel::Allocator *mr) {
    minreal_argcheck(sys);
    if (tol <= 0.0) tol = std::sqrt(std::numeric_limits<keel::f64>::epsilon());

    if (isKind(sys, "tf")) {
        auto [n2, d2] = cancelTf(sys.field("num"), sys.field("den"), tol, mr);
        Value s = taggedStruct("tf", sampleTimeOf(sys), mr);
        s.field("num") = std::move(n2);
        s.field("den") = std::move(d2);
        return s;
    }
    if (isKind(sys, "zpk")) {
        auto [num, den] = zp2tf(sys.field("z"), sys.field("p"), sys.field("k"), mr);
        auto [n2, d2] = cancelTf(num, den, tol, mr);
        Value s = taggedStruct("tf", sampleTimeOf(sys), mr);   // returns tf form
        s.field("num") = std::move(n2);
        s.field("den") = std::move(d2);
        return s;
    }
    if (isKind(sys, "ss")) {
        const Value &B = sys.field("B");
        const Value &C = sys.field("C");
        if (C.dims().rows() != 1 || B.dims().cols() != 1)
            throw Error("minreal: only SISO state-space is supported",
                        0, 0, "minreal", "", "numkit:minreal:mimo");
        auto [num, den] = ss2tf(sys.field("A"), B, C, sys.field("D"), 1, mr);
        auto [n2, d2] = cancelTf(num, den, tol, mr);
        StateSpace r = tf2ss(n2, d2, mr);
        Value s = taggedStruct("ss", sampleTimeOf(sys), mr);
        s.field("A") = std::move(r.A);
        s.field("B") = std::move(r.B);
        s.field("C") = std::move(r.C);
        s.field("D") = std::move(r.D);
        return s;
    }
    throw Error("minreal: expected an LTI struct (tf / zpk / ss)",
                0, 0, "minreal", "", "numkit:minreal:kind");
}

// No `*_reg` adapters are provided here — every conversion entry
// point (tf2zp, zp2tf in toolboxes/builtin; tf2ss, ss2tf in toolboxes/signal)
// is already registered as a builtin. The C++ APIs above are kept
// public so upcoming control cycles (e.g. step / lsim / feedback)
// can compose them internally without paying the runtime dispatch
// tax on the builtin lookup. (minreal's CallContext wrapper lives in
// lti_reg.cpp alongside the other model-level builtins.)

} // namespace numkit::control
