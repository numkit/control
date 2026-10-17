// toolboxes/control/src/response/response.cpp
//
// step / impulse / lsim — time-domain responses.
//
// For a continuous SISO model x' = Ax + Bu, y = Cx + Du, we
// discretise under a zero-order hold:
//     A_d = exp(A·dt),
//     B_d = integral_0^dt exp(A·τ) dτ · B
//         = first-block of exp([A B; 0 0]·dt)  (Van Loan).
// The matrix exponential uses 6th-order Padé with scaling and
// squaring — adequate for the well-conditioned A's seen in
// textbook control problems and avoids any external linalg dep.
//
// For a discrete model (Ts > 0) we just iterate the recurrence as-is.
// The simulation grid for continuous mode defaults to ≈ 8 / |Re λ_min|
// when the user passes no time argument; otherwise t is interpreted
// as either tFinal (scalar) or the explicit grid (vector).

#include <numkit/control/response/response.hpp>
#include <numkit/control/lti/lti.hpp>
#include <numkit/control/conversion/conversion.hpp>
#include <numkit/control/props/props.hpp>
#include <numkit/control/internal/numerics.hpp>
#include <numkit/control/internal/type_gates.hpp>

#include <numkit/builtin/polyfun.hpp>

// Compute-only TU: Value substrate + Error, no engine. The step/impulse/lsim
// builtins (CallContext wrappers) live in response_reg.cpp.
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <utility>
#include <vector>

namespace numkit::control {

namespace {

using Mat = internal::Mat;
using Vec = internal::Vec;
using internal::expm;
using internal::solveInPlace;

Mat zeros(size_t r, size_t c) { return Mat(r * c, 0.0); }
Mat eye(size_t n) { Mat I(n * n, 0.0); for (size_t i = 0; i < n; ++i) I[i * n + i] = 1.0; return I; }


// Bring an LTI sys (any kind) into ss-form coefficient buffers.
struct SS {
    size_t n;          // state dim
    Mat A;             // n×n
    Vec B;             // n
    Vec C;             // n
    double D;          // SISO
    double Ts;
};

SS toSSiso(const Value &sys, const char *fn, std::pmr::memory_resource *mr) {
    Value Av, Bv, Cv, Dv;
    double Ts = 0.0;
    if (sys.isStruct() && sys.hasField("kind")) {
        Ts = sys.field("Ts").toScalar();
        const std::string k = sys.field("kind").toString();
        if (k == "ss") {
            Av = sys.field("A"); Bv = sys.field("B");
            Cv = sys.field("C"); Dv = sys.field("D");
        } else if (k == "tf") {
            auto ss = tf2ss(sys.field("num"), sys.field("den"), mr);
            Av = std::move(ss.A); Bv = std::move(ss.B);
            Cv = std::move(ss.C); Dv = std::move(ss.D);
        } else if (k == "zpk") {
            auto [num, den] = zp2tf(sys.field("z"), sys.field("p"), sys.field("k"), mr);
            auto ss = tf2ss(num, den, mr);
            Av = std::move(ss.A); Bv = std::move(ss.B);
            Cv = std::move(ss.C); Dv = std::move(ss.D);
        } else {
            throw Error(std::string(fn) + ": input 1 (sys) has unknown LTI "
                        "kind '" + k + "'",
                        0, 0, fn, "", std::string("numkit:") + fn + ":kind");
        }
    } else {
        throw Error(std::string(fn) + ": input 1 (sys) must be an LTI struct "
                    "(tf/zpk/ss)",
                    0, 0, fn, "", std::string("numkit:") + fn + ":kind");
    }

    SS s;
    s.n = Av.dims().rows();
    s.Ts = Ts;
    s.A.resize(s.n * s.n);
    for (size_t i = 0; i < s.n * s.n; ++i) s.A[i] = Av.elemAsDouble(i);
    s.B.resize(s.n);
    for (size_t i = 0; i < s.n; ++i) s.B[i] = Bv.elemAsDouble(i);
    s.C.resize(s.n);
    // C is 1×n — column-major elem 0..n-1 are exactly the row.
    for (size_t i = 0; i < s.n; ++i) s.C[i] = Cv.elemAsDouble(i * 1 + 0);
    s.D = (Dv.numel() == 0) ? 0.0 : Dv.toScalar();
    return s;
}

// Van Loan ZOH: M = [[A,B],[0,0]], expm(M*dt) → top-left A_d, top-right B_d.
void zohDiscretise(const Mat &A, const Vec &B, size_t n, double dt,
                   Mat &Ad, Vec &Bd)
{
    const size_t m = n + 1;
    Mat M = zeros(m, m);
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < n; ++i)
            M[j * m + i] = A[j * n + i] * dt;
    for (size_t i = 0; i < n; ++i)
        M[n * m + i] = B[i] * dt;
    Mat E = expm(M, m);
    Ad.assign(n * n, 0.0);
    Bd.assign(n, 0.0);
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < n; ++i)
            Ad[j * n + i] = E[j * m + i];
    for (size_t i = 0; i < n; ++i) Bd[i] = E[n * m + i];
}

// Decide a default time grid for continuous models: ~8 time
// constants of the slowest stable mode, ~50 samples per slowest pole.
struct Grid { Vec t; double dt; };

Grid pickGrid(const Value &sys, std::pmr::memory_resource *mr) {
    // Find pole with smallest |real|, use that to set Tfinal; fastest
    // pole sets dt.
    Value pV = pole(sys, mr);
    const size_t np = pV.numel();
    double minAbsReal = 1.0;
    double maxAbsReal = 1.0;
    if (np > 0) {
        std::vector<std::complex<double>> ps(np);
        if (pV.type() == ValueType::COMPLEX) {
            const std::complex<double> *src = pV.complexData();
            for (size_t i = 0; i < np; ++i) ps[i] = src[i];
        } else {
            for (size_t i = 0; i < np; ++i)
                ps[i] = std::complex<double>(pV.elemAsDouble(i), 0.0);
        }
        bool anyStable = false;
        for (auto &p : ps) {
            const double ar = std::abs(p.real());
            const double mag = std::abs(p);
            if (p.real() < 0.0 && mag > 0.0) {
                if (!anyStable || ar < minAbsReal) minAbsReal = ar;
                anyStable = true;
            }
            if (mag > maxAbsReal) maxAbsReal = mag;
        }
        if (!anyStable) minAbsReal = 1.0;
    }
    // MATLAB convention: Tfinal corresponds to e^(-5.8) ~= 0.003 decay
    // of the slowest stable mode (~99.7% settled). Default sample
    // count ~= 127 for simple 1st-order cases; higher-order systems
    // get more samples via the maxAbsReal-based dt cap.
    const double Tfinal = 5.80251 / minAbsReal;
    const double dt = std::min(Tfinal / 126.0, 1.0 / (4.0 * maxAbsReal));
    size_t N = static_cast<size_t>(std::ceil(Tfinal / dt)) + 1;
    if (N < 60) N = 60;
    if (N > 1000) N = 1000;
    const double dtFinal = Tfinal / static_cast<double>(N - 1);
    Vec t(N);
    for (size_t i = 0; i < N; ++i) t[i] = i * dtFinal;
    return {t, dtFinal};
}

Vec readTimeArg(const Value &sys, const Value &tArg, std::pmr::memory_resource *mr)
{
    if (tArg.numel() == 0) {
        return pickGrid(sys, mr).t;
    }
    if (tArg.numel() == 1) {
        const double Tfinal = tArg.toScalar();
        const Grid g = pickGrid(sys, mr);
        // Re-tile at the same dt up to user's Tfinal.
        const double dt = g.dt;
        const size_t N = static_cast<size_t>(std::ceil(Tfinal / dt)) + 1;
        Vec t(N);
        for (size_t i = 0; i < N; ++i) t[i] = i * dt;
        return t;
    }
    Vec t(tArg.numel());
    for (size_t i = 0; i < tArg.numel(); ++i) t[i] = tArg.elemAsDouble(i);
    return t;
}

Value rowFromVec(const Vec &v, std::pmr::memory_resource *mr) {
    Value r = Value::matrix(v.size(), 1, ValueType::DOUBLE, mr);
    if (!v.empty()) std::copy(v.begin(), v.end(), r.doubleDataMut());
    return r;
}

// State trajectory → MATLAB-style N×n matrix (rows = time, cols = states).
// xTraj is already column-major time×state, so it copies straight in.
Value matFromTraj(const Vec &xTraj, size_t N, size_t n, std::pmr::memory_resource *mr) {
    Value m = Value::matrix(N, n, ValueType::DOUBLE, mr);
    if (!xTraj.empty()) std::copy(xTraj.begin(), xTraj.end(), m.doubleDataMut());
    return m;
}

// Run the discrete simulation for a SISO system on input u[0..N-1]
// (length N matches t). Returns y[0..N-1].
// xTraj (optional, size N*n): the state trajectory in column-major
// time×state layout, xTraj[s*N + k] = state s at time t[k].
Vec simulate(const SS &sys, const Vec &t, const Vec &u, const Vec &x0,
             Vec *xTraj = nullptr)
{
    const size_t N = t.size();
    const size_t n = sys.n;
    Vec x = x0;
    if (x.size() != n) x.assign(n, 0.0);
    Vec y(N, 0.0);
    if (xTraj) xTraj->assign(N * n, 0.0);
    auto record = [&](size_t k) {
        if (xTraj) for (size_t i = 0; i < n; ++i) (*xTraj)[i * N + k] = x[i];
    };

    if (sys.Ts > 0.0) {
        // Already discrete.
        for (size_t k = 0; k < N; ++k) {
            record(k);
            double yk = sys.D * u[k];
            for (size_t i = 0; i < n; ++i) yk += sys.C[i] * x[i];
            y[k] = yk;
            // x_{k+1} = A·x + B·u
            Vec xn(n, 0.0);
            for (size_t i = 0; i < n; ++i) {
                double s = sys.B[i] * u[k];
                for (size_t j = 0; j < n; ++j)
                    s += sys.A[j * n + i] * x[j];
                xn[i] = s;
            }
            x = std::move(xn);
        }
        return y;
    }

    // Continuous: ZOH-discretise per actual dt. If t is uniform, do
    // one expm; otherwise re-discretise per step (more expensive but
    // robust).
    bool uniform = true;
    double dt = (N >= 2) ? (t[1] - t[0]) : 0.0;
    for (size_t k = 2; k < N && uniform; ++k)
        if (std::abs((t[k] - t[k - 1]) - dt) > 1e-12 * std::max(1.0, std::abs(dt)))
            uniform = false;

    if (uniform && dt > 0.0) {
        Mat Ad; Vec Bd;
        zohDiscretise(sys.A, sys.B, n, dt, Ad, Bd);
        for (size_t k = 0; k < N; ++k) {
            record(k);
            double yk = sys.D * u[k];
            for (size_t i = 0; i < n; ++i) yk += sys.C[i] * x[i];
            y[k] = yk;
            if (k + 1 < N) {
                Vec xn(n, 0.0);
                for (size_t i = 0; i < n; ++i) {
                    double s = Bd[i] * u[k];
                    for (size_t j = 0; j < n; ++j)
                        s += Ad[j * n + i] * x[j];
                    xn[i] = s;
                }
                x = std::move(xn);
            }
        }
    } else {
        // Non-uniform grid: per-step expm.
        for (size_t k = 0; k < N; ++k) {
            record(k);
            double yk = sys.D * u[k];
            for (size_t i = 0; i < n; ++i) yk += sys.C[i] * x[i];
            y[k] = yk;
            if (k + 1 < N) {
                const double dtk = t[k + 1] - t[k];
                Mat Ad; Vec Bd;
                zohDiscretise(sys.A, sys.B, n, dtk, Ad, Bd);
                Vec xn(n, 0.0);
                for (size_t i = 0; i < n; ++i) {
                    double s = Bd[i] * u[k];
                    for (size_t j = 0; j < n; ++j)
                        s += Ad[j * n + i] * x[j];
                    xn[i] = s;
                }
                x = std::move(xn);
            }
        }
    }
    return y;
}

} // anonymous

std::pair<Value, Value>
step_response(const Value &sys, const Value &tArg,
              Value *xOut, std::pmr::memory_resource *mr)
{
    SS s = toSSiso(sys, "step", mr);
    Vec t = readTimeArg(sys, tArg, mr);
    Vec u(t.size(), 1.0);   // unit step
    Vec x0(s.n, 0.0);
    Vec xTraj;
    Vec y = simulate(s, t, u, x0, xOut ? &xTraj : nullptr);
    if (xOut) *xOut = matFromTraj(xTraj, t.size(), s.n, mr);
    return {rowFromVec(y, mr), rowFromVec(t, mr)};
}

std::pair<Value, Value>
impulse_response(const Value &sys, const Value &tArg,
                 Value *xOut, std::pmr::memory_resource *mr)
{
    SS s = toSSiso(sys, "impulse", mr);
    Vec t = readTimeArg(sys, tArg, mr);
    Vec x0(s.n, 0.0);
    Vec u(t.size(), 0.0);

    if (s.Ts > 0.0) {
        // Discrete: u[0] = 1, then 0.
        if (!u.empty()) u[0] = 1.0;
    } else {
        // Continuous: impulse ⇒ x(0+) = B (after differentiating
        // through the delta in u(t)). Simulate with u ≡ 0 and IC = B.
        x0 = s.B;
    }
    Vec xTraj;
    Vec y = simulate(s, t, u, x0, xOut ? &xTraj : nullptr);
    if (xOut) *xOut = matFromTraj(xTraj, t.size(), s.n, mr);
    return {rowFromVec(y, mr), rowFromVec(t, mr)};
}

Value lsim(const Value &sys, const Value &uIn, const Value &tIn,
           const Value &x0In,
           Value *xOut, std::pmr::memory_resource *mr)
{
    SS s = toSSiso(sys, "lsim", mr);

    // lsim argcheck (argcheck separation). Classes probed R2025b:
    // u coerces every NUMERIC class (int8 u accepted) but char/cell
    // are rejected (Control:analysis:rfinputs02/18), and complex u
    // yields a complex response; t must be a real
    // double/single vector (int8/complex t rejected — rfinputs14); x0
    // coerces numerics, complex x0 gives a complex response,
    // and a non-empty x0 with the wrong length is a hard error
    // (reference: timeresp1).
    internal::requireNumericInput(uIn, "lsim", "u", 2);
    {
        const ValueType tt = tIn.type();
        if (tt != ValueType::DOUBLE && tt != ValueType::SINGLE)
            throw Error(std::string("lsim: input 3 (t) must be a real ")
                        + "double or single time vector (got a "
                        + internal::shapeOf(tIn) + " " + mtypeName(tt)
                        + " array).",
                        0, 0, "lsim", "", "numkit:lsim:t");
    }
    internal::requireNumericInput(x0In, "lsim", "x0", 4);

    if (tIn.numel() < 2)
        throw Error("lsim: t must have at least 2 samples",
                    0, 0, "lsim", "", "numkit:lsim:t");
    Vec t(tIn.numel());
    for (size_t i = 0; i < tIn.numel(); ++i) t[i] = tIn.elemAsDouble(i);
    if (uIn.numel() != t.size())
        throw Error("lsim: u and t must be the same length",
                    0, 0, "lsim", "", "numkit:lsim:size");
    if (x0In.numel() != 0 && x0In.numel() != s.n)
        throw Error("lsim: input 4 (x0) must have one entry per state "
                    "(got " + std::to_string(x0In.numel())
                    + " entries for " + std::to_string(s.n) + " states).",
                    0, 0, "lsim", "", "numkit:lsim:x0");

    if (uIn.isComplex() || x0In.isComplex()) {
        const size_t N = t.size();
        Vec uRe(N, 0.0), uIm(N, 0.0);
        if (uIn.isComplex()) {
            const std::complex<double> *cd = uIn.complexData();
            for (size_t i = 0; i < N; ++i) { uRe[i] = cd[i].real(); uIm[i] = cd[i].imag(); }
        } else {
            for (size_t i = 0; i < N; ++i) uRe[i] = uIn.elemAsDouble(i);
        }
        Vec x0Re(s.n, 0.0), x0Im(s.n, 0.0);
        if (x0In.numel() != 0) {
            if (x0In.isComplex()) {
                const std::complex<double> *cd = x0In.complexData();
                for (size_t i = 0; i < s.n; ++i) { x0Re[i] = cd[i].real(); x0Im[i] = cd[i].imag(); }
            } else {
                for (size_t i = 0; i < s.n; ++i) x0Re[i] = x0In.elemAsDouble(i);
            }
        }
        Vec xTrajRe, xTrajIm;
        Vec yRe = simulate(s, t, uRe, x0Re, xOut ? &xTrajRe : nullptr);
        Vec yIm = simulate(s, t, uIm, x0Im, xOut ? &xTrajIm : nullptr);
        if (xOut) {
            Value xm = Value::matrix(N, s.n, ValueType::COMPLEX, mr);
            std::complex<double> *xd = xm.complexDataMut();
            for (size_t i = 0; i < N * s.n; ++i) xd[i] = std::complex<double>(xTrajRe[i], xTrajIm[i]);
            *xOut = std::move(xm);
        }
        Value yOut = Value::matrix(N, 1, ValueType::COMPLEX, mr);
        std::complex<double> *yd = yOut.complexDataMut();
        for (size_t i = 0; i < N; ++i) yd[i] = std::complex<double>(yRe[i], yIm[i]);
        return yOut;
    }

    Vec u(t.size());
    for (size_t i = 0; i < t.size(); ++i) u[i] = uIn.elemAsDouble(i);
    Vec x0(s.n, 0.0);
    if (x0In.numel() != 0) {
        for (size_t i = 0; i < s.n; ++i) x0[i] = x0In.elemAsDouble(i);
    }
    Vec xTraj;
    Vec y = simulate(s, t, u, x0, xOut ? &xTraj : nullptr);
    if (xOut) *xOut = matFromTraj(xTraj, t.size(), s.n, mr);
    return rowFromVec(y, mr);
}

std::pair<Value, Value>
initial_response(const Value &sys, const Value &x0In, const Value &tArg,
                 Value *xOut, std::pmr::memory_resource *mr)
{
    SS s = toSSiso(sys, "initial", mr);
    Vec t = readTimeArg(sys, tArg, mr);
    Vec u(t.size(), 0.0);            // zero input
    Vec x0(s.n, 0.0);
    if (x0In.numel() == s.n)
        for (size_t i = 0; i < s.n; ++i) x0[i] = x0In.elemAsDouble(i);
    else if (x0In.numel() != 0)
        throw Error("initial: x0 length must equal the number of states",
                    0, 0, "initial", "", "numkit:initial:x0");
    Vec xTraj;
    Vec y = simulate(s, t, u, x0, xOut ? &xTraj : nullptr);
    if (xOut) *xOut = matFromTraj(xTraj, t.size(), s.n, mr);
    return {rowFromVec(y, mr), rowFromVec(t, mr)};
}

} // namespace numkit::control
