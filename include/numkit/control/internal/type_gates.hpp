/// @file type_gates.hpp
/// @ingroup group_control
// toolboxes/control/include/nk/control/internal/type_gates.hpp
//
// Shared <fn>_argcheck cores for toolboxes/control (argcheck
// separation, dev-docs/todo/argcheck_separation.md — boundary rule:
// raw arguments in, all message digestion here; call sites are one
// pass-through line). PRIVATE to control TUs — not part of the public
// library API.
//
// Rejection classes reference-probed against R2025b on 2026-09-18
// (build/probe9..11_control_ref.m). The reference coerces broad — and
// per-entry different — numeric classes: lyap/dlyap even take int8 and
// char; ctrb/obsv/acker reject integer matrices at the matmul layer;
// tf's coefficients reject logical while filt's numerator accepts even
// strings; lsim coerces int8 u but rejects int8 t. The gates below
// mirror those per-entry sets instead of one blanket rule.

#pragma once

#include <string>
#include <string_view>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/value_type.hpp>   // mtypeName (got-clauses)

namespace numkit::control::internal {

/// @addtogroup group_control
/// @{

/// Render "2x3x4" for got-clauses.
inline std::string shapeOf(const Value &v)
{
    const Dims &d = v.dims();
    std::string s = std::to_string(d.dim(0));
    for (int k = 1; k < d.ndim(); ++k) {
        s += "x";
        s += std::to_string(d.dim(k));
    }
    return s;
}

/// @brief Coercion-set gate: double / complex / single / logical /
/// integer / char pass (the element reads coerce exactly like the
/// reference's double() paths); cell / string / struct are the
/// inputType rejection class.
///
/// Used by the lyap/dlyap family where the reference accepts every
/// numeric class plus char (probed: lyap(int8 A), lyap('12' A) both
/// compute).
inline void requireCoercibleMatrix(const Value &v, std::string_view fn,
                                   std::string_view argName, int argNum)
{
    const ValueType t = v.type();
    if (t == ValueType::DOUBLE || t == ValueType::COMPLEX
        || t == ValueType::SINGLE || t == ValueType::LOGICAL
        || t == ValueType::CHAR || isIntegerType(t))
        return;
    throw Error(std::string(fn) + ": input " + std::to_string(argNum)
                    + " (" + std::string(argName) + ") must be a numeric "
                    + "or char matrix (got a " + shapeOf(v) + " "
                    + mtypeName(t) + " array).",
                0, 0, std::string(fn), "",
                "nk:" + std::string(fn) + ":inputType");
}

/// @brief Numeric-only gate (no char): for entries whose reference
/// rejects char/code-value coercion on a given argument (acker's pole
/// vector — MATLAB:zeros:invalidStringInput) while still coercing
/// single/logical/integer.
inline void requireNumericInput(const Value &v, std::string_view fn,
                                std::string_view argName, int argNum)
{
    const ValueType t = v.type();
    if (t == ValueType::DOUBLE || t == ValueType::COMPLEX
        || t == ValueType::SINGLE || t == ValueType::LOGICAL
        || isIntegerType(t))
        return;
    throw Error(std::string(fn) + ": input " + std::to_string(argNum)
                    + " (" + std::string(argName) + ") must be a numeric "
                    + "vector (got a " + shapeOf(v) + " " + mtypeName(t)
                    + " array).",
                0, 0, std::string(fn), "",
                "nk:" + std::string(fn) + ":inputType");
}

/// @brief Integer-matrix rejection for the ctrb/obsv/acker/ss2ss
/// family: the reference fails on integer operands inside its matmul
/// (MATLAB:mtimes:integerNotSupported / mrdivide:integerLinearAlgebra),
/// while double/single/logical/char coerce fine.
inline void requireNonIntegerMatrix(const Value &v, std::string_view fn,
                                    std::string_view argName, int argNum)
{
    if (!isIntegerType(v.type()))
        return;
    throw Error(std::string(fn) + ": input " + std::to_string(argNum)
                    + " (" + std::string(argName) + ") must not be an "
                    + "integer matrix — integer matrix products are "
                    + "unsupported; convert with double() (got a "
                    + shapeOf(v) + " " + mtypeName(v.type()) + " array).",
                0, 0, std::string(fn), "",
                "nk:" + std::string(fn) + ":integerUnsupported");
}

/// @brief Real-kernel stub for entries whose reference computes
/// complex results but whose kernel here is double-only
/// (lyap/dlyap A,Q; ctrb/obsv A,B,C; acker A; ss2ss T; lsim u,x0).
/// Rejecting honestly beats the silent imaginary-part drop the raw
/// elemAsDouble reads used to perform (bugs/opened/control/complex-kernels.md).
inline void requireRealKernelInput(const Value &v, std::string_view fn,
                                   std::string_view argName, int argNum)
{
    if (!v.isComplex())
        return;
    throw Error(std::string(fn) + ": input " + std::to_string(argNum)
                    + " (" + std::string(argName) + ") is complex — complex "
                    + "kernels are not supported yet; the reference "
                    + "computes a complex result here, so refusing beats "
                    + "returning a wrong real one (got a " + shapeOf(v)
                    + " complex array).",
                0, 0, std::string(fn), "",
                "nk:" + std::string(fn) + ":complexUnsupported");
}

/// @brief Single-result stub: acker with any single input returns a
/// single-class result in the reference; the kernel is double-only, so
/// single is rejected with its own identifier instead of a wrong-class
/// answer (bugs/opened/control/single-float-inputs.md).
inline void requireDoubleOnlyInput(const Value &v, std::string_view fn,
                                   std::string_view argName, int argNum)
{
    if (v.type() != ValueType::SINGLE)
        return;
    throw Error(std::string(fn) + ": input " + std::to_string(argNum)
                    + " (" + std::string(argName) + ") is single — single "
                    + "inputs produce a single result in the reference "
                    + "engine; convert with double() (got a " + shapeOf(v)
                    + " single array).",
                0, 0, std::string(fn), "",
                "nk:" + std::string(fn) + ":singleUnsupported");
}

/// @brief LTI-model gate for sys-args (predicates, gramians,
/// transforms): the argument must be a struct tagged kind ∈
/// {tf, zpk, ss}; frd is accepted only where the reference takes it
/// (issiso/isproper do; order/gram reject it — probed R2025b).
/// Everything else is the inputType rejection class (the reference
/// answers MATLAB:UndefinedFunction on untyped args).
inline void requireLtiModel(const Value &sys, std::string_view fn,
                            bool acceptFrD = false)
{
    if (sys.isStruct() && sys.hasField("kind")) {
        const std::string k = sys.field("kind").toString();
        if (k == "tf" || k == "zpk" || k == "ss" || (acceptFrD && k == "frd"))
            return;
    }
    std::string msg = std::string(fn) + ": input 1 (sys) must be an LTI "
                    + "model struct (tf/zpk/ss";
    if (acceptFrD) msg += "/frd";
    msg += ") — built by tf/zpk/ss";
    if (acceptFrD) msg += "/frd";
    msg += " (got ";
    msg += sys.isStruct() ? std::string("an untagged struct")
                          : ("a " + std::string(mtypeName(sys.type())));
    msg += " array).";
    throw Error(std::move(msg), 0, 0, std::string(fn), "",
                "nk:" + std::string(fn) + ":inputType");
}

/// @brief Coerce a coercible operand (single / logical / char /
/// integer; string too when the caller's probed contract allows it)
/// to a double matrix of the same shape; complex and non-coercible
/// classes (cell / struct) pass through UNCHANGED so the caller's
/// type gate voices the rejection. Callers gate the accepted classes
/// per-function first — the reference's coercion set differs per
/// entry (probed R2025b 2026-09-18).
inline Value coerceToDoubleMatrix(const Value &v,
                                  std::pmr::memory_resource *mr = nullptr)
{
    const ValueType t = v.type();
    if (t == ValueType::DOUBLE || t == ValueType::COMPLEX)
        return v;
    if (t == ValueType::STRING) {
        // String scalars coerce like char rows (filt's numerator).
        const std::string sv = v.toString();
        Value out = Value::matrix(1, sv.size(), ValueType::DOUBLE, mr);
        double *od = out.doubleDataMut();
        for (std::size_t i = 0; i < sv.size(); ++i)
            od[i] = static_cast<double>(static_cast<unsigned char>(sv[i]));
        return out;
    }
    if (t == ValueType::SINGLE || t == ValueType::LOGICAL
        || t == ValueType::CHAR || isIntegerType(t)) {
        Value out = Value::matrix(v.dims().rows(), v.dims().cols(),
                                  ValueType::DOUBLE, mr);
        const std::size_t n = v.numel();
        double *od = out.doubleDataMut();
        for (std::size_t i = 0; i < n; ++i)
            od[i] = v.elemAsDouble(i);
        return out;
    }
    return v;
}

/// @} group_control

} // namespace numkit::control::internal
