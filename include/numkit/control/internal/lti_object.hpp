/// @file lti_object.hpp
/// @ingroup group_control
// Internal LTI BuiltinClass object payload & accessors for tf/ss/zpk/frd.
// PRIVATE to control TUs — not part of the public library API.

#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/smart_ptr.h>
#include <keel/string.h>

#include <numkit/value/object.hpp>
#include <numkit/value/value.hpp>


namespace numkit::control::internal {

/// @addtogroup group_control
/// @{

struct LtiPayload final : public NativePayload
{
    Value data; // underlying struct storing kind, Ts, model arrays, and metadata

    explicit LtiPayload(Value d) : data(keel::move(d)) {}

    keel::Arc<NativePayload> clone(keel::Allocator *alloc = nullptr) const override
    {
        keel::Allocator *a = alloc ? alloc : keel::default_allocator();
        return keel::make_arc_in<LtiPayload>(a, data);
    }
};

inline bool isLtiClassName(keel::StringRef cls) noexcept
{
    return cls == "tf" || cls == "ss" || cls == "zpk" || cls == "frd";
}

inline bool isLtiObject(const Value &v) noexcept
{
    return v.isObject() && v.objectCount() == 1
           && isLtiClassName(v.objectClassNameRef());
}

inline Value wrapLtiObject(keel::StringRef clsName, Value structData, keel::Allocator *mr)
{
    keel::Allocator *a = mr ? mr : keel::default_allocator();
    auto state = keel::make_arc_in<ObjectState>(a, a);
    state->native = keel::make_arc_in<LtiPayload>(a, keel::move(structData));
    return Value::object(clsName, keel::move(state), /*isHandle=*/false, a);
}

inline bool isLtiModel(const Value &v) noexcept
{
    if (v.isObject() && v.objectCount() == 1 && isLtiClassName(v.objectClassNameRef())) {
        const ObjectState *st = v.objectStateConst();
        return st && st->native;
    }
    return v.isStruct() && !v.isStructArray() && v.numel() == 1 && v.hasField("kind");
}

inline const Value &ltiStruct(const Value &v) noexcept
{
    if (v.isObject() && v.objectCount() == 1 && isLtiClassName(v.objectClassNameRef())) {
        const ObjectState *st = v.objectStateConst();
        if (st && st->native)
            return static_cast<const LtiPayload *>(st->native.get())->data;
    }
    return v;
}

inline Value *ltiDataMutPtr(Value &v)
{
    if (v.isObject() && v.objectCount() == 1 && isLtiClassName(v.objectClassNameRef())) {
        ObjectState *st = v.objectStateMut();
        if (st && st->native)
            return &static_cast<LtiPayload *>(st->native.get())->data;
        return nullptr;
    }
    return nullptr;
}

inline keel::f64 ltiSampleTime(const Value &v, keel::f64 defaultTs = 0.0)
{
    if (!isLtiModel(v))
        return defaultTs;
    const Value &d = ltiStruct(v);
    if (d.hasField("Ts"))
        return d.field("Ts").toScalar();
    return 0.0;
}

inline keel::String ltiKindOf(const Value &v)
{
    if (!isLtiModel(v))
        return keel::String{};
    const Value &d = ltiStruct(v);
    if (!d.hasField("kind"))
        return keel::String{};
    const Value &kf = d.field("kind");
    if (!kf.isChar() && !kf.isString())
        return keel::String{};
    return kf.toString();
}

inline bool isLtiKind(const Value &v, keel::StringRef want)
{
    return ltiKindOf(v) == want;
}

/// @}

} // namespace numkit::control::internal
