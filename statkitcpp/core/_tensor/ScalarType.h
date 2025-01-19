#pragma once

#include <cstdint>
#include <cstddef>
#include <ostream>
#include <stdexcept>
#include <type_traits>

namespace statkitcpp {

#define SCALAR_TYPES(_) \
    _(int8_t, Char) \
    _(int16_t, Short) \
    _(int, Int) \
    _(int64_t, Long) \
    _(float, Float) \
    _(double, Double) \
    _(bool, Bool)

#define SCALAR_TYPES_NOBOOL(_) \
    _(int8_t, Char) \
    _(int16_t, Short) \
    _(int, Int) \
    _(int64_t, Long) \
    _(float, Float) \
    _(double, Double)

enum class ScalarType {
    #define DEFINE_ENUM_TYPES_(_1, n) n,
    SCALAR_TYPES(DEFINE_ENUM_TYPES_)
    #undef DEFINE_ENUM_TYPES_
    NumOptions
};

template <ScalarType N>
struct ScalarTypeToCPPType;

#define DEFINE_SCALARTOCPPTYPE(cpp_type, scalar_type) \
template <> \
struct ScalarTypeToCPPType<ScalarType::scalar_type> { \
    using type = cpp_type; \
    static type t; \
}; 

SCALAR_TYPES(DEFINE_SCALARTOCPPTYPE)
#undef DEFINE_SCALARTOCPPTYPE

template <ScalarType N>
using ScalarTypeToCPPTypeT = typename ScalarTypeToCPPType<N>::type;

template <typename T>
struct CppTypeToScalarType;

#define DEFINE_CPPTYPETOSCALARTYPE(cpp_type, scalar_type) \
template <> \
struct CppTypeToScalarType<cpp_type> \
    : std::integral_constant<ScalarType, ScalarType::scalar_type> { \
};

SCALAR_TYPES(DEFINE_CPPTYPETOSCALARTYPE)

#undef DEFINE_CPPTYPETOSCALARTYPE

inline const char* ToString(ScalarType t) {
#define DEFINE_NAME(_, name) \
    case ScalarType::name: \
        return #name;

    switch(t) {
        SCALAR_TYPES(DEFINE_NAME)
        default:
            return "UNKNOWN_SCALAR";
    }
#undef DEFINE_NAME
}

inline size_t ItemSize(ScalarType t) {
#define DEFINE_ITEMSIZE(ctype, name) \
    case ScalarType::name: \
        return sizeof(ctype);

    switch (t) {
        SCALAR_TYPES(DEFINE_ITEMSIZE)
        default:
            throw std::runtime_error{"Unknown ScalarType"};
    }
#undef DEFINE_ITEMSIZE
}

inline bool IsIntegral(ScalarType t) {
    bool is_integral = 
    (t == ScalarType::Char || t == ScalarType::Long ||
     t == ScalarType::Int || t == ScalarType::Short);
    return is_integral;
}

inline bool IsFloatingType(ScalarType t) {
    return (t == ScalarType::Double || t == ScalarType::Float);
}

inline bool CanCast(const ScalarType from, const ScalarType to) {
    if (IsFloatingType(from) && IsIntegral(to)) {
        return false;
    }
    if (from != ScalarType::Bool && to == ScalarType::Bool) {
        return false;
    }
    return true;
}

inline std::ostream& operator<<(
    std::ostream& stream,
    ScalarType scalar_type) {
    return stream << ToString(scalar_type);
}

ScalarType PromoteTypes(ScalarType a, ScalarType b);

std::string GetDTypeName(ScalarType scalar_type);

}